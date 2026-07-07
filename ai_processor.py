"""
Voxtext AI Processor — pluggable, bring-your-own-agent text operations.

Two operations:
  • format    — clean a raw transcript into readable prose (TXT / HTML / MD only)
  • translate — translate a transcript while preserving segment timing (VTT)

Two backends behind one interface:
  • ClipboardBackend — works TODAY, no API key. Builds a prompt, hands it
    to the clipboard for the user to paste into their sanctioned agent
    (Copilot, Claude, ChatGPT, etc.), then parses the pasted-back result.
  • ApiBackend       — STUB. Snaps in when an endpoint is sanctioned.
    Same interface, no UI changes required.

Design note: the prompt engineering lives HERE, not in the user's head.

IMPORTANT — two very different jobs, two very different approaches:

  format    — sends the full joined transcript as plain prose. The agent
              rewrites it for readability. Segment tags are NOT used because
              restructuring prose across Whisper's 5-second chunks is
              meaningless. Result is applied only to prose formats
              (TXT, HTML, MD) — never to SRT or VTT, where word-for-word
              accuracy tied to timestamps is required.

  translate — sends segment-tagged text. The agent translates each line
              in place, tags preserved, so timing survives the round-trip.
              Always written to VTT — timing is the whole point.
"""

import re
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Segment tagging — used for TRANSLATE only
# ----------------------------------------------------------------------
# Each segment is emitted as:   [[N]] text of the segment
# The [[N]] tag is stable, numeric, and unlikely to be reformatted by an
# LLM. We reattach translations to ORIGINAL timings by N, so even if the
# agent reorders or drops lines we can detect it rather than ship a
# broken VTT.

_TAG_RE = re.compile(r"^\s*\[\[(\d+)\]\]\s*(.*)$")


def segments_to_tagged_text(segments):
    """Turn Whisper segments into tagged, agent-ready lines (translate only)."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip()
        lines.append(f"[[{i}]] {text}")
    return "\n".join(lines)


def segments_to_plain_text(segments):
    """Join all segment text into a single prose block (format only)."""
    return " ".join((seg.get("text") or "").strip() for seg in segments).strip()


def parse_tagged_text(tagged_text):
    """
    Parse agent output back into {segment_number: translated_text}.
    Tolerant of blank lines and stray prose; only [[N]] lines are kept.
    """
    out = {}
    for line in tagged_text.splitlines():
        m = _TAG_RE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


@dataclass
class ProcessResult:
    """Result of applying agent output back onto the original segments."""
    segments: list = field(default_factory=list)  # new segments (timing preserved)
    text: str = ""                                  # joined plain text
    missing_ids: list = field(default_factory=list) # segments the agent dropped
    extra_ids: list = field(default_factory=list)    # ids the agent invented
    ok: bool = True


def apply_to_segments(original_segments, parsed_map):
    """
    Reattach parsed agent text to ORIGINAL segment timings by number.
    Flags any mismatch instead of silently corrupting the VTT.
    Used by TRANSLATE only.
    """
    expected = set(range(1, len(original_segments) + 1))
    got = set(parsed_map.keys())

    missing = sorted(expected - got)
    extra = sorted(got - expected)

    new_segments = []
    for i, seg in enumerate(original_segments, start=1):
        new_seg = dict(seg)
        if i in parsed_map:
            new_seg["text"] = parsed_map[i]
        # if missing, keep original text as a safe fallback
        new_segments.append(new_seg)

    text = " ".join(s.get("text", "").strip() for s in new_segments).strip()

    return ProcessResult(
        segments=new_segments,
        text=text,
        missing_ids=missing,
        extra_ids=extra,
        ok=(not missing and not extra),
    )


def plain_text_result(original_segments, cleaned_text):
    """
    Wrap cleaned prose text in a ProcessResult.
    Segments retain original timing; text field holds the cleaned prose.
    Used by FORMAT only — SRT/VTT are never touched by this path.
    """
    return ProcessResult(
        segments=original_segments,   # original timing preserved, untouched
        text=cleaned_text.strip(),
        missing_ids=[],
        extra_ids=[],
        ok=True,
    )


# ----------------------------------------------------------------------
# VTT file parser
# ----------------------------------------------------------------------
# Reads an externally-authored WebVTT file and returns the same
# [{start, end, text}] segment list that Whisper produces, so the
# existing translate pipeline can work on it unchanged.

_VTT_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
)
# Also accept the comma variant (SRT-style) in case someone has one
_VTT_TIMESTAMP_COMMA_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _vtt_ts_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt_file(path):
    """
    Parse a WebVTT file into Whisper-compatible segments:
        [{"start": float, "end": float, "text": str}, ...]

    Handles:
    - WEBVTT header (required) and optional STYLE / NOTE / REGION blocks
    - Cue identifiers (numeric or named) on the line before the timestamp
    - Multi-line cue text (joined with a space)
    - Cue settings on the timestamp line (line:80% etc.) — stripped cleanly
    - <tag> markup inside cue text — stripped out

    Raises ValueError if the file doesn't look like a valid VTT.
    """
    with open(path, encoding="utf-8-sig") as f:   # utf-8-sig strips BOM if present
        raw = f.read()

    lines = raw.splitlines()
    if not lines or not lines[0].strip().startswith("WEBVTT"):
        raise ValueError("Not a valid WebVTT file — missing WEBVTT header.")

    segments = []
    i = 1  # skip the WEBVTT line
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Skip blank lines, NOTE blocks, STYLE blocks, REGION blocks
        if not line:
            i += 1
            continue
        if line.startswith(("NOTE", "STYLE", "REGION")):
            # Skip until the next blank line
            i += 1
            while i < n and lines[i].strip():
                i += 1
            continue

        # Check whether this line IS a timestamp, or the NEXT line is
        # (allowing for an optional cue identifier line first).
        ts_line = None
        ts_match = _VTT_TIMESTAMP_COMMA_RE.search(line)
        if ts_match:
            ts_line = line
        else:
            # Could be a cue identifier — peek ahead
            if i + 1 < n:
                next_line = lines[i + 1].strip()
                ts_match = _VTT_TIMESTAMP_COMMA_RE.search(next_line)
                if ts_match:
                    i += 1          # skip the identifier
                    ts_line = next_line

        if ts_line is None or ts_match is None:
            i += 1
            continue

        start = _vtt_ts_to_seconds(*ts_match.group(1, 2, 3, 4))
        end   = _vtt_ts_to_seconds(*ts_match.group(5, 6, 7, 8))
        i += 1

        # Collect cue payload lines until the next blank line
        text_lines = []
        while i < n and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        # Join, strip inline <tags> (e.g. <c>, <b>, speaker tags)
        raw_text = " ".join(text_lines)
        clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()

        if clean_text:
            segments.append({"start": start, "end": end, "text": clean_text})

    if not segments:
        raise ValueError("No cues found in the VTT file — is it empty?")

    return segments


# ----------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------

# Languages offered in the UI.
# Keep this list to languages with strong LLM support.
# Portuguese removed — pt-BR and pt-PT are different enough that
# a single "Portuguese" entry produces inconsistent results.
# Add back as separate entries (pt-BR / pt-PT) when ready.
LANGUAGES = {
    "fr": "French (France)",
    "es": "Spanish (Latin America)",
    "de": "German",
    "it": "Italian",
}

# Language-specific guidance injected into translate prompts.
# Keeps the main prompt clean while letting us tune per-language register.
_LANG_GUIDANCE = {
    "fr": (
        "Use contemporary French as spoken and written in France today. "
        "Avoid dated, overly formal, or bureaucratic phrasing. "
        "Prefer natural spoken register over stiff written register."
    ),
    "es": (
        "Use modern, neutral Spanish suitable for a broad Latin American audience. "
        "Avoid Spain-specific vocabulary (vosotros, coger, ordenador, etc.). "
        "Prefer clear, contemporary phrasing over overly formal or literal translation."
    ),
    "de": (
        "Use clear, modern standard German (Hochdeutsch). "
        "Avoid overly bureaucratic or stilted phrasing. "
        "Prefer natural spoken register."
    ),
    "it": (
        "Use clear, contemporary Italian. "
        "Avoid overly formal or archaic phrasing. "
        "Prefer natural, modern register."
    ),
}


def build_translate_prompt(tagged_text, target_lang_code, context_hint=""):
    """
    Build a translation prompt for the segment-tagged wire format.
    Timing is preserved by the [[N]] tags — the agent must not alter them.
    """
    lang_name = LANGUAGES.get(target_lang_code, target_lang_code)
    lang_guidance = _LANG_GUIDANCE.get(target_lang_code, "")

    # Context line: user-supplied hint wins; otherwise omit entirely.
    # We do NOT assume any content type — the user knows what the file is.
    context_line = f"\nContent context: {context_hint.strip()}" if context_hint.strip() else ""

    return f"""You are a professional localization translator. Translate the
following transcript into {lang_name}.{context_line}

Language guidance: {lang_guidance}

STRICT RULES — follow exactly:
1. Each line begins with a tag like [[1]], [[2]], etc. KEEP every tag
   exactly as written, in the same order.
2. Translate ONLY the text after each tag. Do not translate the tags.
3. Do NOT merge, split, add, or remove lines. One tag in = one tag out.
4. Do NOT renumber. [[7]] stays [[7]].
5. Output ONLY the tagged lines. No preamble, no notes, no explanation.

Transcript to translate:

{tagged_text}
"""


def build_format_prompt(plain_text, context_hint=""):
    """
    Build a readability prompt for plain prose text.
    This operates on the full joined transcript — NOT on segment-tagged lines.
    The result is used for TXT, HTML, and MD output only.
    SRT and VTT are never passed through this path.

    Goal: make the transcript easy to read without changing any words.
    Punctuation, capitalization, and paragraph breaks only.

    context_hint lets the caller describe what kind of content this is
    (e.g. "podcast interview", "lecture recording", "training narration").
    If blank, the prompt stays fully content-neutral.
    """
    context_line = (
        f"\nContent type: {context_hint.strip()}"
        if context_hint.strip()
        else ""
    )

    return f"""You are a transcript formatter. The following is a raw automatic
transcription of spoken audio.{context_line}

Your job is to make this transcript easy to read. You are NOT editing
the content — every word stays exactly as transcribed.

WHAT TO DO:
- Add correct punctuation (periods, commas, question marks, etc.)
- Fix capitalization (sentences, proper nouns, the word "I")
- Fix obvious transcription errors where a misheard word is clearly wrong
  (e.g. "Monterray" → "Monterey", "teh" → "the")
- Break the continuous wall of text into paragraphs where the topic or
  speaker naturally shifts — use your judgment, don't over-break it

WHAT NOT TO DO:
- Do not remove any words — not fillers, not false starts, not anything
- Do not paraphrase, summarize, or reword anything
- Do not add words that were not in the original
- Do not add headers, bullet points, or any formatting markup
- Do not translate

OUTPUT:
Return only the formatted transcript. No preamble, no explanation,
no notes about what you changed.

Raw transcript:

{plain_text}
"""


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

class ClipboardBackend:
    """
    Works today. No credentials. The UI drives this:
      1. .build_outgoing(...) -> prompt string to copy to clipboard
      2. user pastes into their agent, copies the reply
      3. .parse_incoming(...) -> ProcessResult applied to original segments

    For FORMAT operations, parse_incoming expects plain prose back
    (not tagged lines) and wraps it in a ProcessResult via plain_text_result.

    For TRANSLATE operations, parse_incoming expects tagged lines back
    and reattaches them to original segment timings.
    """
    name = "clipboard"
    requires_paste_back = True

    def build_outgoing(self, segments, operation, target_lang_code=None,
                       context_hint=""):
        if operation == "translate":
            tagged = segments_to_tagged_text(segments)
            return build_translate_prompt(tagged, target_lang_code, context_hint)
        elif operation == "format":
            plain = segments_to_plain_text(segments)
            return build_format_prompt(plain, context_hint)
        raise ValueError(f"Unknown operation: {operation}")

    def parse_incoming(self, original_segments, pasted_text, operation="translate"):
        if operation == "format":
            # Plain prose back — no tag parsing needed
            return plain_text_result(original_segments, pasted_text)
        else:
            # Tagged lines back — reattach to original timings
            parsed = parse_tagged_text(pasted_text)
            return apply_to_segments(original_segments, parsed)


class ApiBackend:
    """
    STUB — snaps in when an endpoint is sanctioned.
    Same interface as ClipboardBackend but does the round-trip invisibly:
    no paste-back, just call the endpoint and return a ProcessResult.
    """
    name = "api"
    requires_paste_back = False

    def __init__(self, endpoint=None, credential_provider=None):
        # credential_provider is a callable so keys never live in this code
        self.endpoint = endpoint
        self.credential_provider = credential_provider

    def process(self, segments, operation, target_lang_code=None,
                context_hint=""):
        raise NotImplementedError(
            "API backend not yet enabled. Use clipboard mode for now."
        )
