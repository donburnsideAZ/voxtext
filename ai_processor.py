"""
Voxtext manifest builder.

Voxtext does not translate anything. This module only prepares the small
JSON "manifest" files used by the optional Translation Manifest Delivery
feature (see TRANSLATION_MANIFEST_DELIVERY.md and MANIFEST_SCHEMA.txt):

  * a per-video manifest holding the segment timings and the segment text,
    in the source language, ready for an outside automation to pick up
  * a job manifest listing the per-video manifests from one run

It also holds the list of target languages offered in the UI. Everything
here is plain data-shaping: no network calls, no prompts, no clipboard.
"""

import json
from pathlib import Path


# ----------------------------------------------------------------------
# Segment tagging
# ----------------------------------------------------------------------
# Each segment is emitted as:   [[N]] text of the segment
# The [[N]] tag is stable and numeric. The downstream flow matches its
# output back to the ORIGINAL timings by N, so a dropped or merged line
# can be detected instead of shipping a broken caption file.

def segments_to_tagged_text(segments):
    """Turn Whisper segments into tagged [[id]] text lines."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = (seg.get("text") or "").strip()
        lines.append(f"[[{i}]] {text}")
    return "\n".join(lines)



# ----------------------------------------------------------------------
# Translation manifests — handoff to an automated translation flow
# ----------------------------------------------------------------------
# Voxtext-side half of the manifest contract in MANIFEST_SCHEMA.txt. The
# per-video manifest carries the segment text (tagged [[id]] form) plus
# the timings, the target language, and an optional context hint, so the
# flow on the other end has everything it needs without talking to Voxtext.

def _format_manifest_timestamp(seconds):
    """
    Format seconds as Voxtext's exact VTT timestamp — zero-padded
    HH:MM:SS.mmm, '.' separator, milliseconds TRUNCATED (not rounded) to
    match TranscriptionWorker._ts(comma=False) byte-for-byte -- rounding
    here previously caused occasional 1ms drift from Voxtext's own VTT
    output on real translated files.
    """
    seconds = seconds or 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_manifest_segments(segments):
    """
    Convert Whisper segments into the manifest's {id, start, end} shape —
    1-based, sequential ids, same numbering as segments_to_tagged_text()
    (which carries the text side, via translatePrompt) so the ids in the
    text and the ids in the timings always agree. start/end are pre-formatted HH:MM:SS.mmm strings — the flow
    consumes these directly when it writes the output VTT.
    """
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append({
            "id": i,
            "start": _format_manifest_timestamp(seg.get("start", 0.0)),
            "end": _format_manifest_timestamp(seg.get("end", 0.0)),
        })
    return out


def build_translation_manifest(video_id, course_name, source_language,
                                target_language, segments, source_folder=None,
                                context_hint=""):
    """
    Build one per-video manifest dict — the shape actually tested and
    working against a live SharePoint/OneDrive + Power Automate flow (courseName
    / sourceLanguage / targetLanguage / sourceFolder / sourceFileName /
    translatePrompt / segments). Written next to the source video (not the
    watched job manifest folder) so it never self-triggers the flow;
    see build_job_manifest() for the file that actually lands there.

    translatePrompt is the segment text in [[id]] tagged form, as built
    by segments_to_tagged_text() — pre-built so the flow doesn't need its
    own compose step. It is in the source language and contains no
    instructions; the flow adds those. sourceFileName is the source-language VTT transcript Voxtext
    writes alongside the video (<video_id>_transcript.vtt).

    context_hint is optional register/terminology guidance from the
    Voxtext UI (e.g. "compliance training video"). It's prepended to
    translatePrompt as a single "Content context: ..." line, so it rides
    along with the existing field and needs no flow-side change.
    Omitted entirely when blank.

    source_folder (already SharePoint-relative — see
    sharepoint_relative_path()) is what the flow uses as the destination
    folder for the translated output, so it rides on every per-video
    manifest as well as the job manifest. Omitted entirely when the
    caller couldn't compute it.
    """
    manifest = {
        "courseName": course_name,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
    }
    if source_folder:
        manifest["sourceFolder"] = source_folder
    manifest["sourceFileName"] = f"{video_id}_transcript.vtt"
    context_line = f"Content context: {context_hint.strip()}\n\n" if context_hint.strip() else ""
    manifest["translatePrompt"] = context_line + segments_to_tagged_text(segments)
    manifest["segments"] = build_manifest_segments(segments)
    return manifest


def write_manifest_file(manifest, out_path):
    """
    Write a manifest (per-video or job-level) as JSON, deleting any
    existing file first rather than overwriting in place — per
    MANIFEST_SCHEMA.txt, SharePoint/OneDrive's "file created" trigger can
    register an in-place overwrite as a *modification* and silently skip
    the flow. Returns the Path written.
    """
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out_path


def sharepoint_relative_path(local_folder, sync_root_local, sync_root_sharepoint):
    """
    Convert a local OneDrive-synced folder path into the SharePoint
    document-library-relative path the flow's "Get file content by path"
    step actually needs (e.g. "/Shared Documents/Training/Onboarding").

    Two anchors are required, both set in Options — Translation Manifest
    Delivery, because OneDrive's local sync-folder naming (e.g.
    "Contoso - Training") doesn't cleanly map onto the SharePoint
    library path (e.g. "/Shared Documents/Training") by just swapping a
    fixed prefix -- the two can diverge mid-folder-name. So the user picks
    the local folder (sync_root_local) AND types what it equals on
    SharePoint (sync_root_sharepoint); everything in local_folder past
    sync_root_local is appended onto sync_root_sharepoint as-is.

    Returns None (rather than guessing) if either anchor isn't set or
    local_folder isn't under sync_root_local — callers should treat that
    as "can't compute sourceFolder" and warn, not silently send a bad
    path.
    """
    if not sync_root_local or not sync_root_sharepoint or not local_folder:
        return None
    try:
        rel = Path(local_folder).resolve().relative_to(Path(sync_root_local).resolve())
    except (ValueError, OSError):
        return None
    prefix = sync_root_sharepoint.rstrip("/")
    rel_str = str(rel).replace("\\", "/")
    if rel_str == ".":
        return prefix
    return f"{prefix}/{rel_str}"


def build_job_manifest(course_name, manifest_filenames, source_folder=None):
    """
    Build the batch-level job manifest — the ONLY file Voxtext drops into
    the watched job manifest folder itself (per-video manifests stay
    in the source folder alongside their video, so they never land in the
    watched folder and self-trigger the flow). The flow's outer "For each"
    loop iterates manifestFiles, one pass per video; it builds each
    per-video manifest's SharePoint path
    from this sourceFolder + filename, so it must already be in
    SharePoint-relative form (see sharepoint_relative_path()) — a raw
    local Windows/Mac path here makes SharePoint-based flows fail.

    source_folder is omitted entirely when the caller couldn't compute it
    (no sync root configured, or the run's folder isn't under it) rather
    than sending something wrong.
    """
    job = {
        "courseName": course_name,
        "manifestFiles": list(manifest_filenames),
    }
    if source_folder:
        job["sourceFolder"] = source_folder
    return job


# ----------------------------------------------------------------------
# Target languages
# ----------------------------------------------------------------------

# Target languages offered in the UI.
# Portuguese removed — pt-BR and pt-PT are different enough that
# a single "Portuguese" entry produces inconsistent results.
# Add back as separate entries (pt-BR / pt-PT) when ready.
LANGUAGES = {
    "fr": "French (France)",
    "es": "Spanish (Latin America)",
    "de": "German",
    "it": "Italian",
}
