"""
Voxtext - Simple, Local Audio/Video Transcription
Part of the Vox Suite - From the makers of Voxsmith

Powered by OpenAI Whisper
100% local processing - No API keys - No subscriptions
Your privacy matters.

PyQt6 Version - Cross-platform native appearance

TO HIDE TERMINAL WINDOW:
When packaging with PyInstaller, use: --noconsole flag
Example: pyinstaller --onefile --noconsole --name Voxtext voxtext_pyqt.py
"""

import sys
import os
import threading
import multiprocessing
from pathlib import Path
import json
import time
import tempfile
import webbrowser

import ai_processor

# ---------------------------------------------------------------------------
# Windowed-build stdout/stderr guard  (fixes "'NoneType' has no attribute 'write'")
# ---------------------------------------------------------------------------
# PyInstaller's --noconsole / --windowed mode runs with no console attached, so
# sys.stdout and sys.stderr are None. Whisper draws a tqdm progress bar during
# transcription, and tqdm writes to sys.stderr — which crashes immediately with
# "'NoneType' object has no attribute 'write'" the moment Transcribe is clicked.
# Point both streams at os.devnull when they're missing so those writes go
# harmlessly nowhere. (Only triggers in the packaged app; source runs are fine.)
for _stream in ("stdout", "stderr"):
    if getattr(sys, _stream, None) is None:
        try:
            setattr(sys, _stream, open(os.devnull, "w"))
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Windows: suppress the console window ffmpeg/whisper flash on screen
# ---------------------------------------------------------------------------
# --noconsole means Voxtext itself has no console — but Whisper shells out to
# ffmpeg via subprocess for every file it loads, and Windows spawns a brand
# new (visible) console for a child process whenever the parent has none and
# doesn't say otherwise. That's the blank terminal window that flashes open
# during transcription. Patching Popen to always pass CREATE_NO_WINDOW on
# Windows fixes it for every subprocess call the app makes (ffmpeg, pip
# during Install Whisper, etc.) without touching the whisper package itself.
if sys.platform == "win32":
    import subprocess as _subprocess
    _CREATE_NO_WINDOW = 0x08000000  # subprocess.CREATE_NO_WINDOW
    _orig_popen_init = _subprocess.Popen.__init__

    def _no_window_popen_init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
        _orig_popen_init(self, *args, **kwargs)

    _subprocess.Popen.__init__ = _no_window_popen_init

# Ensure FFmpeg is on PATH before any Whisper usage.
#
# Two problems this solves:
#   1. A bundled ffmpeg/ directory next to the script (or in PyInstaller's
#      _MEIPASS) should always be findable.
#   2. macOS .app bundles launched from Finder get a sanitized PATH
#      (just /usr/bin:/bin:/usr/sbin:/sbin), which omits Homebrew's
#      /opt/homebrew/bin (Apple Silicon) and /usr/local/bin (Intel).
#      So `brew install ffmpeg` works from a terminal but the built .app
#      can't find it. Same story (less severe) on Windows/Linux. We probe
#      common install locations and prepend the first one that actually
#      contains an ffmpeg binary.
def _ensure_ffmpeg_on_path():
    import platform
    try:
        candidate_dirs = []

        # 1. Bundled ffmpeg/ shipped with the app.
        #    --onefile extracts to _MEIPASS; --onedir puts data next to the
        #    exe inside _internal/. Probe both, plus the exe dir itself.
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            candidate_dirs.append(os.path.join(exe_dir, "ffmpeg"))
            candidate_dirs.append(os.path.join(exe_dir, "_internal", "ffmpeg"))
            candidate_dirs.append(exe_dir)
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                candidate_dirs.append(os.path.join(meipass, "ffmpeg"))
                candidate_dirs.append(meipass)
        else:
            base = os.path.dirname(os.path.abspath(__file__))
            candidate_dirs.append(os.path.join(base, "ffmpeg"))

        # 2. Common system install locations by platform
        system = platform.system()
        home = os.path.expanduser("~")

        if system == "Darwin":
            candidate_dirs.extend([
                "/opt/homebrew/bin",   # Apple Silicon Homebrew
                "/usr/local/bin",      # Intel Homebrew
                "/opt/local/bin",      # MacPorts
            ])
        elif system == "Windows":
            localappdata = os.environ.get(
                "LOCALAPPDATA", os.path.join(home, "AppData", "Local")
            )
            programdata = os.environ.get("ProgramData", r"C:\ProgramData")
            candidate_dirs.extend([
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\Program Files (x86)\ffmpeg\bin",
                # Chocolatey
                os.path.join(programdata, "chocolatey", "bin"),
                # Scoop — per-user and global
                os.path.join(home, "scoop", "shims"),
                os.path.join(home, "scoop", "apps", "ffmpeg", "current", "bin"),
                r"C:\ProgramData\scoop\shims",
                # winget shim directory
                os.path.join(localappdata, "Microsoft", "WinGet", "Links"),
            ])
            # winget unpacks FFmpeg into a versioned folder under Packages,
            # e.g. ...\Packages\Gyan.FFmpeg_...\ffmpeg-7.1-full_build\bin
            # The version string is unpredictable, so glob for it.
            try:
                import glob
                pkg_root = os.path.join(localappdata, "Microsoft", "WinGet", "Packages")
                if os.path.isdir(pkg_root):
                    candidate_dirs.extend(
                        glob.glob(os.path.join(pkg_root, "*FFmpeg*", "**", "bin"),
                                  recursive=True)
                    )
            except Exception:
                pass
        else:  # Linux and other Unix-likes
            candidate_dirs.extend([
                "/usr/local/bin",
                "/usr/bin",
                "/snap/bin",
                os.path.join(home, ".local", "bin"),
            ])

        ffmpeg_name = "ffmpeg.exe" if system == "Windows" else "ffmpeg"
        current_path = os.environ.get("PATH", "")
        current_parts = current_path.split(os.pathsep) if current_path else []

        for d in candidate_dirs:
            if not d:
                continue
            try:
                if os.path.isdir(d) and os.path.isfile(os.path.join(d, ffmpeg_name)):
                    if d not in current_parts:
                        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                        current_parts.insert(0, d)
            except Exception:
                continue
    except Exception:
        pass

_ensure_ffmpeg_on_path()

# Check whether openai-whisper is installed WITHOUT importing it. A real
# `import whisper` would pull in torch (3-5 seconds) at every launch, even
# though the only thing we need here is a boolean flag for the UI. The
# transcription worker thread does its own lazy import when the user
# actually clicks Transcribe — that's where the cost belongs.
import importlib.util as _importlib_util
WHISPER_AVAILABLE = _importlib_util.find_spec("whisper") is not None
if not WHISPER_AVAILABLE:
    print("Note: openai-whisper not installed. Use 'Install Whisper' button or run: pip install openai-whisper")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QButtonGroup, QFrame, QComboBox,
    QLineEdit, QTextEdit, QScrollArea, QDialog
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent


# === TPC DESIGN LANGUAGE COLORS ===
COLORS = {
    # Backgrounds
    'background':       '#ffffff',   # Main window
    'surface':          '#ffffff',   # Cards, inputs
    'panel_bg':         '#f8f8f8',   # Light panel background (cards)
    'panel_sidebar':    '#f1f1f1',   # Sidebar-style background

    # Borders
    'border':           '#e0e0e0',   # Standard
    'border_light':     '#e8e8e8',   # Subtle
    'border_input':     '#cccccc',   # Input fields

    # Text
    'text_primary':     '#000000',   # Headlines
    'text':             '#333333',   # Body / labels
    'text_tertiary':    '#555555',   # Field labels
    'text_muted':       '#666666',   # Descriptions, hints
    'text_disabled':    '#888888',   # Disabled / timestamps
    'text_placeholder': '#999999',   # Placeholder
    'text_light':       '#ffffff',   # On dark backgrounds

    # Brand
    'blue':             '#4a9eff',   # TPC Blue — accent, links, selection
    'blue_hover':       '#5aafff',
    'blue_pressed':     '#3a8eef',
    'green':            '#2ecc71',   # TPC Green — primary action (Transcribe)
    'green_hover':      '#3ddc81',
    'green_pressed':    '#27ae60',
    'navy':             '#0f3460',   # Drop zone only (prominent, needs contrast)
    'navy_hover':       '#1a4a7a',

    # Semantic
    'secondary':        '#95a5a6',   # Gray buttons
    'secondary_hover':  '#7f8c8d',
    'danger':           '#e74c3c',   # Cancel / error
    'danger_hover':     '#c0392b',
    'warning':          '#e67e22',   # Install Whisper
    'success':          '#27ae60',

    # Info banner
    'info_bg':          '#e8f4fd',
    'info_border':      '#4a9eff',
    'info_text':        '#1a5a9e',
}


# ---------------------------------------------------------------------------
# Shared stylesheet helper — keeps panel CSS in one place
# ---------------------------------------------------------------------------
def _card_style(bg=None, border=None, radius=8):
    bg = bg or COLORS['panel_bg']
    border = border or COLORS['border']
    return f"""
        QFrame {{
            background-color: {bg};
            border: 1px solid {border};
            border-radius: {radius}px;
        }}
    """


# ---------------------------------------------------------------------------
# Persistent settings (start folder, etc.) — stored under the OS app-data dir
# ---------------------------------------------------------------------------
def _voxtext_appdata():
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or tempfile.gettempdir()
    root = os.path.join(base, "Voxtext")
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        pass
    return root


def _settings_path():
    return os.path.join(_voxtext_appdata(), "settings.json")


def load_settings() -> dict:
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(data: dict):
    try:
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_setting(key, default=None):
    return load_settings().get(key, default)


def set_setting(key, value):
    data = load_settings()
    data[key] = value
    save_settings(data)


def _start_dir():
    """Directory that Browse dialogs should open to (start folder, else '')."""
    return get_setting("start_folder", "") or ""


# ---------------------------------------------------------------------------
# Live Whisper progress
# ---------------------------------------------------------------------------
# model.transcribe() has no progress callback, but inside whisper/transcribe.py
# it drives a tqdm bar (total = audio frames, update() after every ~30s window)
# whenever verbose=False. We temporarily swap that module's `tqdm` for a tiny
# stand-in that reports frames-done / total back to the worker instead of
# drawing to stderr. Any exception raised from update() propagates out of
# transcribe(), which is how Cancel now stops a file mid-transcription.
# If a future whisper release changes this internals, the swap is a no-op
# and progress simply falls back to the old jump-at-the-end behaviour.
class _WhisperCancelled(Exception):
    """Raised from inside Whisper's loop when the user clicks Cancel."""


class _whisper_progress:
    def __init__(self, on_fraction, is_cancelled):
        self.on_fraction  = on_fraction
        self.is_cancelled = is_cancelled
        self._module = None
        self._orig   = None

    def __enter__(self):
        import importlib, types
        try:
            # import_module returns the submodule itself — `whisper.transcribe`
            # as an attribute is the transcribe() function, not the module.
            mod = importlib.import_module("whisper.transcribe")
        except Exception:
            return self
        if not hasattr(mod, "tqdm"):
            return self

        on_fraction, is_cancelled = self.on_fraction, self.is_cancelled

        class _Bar:
            def __init__(self, *args, total=None, **kwargs):
                self.total = total or 0
                self.n = 0
                self._t_last = time.monotonic()
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def update(self, n=1):
                self.n += n
                now = time.monotonic()
                step_secs, self._t_last = now - self._t_last, now
                if is_cancelled():
                    raise _WhisperCancelled()
                if self.total:
                    # frac done, size of the step just taken (as a fraction),
                    # and how long that step took — the UI uses the last two
                    # to glide toward the next expected update.
                    on_fraction(max(0.0, min(1.0, self.n / self.total)),
                                n / self.total, step_secs)
            def close(self):
                pass

        self._module = mod
        self._orig   = mod.tqdm
        mod.tqdm = types.SimpleNamespace(tqdm=_Bar)
        return self

    def __exit__(self, *exc):
        if self._module is not None:
            self._module.tqdm = self._orig
        return False


class TranscriptionWorker(QThread):
    """Worker thread for transcription to keep UI responsive"""
    progress = pyqtSignal(str, int)   # message, percentage
    # Smoothing hint: "the next real update should land at <pct> in about
    # <seconds>". The main window glides the bar toward it between updates.
    progress_hint = pyqtSignal(float, float)
    finished = pyqtSignal(list)        # list of created files
    error    = pyqtSignal(str)         # error message

    def __init__(self, file_path, model_name, output_formats, output_dir, lms_settings=None,
                 manifest_settings=None):
        super().__init__()
        self.file_path         = file_path
        self.model_name        = model_name
        self.output_formats    = output_formats
        self.output_dir        = output_dir
        self.lms_settings      = lms_settings or {}
        self.manifest_settings = manifest_settings or {}
        self.cancelled         = False
        self.model             = None

    def run(self):
        try:
            import sys as _sys
            import whisper as _whisper_mod
            # If the imported `whisper` module has no load_model, something is
            # off with the install or the import path. Dump enough info into the
            # error dialog that we can actually diagnose it instead of guessing.
            if not hasattr(_whisper_mod, 'load_model'):
                mod_file = getattr(_whisper_mod, '__file__', '<unknown>')
                public_attrs = [a for a in dir(_whisper_mod) if not a.startswith('_')]
                attr_preview = ', '.join(public_attrs[:8]) if public_attrs else '(none)'
                raise ImportError(
                    "openai-whisper isn't loading correctly.\n\n"
                    f"Python:  {_sys.executable}\n"
                    f"Module:  {mod_file}\n"
                    f"Attrs:   {attr_preview}\n\n"
                    "Likely causes:\n"
                    "• A 'whisper.py' file or 'whisper/' folder on the import path\n"
                    "• Wrong PyPI package installed (e.g. 'whisper' instead of 'openai-whisper')\n"
                    "• Voxtext is running on a different Python than where openai-whisper lives"
                )
            whisper = _whisper_mod
            import ssl, urllib.request

            if self.cancelled:
                return

            self.progress.emit(f"Loading {self.model_name} model…", 5)
            try:
                self.model = whisper.load_model(self.model_name)
            except urllib.error.URLError as ssl_error:
                if "CERTIFICATE_VERIFY_FAILED" in str(ssl_error):
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.model = whisper.load_model(self.model_name)
                else:
                    raise

            if self.cancelled:
                return

            self.progress.emit("Model loaded. Transcribing audio…", 10)
            # language=None lets Whisper auto-detect the spoken language from
            # the first ~30s of audio. Hardcoding "en" here was forcing every
            # file to be decoded as English, which is why French (and other
            # non-English) source audio was coming back as mangled English text.
            # Map Whisper's 0..1 into the 10–80% band of the bar.
            last_pct = [10]
            def _on_fraction(frac, step, step_secs):
                pct = 10 + int(frac * 70)
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    self.progress.emit("Transcribing audio…", pct)
                if frac < 1.0:
                    self.progress_hint.emit(10 + min(1.0, frac + step) * 70, step_secs)

            with _whisper_progress(_on_fraction, lambda: self.cancelled):
                result = self.model.transcribe(self.file_path, language=None,
                                               task="transcribe", verbose=False)

            if self.cancelled:
                return

            self.progress.emit("Transcription done. Writing files…", 80)

            base_path = Path(self.file_path).stem
            created_files = []
            num_formats = len(self.output_formats)
            progress_per_format = 15 / num_formats if num_formats > 0 else 15
            current_progress = 80

            format_writers = {
                'txt':  lambda p: self._write_txt(result, p),
                'srt':  lambda p: self._write_srt(result, p),
                'vtt':  lambda p: self._write_vtt(result, p),
                'html': lambda p: self._write_html(result, p),
                'md':   lambda p: self._write_markdown(result, p),
                'json': lambda p: self._write_json(result, p),
            }
            extensions = {
                'txt': '_transcript.txt', 'srt': '_transcript.srt',
                'vtt': '_transcript.vtt', 'html': '_transcript.html',
                'md': '_transcript.md',   'json': '_transcript.json',
            }

            for fmt in ['txt', 'srt', 'vtt', 'html', 'md', 'json']:
                if fmt not in self.output_formats:
                    continue
                if self.cancelled:
                    return
                out = self.output_dir / f"{base_path}{extensions[fmt]}"
                format_writers[fmt](out)
                created_files.append(str(out))
                current_progress += progress_per_format
                self.progress.emit(f"Created {out.name}", int(current_progress))

            if self.manifest_settings.get('enabled') and not self.cancelled:
                try:
                    manifest_path = self._write_manifest(result, base_path, self.output_dir)
                    created_files.append(str(manifest_path))
                    self.progress.emit(f"Created {manifest_path.name}", 97)
                except Exception as manifest_exc:
                    # A manifest problem shouldn't sink an otherwise-successful
                    # transcription — surface it and keep the real outputs.
                    self.progress.emit(f"Translation manifest failed: {manifest_exc}", 97)

            if not self.cancelled:
                self.finished.emit(created_files)

        except _WhisperCancelled:
            return

        except FileNotFoundError as e:
            if not self.cancelled:
                import shutil as _shutil

                # Do NOT test for the word "ffmpeg" in the message — Windows
                # raises this as:
                #   [WinError 2] The system cannot find the file specified
                # with no filename attached, because CreateProcess failed
                # before one was recorded. Substring matching silently misses
                # it and the user gets a bare WinError instead of this help.
                #
                # Identify it structurally instead: errno 2 (or winerror 2)
                # plus ffmpeg being unresolvable on PATH. The source media
                # file is known to exist — it was selected in the UI — so a
                # missing-file error here can only be the executable.
                found = _shutil.which("ffmpeg")
                looks_like_missing_exe = (
                    getattr(e, "winerror", None) == 2
                    or e.errno == 2
                    or "ffmpeg" in str(e).lower()
                    or "ffmpeg" in str(getattr(e, "filename", "") or "").lower()
                )

                if found is None and looks_like_missing_exe:
                    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    path_preview = "\n  ".join(path_dirs) if path_dirs else "(empty)"
                    error_msg = (
                        "FFmpeg not found.\n\n"
                        "Voxtext needs FFmpeg to read audio and video files. "
                        "It isn't installed on this computer, or it isn't on "
                        "the system PATH.\n\n"
                        "Windows — pick one:\n"
                        "  winget install Gyan.FFmpeg\n"
                        "  choco install ffmpeg\n"
                        "  or download from ffmpeg.org and unzip to C:\\ffmpeg\n\n"
                        "Mac:\n  brew install ffmpeg\n\n"
                        "Linux:\n  sudo apt install ffmpeg\n\n"
                        "Quit and reopen Voxtext after installing so it picks "
                        "up the updated PATH.\n\n"
                        "─── Diagnostic ───\n"
                        "shutil.which('ffmpeg'): (not found on PATH)\n"
                        f"Original error: {e}\n"
                        f"PATH searched:\n  {path_preview}"
                    )
                else:
                    error_msg = str(e)
                self.error.emit(error_msg)
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))

    def _write_txt(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result['text'])

    def _write_srt(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(result['segments'], start=1):
                f.write(f"{i}\n")
                f.write(f"{self._ts(seg['start'])} --> {self._ts(seg['end'])}\n")
                f.write(f"{seg['text'].strip()}\n\n")

    def _write_vtt(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n")
            if self.lms_settings.get('enabled'):
                css = self.lms_settings.get('css', '').strip()
                if css:
                    f.write(f"\nSTYLE\n{css}\n")
            f.write("\n")
            cue = ""
            if self.lms_settings.get('enabled'):
                c = self.lms_settings.get('cue_settings', '').strip()
                if c:
                    cue = " " + c
            for seg in result['segments']:
                f.write(f"{self._ts(seg['start'], comma=False)} --> {self._ts(seg['end'], comma=False)}{cue}\n")
                f.write(f"{seg['text'].strip()}\n\n")

    def _write_html(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"<h3>Full Transcript</h3>\n<details>\n<summary>Click to expand transcript</summary>\n\n{result['text']}\n\n</details>\n\n<p><em>Transcribed with Voxtext using OpenAI Whisper</em></p>")

    def _write_markdown(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcript\n\n{result['text']}\n\n---\n\n*Transcribed with Voxtext using OpenAI Whisper*\n")

    def _write_json(self, result, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _ts(self, seconds, comma=True):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        sep = ',' if comma else '.'
        return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

    def cancel(self):
        self.cancelled = True

    def _write_manifest(self, result, base_path, out_dir):
        """
        Build and write one per-video translation manifest (see
        ai_processor.build_translation_manifest / MANIFEST_SCHEMA.txt).
        Shared by single-file and batch runs — base_path/out_dir are
        passed explicitly since BatchTranscriptionWorker never sets
        self.output_dir.
        """
        ms = self.manifest_settings
        source_folder_sp = ai_processor.sharepoint_relative_path(
            out_dir, ms.get('sync_root_local', ''), ms.get('sync_root_sp', '')
        )
        manifest = ai_processor.build_translation_manifest(
            video_id=str(base_path),
            course_name=(ms.get('course_name') or str(base_path)),
            source_language=(result.get('language') or 'en'),
            target_language=ms.get('target_lang', 'fr'),
            segments=result.get('segments', []),
            source_folder=source_folder_sp,
            context_hint=ms.get('context', ''),
        )
        out = Path(out_dir) / f"{base_path}.manifest.json"
        return ai_processor.write_manifest_file(manifest, out)


class BatchTranscriptionWorker(TranscriptionWorker):
    """
    Batch version of TranscriptionWorker — transcribes every media file in a
    list, loading the Whisper model ONCE rather than once per file. Reuses
    the format-writing methods (_write_txt, _write_srt, etc.) unchanged, so
    a fix to how any format is written applies to both single-file and
    batch runs automatically, and output filenames match the single-file
    worker exactly (e.g. "clip_transcript.vtt") so the two modes recognize
    each other's output when skip-existing is on.

    Per-file error isolation: one bad file is recorded as a failure and the
    run continues — a corrupt file three files in shouldn't lose the other
    fifty. Cancellation is checked between files (a file already mid-
    transcribe finishes before the batch stops, same as Whisper's own
    behaviour in the single-file worker).
    """
    file_finished  = pyqtSignal(str, list)   # filename, created file paths
    file_failed    = pyqtSignal(str, str)    # filename, error message
    batch_finished = pyqtSignal(list, list)  # done filenames, [(filename, error), ...]

    OUTPUT_SUFFIXES = {
        'txt': '_transcript.txt', 'srt': '_transcript.srt',
        'vtt': '_transcript.vtt', 'html': '_transcript.html',
        'md': '_transcript.md',   'json': '_transcript.json',
    }

    def __init__(self, file_paths, model_name, output_formats, lms_settings=None,
                 skip_existing=True, manifest_settings=None):
        # Parent __init__ wants a single file_path/output_dir — pass placeholders;
        # run() is fully overridden below and never touches self.file_path/output_dir.
        super().__init__(file_paths[0] if file_paths else "", model_name,
                          output_formats, None, lms_settings, manifest_settings)
        self.file_paths    = list(file_paths)
        self.skip_existing = skip_existing

    def _expected_outputs(self, media_path):
        return [media_path.parent / f"{media_path.stem}{suffix}"
                for fmt, suffix in self.OUTPUT_SUFFIXES.items()
                if fmt in self.output_formats]

    def _writer_for(self, fmt):
        return {
            'txt': self._write_txt, 'srt': self._write_srt,
            'vtt': self._write_vtt, 'html': self._write_html,
            'md': self._write_markdown, 'json': self._write_json,
        }[fmt]

    def run(self):
        try:
            import sys as _sys
            import whisper as _whisper_mod
            import ssl, urllib.error
            if not hasattr(_whisper_mod, 'load_model'):
                raise ImportError(
                    "openai-whisper isn't loading correctly. Check Options for "
                    "the same diagnostics shown on a single-file transcription."
                )
            whisper = _whisper_mod

            if self.cancelled:
                return

            pending, skipped = [], 0
            for fp in self.file_paths:
                p = Path(fp)
                if self.skip_existing and self._expected_outputs(p) and \
                        all(o.exists() for o in self._expected_outputs(p)):
                    skipped += 1
                    continue
                pending.append(p)

            if not pending:
                self.progress.emit(
                    f"Nothing to do — all {skipped} file(s) already have output.", 100
                )
                self.batch_finished.emit([], [])
                return

            self.progress.emit(
                f"Loading {self.model_name} model (once for the whole batch)…", 2
            )
            try:
                self.model = whisper.load_model(self.model_name)
            except urllib.error.URLError as ssl_error:
                if "CERTIFICATE_VERIFY_FAILED" in str(ssl_error):
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.model = whisper.load_model(self.model_name)
                else:
                    raise

            if self.cancelled:
                return

            done, failed = [], []
            total = len(pending)

            for i, media in enumerate(pending, start=1):
                if self.cancelled:
                    break

                start_pct = 5 + int(((i - 1) / total) * 90)
                self.progress.emit(f"[{i}/{total}] {media.name} — transcribing…", start_pct)

                try:
                    # Auto-detect language per file (see note in TranscriptionWorker.run)
                    # Smooth progress within this file's slice of the bar,
                    # leaving a little headroom for writing its outputs.
                    end_pct  = 5 + int((i / total) * 90)
                    span     = max(0, end_pct - start_pct - 1)
                    last_pct = [start_pct]
                    def _on_fraction(frac, step, step_secs, _i=i, _name=media.name,
                                     _start=start_pct, _span=span, _last=last_pct):
                        pct = _start + int(frac * _span)
                        if pct != _last[0]:
                            _last[0] = pct
                            self.progress.emit(
                                f"[{_i}/{total}] {_name} — transcribing… {int(frac * 100)}%", pct
                            )
                        if frac < 1.0:
                            self.progress_hint.emit(
                                _start + min(1.0, frac + step) * _span, step_secs
                            )

                    with _whisper_progress(_on_fraction, lambda: self.cancelled):
                        result = self.model.transcribe(str(media), language=None,
                                                       task="transcribe", verbose=False)
                    created = []
                    for fmt in self.output_formats:
                        out = media.parent / f"{media.stem}{self.OUTPUT_SUFFIXES[fmt]}"
                        self._writer_for(fmt)(result, out)
                        created.append(str(out))
                    if self.manifest_settings.get('enabled'):
                        try:
                            manifest_path = self._write_manifest(result, media.stem, media.parent)
                            created.append(str(manifest_path))
                        except Exception as manifest_exc:
                            # One video's manifest failing shouldn't fail the video's
                            # own transcription outputs, or the rest of the batch.
                            self.progress.emit(
                                f"[{i}/{total}] {media.name} — translation manifest failed: {manifest_exc}",
                                start_pct
                            )
                    done.append(media.name)
                    self.file_finished.emit(media.name, created)
                    self.progress.emit(f"[{i}/{total}] {media.name} — done",
                                        5 + int((i / total) * 90))
                except _WhisperCancelled:
                    break
                except Exception as exc:
                    failed.append((media.name, str(exc)))
                    self.file_failed.emit(media.name, str(exc))

            if not self.cancelled:
                self.progress.emit("Batch complete.", 100)
                self.batch_finished.emit(done, failed)

        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class VoxtextOptionsDialog(QDialog):
    """Options — Start Folder plus the model-maintenance and help actions that
    used to live in the menu bar."""

    def __init__(self, parent):
        super().__init__(parent)
        self.win = parent
        self.setWindowTitle("Options")
        self.setModal(True)
        self.setMinimumWidth(580)
        self._build()
        self.setStyleSheet(f"QDialog {{ background-color: {COLORS['background']}; }}")

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setFont(self.win.f_sec)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; letter-spacing: 1px; border: none;")
        return lbl

    def _ghost(self, text, slot, width=None):
        b = QPushButton(text)
        b.setFont(self.win.f_body)
        b.setStyleSheet(self.win._btn_secondary())
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(slot)
        if width:
            b.setFixedWidth(width)
        return b

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 28, 28, 24)
        lay.setSpacing(16)

        title = QLabel("Options")
        title.setFont(QFont("SF Pro Display", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        lay.addWidget(title)

        # — Start folder —
        lay.addWidget(self._section("START FOLDER"))
        desc = QLabel("Where Browse opens when you pick a file. Leave empty to use your last location.")
        desc.setFont(self.win.f_body)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        lay.addWidget(desc)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.folder_edit = QLineEdit(get_setting("start_folder", "") or "")
        self.folder_edit.setFont(self.win.f_body)
        self.folder_edit.setStyleSheet(self.win._input_style())
        self.folder_edit.setPlaceholderText("No default set \u2014 uses your last location")
        row.addWidget(self.folder_edit, 1)
        row.addWidget(self._ghost("Browse\u2026", self._browse_folder, width=110))
        row.addWidget(self._ghost("Clear", self.folder_edit.clear, width=80))
        lay.addLayout(row)

        # — Translation Manifest Delivery (off by default — a power-user
        # feature for teams running an automated translation flow, not
        # something every Voxtext user should see) —
        lay.addWidget(self._section("TRANSLATION MANIFEST DELIVERY"))
        tmd_desc = QLabel(
            "Off by default. Turn this on only if you have an automated "
            "translation flow (e.g. a Power Automate workflow) "
            "watching a cloud folder for Voxtext's manifest files."
        )
        tmd_desc.setFont(self.win.f_body)
        tmd_desc.setWordWrap(True)
        tmd_desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        lay.addWidget(tmd_desc)

        self.manifest_delivery_enabled = QCheckBox("Enable Translation Manifest Delivery")
        self.manifest_delivery_enabled.setFont(self.win.f_body)
        self.manifest_delivery_enabled.setStyleSheet(self.win._checkbox_style())
        self.manifest_delivery_enabled.setChecked(
            bool(get_setting("translation_manifest_feature_enabled", False))
        )
        self.manifest_delivery_enabled.stateChanged.connect(self._toggle_manifest_delivery_rows)
        lay.addWidget(self.manifest_delivery_enabled)

        jm_lbl = QLabel("Job manifest folder \u2014 the watched cloud folder the flow triggers on:")
        jm_lbl.setFont(self.win.f_label)
        jm_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        lay.addWidget(jm_lbl)

        tq_row = QHBoxLayout()
        tq_row.setSpacing(8)
        self.queue_folder_edit = QLineEdit(get_setting("translation_queue_folder", "") or "")
        self.queue_folder_edit.setFont(self.win.f_body)
        self.queue_folder_edit.setStyleSheet(self.win._input_style())
        self.queue_folder_edit.setPlaceholderText("Not set \u2014 job manifests won't be sent")
        tq_row.addWidget(self.queue_folder_edit, 1)
        self._queue_browse_btn = self._ghost("Browse\u2026", self._browse_queue_folder, width=110)
        self._queue_clear_btn = self._ghost("Clear", self.queue_folder_edit.clear, width=80)
        tq_row.addWidget(self._queue_browse_btn)
        tq_row.addWidget(self._queue_clear_btn)
        lay.addLayout(tq_row)

        sr_lbl = QLabel(
            "Base cloud folder \u2014 the local folder that mirrors your SharePoint "
            "library (e.g. your OneDrive-synced site folder):"
        )
        sr_lbl.setFont(self.win.f_label)
        sr_lbl.setWordWrap(True)
        sr_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        lay.addWidget(sr_lbl)

        sr_row = QHBoxLayout()
        sr_row.setSpacing(8)
        self.sync_root_local_edit = QLineEdit(get_setting("sharepoint_sync_root_local", "") or "")
        self.sync_root_local_edit.setFont(self.win.f_body)
        self.sync_root_local_edit.setStyleSheet(self.win._input_style())
        self.sync_root_local_edit.setPlaceholderText("Not set \u2014 sourceFolder won't be sent")
        sr_row.addWidget(self.sync_root_local_edit, 1)
        self._sync_root_browse_btn = self._ghost("Browse\u2026", self._browse_sync_root, width=110)
        self._sync_root_clear_btn = self._ghost("Clear", self.sync_root_local_edit.clear, width=80)
        sr_row.addWidget(self._sync_root_browse_btn)
        sr_row.addWidget(self._sync_root_clear_btn)
        lay.addLayout(sr_row)

        sp_lbl = QLabel("Its SharePoint path \u2014 what that folder equals on SharePoint:")
        sp_lbl.setFont(self.win.f_label)
        sp_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        lay.addWidget(sp_lbl)

        self.sync_root_sp_edit = QLineEdit(get_setting("sharepoint_sync_root_path", "") or "")
        self.sync_root_sp_edit.setFont(self.win.f_body)
        self.sync_root_sp_edit.setStyleSheet(self.win._input_style())
        self.sync_root_sp_edit.setPlaceholderText("e.g. /Shared Documents/Training")
        lay.addWidget(self.sync_root_sp_edit)

        self._toggle_manifest_delivery_rows(self.manifest_delivery_enabled.checkState().value)

        # — Models —
        lay.addWidget(self._section("MODELS"))
        mrow = QHBoxLayout()
        mrow.setSpacing(8)
        mrow.addWidget(self._ghost("Manage Models\u2026", self.win.manage_models))
        mrow.addWidget(self._ghost("Clear Model Cache\u2026", self.win.clear_cache))
        mrow.addWidget(self._ghost("Open Model Folder", self.win.open_model_folder))
        mrow.addStretch()
        lay.addLayout(mrow)

        # — Help —
        lay.addWidget(self._section("HELP"))
        hrow = QHBoxLayout()
        hrow.setSpacing(8)
        hrow.addWidget(self._ghost("Getting Started", self.win.show_getting_started))
        hrow.addWidget(self._ghost("Processing Speed Guide", self.win.show_speed_guide))
        hrow.addWidget(self._ghost("Visit Website", lambda: webbrowser.open("https://donburnside.com")))
        hrow.addStretch()
        lay.addLayout(hrow)

        lay.addStretch()

        # — Buttons —
        brow = QHBoxLayout()
        brow.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setFont(self.win.f_body)
        cancel.setStyleSheet(self.win._btn_secondary())
        cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel.setFixedWidth(100)
        cancel.clicked.connect(self.reject)
        brow.addWidget(cancel)

        save = QPushButton("Save")
        save.setFont(self.win.f_bold)
        save.setStyleSheet(self.win._btn_blue())  # blue = confirm; green stays for run
        save.setCursor(Qt.CursorShape.PointingHandCursor)
        save.setFixedWidth(120)
        save.clicked.connect(self._save)
        brow.addWidget(save)
        lay.addLayout(brow)

    def _browse_folder(self):
        start = self.folder_edit.text().strip() or _start_dir()
        d = QFileDialog.getExistingDirectory(self, "Select Start Folder", start)
        if d:
            self.folder_edit.setText(d)

    def _browse_queue_folder(self):
        start = self.queue_folder_edit.text().strip() or _start_dir()
        d = QFileDialog.getExistingDirectory(self, "Select Job Manifest Folder", start)
        if d:
            self.queue_folder_edit.setText(d)

    def _browse_sync_root(self):
        start = self.sync_root_local_edit.text().strip() or _start_dir()
        d = QFileDialog.getExistingDirectory(self, "Select Base Cloud Folder", start)
        if d:
            self.sync_root_local_edit.setText(d)

    def _toggle_manifest_delivery_rows(self, state):
        enabled = state == Qt.CheckState.Checked.value
        for w in (self.queue_folder_edit, self._queue_browse_btn, self._queue_clear_btn,
                  self.sync_root_local_edit, self._sync_root_browse_btn, self._sync_root_clear_btn,
                  self.sync_root_sp_edit):
            w.setEnabled(enabled)

    def _save(self):
        set_setting("start_folder", self.folder_edit.text().strip())
        set_setting("translation_manifest_feature_enabled", self.manifest_delivery_enabled.isChecked())
        set_setting("translation_queue_folder", self.queue_folder_edit.text().strip())
        set_setting("sharepoint_sync_root_local", self.sync_root_local_edit.text().strip())
        set_setting("sharepoint_sync_root_path", self.sync_root_sp_edit.text().strip())
        self.accept()


class VoxtextWindow(QMainWindow):

    # Signal used by install_whisper to marshal completion back to the
    # main (GUI) thread — macOS will crash if QMessageBox is created off-main.
    install_finished = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.selected_file  = None
        self.worker         = None
        self.start_time     = None
        self.timer_id       = None
        self.batch_files    = []   # queued paths when in batch (folder) mode
        self.batch_folder_path = None       # source folder when batch came from a folder
        self._batch_manifest_files = []     # per-video manifest.json paths from the last batch run

        self.vtt_presets = {
            'Custom':        {'cue_settings': '',          'css': ''},
            'LMS Standard':  {'cue_settings': 'line:80%',  'css': '::cue {\n  background-color: rgb(0, 0, 0, 60%);\n  line-height: 1.5em;\n}'},
            'Lower Third':   {'cue_settings': 'line:90% align:start', 'css': '::cue {\n  background-color: rgba(0,0,0,0.8);\n  color: #ffffff;\n  font-size: 1.1em;\n}'},
            'High Contrast': {'cue_settings': 'line:85%',  'css': '::cue {\n  background-color: #000000;\n  color: #ffff00;\n  font-weight: bold;\n}'},
        }

        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

        self.init_ui()
        self.setAcceptDrops(True)

        # Connect install signal — runs handler on main thread automatically
        self.install_finished.connect(self._on_install_finished)

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Voxtext")

        # Window — landscape, matching Voxsmith / VoxPrep. The body scrolls,
        # so a shorter screen just scrolls rather than clipping the window.
        WIN_W, WIN_H = 1160, 720
        self.setMinimumSize(980, 560)
        self.resize(WIN_W, WIN_H)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - WIN_W) // 2, (screen.height() - WIN_H) // 2)

        # Fonts — one consistent size throughout, SF Pro / system-ui
        self.f_body  = QFont("SF Pro Display", 13)
        self.f_bold  = QFont("SF Pro Display", 13, QFont.Weight.Bold)
        self.f_small = QFont("SF Pro Display", 13)   # same size, no more tiny text
        self.f_label = QFont("SF Pro Display", 13, QFont.Weight.DemiBold)
        self.f_title = QFont("SF Pro Display", 18, QFont.Weight.DemiBold)
        self.f_sec   = QFont("SF Pro Display", 11, QFont.Weight.DemiBold)  # section headers only

        # Keyboard shortcuts (menu bar removed — preserve muscle memory)
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.browse_file)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.clear_or_cancel)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

        central = QWidget()
        central.setStyleSheet(f"background-color: {COLORS['background']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 16, 24, 14)
        root.setSpacing(12)

        # ── Flush header: file identity left, chrome + Browse right ──────────
        header = QHBoxLayout()
        header.setSpacing(16)

        id_col = QVBoxLayout()
        id_col.setSpacing(3)

        self.hdr_name = QLabel("No file selected")
        self.hdr_name.setFont(self.f_title)
        self.hdr_name.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        id_col.addWidget(self.hdr_name)

        self.hdr_sub = QLabel("Drag & drop a file anywhere, or Browse to select one")
        self.hdr_sub.setFont(self.f_body)
        self.hdr_sub.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        id_col.addWidget(self.hdr_sub)

        self.hdr_meta = QLabel("")
        self.hdr_meta.setFont(self.f_small)
        self.hdr_meta.setStyleSheet(f"color: {COLORS['text_disabled']}; border: none;")
        self.hdr_meta.hide()
        id_col.addWidget(self.hdr_meta)

        header.addLayout(id_col, 1)

        # Right: Batch, Browse, Options, About — one row (matches VoxPrep)
        hdr_right = QHBoxLayout()
        hdr_right.setSpacing(8)
        hdr_right.addStretch()

        self.batch_btn = QPushButton("Batch Folder\u2026")
        self.batch_btn.setFont(self.f_body)
        self.batch_btn.setStyleSheet(self._btn_secondary())
        self.batch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_btn.clicked.connect(self.browse_batch_folder)
        hdr_right.addWidget(self.batch_btn)

        self.browse_btn = QPushButton("Browse\u2026")
        self.browse_btn.setFont(self.f_body)
        self.browse_btn.setStyleSheet(self._btn_secondary())
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.clicked.connect(self.browse_file)
        hdr_right.addWidget(self.browse_btn)

        self.options_btn = QPushButton("\u2699  Options")
        self.options_btn.setFont(self.f_body)
        self.options_btn.setStyleSheet(self._btn_chrome())
        self.options_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.options_btn.clicked.connect(self.open_options)
        hdr_right.addWidget(self.options_btn)

        self.about_btn = QPushButton("About")
        self.about_btn.setFont(self.f_body)
        self.about_btn.setStyleSheet(self._btn_chrome())
        self.about_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.about_btn.clicked.connect(self.show_about)
        hdr_right.addWidget(self.about_btn)

        header.addLayout(hdr_right)
        root.addLayout(header)

        # Hairline divider under header
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {COLORS['border']}; border: none;")
        root.addWidget(divider)

        # ── Batch bar (hidden until a folder / multiple files are queued) ────
        self.batch_bar = QFrame()
        self.batch_bar.setStyleSheet(f"""
            background-color: {COLORS['info_bg']};
            border: 1px solid {COLORS['info_border']};
            border-radius: 6px;
        """)
        batch_bar_layout = QHBoxLayout(self.batch_bar)
        batch_bar_layout.setContentsMargins(12, 8, 12, 8)
        batch_bar_layout.setSpacing(10)

        self.batch_info_label = QLabel("")
        self.batch_info_label.setFont(self.f_body)
        self.batch_info_label.setStyleSheet(f"color: {COLORS['info_text']}; border: none;")
        self.batch_info_label.setWordWrap(True)
        batch_bar_layout.addWidget(self.batch_info_label, 1)

        self.batch_skip_existing = QCheckBox("Skip files that already have output")
        self.batch_skip_existing.setFont(self.f_body)
        self.batch_skip_existing.setStyleSheet(self._checkbox_style())
        self.batch_skip_existing.setChecked(True)
        batch_bar_layout.addWidget(self.batch_skip_existing)

        self.batch_clear_btn = QPushButton("Clear batch")
        self.batch_clear_btn.setFont(self.f_body)
        self.batch_clear_btn.setStyleSheet(self._btn_secondary())
        self.batch_clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_clear_btn.clicked.connect(self.clear_batch)
        batch_bar_layout.addWidget(self.batch_clear_btn)

        self.batch_bar.setVisible(False)
        root.addWidget(self.batch_bar)

        # ── Whisper warning (full width, conditional) ───────────────────────
        self.warn_label = QLabel("\u26a0  Whisper not installed \u2014 use 'Install Whisper' below to get started.")
        self.warn_label.setFont(self.f_body)
        self.warn_label.setStyleSheet(f"""
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 6px;
            color: #856404;
            padding: 8px 12px;
        """)
        self.warn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.warn_label.setVisible(not WHISPER_AVAILABLE)
        root.addWidget(self.warn_label)

        # ── Body (scrollable): Model+Formats row, then full-width manifest band ────
        # Wrapping the middle in a scroll area means the window keeps a fixed
        # landscape size; content scrolls only when it overflows (VTT+LMS open).
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        body_widget = QWidget()
        body_widget.setStyleSheet("background: transparent;")
        body = QVBoxLayout(body_widget)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        self._body_layout = body   # LMS panel is appended here later

        # — Top row: Model quality | Output formats (half width each) ────────
        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        model_card = self._make_card()
        mc_layout  = QVBoxLayout(model_card)
        mc_layout.setContentsMargins(16, 14, 16, 14)
        mc_layout.setSpacing(4)

        mc_layout.addWidget(self._section_header("MODEL QUALITY"))

        self.model_group = QButtonGroup(self)
        models = [
            ('small',  'Good quality'),
            ('medium', 'Professional quality'),
            ('large',  'Maximum accuracy'),
        ]
        for name, quality in models:
            rb = QRadioButton(name.capitalize())
            rb.setFont(self.f_body)
            rb.setStyleSheet(self._radio_style())
            rb.setProperty("model_name", name)
            self.model_group.addButton(rb)

            desc = QLabel(quality)
            desc.setFont(self.f_small)
            desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none; padding-left: 22px;")

            mc_layout.addWidget(rb)
            mc_layout.addWidget(desc)

            if name == 'medium':
                rb.setChecked(True)

        col1 = QVBoxLayout()
        col1.setSpacing(12)
        col1.addWidget(model_card)
        col1.addStretch()

        fmt_card   = self._make_card()
        fmt_layout = QVBoxLayout(fmt_card)
        fmt_layout.setContentsMargins(16, 14, 16, 14)
        fmt_layout.setSpacing(4)

        fmt_layout.addWidget(self._section_header("OUTPUT FORMATS"))

        self.format_checks = {}
        formats = [
            ('txt',  'Text (.txt)',      True),
            ('srt',  'SRT (.srt)',       False),
            ('vtt',  'WebVTT (.vtt)',    False),
            ('html', 'HTML (.html)',     False),
            ('md',   'Markdown (.md)',   False),
            ('json', 'JSON',             False),
        ]
        fmt_grid = QHBoxLayout()
        fmt_grid.setSpacing(12)
        col_a = QVBoxLayout(); col_a.setSpacing(2)
        col_b = QVBoxLayout(); col_b.setSpacing(2)
        for i, (key, label, default) in enumerate(formats):
            cb = QCheckBox(label)
            cb.setFont(self.f_body)
            cb.setStyleSheet(self._checkbox_style())
            cb.setChecked(default)
            self.format_checks[key] = cb
            (col_a if i < 3 else col_b).addWidget(cb)
        fmt_grid.addLayout(col_a)
        fmt_grid.addLayout(col_b)
        fmt_layout.addLayout(fmt_grid)

        hint = QLabel("Files saved next to your source file")
        hint.setFont(self.f_small)
        hint.setStyleSheet(f"color: {COLORS['text_muted']}; border: none; margin-top: 4px;")
        fmt_layout.addWidget(hint)

        col2 = QVBoxLayout()
        col2.setSpacing(12)
        col2.addWidget(fmt_card)
        col2.addStretch()

        top_row.addLayout(col1, 1)
        top_row.addLayout(col2, 1)
        body.addLayout(top_row)

        scroll.setWidget(body_widget)
        root.addWidget(scroll, 1)

        # — LMS VTT styling (full-width, hidden until VTT checked) ————————————
        self.lms_panel = self._make_card()
        lms_layout = QVBoxLayout(self.lms_panel)
        lms_layout.setContentsMargins(16, 14, 16, 14)
        lms_layout.setSpacing(6)
        self.lms_panel.setVisible(False)

        lms_layout.addWidget(self._section_header("LMS VTT STYLING"))

        lms_hdr = QHBoxLayout()
        self.lms_enabled = QCheckBox("Enable LMS Styling")
        self.lms_enabled.setFont(self.f_body)
        self.lms_enabled.setStyleSheet(self._checkbox_style())
        self.lms_enabled.stateChanged.connect(self.toggle_lms_options)
        lms_hdr.addWidget(self.lms_enabled)

        preset_lbl = QLabel("Preset:")
        preset_lbl.setFont(self.f_label)
        preset_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        lms_hdr.addWidget(preset_lbl)

        self.preset_combo = QComboBox()
        self.preset_combo.setFont(self.f_body)
        self.preset_combo.setMinimumWidth(160)
        self.preset_combo.setStyleSheet(self._combo_style())
        self.preset_combo.addItems(list(self.vtt_presets.keys()))
        self.preset_combo.setCurrentText("LMS Standard")
        self.preset_combo.currentTextChanged.connect(self.apply_vtt_preset)
        lms_hdr.addWidget(self.preset_combo)
        lms_hdr.addStretch()
        lms_layout.addLayout(lms_hdr)

        self.lms_options = QWidget()
        self.lms_options.setStyleSheet("background: transparent;")
        opts_layout = QVBoxLayout(self.lms_options)
        opts_layout.setContentsMargins(0, 4, 0, 0)
        opts_layout.setSpacing(6)

        cue_row = QHBoxLayout()
        cue_lbl = QLabel("Cue settings:")
        cue_lbl.setFont(self.f_label)
        cue_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        cue_row.addWidget(cue_lbl)
        self.cue_settings = QLineEdit("line:80%")
        self.cue_settings.setFont(self.f_body)
        self.cue_settings.setStyleSheet(self._input_style())
        cue_row.addWidget(self.cue_settings)
        opts_layout.addLayout(cue_row)

        css_lbl = QLabel("STYLE block:")
        css_lbl.setFont(self.f_label)
        css_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        opts_layout.addWidget(css_lbl)

        self.css_text = QTextEdit()
        self.css_text.setFont(self.f_small)
        self.css_text.setStyleSheet(self._input_style())
        self.css_text.setMaximumHeight(100)
        self.css_text.setPlainText("::cue {\n  background-color: rgb(0, 0, 0, 60%);\n  line-height: 1.5em;\n}")
        opts_layout.addWidget(self.css_text)

        self.lms_options.setVisible(False)
        lms_layout.addWidget(self.lms_options)

        # — Translation Manifest (full-width; hidden unless enabled in
        # Options — Translation Manifest Delivery, since this is a
        # power-user feature for teams running an automated translation
        # flow, not something every Voxtext user should see) —
        self.manifest_panel = self._make_card()
        manifest_layout = QVBoxLayout(self.manifest_panel)
        manifest_layout.setContentsMargins(16, 14, 16, 14)
        manifest_layout.setSpacing(6)

        manifest_layout.addWidget(self._section_header("TRANSLATION MANIFEST"))

        manifest_hdr = QHBoxLayout()
        self.manifest_enabled = QCheckBox("Generate Translation Manifest")
        self.manifest_enabled.setFont(self.f_body)
        self.manifest_enabled.setStyleSheet(self._checkbox_style())
        self.manifest_enabled.stateChanged.connect(self.toggle_manifest_options)
        manifest_hdr.addWidget(self.manifest_enabled)

        target_lbl = QLabel("Target language:")
        target_lbl.setFont(self.f_label)
        target_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        manifest_hdr.addWidget(target_lbl)

        self.manifest_lang_combo = QComboBox()
        self.manifest_lang_combo.setFont(self.f_body)
        self.manifest_lang_combo.setMinimumWidth(170)
        self.manifest_lang_combo.setStyleSheet(self._combo_style())
        for code, name in ai_processor.LANGUAGES.items():
            self.manifest_lang_combo.addItem(name, code)
        manifest_hdr.addWidget(self.manifest_lang_combo)
        manifest_hdr.addStretch()
        manifest_layout.addLayout(manifest_hdr)

        self.manifest_options = QWidget()
        self.manifest_options.setStyleSheet("background: transparent;")
        mopts_layout = QVBoxLayout(self.manifest_options)
        mopts_layout.setContentsMargins(0, 4, 0, 0)
        mopts_layout.setSpacing(6)

        course_row = QHBoxLayout()
        course_lbl = QLabel("Course / batch name:")
        course_lbl.setFont(self.f_label)
        course_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        course_row.addWidget(course_lbl)
        self.manifest_course = QLineEdit()
        self.manifest_course.setFont(self.f_body)
        self.manifest_course.setStyleSheet(self._input_style())
        self.manifest_course.setPlaceholderText("Names the job manifest — defaults to the source folder name")
        course_row.addWidget(self.manifest_course)
        mopts_layout.addLayout(course_row)

        context_row = QHBoxLayout()
        context_lbl = QLabel("Context hint:")
        context_lbl.setFont(self.f_label)
        context_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        context_row.addWidget(context_lbl)
        self.manifest_context = QLineEdit()
        self.manifest_context.setFont(self.f_body)
        self.manifest_context.setStyleSheet(self._input_style())
        self.manifest_context.setPlaceholderText('Optional — e.g. "compliance training video" (register/terminology only)')
        context_row.addWidget(self.manifest_context)
        mopts_layout.addLayout(context_row)

        manifest_hint = QLabel(
            "Writes one <video>.manifest.json next to each source file, plus one "
            "job manifest in your Job Manifest Folder (set in Options) that "
            "kicks off your automated translation flow."
        )
        manifest_hint.setFont(self.f_small)
        manifest_hint.setWordWrap(True)
        manifest_hint.setStyleSheet(f"color: {COLORS['text_muted']}; border: none; margin-top: 2px;")
        mopts_layout.addWidget(manifest_hint)

        self.manifest_options.setVisible(False)
        manifest_layout.addWidget(self.manifest_options)

        self.manifest_panel.setVisible(bool(get_setting("translation_manifest_feature_enabled", False)))

        self._body_layout.addWidget(self.lms_panel)
        self._body_layout.addWidget(self.manifest_panel)
        self._body_layout.addStretch()

        # Connect VTT toggle
        self.format_checks['vtt'].stateChanged.connect(self.on_vtt_toggle)

        # ── Bottom action bar ───────────────────────────────────────────────
        bottom_div = QFrame()
        bottom_div.setFixedHeight(1)
        bottom_div.setStyleSheet(f"background-color: {COLORS['border']}; border: none;")
        root.addWidget(bottom_div)

        prog_row = QHBoxLayout()
        prog_row.setSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        # 0–1000 internally so the smoothing glide moves in sub-percent
        # steps; everything else talks in 0–100 via _set_progress().
        self.progress_bar.setMaximum(1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: {COLORS['border']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['green']};
                border-radius: 4px;
            }}
        """)
        prog_row.addWidget(self.progress_bar)

        self.progress_label = QLabel("0%")
        self.progress_label.setFont(self.f_bold)
        self.progress_label.setStyleSheet(f"color: {COLORS['text']}; border: none;")
        self.progress_label.setFixedWidth(44)
        prog_row.addWidget(self.progress_label)

        # Smooth progress: real updates from Whisper arrive once per ~30s
        # chunk. Between them this timer eases the bar toward where the next
        # update is expected, always stopping short of it, so the bar never
        # runs ahead of real work and never moves backwards.
        self._prog_display = 0.0     # what the bar shows (0–100, float)
        self._prog_real    = 0.0     # last real value from the worker
        self._prog_target  = None    # expected next real value, or None
        self._prog_tau     = 1.0     # easing time constant (seconds)
        self._prog_timer   = QTimer(self)
        self._prog_timer.setInterval(50)
        self._prog_timer.timeout.connect(self._tick_progress)

        root.addLayout(prog_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.status_label = QLabel("Ready to transcribe")
        self.status_label.setFont(self.f_body)
        self.status_label.setStyleSheet(f"color: {COLORS['text']}; border: none;")
        action_row.addWidget(self.status_label)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setFont(self.f_small)
        self.elapsed_label.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        action_row.addWidget(self.elapsed_label)

        action_row.addStretch()

        if not WHISPER_AVAILABLE:
            self.install_btn = QPushButton("Install Whisper")
            self.install_btn.setFont(self.f_bold)
            self.install_btn.setStyleSheet(self._btn_custom(COLORS['warning'], '#d35400'))
            self.install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.install_btn.clicked.connect(self.install_whisper)
            action_row.addWidget(self.install_btn)

        self.transcribe_btn = QPushButton("\u25b6  Transcribe")
        self.transcribe_btn.setFont(self.f_bold)
        self.transcribe_btn.setStyleSheet(self._btn_green())
        self.transcribe_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.transcribe_btn.setEnabled(WHISPER_AVAILABLE)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        action_row.addWidget(self.transcribe_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFont(self.f_body)
        self.clear_btn.setStyleSheet(self._btn_secondary())
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_or_cancel)
        action_row.addWidget(self.clear_btn)

        root.addLayout(action_row)

    # -----------------------------------------------------------------------
    # Style helpers
    # -----------------------------------------------------------------------
    def _make_card(self):
        f = QFrame()
        f.setStyleSheet(_card_style())
        return f

    def _section_header(self, text):
        lbl = QLabel(text)
        lbl.setFont(self.f_sec)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; letter-spacing: 1px; border: none; margin-bottom: 4px;")
        return lbl

    def _radio_style(self):
        return f"""
            QRadioButton {{
                color: {COLORS['text']};
                padding: 3px 0;
                background: transparent;
                border: none;
            }}
            QRadioButton::indicator {{
                width: 14px; height: 14px;
                border-radius: 7px;
                border: 2px solid {COLORS['border_input']};
                background: {COLORS['surface']};
            }}
            QRadioButton::indicator:checked {{
                background: {COLORS['blue']};
                border-color: {COLORS['blue']};
            }}
            QRadioButton::indicator:hover {{
                border-color: {COLORS['blue']};
            }}
        """

    def _checkbox_style(self):
        return f"""
            QCheckBox {{
                color: {COLORS['text']};
                padding: 3px 0;
                background: transparent;
                border: none;
            }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border-radius: 3px;
                border: 2px solid {COLORS['border_input']};
                background: {COLORS['surface']};
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['blue']};
                border-color: {COLORS['blue']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {COLORS['blue']};
            }}
        """

    def _combo_style(self):
        return f"""
            QComboBox {{
                background: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_input']};
                border-radius: 6px;
                padding: 5px 10px;
                font-size: 13px;
            }}
            QComboBox:focus {{ border-color: {COLORS['blue']}; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['surface']};
                color: {COLORS['text']};
                selection-background-color: {COLORS['info_bg']};
                selection-color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
            }}
        """

    def _input_style(self):
        return f"""
            QLineEdit, QTextEdit {{
                background: {COLORS['surface']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border_input']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }}
            QLineEdit:focus, QTextEdit:focus {{ border-color: {COLORS['blue']}; }}
        """

    def _btn_green(self):
        return f"""
            QPushButton {{
                background: {COLORS['green']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 28px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton:hover  {{ background: {COLORS['green_hover']}; }}
            QPushButton:pressed {{ background: {COLORS['green_pressed']}; }}
            QPushButton:disabled {{ background: {COLORS['border']}; color: {COLORS['text_muted']}; }}
        """

    def _btn_blue(self):
        return f"""
            QPushButton {{
                background: {COLORS['blue']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 7px 16px;
                font-size: 13px;
            }}
            QPushButton:hover   {{ background: {COLORS['blue_hover']}; }}
            QPushButton:pressed  {{ background: {COLORS['blue_pressed']}; }}
            QPushButton:disabled {{ background: {COLORS['secondary']}; color: #e0e0e0; }}
        """

    def _btn_secondary(self):
        return f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border_input']};
                border-radius: 6px;
                padding: 9px 20px;
                font-size: 13px;
            }}
            QPushButton:hover  {{ background: {COLORS['panel_bg']}; color: {COLORS['text']}; }}
            QPushButton:pressed {{ background: #e4e4e4; }}
        """

    def _btn_chrome(self):
        return f"""
            QPushButton {{
                background: transparent;
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 5px 14px;
                font-size: 13px;
            }}
            QPushButton:hover  {{ background: {COLORS['panel_bg']}; color: {COLORS['text']}; border-color: {COLORS['border_input']}; }}
            QPushButton:pressed {{ background: #e4e4e4; }}
        """

    def _btn_danger(self):
        return f"""
            QPushButton {{
                background: {COLORS['danger']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 9px 20px;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: {COLORS['danger_hover']}; }}
        """

    def _btn_custom(self, bg, hover):
        return f"""
            QPushButton {{
                background: {bg};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background: {hover}; }}
        """

    def _ai_button_style(self):
        return self._btn_blue()

    # -----------------------------------------------------------------------
    # Drag and Drop
    # -----------------------------------------------------------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls]

        # A single folder dropped -> scan it and queue a batch
        if len(paths) == 1 and os.path.isdir(paths[0]):
            self._load_batch_folder(paths[0])
            return

        files = [p for p in paths if os.path.isfile(p)]

        # Multiple files dropped at once -> batch those exact files
        if len(files) > 1:
            self._load_batch_files(files, source_label=f"{len(files)} dropped files")
            return

        # A single file -> existing single-file flow, unchanged
        if files:
            self._load_file(files[0])

    # -----------------------------------------------------------------------
    # File selection
    # -----------------------------------------------------------------------
    def browse_file(self):
        if self.worker and self.worker.isRunning():
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Audio or Video File", _start_dir(),
            "Media Files (*.mp3 *.mp4 *.wav *.m4a *.flac *.ogg *.mov *.avi *.mkv *.webm);;"
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;"
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;"
            "All Files (*.*)"
        )
        if fp:
            self._load_file(fp)

    def _load_file(self, fp):
        self.clear_batch(silent=True)
        self.selected_file = fp
        self._set_file_header(fp)
        self.status_label.setText("File loaded. Ready to transcribe.")
        if not self.manifest_course.text().strip():
            self.manifest_course.setText(Path(fp).parent.name)

    # -----------------------------------------------------------------------
    # Batch (folder) selection
    # -----------------------------------------------------------------------
    def browse_batch_folder(self):
        if self.worker and self.worker.isRunning():
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select a Folder of Audio/Video Files", _start_dir()
        )
        if folder:
            self._load_batch_folder(folder)

    def _scan_media_files(self, folder):
        exts = self.audio_extensions | self.video_extensions
        found = []
        for name in sorted(os.listdir(folder)):
            fp = os.path.join(folder, name)
            if os.path.isfile(fp) and Path(fp).suffix.lower() in exts:
                found.append(fp)
        return found

    def _load_batch_folder(self, folder):
        files = self._scan_media_files(folder)
        if not files:
            QMessageBox.warning(self, "No Media Files",
                                f"No audio or video files found in:\n{folder}")
            return
        self._load_batch_files(files, source_label=Path(folder).name, folder_path=folder)

    def _load_batch_files(self, files, source_label, folder_path=None):
        self.selected_file = None
        self.batch_files = files
        self.batch_folder_path = folder_path
        n = len(files)
        self.hdr_name.setText(f"{n} file{'s' if n != 1 else ''} queued")
        self.hdr_sub.setText(folder_path or source_label)
        self.hdr_meta.setText(f"Batch — {source_label}")
        self.hdr_meta.show()

        self.batch_info_label.setText(
            f"\U0001F4E6  {n} file(s) queued from {source_label}. "
            f"Each file's output is saved next to it, same as single-file mode."
        )
        self.batch_bar.setVisible(True)
        self.status_label.setText("Ready to batch transcribe.")
        if not self.manifest_course.text().strip():
            self.manifest_course.setText(
                Path(folder_path).name if folder_path else source_label
            )

    def clear_batch(self, silent=False):
        self.batch_files = []
        self.batch_folder_path = None
        self.batch_bar.setVisible(False)
        if not silent:
            self._reset_header()
            self.status_label.setText("Ready to transcribe")

    # ── Flush-header helpers ────────────────────────────────────────────────
    def _reset_header(self):
        self.hdr_name.setText("No file selected")
        self.hdr_sub.setText("Drag & drop a file anywhere, or Browse to select one")
        self.hdr_meta.hide()

    def _set_file_header(self, fp, note=None):
        ext = Path(fp).suffix.lower()
        kind = "Audio" if ext in self.audio_extensions else ("Video" if ext in self.video_extensions else "File")
        try:
            size = float(os.path.getsize(fp))
            unit = "B"
            for u in ("B", "KB", "MB", "GB"):
                unit = u
                if size < 1024 or u == "GB":
                    break
                size /= 1024
            sz = f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        except Exception:
            sz = ""
        meta = kind + (f"  \u00b7  {sz}" if sz else "")
        if note:
            meta += f"  \u00b7  {note}"
        self.hdr_name.setText(os.path.basename(fp))
        self.hdr_sub.setText(os.path.dirname(fp) or fp)
        self.hdr_meta.setText(meta)
        self.hdr_meta.show()

    def open_options(self):
        VoxtextOptionsDialog(self).exec()
        self.manifest_panel.setVisible(bool(get_setting("translation_manifest_feature_enabled", False)))
        if not self.manifest_panel.isVisible():
            self.manifest_enabled.setChecked(False)


    # -----------------------------------------------------------------------
    # LMS panel controls
    # -----------------------------------------------------------------------
    def on_vtt_toggle(self, state):
        self.lms_panel.setVisible(state == Qt.CheckState.Checked.value)

    def toggle_lms_options(self, state):
        self.lms_options.setVisible(state == Qt.CheckState.Checked.value)

    def apply_vtt_preset(self, name):
        if name in self.vtt_presets:
            p = self.vtt_presets[name]
            self.cue_settings.setText(p['cue_settings'])
            self.css_text.setPlainText(p['css'])

    def get_lms_settings(self):
        return {
            'enabled':      self.lms_enabled.isChecked(),
            'cue_settings': self.cue_settings.text(),
            'css':          self.css_text.toPlainText(),
        }

    # -----------------------------------------------------------------------
    # Translation Manifest panel controls (automated translation flow handoff)
    # -----------------------------------------------------------------------
    def toggle_manifest_options(self, state):
        self.manifest_options.setVisible(state == Qt.CheckState.Checked.value)

    def get_manifest_settings(self):
        return {
            'enabled':         bool(get_setting("translation_manifest_feature_enabled", False))
                           and self.manifest_enabled.isChecked(),
            'target_lang':     self.manifest_lang_combo.currentData(),
            'context':         self.manifest_context.text(),
            'course_name':     self.manifest_course.text().strip(),
            'queue_folder':    get_setting("translation_queue_folder", "") or "",
            'sync_root_local': get_setting("sharepoint_sync_root_local", "") or "",
            'sync_root_sp':    get_setting("sharepoint_sync_root_path", "") or "",
        }

    def _selected_output_formats(self):
        """
        The output formats the user ticked — plus VTT whenever manifest
        generation is on. Every manifest's sourceFileName points at
        <name>_transcript.vtt, so that file has to exist even if the user
        didn't tick VTT themselves. Still returns [] when nothing was
        ticked, so the usual "pick a format" prompt fires first.
        """
        formats = [k for k, cb in self.format_checks.items() if cb.isChecked()]
        if formats and 'vtt' not in formats and self.get_manifest_settings()['enabled']:
            formats.append('vtt')
        return formats

    def _maybe_write_job_manifest(self, candidate_files, source_folder):
        """
        After a run that had manifest generation on, roll the per-video
        manifest(s) it created into one job manifest and drop that into the
        Job Manifest Folder — the only file that lands in the watched
        folder itself (see ai_processor.build_job_manifest).
        """
        ms = self.get_manifest_settings()
        if not ms['enabled']:
            return

        manifest_files = [f for f in candidate_files if str(f).endswith('.manifest.json')]
        if not manifest_files:
            return

        queue_folder = ms['queue_folder']
        if not queue_folder:
            self.status_label.setText(
                "Translation manifest(s) created, but no Job Manifest Folder is "
                "set in Options — job manifest wasn't sent."
            )
            return

        source_folder = Path(source_folder) if source_folder else None
        course_name = ms['course_name'] or (source_folder.name if source_folder else "Voxtext")

        source_folder_sp = ai_processor.sharepoint_relative_path(
            source_folder, ms['sync_root_local'], ms['sync_root_sp']
        ) if source_folder else None

        job = ai_processor.build_job_manifest(
            course_name, [Path(f).name for f in manifest_files], source_folder=source_folder_sp
        )
        job_name = f"{course_name}_{time.strftime('%Y%m%d_%H%M%S')}.job.manifest.json"

        try:
            ai_processor.write_manifest_file(job, Path(queue_folder) / job_name)
            if source_folder_sp:
                self.status_label.setText(
                    f"Translation manifest(s) ready — job file sent to {Path(queue_folder).name}."
                )
            else:
                self.status_label.setText(
                    "Job manifest sent, but sourceFolder couldn't be computed — set your "
                    "Base Cloud Folder in Options so the flow can find the per-video manifests."
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Job Manifest",
                f"Per-video manifest(s) were created, but the job manifest couldn't "
                f"be written to your Job Manifest Folder:\n\n{e}"
            )

    # -----------------------------------------------------------------------
    # Transcription
    # -----------------------------------------------------------------------
    def start_transcription(self):
        if self.batch_files:
            self.start_batch_transcription()
            return

        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please select an audio or video file first."); return
        if not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Whisper Not Installed", "Please install Whisper first."); return

        output_formats = self._selected_output_formats()
        if not output_formats:
            QMessageBox.warning(self, "No Format", "Please select at least one output format."); return

        selected_btn = self.model_group.checkedButton()
        model_name   = selected_btn.property("model_name") if selected_btn else "medium"

        self.transcribe_btn.setEnabled(False)
        self.clear_btn.setText("Cancel")
        self.clear_btn.setStyleSheet(self._btn_danger())
        self._set_progress(0, reset=True)
        self.start_time = time.time()
        self.update_elapsed_time()

        self.worker = TranscriptionWorker(
            self.selected_file, model_name, output_formats,
            Path(self.selected_file).parent, self.get_lms_settings(),
            self.get_manifest_settings()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.progress_hint.connect(self.on_progress_hint)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, message, percentage):
        self.status_label.setText(message)
        self._set_progress(percentage)

    def on_progress_hint(self, target, step_secs):
        """Worker expects the next real update at `target` in ~step_secs."""
        if target <= self._prog_real:
            return
        # Leave a sliver of headroom so the glide visibly waits for the real
        # update rather than parking exactly on the expected value.
        self._prog_target = target - 0.4
        # tau = step/3 → covers ~95% of the gap by the expected arrival, then
        # keeps slowing if Whisper runs long on this chunk.
        self._prog_tau = max(0.3, step_secs / 3.0)
        if not self._prog_timer.isActive():
            self._prog_timer.start()

    def _set_progress(self, pct, reset=False):
        """Apply a real progress value (0–100). Never moves backwards unless reset."""
        pct = float(max(0, min(100, pct)))
        self._prog_real = pct
        self._prog_target = None          # wait for a fresh hint
        if reset:
            self._prog_display = pct
            self._prog_timer.stop()
        else:
            self._prog_display = max(self._prog_display, pct)
        if pct >= 100:
            self._prog_timer.stop()
        self._render_progress()

    def _tick_progress(self):
        if self._prog_target is None or self._prog_display >= self._prog_target:
            return
        gap = self._prog_target - self._prog_display
        self._prog_display += gap * min(1.0, (self._prog_timer.interval() / 1000.0) / self._prog_tau)
        self._render_progress()

    def _render_progress(self):
        self.progress_bar.setValue(int(round(self._prog_display * 10)))
        self.progress_label.setText(f"{int(self._prog_display)}%")

    def on_finished(self, created_files):
        self._set_progress(100)
        self.status_label.setText("Transcription complete!")
        self.reset_buttons()

        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_label.setText(f"Completed in {elapsed // 60}m {elapsed % 60}s")

        self._maybe_write_job_manifest(created_files, source_folder=Path(self.selected_file).parent)

        files_list = "\n".join([f"• {Path(f).name}" for f in created_files])
        QMessageBox.information(
            self, "Done!",
            f"Transcription complete!\n\nCreated files:\n{files_list}\n\nSaved to:\n{Path(self.selected_file).parent}"
        )

    def on_error(self, error_msg):
        self._set_progress(0, reset=True)
        self.status_label.setText("Something went wrong. Please try again.")
        self.elapsed_label.setText("")
        self.reset_buttons()
        QMessageBox.critical(self, "Transcription Error",
                             f"Transcription failed:\n\n{error_msg}\n\nPlease check the file and try again.")

    def start_batch_transcription(self):
        if not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Whisper Not Installed", "Please install Whisper first."); return

        output_formats = self._selected_output_formats()
        if not output_formats:
            QMessageBox.warning(self, "No Format", "Please select at least one output format."); return

        selected_btn = self.model_group.checkedButton()
        model_name   = selected_btn.property("model_name") if selected_btn else "medium"

        self.transcribe_btn.setEnabled(False)
        self.clear_btn.setText("Cancel")
        self.clear_btn.setStyleSheet(self._btn_danger())
        self._set_progress(0, reset=True)
        self.start_time = time.time()
        self.update_elapsed_time()

        self._batch_manifest_files = []

        self.worker = BatchTranscriptionWorker(
            self.batch_files, model_name, output_formats,
            self.get_lms_settings(), skip_existing=self.batch_skip_existing.isChecked(),
            manifest_settings=self.get_manifest_settings()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.progress_hint.connect(self.on_progress_hint)
        self.worker.file_finished.connect(self._on_batch_file_finished)
        self.worker.file_failed.connect(self._on_batch_file_failed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def _on_batch_file_finished(self, filename, created_files):
        for f in created_files:
            if str(f).endswith('.manifest.json'):
                self._batch_manifest_files.append(f)

    def _on_batch_file_failed(self, filename, error_msg):
        self.status_label.setText(f"⚠ {filename} failed — continuing with the rest…")

    def on_batch_finished(self, done, failed):
        self._set_progress(100)
        self.reset_buttons()

        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_label.setText(f"Completed in {elapsed // 60}m {elapsed % 60}s")

        summary = f"Done: {len(done)}   Failed: {len(failed)}"
        self.status_label.setText(summary)

        batch_source_folder = self.batch_folder_path or (
            Path(self.batch_files[0]).parent if self.batch_files else None
        )
        self._maybe_write_job_manifest(self._batch_manifest_files, source_folder=batch_source_folder)

        msg = f"Batch transcription complete!\n\n{summary}"
        if failed:
            fail_lines = "\n".join(f"• {name}: {err}" for name, err in failed[:10])
            msg += f"\n\nFailures:\n{fail_lines}"
            if len(failed) > 10:
                msg += f"\n…and {len(failed) - 10} more."
            QMessageBox.warning(self, "Batch Complete — Some Failures", msg)
        else:
            QMessageBox.information(self, "Batch Complete", msg)

    def reset_buttons(self):
        self.transcribe_btn.setEnabled(True)
        self.clear_btn.setText("Clear")
        self.clear_btn.setStyleSheet(self._btn_secondary())

    def clear_or_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.quit()
            self.worker.wait()
            self.status_label.setText("Transcription cancelled")
            self._set_progress(0, reset=True)
            self.elapsed_label.setText("")
            self.reset_buttons()
        else:
            self.selected_file = None
            self._reset_header()
            self.clear_batch(silent=True)
            self.status_label.setText("Ready to transcribe")
            self._set_progress(0, reset=True)
            self.elapsed_label.setText("")

    def update_elapsed_time(self):
        if self.worker and self.worker.isRunning() and self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_label.setText(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")
            self.timer_id = self.startTimer(1000)

    def timerEvent(self, event):
        if self.worker and self.worker.isRunning() and self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_label.setText(f"Elapsed: {elapsed // 60}m {elapsed % 60}s")
        else:
            if self.timer_id:
                self.killTimer(self.timer_id)
                self.timer_id = None

    # -----------------------------------------------------------------------
    # Whisper installation
    # -----------------------------------------------------------------------
    def install_whisper(self):
        """Run pip install in a background thread; signal back to main thread on completion."""
        self.status_label.setText("Installing Whisper… this may take a minute.")

        def install_in_thread():
            # IMPORTANT: Never touch the UI from here.
            # Use sys.executable so we install into THIS Python, not some stray 'pip'.
            try:
                import subprocess

                pip_cmd = [sys.executable, "-m", "pip", "install", "openai-whisper"]

                # First attempt: plain pip install
                r = subprocess.run(pip_cmd, capture_output=True, text=True)

                # If pip refused due to PEP 668 (newer Python on Linux/Mac with
                # externally-managed-environment marker), retry with the override.
                if r.returncode != 0 and "externally-managed-environment" in (r.stderr + r.stdout):
                    r = subprocess.run(
                        pip_cmd + ["--break-system-packages"],
                        capture_output=True, text=True
                    )

                ok = (r.returncode == 0)
                # Combine stderr + stdout — pip writes useful info to either
                msg = (r.stderr or "") + ("\n" + r.stdout if r.stdout and not ok else "")
                msg = msg.strip()
            except Exception as e:
                ok = False
                msg = str(e)

            # Emit signal — Qt will dispatch _on_install_finished onto main thread.
            self.install_finished.emit(ok, msg)

        threading.Thread(target=install_in_thread, daemon=True).start()

    def _on_install_finished(self, success, error):
        """Handle install completion — runs on main thread via install_finished signal."""
        if success:
            global WHISPER_AVAILABLE
            WHISPER_AVAILABLE = True
            self.status_label.setText("Whisper installed successfully!")
            self.transcribe_btn.setEnabled(True)

            # Hide the warning banner and Install button — no longer needed
            if hasattr(self, 'warn_label'):
                self.warn_label.setVisible(False)
            if hasattr(self, 'install_btn'):
                self.install_btn.setVisible(False)

            QMessageBox.information(
                self, "Done!",
                "Whisper installed!\n\nYou can now transcribe audio and video files."
            )
        else:
            self.status_label.setText("Installation failed — see error.")
            QMessageBox.critical(
                self, "Installation Error",
                f"Failed to install Whisper:\n\n{error}\n\nTry: pip install openai-whisper"
            )

    # -----------------------------------------------------------------------
    # Menu actions
    # -----------------------------------------------------------------------
    def _whisper_cache_dir(self):
        """Return Whisper's model cache directory.

        This matches Whisper's own logic exactly: it honors XDG_CACHE_HOME if
        set, otherwise falls back to ~/.cache — on ALL platforms, INCLUDING
        Windows. Whisper does not use AppData. (The old code looked in
        ~/AppData/Local/whisper on Windows, which is why the Tools menu never
        found the models that were actually sitting in ~/.cache/whisper.)
        """
        default = os.path.join(os.path.expanduser("~"), ".cache")
        return Path(os.getenv("XDG_CACHE_HOME", default)) / "whisper"

    def clear_cache(self):
        import shutil
        cache_dir = self._whisper_cache_dir()

        if not cache_dir.exists():
            QMessageBox.information(self, "Cache Empty", "No model cache found."); return
        size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1048576
        if QMessageBox.question(self, "Clear Cache",
            f"Model cache is {size_mb:.1f} MB.\n\nClear all downloaded models?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(cache_dir)
                QMessageBox.information(self, "Done", f"Cleared {size_mb:.1f} MB.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def manage_models(self):
        cache_dir = self._whisper_cache_dir()
        if not cache_dir.exists():
            QMessageBox.information(self, "No Models", "No models have been downloaded yet."); return
        models = list(cache_dir.glob("*.pt"))
        if not models:
            QMessageBox.information(self, "No Models", "No models found in cache."); return
        info = "\n".join(f"• {m.stem}  ({m.stat().st_size/1048576:.0f} MB)" for m in sorted(models))
        QMessageBox.information(self, "Downloaded Models", f"{info}\n\nLocation: {cache_dir}")

    def open_model_folder(self):
        import platform, subprocess
        cache_dir = self._whisper_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if platform.system() == "Darwin":  subprocess.run(["open",     str(cache_dir)])
            elif platform.system() == "Windows": subprocess.run(["explorer", str(cache_dir)])
            else:                                 subprocess.run(["xdg-open", str(cache_dir)])
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_getting_started(self):
        QMessageBox.information(self, "Getting Started", """Getting Started with Voxtext

1. SELECT A FILE
   Click the drop zone or drag & drop an audio/video file
   Supported: MP3, MP4, WAV, M4A, MOV, and more

2. CHOOSE A MODEL
   Small:  Good quality, faster
   Medium: Professional quality (recommended)
   Large:  Maximum accuracy, slowest
   First use downloads the model once, then it stays cached.

3. SELECT OUTPUT FORMATS
   Text (.txt): Plain transcript
   SRT (.srt): Subtitles with timestamps
   WebVTT (.vtt): Web video captions
   HTML (.html): Collapsible block for WordPress
   Markdown (.md): For documentation
   JSON: Full data with timestamps

   Checking WebVTT reveals LMS VTT Styling, where you can
   add ::cue CSS and pick a preset for captions in your LMS.

4. CLICK TRANSCRIBE
   Processing: roughly 2-5× the audio length
   Files saved next to your source file

5. TRANSLATION MANIFEST DELIVERY (power users only)
   Off by default — turn it on in Options if you have an automated
   translation flow watching a cloud folder. Once enabled, check
   "Generate Translation Manifest" on the main screen and pick a target
   language to write a <video>.manifest.json next to each source file,
   plus one job manifest in your Job Manifest Folder (set in Options)
   that kicks off the flow.

TIPS:
• Your audio stays on your computer
• Models download once and cache forever""")

    def show_speed_guide(self):
        QMessageBox.information(self, "Processing Speed Guide", """Processing Speed Guide

WHY IS IT SLOW?
Everything runs locally. Speed depends on model
size, your CPU/GPU, and audio length.

TYPICAL TIMES (for 10 min of audio):

MODERN COMPUTER (2020+):
• Small:  2 – 3 min
• Medium: 3 – 5 min
• Large:  5 – 10 min

OLDER COMPUTER:
• Small:  5 – 10 min
• Medium / Large: slow — be patient

RECOMMENDATIONS:
• First pass / drafts: Small
• Podcasts & final output: Medium
• Maximum accuracy: Large, modern PC only

REMEMBER:
• Runs locally — your audio stays on your computer
• Models download once, cached forever""")

    def show_about(self):
        QMessageBox.about(self, "About Voxtext", """<h2>Voxtext</h2>
<p>Version 1.8.2</p>
<p>Local audio &amp; video transcription, powered by OpenAI Whisper.
Processing happens on your computer.</p>

<p>Created by Don Burnside<br>
Part of the Vox Suite</p>

<p><a href="https://donburnside.com">donburnside.com</a></p>""")


def main():
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setApplicationName("Voxtext")
    app.setOrganizationName("Vox Suite")

    font = QFont("SF Pro Display", 13)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    window = VoxtextWindow()
    window.show()

    from PyQt6.QtCore import QTimer
    QTimer.singleShot(0, window.update)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
