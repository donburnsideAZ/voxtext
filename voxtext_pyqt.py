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

        # 1. Bundled ffmpeg/ next to script or in PyInstaller _MEIPASS
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs.append(os.path.join(base, "ffmpeg"))

        # 2. Common system install locations by platform
        system = platform.system()
        if system == "Darwin":
            candidate_dirs.extend([
                "/opt/homebrew/bin",   # Apple Silicon Homebrew
                "/usr/local/bin",      # Intel Homebrew
                "/opt/local/bin",      # MacPorts
            ])
        elif system == "Windows":
            candidate_dirs.extend([
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\Program Files (x86)\ffmpeg\bin",
            ])
        else:  # Linux and other Unix-likes
            candidate_dirs.extend([
                "/usr/local/bin",
                "/usr/bin",
                "/snap/bin",
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
from PyQt6.QtCore import Qt, QThread, pyqtSignal
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


class TranscriptionWorker(QThread):
    """Worker thread for transcription to keep UI responsive"""
    progress = pyqtSignal(str, int)   # message, percentage
    finished = pyqtSignal(list)        # list of created files
    error    = pyqtSignal(str)         # error message

    def __init__(self, file_path, model_name, output_formats, output_dir, lms_settings=None):
        super().__init__()
        self.file_path      = file_path
        self.model_name     = model_name
        self.output_formats = output_formats
        self.output_dir     = output_dir
        self.lms_settings   = lms_settings or {}
        self.cancelled      = False
        self.model          = None
        self.result         = None   # retained for AI ops

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

            self.progress.emit(f"Loading {self.model_name} model (first time downloads)…", 5)
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
            result = self.model.transcribe(self.file_path, language="en", task="transcribe")

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

            if not self.cancelled:
                self.result = result
                self.finished.emit(created_files)

        except FileNotFoundError as e:
            if not self.cancelled:
                if 'ffmpeg' in str(e).lower():
                    # Dump enough info to actually diagnose where we looked.
                    import shutil as _shutil
                    found = _shutil.which("ffmpeg") or "(not found on PATH)"
                    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    path_preview = "\n  ".join(path_dirs) if path_dirs else "(empty)"
                    error_msg = (
                        "FFmpeg not found!\n\n"
                        "Whisper requires FFmpeg to process audio/video files.\n\n"
                        "To install on Mac:\n  brew install ffmpeg\n\n"
                        "To install on Windows:\n  Download from ffmpeg.org\n\n"
                        "To install on Linux:\n  sudo apt install ffmpeg\n\n"
                        "─── Diagnostic ───\n"
                        f"shutil.which('ffmpeg'): {found}\n"
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

    def _save(self):
        set_setting("start_folder", self.folder_edit.text().strip())
        self.accept()


class VoxtextWindow(QMainWindow):

    # Signal used by install_whisper to marshal completion back to the
    # main (GUI) thread — macOS will crash if QMessageBox is created off-main.
    install_finished = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.selected_file  = None
        self.worker         = None
        self.last_result    = None
        self.start_time     = None
        self.timer_id       = None

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
        self.f_title = QFont("SF Pro Display", 22, QFont.Weight.Bold)
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

        # Right: chrome (Options, About) stacked ABOVE Browse
        hdr_right = QVBoxLayout()
        hdr_right.setSpacing(8)

        chrome_row = QHBoxLayout()
        chrome_row.setSpacing(8)
        chrome_row.addStretch()

        self.options_btn = QPushButton("\u2699  Options")
        self.options_btn.setFont(self.f_body)
        self.options_btn.setStyleSheet(self._btn_chrome())
        self.options_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.options_btn.clicked.connect(self.open_options)
        chrome_row.addWidget(self.options_btn)

        self.about_btn = QPushButton("About")
        self.about_btn.setFont(self.f_body)
        self.about_btn.setStyleSheet(self._btn_chrome())
        self.about_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.about_btn.clicked.connect(self.show_about)
        chrome_row.addWidget(self.about_btn)

        hdr_right.addLayout(chrome_row)

        browse_row = QHBoxLayout()
        browse_row.addStretch()
        self.browse_btn = QPushButton("Browse\u2026")
        self.browse_btn.setFont(self.f_body)
        self.browse_btn.setStyleSheet(self._btn_secondary())
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setFixedWidth(130)
        self.browse_btn.clicked.connect(self.browse_file)
        browse_row.addWidget(self.browse_btn)
        hdr_right.addLayout(browse_row)

        header.addLayout(hdr_right)
        root.addLayout(header)

        # Hairline divider under header
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {COLORS['border']}; border: none;")
        root.addWidget(divider)

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

        # ── Body (scrollable): Model+Formats row, then full-width AI band ────
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
            ('small',  '466 MB', 'Good quality'),
            ('medium', '1.5 GB', 'Professional quality'),
            ('large',  '3 GB',   'Maximum accuracy'),
        ]
        for name, size, quality in models:
            rb = QRadioButton(f"{name.capitalize()}  ({size})")
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

        # — Full-width AI Processing band (three actions, side by side) ───────
        ai_card   = self._make_card()
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(16, 14, 16, 16)
        ai_layout.setSpacing(10)

        ai_layout.addWidget(self._section_header("AI PROCESSING"))

        self.ai_note = QLabel("\u2726  Transcribe a file first to enable AI operations")
        self.ai_note.setFont(self.f_body)
        self.ai_note.setStyleSheet(f"""
            color: {COLORS['info_text']};
            background-color: {COLORS['info_bg']};
            border: 1px solid {COLORS['info_border']};
            border-radius: 6px;
            padding: 8px 12px;
        """)
        self.ai_note.setWordWrap(True)
        ai_layout.addWidget(self.ai_note)

        def _ai_heading(text):
            l = QLabel(text); l.setFont(self.f_label)
            l.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
            return l

        def _ai_desc(text):
            l = QLabel(text); l.setFont(self.f_body); l.setWordWrap(True)
            l.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
            return l

        def _vline():
            v = QFrame(); v.setFrameShape(QFrame.Shape.VLine)
            v.setStyleSheet(f"border: none; border-left: 1px solid {COLORS['border']};")
            v.setFixedWidth(1)
            return v

        ai_cols = QHBoxLayout()
        ai_cols.setSpacing(20)

        # Sub-column 1: Format transcript
        c1 = QVBoxLayout(); c1.setSpacing(6)
        c1.addWidget(_ai_heading("Format transcript"))
        c1.addWidget(_ai_desc("Improve readability of the raw text."))
        self.format_ai_btn = QPushButton("Clean Up Transcript\u2026")
        self.format_ai_btn.setFont(self.f_bold)
        self.format_ai_btn.setEnabled(False)
        self.format_ai_btn.setStyleSheet(self._btn_blue())
        self.format_ai_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.format_ai_btn.clicked.connect(lambda: self.open_ai_dialog("format"))
        c1.addWidget(self.format_ai_btn)
        c1.addStretch()

        # Sub-column 2: Translate transcript
        c2 = QVBoxLayout(); c2.setSpacing(6)
        c2.addWidget(_ai_heading("Translate transcript"))
        c2.addWidget(_ai_desc("Translate to another language, preserving segment timing. Delivers a WebVTT file."))
        lang_row = QHBoxLayout(); lang_row.setSpacing(8)
        from ai_processor import LANGUAGES
        self.ai_lang_combo = QComboBox()
        self.ai_lang_combo.setFont(self.f_body)
        self.ai_lang_combo.setStyleSheet(self._combo_style())
        for code, name in LANGUAGES.items():
            self.ai_lang_combo.addItem(name, code)
        lang_row.addWidget(self.ai_lang_combo, 1)
        self.translate_btn = QPushButton("Translate\u2026")
        self.translate_btn.setFont(self.f_bold)
        self.translate_btn.setEnabled(False)
        self.translate_btn.setStyleSheet(self._btn_blue())
        self.translate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_btn.clicked.connect(lambda: self.open_ai_dialog("translate"))
        lang_row.addWidget(self.translate_btn)
        c2.addLayout(lang_row)
        c2.addStretch()

        # Sub-column 3: Translate an existing VTT
        c3 = QVBoxLayout(); c3.setSpacing(6)
        c3.addWidget(_ai_heading("Translate a VTT file"))
        c3.addWidget(_ai_desc("Translate an existing WebVTT file you created manually."))
        self.translate_vtt_btn = QPushButton("Translate VTT File\u2026")
        self.translate_vtt_btn.setFont(self.f_bold)
        self.translate_vtt_btn.setStyleSheet(self._btn_blue())
        self.translate_vtt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_vtt_btn.clicked.connect(self.translate_vtt_file)
        c3.addWidget(self.translate_vtt_btn)
        c3.addStretch()

        ai_cols.addLayout(c1, 1)
        ai_cols.addWidget(_vline())
        ai_cols.addLayout(c2, 1)
        ai_cols.addWidget(_vline())
        ai_cols.addLayout(c3, 1)
        ai_layout.addLayout(ai_cols)

        body.addWidget(ai_card)

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

        self._body_layout.addWidget(self.lms_panel)
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
        self.progress_bar.setMaximum(100)
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
        if urls:
            fp = urls[0].toLocalFile()
            if os.path.isfile(fp):
                self._load_file(fp)

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
        self.selected_file = fp
        self._set_file_header(fp)
        self.status_label.setText("File loaded. Ready to transcribe.")

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
    # Transcription
    # -----------------------------------------------------------------------
    def start_transcription(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please select an audio or video file first."); return
        if not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Whisper Not Installed", "Please install Whisper first."); return

        output_formats = [k for k, cb in self.format_checks.items() if cb.isChecked()]
        if not output_formats:
            QMessageBox.warning(self, "No Format", "Please select at least one output format."); return

        selected_btn = self.model_group.checkedButton()
        model_name   = selected_btn.property("model_name") if selected_btn else "medium"

        self.transcribe_btn.setEnabled(False)
        self.clear_btn.setText("Cancel")
        self.clear_btn.setStyleSheet(self._btn_danger())
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        self.start_time = time.time()
        self.update_elapsed_time()

        self.worker = TranscriptionWorker(
            self.selected_file, model_name, output_formats,
            Path(self.selected_file).parent, self.get_lms_settings()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, message, percentage):
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"{percentage}%")

    def on_finished(self, created_files):
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        self.status_label.setText("Transcription complete!")
        self.reset_buttons()

        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_label.setText(f"Completed in {elapsed // 60}m {elapsed % 60}s")

        if self.worker and getattr(self.worker, 'result', None):
            self.last_result = self.worker.result
            has_segs = bool(self.last_result.get('segments'))
            self.translate_btn.setEnabled(has_segs)
            self.format_ai_btn.setEnabled(has_segs)
            if has_segs:
                self.ai_note.setVisible(False)

        files_list = "\n".join([f"• {Path(f).name}" for f in created_files])
        QMessageBox.information(
            self, "Done!",
            f"Transcription complete!\n\nCreated files:\n{files_list}\n\nSaved to:\n{Path(self.selected_file).parent}"
        )

    def on_error(self, error_msg):
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        self.status_label.setText("Something went wrong. Please try again.")
        self.elapsed_label.setText("")
        self.reset_buttons()
        QMessageBox.critical(self, "Transcription Error",
                             f"Transcription failed:\n\n{error_msg}\n\nPlease check the file and try again.")

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
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")
            self.elapsed_label.setText("")
            self.reset_buttons()
        else:
            self.selected_file = None
            self._reset_header()
            self.status_label.setText("Ready to transcribe")
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")
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
    # Translate an externally-authored VTT file
    # -----------------------------------------------------------------------
    def translate_vtt_file(self):
        """
        Browse for an existing .vtt file, parse it into segments, then open
        the standard AI translate dialog.  The parsed segments are stored in
        self.last_result temporarily so open_ai_dialog can work unchanged.
        self.selected_file is also set to the VTT path so the output file
        lands in the same folder with a sensible name.
        """
        import ai_processor as ai

        fp, _ = QFileDialog.getOpenFileName(
            self, "Select WebVTT File", _start_dir(),
            "WebVTT Files (*.vtt);;All Files (*.*)"
        )
        if not fp:
            return

        try:
            segments = ai.parse_vtt_file(fp)
        except Exception as e:
            QMessageBox.critical(self, "Could not read VTT",
                                 f"Failed to parse the VTT file:\n\n{e}")
            return

        # Temporarily load the VTT as the active file/result so the
        # existing dialog path works without modification.
        self.selected_file = fp
        self.last_result   = {"segments": segments, "text": ""}

        # Update the header so the user sees what's loaded
        self._set_file_header(fp, note="VTT — ready to translate")

        self.open_ai_dialog("translate")

    # -----------------------------------------------------------------------
    # AI dialog (clipboard round-trip)
    # -----------------------------------------------------------------------
    def open_ai_dialog(self, operation):
        import ai_processor as ai

        if not self.last_result or not self.last_result.get('segments'):
            QMessageBox.warning(self, "Nothing to process",
                                "Transcribe a file first, then translate or clean up.")
            return

        # FORMAT only writes prose formats — SRT and VTT are word-for-word
        # timed files and must never be modified by the cleanup operation.
        PROSE_FORMATS = {'txt', 'html', 'md'}
        TIMED_FORMATS = {'srt', 'vtt'}

        segments  = self.last_result['segments']
        backend   = ai.ClipboardBackend()
        lang_code = self.ai_lang_combo.currentData() if operation == "translate" else None
        lang_name = self.ai_lang_combo.currentText() if operation == "translate" else ""
        title     = f"Translate to {lang_name}" if operation == "translate" else "Clean Up Transcript"

        # For format, figure out which prose formats are currently checked.
        # Warn early if none are — no point building the dialog.
        if operation == "format":
            checked_all   = [k for k, cb in self.format_checks.items() if cb.isChecked()]
            prose_checked = [k for k in checked_all if k in PROSE_FORMATS]
            timed_checked = [k for k in checked_all if k in TIMED_FORMATS]
            if not prose_checked:
                msg = (
                    "Clean Up works on Text, HTML, and Markdown output.\n\n"
                    "None of those formats are currently checked.\n\n"
                    "Check at least one prose format (TXT, HTML, or MD) and try again.\n\n"
                    "Note: SRT and VTT are not modified by Clean Up — they preserve\n"
                    "the original word-for-word transcript with timestamps."
                )
                QMessageBox.warning(self, "No prose formats selected", msg)
                return

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(620)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(12)

        # — Optional context hint ——————————————————————————————————————————
        # Lets the user tell the agent what kind of content this is.
        # Entirely optional — leaving it blank keeps the prompt content-neutral.
        hint_lbl = QLabel("Content type (optional — helps the agent understand context):")
        hint_lbl.setFont(self.f_body)
        hint_lbl.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        hint_lbl.setWordWrap(True)
        v.addWidget(hint_lbl)

        hint_input = QLineEdit()
        hint_input.setFont(self.f_body)
        hint_input.setStyleSheet(self._input_style())
        hint_input.setPlaceholderText(
            "e.g. podcast interview, lecture recording, training narration, product demo…"
        )
        v.addWidget(hint_input)

        # — SRT/VTT notice for format operation ——————————————————————————
        if operation == "format" and timed_checked:
            timed_names = ", ".join(f".{f.upper()}" for f in timed_checked)
            notice = QLabel(
                f"ℹ  {timed_names} will not be modified — timed formats preserve the "
                f"original word-for-word transcript. Only "
                f"{', '.join('.' + f for f in prose_checked)} will be saved."
            )
            notice.setFont(self.f_small)
            notice.setWordWrap(True)
            notice.setStyleSheet(f"""
                color: {COLORS['info_text']};
                background-color: {COLORS['info_bg']};
                border: 1px solid {COLORS['info_border']};
                border-radius: 6px;
                padding: 8px 12px;
            """)
            v.addWidget(notice)

        # — Step 1 ————————————————————————————————————————————————————————
        step1 = QLabel("Step 1 — Copy this prompt and paste it into your agent (Copilot, Claude, ChatGPT, etc.)")
        step1.setFont(self.f_bold)
        step1.setWordWrap(True)
        v.addWidget(step1)

        # Prompt box — rebuilt when hint changes so the prompt stays current
        prompt_box = QTextEdit()
        prompt_box.setReadOnly(True)
        prompt_box.setMaximumHeight(160)
        prompt_box.setStyleSheet(self._input_style())
        v.addWidget(prompt_box)

        copy_btn = QPushButton("Copy prompt to clipboard")

        def _refresh_prompt():
            context = hint_input.text().strip()
            p = backend.build_outgoing(segments, operation, lang_code, context)
            prompt_box.setPlainText(p)
            copy_btn.setText("Copy prompt to clipboard")

        hint_input.textChanged.connect(_refresh_prompt)
        _refresh_prompt()  # populate on open
        copy_btn.setStyleSheet(self._btn_blue())
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        def do_copy():
            QApplication.clipboard().setText(prompt_box.toPlainText())
            copy_btn.setText("Copied ✓")
        copy_btn.clicked.connect(do_copy)
        v.addWidget(copy_btn)

        # — Step 2 ————————————————————————————————————————————————————————
        step2 = QLabel("Step 2 — Paste your agent's full reply below")
        step2.setFont(self.f_bold)
        step2.setWordWrap(True)
        v.addWidget(step2)

        reply_box = QTextEdit()
        reply_box.setPlaceholderText("Paste the agent's response here…")
        reply_box.setMinimumHeight(130)
        reply_box.setStyleSheet(self._input_style())
        v.addWidget(reply_box)

        # — Buttons ———————————————————————————————————————————————————————
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self._btn_secondary())
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(cancel_btn)

        if operation == "translate":
            apply_label = "Apply & Save VTT"
        else:
            apply_label = f"Apply & Save ({', '.join('.' + f for f in prose_checked)})"

        apply_btn = QPushButton(apply_label)
        apply_btn.setStyleSheet(self._btn_blue())
        btn_row.addWidget(apply_btn)
        v.addLayout(btn_row)

        # — Apply logic ———————————————————————————————————————————————————
        def do_apply():
            pasted = reply_box.toPlainText().strip()
            if not pasted:
                QMessageBox.warning(dlg, "Nothing pasted", "Paste the agent's reply before applying.")
                return

            result = backend.parse_incoming(segments, pasted, operation)

            # For translate, warn on segment mismatches (timing integrity check)
            if operation == "translate" and not result.ok:
                msg = []
                if result.missing_ids:
                    msg.append(f"Missing segments: {result.missing_ids} (original text kept).")
                if result.extra_ids:
                    msg.append(f"Unexpected segment numbers: {result.extra_ids} (ignored).")
                proceed = QMessageBox.question(
                    dlg, "Some segments didn't match",
                    "\n".join(msg) + "\n\nSave anyway with these fallbacks?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if proceed != QMessageBox.StandardButton.Yes:
                    return

            base    = Path(self.selected_file).stem
            suffix  = f"_{lang_code}" if operation == "translate" else "_cleaned"
            out_dir = Path(self.selected_file).parent
            writer  = TranscriptionWorker(self.selected_file, "tiny", [], out_dir, self.get_lms_settings())
            saved   = []

            if operation == "translate":
                # Translate always writes VTT — timing is the whole point
                out = out_dir / f"{base}{suffix}.vtt"
                writer._write_vtt({'segments': result.segments}, out)
                saved.append(out.name)
            else:
                # Format/cleanup writes prose formats only — never SRT or VTT.
                # result.text is the cleaned prose; result.segments retain
                # the original timings untouched (used only if JSON is added later).
                prose_result = {'text': result.text, 'segments': result.segments}
                prose_writers = {
                    'txt':  lambda p: writer._write_txt(prose_result, p),
                    'html': lambda p: writer._write_html(prose_result, p),
                    'md':   lambda p: writer._write_markdown(prose_result, p),
                }
                ext_map = {'txt': '.txt', 'html': '.html', 'md': '.md'}
                for fmt in prose_checked:
                    out = out_dir / f"{base}{suffix}{ext_map[fmt]}"
                    prose_writers[fmt](out)
                    saved.append(out.name)

            files_list = "\n".join(f"• {n}" for n in saved)
            QMessageBox.information(dlg, "Saved", f"Saved:\n{files_list}\n\nin {out_dir}")
            dlg.accept()

        apply_btn.clicked.connect(do_apply)
        dlg.exec()

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

5. AI PROCESSING (after a transcription)
   Clean Up: fix punctuation, remove filler words
   Translate: convert to another language, segment timing preserved

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
<p>Version 1.5</p>
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
