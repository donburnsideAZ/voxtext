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
import webbrowser

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
    QLineEdit, QTextEdit, QScrollArea, QDialog, QMenuBar, QMenu,
    QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
from PyQt6.QtGui import QFont, QAction, QDragEnterEvent, QDropEvent, QDesktopServices


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

        # Window — compact but not cramped
        WIN_W, WIN_H = 820, 915
        self.setMinimumSize(780, 850)
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

        self._create_menu_bar()

        central = QWidget()
        central.setStyleSheet(f"background-color: {COLORS['background']};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(28, 14, 28, 16)
        root.setSpacing(10)

        # ── Drop zone (navy — needs strong contrast to invite interaction) ──
        self.drop_zone = QFrame()
        self.drop_zone.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['navy']};
                border: none;
                border-radius: 8px;
            }}
            QFrame:hover {{ background-color: {COLORS['navy_hover']}; }}
        """)
        self.drop_zone.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drop_zone.setFixedHeight(56)
        self.drop_zone.mousePressEvent = lambda e: self.browse_file()

        dz_layout = QVBoxLayout(self.drop_zone)
        dz_layout.setContentsMargins(0, 0, 0, 0)
        self.drop_label = QLabel("Drag and drop a file here, or click to browse")
        self.drop_label.setFont(self.f_body)
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(f"color: {COLORS['text_light']}; border: none;")
        dz_layout.addWidget(self.drop_label)
        root.addWidget(self.drop_zone)

        # ── Whisper warning ─────────────────────────────────────────────────
        # Always create so we can show/hide it; visibility controlled by WHISPER_AVAILABLE.
        self.warn_label = QLabel("⚠  Whisper not installed — click 'Install Whisper' below to get started.")
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

        # ── Selected file display ───────────────────────────────────────────
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(self.f_bold)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        root.addWidget(self.file_label)

        # ── Two-column body ─────────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(14)

        left_col  = QVBoxLayout()
        left_col.setSpacing(12)
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # — LEFT: Model quality ——————————————————————————————————————————————
        model_card = self._make_card()
        mc_layout  = QVBoxLayout(model_card)
        mc_layout.setContentsMargins(16, 14, 16, 14)
        mc_layout.setSpacing(4)

        mc_layout.addWidget(self._section_header("MODEL QUALITY"))

        self.model_group = QButtonGroup(self)
        models = [
            ('small',  '466 MB', 'Good quality',         'Faster — great for a first pass'),
            ('medium', '1.5 GB', 'Professional quality', 'Recommended for final output'),
            ('large',  '3 GB',   'Maximum accuracy',     'Slowest — use for critical work'),
        ]
        for name, size, quality, speed in models:
            rb = QRadioButton(f"{name.capitalize()}  ({size})")
            rb.setFont(self.f_body)
            rb.setStyleSheet(self._radio_style())
            rb.setProperty("model_name", name)
            self.model_group.addButton(rb)

            desc = QLabel(f"{quality}  —  {speed}")
            desc.setFont(self.f_small)
            desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none; padding-left: 22px;")

            mc_layout.addWidget(rb)
            mc_layout.addWidget(desc)

            if name == 'medium':
                rb.setChecked(True)

        left_col.addWidget(model_card)

        # — LEFT: Output formats ————————————————————————————————————————————
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
        # Two columns of checkboxes
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

        left_col.addWidget(fmt_card)
        left_col.addStretch()

        # — RIGHT: AI Processing ————————————————————————————————————————————
        ai_card    = self._make_card()
        ai_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ai_layout  = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(16, 14, 16, 16)
        ai_layout.setSpacing(10)

        ai_layout.addWidget(self._section_header("AI PROCESSING"))

        # Note at top — hidden after first transcription
        self.ai_note = QLabel("✦  Transcribe a file first to enable AI operations")
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

        # — Clean Up ——
        cleanup_lbl = QLabel("Format transcript")
        cleanup_lbl.setFont(self.f_label)
        cleanup_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        ai_layout.addWidget(cleanup_lbl)

        cleanup_desc = QLabel("Fix punctuation, remove filler words, improve readability.")
        cleanup_desc.setFont(self.f_body)
        cleanup_desc.setWordWrap(True)
        cleanup_desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        ai_layout.addWidget(cleanup_desc)

        self.format_ai_btn = QPushButton("Clean Up Transcript…")
        self.format_ai_btn.setFont(self.f_bold)
        self.format_ai_btn.setEnabled(False)
        self.format_ai_btn.setStyleSheet(self._btn_blue())
        self.format_ai_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.format_ai_btn.clicked.connect(lambda: self.open_ai_dialog("format"))
        ai_layout.addWidget(self.format_ai_btn)

        # — Divider ——
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet(f"border: none; border-top: 1px solid {COLORS['border']}; margin: 2px 0;")
        div2.setFixedHeight(1)
        ai_layout.addWidget(div2)

        # — Translate ——
        trans_lbl = QLabel("Translate transcript")
        trans_lbl.setFont(self.f_label)
        trans_lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; border: none;")
        ai_layout.addWidget(trans_lbl)

        trans_desc = QLabel("Translate to another language while preserving segment timing.")
        trans_desc.setFont(self.f_body)
        trans_desc.setWordWrap(True)
        trans_desc.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        ai_layout.addWidget(trans_desc)

        lang_row = QHBoxLayout()
        lang_row.setSpacing(8)

        from ai_processor import LANGUAGES
        self.ai_lang_combo = QComboBox()
        self.ai_lang_combo.setFont(self.f_body)
        self.ai_lang_combo.setStyleSheet(self._combo_style())
        for code, name in LANGUAGES.items():
            self.ai_lang_combo.addItem(name, code)
        lang_row.addWidget(self.ai_lang_combo, 1)

        self.translate_btn = QPushButton("Translate…")
        self.translate_btn.setFont(self.f_bold)
        self.translate_btn.setEnabled(False)
        self.translate_btn.setStyleSheet(self._btn_blue())
        self.translate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_btn.clicked.connect(lambda: self.open_ai_dialog("translate"))
        lang_row.addWidget(self.translate_btn)

        ai_layout.addLayout(lang_row)
        ai_layout.addStretch()

        right_col.addWidget(ai_card, 1)  # stretch=1 so it fills the column

        # — RIGHT: LMS VTT styling (hidden until VTT checked) ————————————————
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

        right_col.addWidget(self.lms_panel)
        right_col.addStretch()

        # Connect VTT toggle
        self.format_checks['vtt'].stateChanged.connect(self.on_vtt_toggle)

        cols.addLayout(left_col, 1)
        cols.addLayout(right_col, 1)
        root.addLayout(cols)

        # ── Progress ────────────────────────────────────────────────────────
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

        # ── Status + elapsed ─────────────────────────────────────────────────
        status_row = QHBoxLayout()

        self.status_label = QLabel("Ready to transcribe")
        self.status_label.setFont(self.f_body)
        self.status_label.setStyleSheet(f"color: {COLORS['text']}; border: none;")
        status_row.addWidget(self.status_label)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setFont(self.f_small)
        self.elapsed_label.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self.elapsed_label)

        root.addLayout(status_row)

        # ── Action buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        if not WHISPER_AVAILABLE:
            self.install_btn = QPushButton("Install Whisper")
            self.install_btn.setFont(self.f_bold)
            self.install_btn.setStyleSheet(self._btn_custom(COLORS['warning'], '#d35400'))
            self.install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.install_btn.clicked.connect(self.install_whisper)
            btn_row.addWidget(self.install_btn)

        self.transcribe_btn = QPushButton("▶  Transcribe")
        self.transcribe_btn.setFont(self.f_bold)
        self.transcribe_btn.setStyleSheet(self._btn_green())
        self.transcribe_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.transcribe_btn.setEnabled(WHISPER_AVAILABLE)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        btn_row.addWidget(self.transcribe_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFont(self.f_body)
        self.clear_btn.setStyleSheet(self._btn_secondary())
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_or_cancel)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

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
    # Menu
    # -----------------------------------------------------------------------
    def _create_menu_bar(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        a = QAction("Open File…", self); a.setShortcut("Ctrl+O"); a.triggered.connect(self.browse_file); file_menu.addAction(a)
        a = QAction("Clear Selection", self); a.setShortcut("Ctrl+K"); a.triggered.connect(self.clear_or_cancel); file_menu.addAction(a)
        file_menu.addSeparator()
        import platform
        if platform.system() != "Darwin":
            a = QAction("Exit", self); a.setShortcut("Ctrl+Q"); a.triggered.connect(self.close); file_menu.addAction(a)

        tools_menu = mb.addMenu("Tools")
        a = QAction("Clear Model Cache…", self); a.triggered.connect(self.clear_cache); tools_menu.addAction(a)
        a = QAction("Manage Models…", self); a.triggered.connect(self.manage_models); tools_menu.addAction(a)
        tools_menu.addSeparator()
        a = QAction("Open Model Folder", self); a.triggered.connect(self.open_model_folder); tools_menu.addAction(a)

        help_menu = mb.addMenu("Help")
        a = QAction("Getting Started", self); a.triggered.connect(self.show_getting_started); help_menu.addAction(a)
        a = QAction("Processing Speed Guide", self); a.triggered.connect(self.show_speed_guide); help_menu.addAction(a)
        help_menu.addSeparator()
        a = QAction("About Voxtext", self); a.triggered.connect(self.show_about); help_menu.addAction(a)
        help_menu.addSeparator()
        a = QAction("Visit Website", self); a.triggered.connect(lambda: webbrowser.open("https://donburnside.com")); help_menu.addAction(a)

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
            self, "Select Audio or Video File", "",
            "Media Files (*.mp3 *.mp4 *.wav *.m4a *.flac *.ogg *.mov *.avi *.mkv *.webm);;"
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;"
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;"
            "All Files (*.*)"
        )
        if fp:
            self._load_file(fp)

    def _load_file(self, fp):
        self.selected_file = fp
        ext = Path(fp).suffix.lower()
        emoji = "🎵" if ext in self.audio_extensions else ("🎬" if ext in self.video_extensions else "📄")
        self.file_label.setText(f"{emoji}  {os.path.basename(fp)}")
        self.file_label.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        self.status_label.setText("File loaded. Ready to transcribe.")

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
            self.file_label.setText("No file selected")
            self.file_label.setStyleSheet(f"color: {COLORS['text_muted']}; border: none;")
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
    def clear_cache(self):
        import platform, shutil
        home = Path.home()
        cache_dir = home / (".cache" if platform.system() != "Windows" else "AppData/Local") / "whisper"
        if platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"

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
        import platform
        home = Path.home()
        cache_dir = home / ".cache" / "whisper"
        if platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"
        if not cache_dir.exists():
            QMessageBox.information(self, "No Models", "No models have been downloaded yet."); return
        models = list(cache_dir.glob("*.pt"))
        if not models:
            QMessageBox.information(self, "No Models", "No models found in cache."); return
        info = "\n".join(f"• {m.stem}  ({m.stat().st_size/1048576:.0f} MB)" for m in sorted(models))
        QMessageBox.information(self, "Downloaded Models", f"{info}\n\nLocation: {cache_dir}")

    def open_model_folder(self):
        import platform, subprocess
        home = Path.home()
        cache_dir = home / ".cache" / "whisper"
        if platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"
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
   Tiny/Base: Fast, good for older computers
   Small: Balanced quality and speed
   Medium: Professional quality (recommended)
   Large: Maximum accuracy (very slow)

3. SELECT OUTPUT FORMATS
   Text (.txt): Plain transcript
   SRT (.srt): Subtitles with timestamps
   WebVTT (.vtt): Web video captions
   HTML (.html): Collapsible for WordPress
   Markdown (.md): For documentation
   JSON: Full data with timestamps

4. CLICK TRANSCRIBE
   Processing: roughly 2-5× the audio length
   Files saved next to your source file

TIPS:
• Use Tiny/Base for testing — surprisingly good!
• Your audio never leaves your computer
• Models download once and cache forever""")

    def show_speed_guide(self):
        QMessageBox.information(self, "Processing Speed Guide", """Processing Speed Guide

WHY IS IT SLOW?
Everything runs locally. Speed depends on model
size, your CPU/GPU, and audio length.

TYPICAL TIMES (for 10 min of audio):

MODERN COMPUTER (2020+):
• Tiny:   30 sec – 1 min
• Base:   1 – 2 min
• Small:  2 – 3 min
• Medium: 3 – 5 min
• Large:  5 – 10 min

OLDER COMPUTER:
• Tiny:   2 – 3 min
• Base:   3 – 5 min
• Medium/Large: not recommended

RECOMMENDATIONS:
• Podcasts: Medium or Small
• Quick drafts: Tiny (it's better than you'd think)
• Maximum accuracy: Large, modern PC only

REMEMBER:
• 100% free, 100% local, 100% private
• Models download once, cached forever""")

    def show_about(self):
        QMessageBox.about(self, "About Voxtext", """<h2>Voxtext</h2>
<p>Version 1.1</p>
<p>Local Audio &amp; Video Transcription</p>

<p>✓ 100% Local Processing<br>
✓ No API Keys Required<br>
✓ No Subscriptions Ever<br>
✓ Your Privacy Matters</p>

<p>Powered by OpenAI Whisper<br>
Open source and completely free</p>

<p>Created by Don Burnside<br>
Part of the Vox Suite</p>

<p><a href="https://donburnside.com">donburnside.com</a></p>
<p><i>Also check out Voxsmith for AI narration</i></p>""")


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
