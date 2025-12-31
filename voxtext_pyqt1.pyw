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

# Ensure bundled FFmpeg is on PATH before any Whisper usage
def _ensure_local_ffmpeg_on_path():
    """
    Add bundled ffmpeg to PATH early in application startup.
    Works for both .py scripts and compiled executables.
    """
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            exe_dir = sys._MEIPASS
        else:
            exe_dir = os.path.dirname(os.path.abspath(__file__))
        
        local_ffmpeg_dir = os.path.join(exe_dir, "ffmpeg")
        
        if os.path.isdir(local_ffmpeg_dir):
            os.environ["PATH"] = local_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_ensure_local_ffmpeg_on_path()

# Try to import Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Note: openai-whisper not installed. Use 'Install Whisper' button or run: pip install openai-whisper")

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QButtonGroup, QFrame, QComboBox,
    QLineEdit, QTextEdit, QScrollArea, QDialog, QMenuBar, QMenu,
    QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
from PyQt6.QtGui import QFont, QAction, QDragEnterEvent, QDropEvent, QDesktopServices


class TranscriptionWorker(QThread):
    """Worker thread for transcription to keep UI responsive"""
    progress = pyqtSignal(str, int)  # message, percentage
    finished = pyqtSignal(list)  # list of created files
    error = pyqtSignal(str)  # error message
    
    def __init__(self, file_path, model_name, output_formats, output_dir, lms_settings=None):
        super().__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.output_formats = output_formats
        self.output_dir = output_dir
        self.lms_settings = lms_settings or {}
        self.cancelled = False
        self.model = None
    
    def run(self):
        try:
            import whisper
            import ssl
            import urllib.request
            
            if self.cancelled:
                return
            
            # Load model
            self.progress.emit(f"Loading {self.model_name} model (first time downloads)...", 5)
            
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
            
            self.progress.emit("Model loaded successfully.", 10)
            
            # Transcribe
            self.progress.emit("Transcribing audio... This may take several minutes.", 20)
            result = self.model.transcribe(
                self.file_path,
                language="en",
                task="transcribe"
            )
            
            if self.cancelled:
                return
            
            self.progress.emit("Transcription complete. Writing output files...", 80)
            
            # Write output files
            base_path = Path(self.file_path).stem
            created_files = []
            num_formats = len(self.output_formats)
            progress_per_format = 15 / num_formats if num_formats > 0 else 15
            current_progress = 80
            
            if 'txt' in self.output_formats:
                if self.cancelled:
                    return
                txt_path = self.output_dir / f"{base_path}_transcript.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                created_files.append(str(txt_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {txt_path.name}", int(current_progress))
            
            if 'srt' in self.output_formats:
                if self.cancelled:
                    return
                srt_path = self.output_dir / f"{base_path}_transcript.srt"
                self._write_srt(result, srt_path)
                created_files.append(str(srt_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {srt_path.name}", int(current_progress))
            
            if 'vtt' in self.output_formats:
                if self.cancelled:
                    return
                vtt_path = self.output_dir / f"{base_path}_transcript.vtt"
                self._write_vtt(result, vtt_path)
                created_files.append(str(vtt_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {vtt_path.name}", int(current_progress))
            
            if 'html' in self.output_formats:
                if self.cancelled:
                    return
                html_path = self.output_dir / f"{base_path}_transcript.html"
                self._write_html(result, html_path)
                created_files.append(str(html_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {html_path.name}", int(current_progress))
            
            if 'md' in self.output_formats:
                if self.cancelled:
                    return
                md_path = self.output_dir / f"{base_path}_transcript.md"
                self._write_markdown(result, md_path)
                created_files.append(str(md_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {md_path.name}", int(current_progress))
            
            if 'json' in self.output_formats:
                if self.cancelled:
                    return
                json_path = self.output_dir / f"{base_path}_transcript.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                created_files.append(str(json_path))
                current_progress += progress_per_format
                self.progress.emit(f"Created {json_path.name}", int(current_progress))
            
            if not self.cancelled:
                self.finished.emit(created_files)
                
        except FileNotFoundError as e:
            if not self.cancelled:
                if 'ffmpeg' in str(e).lower():
                    error_msg = (
                        "FFmpeg not found!\n\n"
                        "Whisper requires FFmpeg to process audio/video files.\n\n"
                        "To install on Mac:\n"
                        "  brew install ffmpeg\n\n"
                        "To install on Windows:\n"
                        "  Download from ffmpeg.org\n\n"
                        "To install on Linux:\n"
                        "  sudo apt install ffmpeg"
                    )
                else:
                    error_msg = str(e)
                self.error.emit(error_msg)
        except Exception as e:
            if not self.cancelled:
                self.error.emit(str(e))
    
    def _write_srt(self, result, output_path):
        """Write SRT subtitle format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], start=1):
                start = self._format_timestamp(segment['start'], use_comma=True)
                end = self._format_timestamp(segment['end'], use_comma=True)
                text = segment['text'].strip()
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def _write_vtt(self, result, output_path):
        """Write WebVTT subtitle format with optional LMS styling"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n")
            
            # Add STYLE block if LMS styling is enabled
            if self.lms_settings.get('enabled'):
                css_content = self.lms_settings.get('css', '').strip()
                if css_content:
                    f.write("\nSTYLE\n")
                    f.write(css_content)
                    f.write("\n")
            
            f.write("\n")
            
            # Get cue settings if LMS styling is enabled
            cue_settings = ""
            if self.lms_settings.get('enabled'):
                cue_settings = self.lms_settings.get('cue_settings', '').strip()
                if cue_settings:
                    cue_settings = " " + cue_settings
            
            for segment in result['segments']:
                start = self._format_timestamp(segment['start'], use_comma=False)
                end = self._format_timestamp(segment['end'], use_comma=False)
                text = segment['text'].strip()
                f.write(f"{start} --> {end}{cue_settings}\n")
                f.write(f"{text}\n\n")
    
    def _write_html(self, result, output_path):
        """Write HTML format with collapsible details tag"""
        html_content = f"""<h3>Full Transcript</h3>
<details>
<summary>Click to expand transcript</summary>

{result['text']}

</details>

<p><em>Transcribed with Voxtext using OpenAI Whisper</em></p>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _write_markdown(self, result, output_path):
        """Write Markdown format"""
        md_content = f"""# Transcript

{result['text']}

---

*Transcribed with Voxtext using OpenAI Whisper*
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _format_timestamp(self, seconds, use_comma=True):
        """Format seconds as SRT/VTT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        separator = ',' if use_comma else '.'
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millisecs:03d}"
    
    def cancel(self):
        self.cancelled = True


class VoxtextWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.selected_file = None
        self.worker = None
        self.start_time = None
        self.timer_id = None
        
        # LMS VTT Styling Presets
        self.vtt_presets = {
            'Custom': {
                'cue_settings': '',
                'css': ''
            },
            'LMS Standard': {
                'cue_settings': 'line:80%',
                'css': '::cue {\n  background-color: rgb(0, 0, 0, 60%);\n  line-height: 1.5em;\n}'
            },
            'Lower Third': {
                'cue_settings': 'line:90% align:start',
                'css': '::cue {\n  background-color: rgba(0, 0, 0, 0.8);\n  color: #ffffff;\n  font-size: 1.1em;\n}'
            },
            'High Contrast': {
                'cue_settings': 'line:85%',
                'css': '::cue {\n  background-color: #000000;\n  color: #ffff00;\n  font-weight: bold;\n}'
            }
        }
        
        # File extensions
        self.audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        
        self.init_ui()
        self.setAcceptDrops(True)
    
    def init_ui(self):
        self.setWindowTitle("Voxtext - Local Transcription")
        self.setMinimumSize(900, 600)
        self.resize(950, 700)
        
        # Standard font size (16pt base, all black)
        self.std_font = QFont("Segoe UI", 16)
        self.std_font_bold = QFont("Segoe UI", 16, QFont.Weight.Bold)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - 950) // 2
        y = (screen.height() - 700) // 2
        self.move(x, y)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Central widget with light background color
        central = QWidget()
        central.setStyleSheet("background-color: #f0f7fa;")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(40, 25, 40, 25)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("Voxtext")
        title.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #000000;")
        layout.addWidget(title)
        
        # Drop zone / file selection area
        self.drop_zone = QFrame()
        self.drop_zone.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #4a90e2;
                border-radius: 8px;
            }
            QFrame:hover {
                background-color: #f0f8ff;
            }
        """)
        self.drop_zone.setCursor(Qt.CursorShape.PointingHandCursor)
        self.drop_zone.setFixedHeight(70)
        self.drop_zone.mousePressEvent = lambda e: self.browse_file()
        
        drop_layout = QVBoxLayout(self.drop_zone)
        self.drop_label = QLabel("Drag and drop file here (or click to browse)")
        self.drop_label.setFont(self.std_font_bold)
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("border: none; color: #000000;")
        drop_layout.addWidget(self.drop_label)
        layout.addWidget(self.drop_zone)
        
        # Whisper warning (if not installed)
        if not WHISPER_AVAILABLE:
            warning = QLabel("⚠️  Whisper not installed. Click 'Install Whisper' below to get started.")
            warning.setFont(self.std_font)
            warning.setStyleSheet("background-color: #cc3333; color: white; padding: 8px; border-radius: 4px;")
            warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(warning)
        
        # Selected file display
        self.file_label = QLabel("No file selected")
        self.file_label.setFont(self.std_font_bold)
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #000000;")
        layout.addWidget(self.file_label)
        
        # === TWO COLUMN LAYOUT: Model (left) | Output (right) ===
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(20)
        
        # LEFT COLUMN: Model selection
        model_group = QGroupBox("Model Quality")
        model_group.setFont(self.std_font_bold)
        model_group.setStyleSheet("""
            QGroupBox { 
                color: #000000; 
                background-color: #f0f7fa;
                border: 1px solid #c8dde8;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            } 
            QGroupBox::title { 
                color: #000000;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        model_layout = QVBoxLayout(model_group)
        
        self.model_group = QButtonGroup(self)
        models = [
            ('tiny', '75MB', 'Basic', 'Very Fast'),
            ('base', '142MB', 'Good', 'Fast'),
            ('small', '466MB', 'Better', 'Moderate'),
            ('medium', '1.5GB', 'Professional', 'Slower'),
            ('large', '3GB', 'Maximum', 'Slowest')
        ]
        
        for name, size, quality, speed in models:
            rb = QRadioButton(f"{name.capitalize()} ({size})")
            rb.setFont(self.std_font)
            rb.setStyleSheet("color: #000000; padding: 6px 0px;")
            rb.setProperty("model_name", name)
            self.model_group.addButton(rb)
            model_layout.addWidget(rb)
            if name == 'medium':
                rb.setChecked(True)
        
        model_layout.addStretch()
        columns_layout.addWidget(model_group)
        
        # RIGHT COLUMN: Output format selection
        format_group = QGroupBox("Output Formats")
        format_group.setFont(self.std_font_bold)
        format_group.setStyleSheet("""
            QGroupBox { 
                color: #000000; 
                background-color: #f0f7fa;
                border: 1px solid #c8dde8;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            } 
            QGroupBox::title { 
                color: #000000;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        format_layout = QVBoxLayout(format_group)
        
        self.format_checks = {}
        
        # All formats in a single column for clarity
        formats = [
            ('txt', 'Text (.txt)', True),
            ('srt', 'SRT (.srt)', False),
            ('vtt', 'WebVTT (.vtt)', False),
            ('html', 'HTML (.html)', False),
            ('md', 'Markdown (.md)', False),
            ('json', 'JSON', False)
        ]
        
        for key, label, default in formats:
            cb = QCheckBox(label)
            cb.setFont(self.std_font)
            cb.setStyleSheet("color: #000000; padding: 6px 0px;")
            cb.setChecked(default)
            self.format_checks[key] = cb
            format_layout.addWidget(cb)
        
        format_layout.addStretch()
        columns_layout.addWidget(format_group)
        
        layout.addLayout(columns_layout)
        
        # Connect VTT checkbox to show/hide LMS panel
        self.format_checks['vtt'].stateChanged.connect(self.on_vtt_toggle)
        
        # LMS Styling Panel (full width below columns)
        self.lms_panel = QGroupBox("LMS VTT Styling")
        self.lms_panel.setFont(self.std_font_bold)
        self.lms_panel.setStyleSheet("""
            QGroupBox { 
                color: #000000; 
                background-color: #f0f7fa;
                border: 1px solid #c8dde8;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            } 
            QGroupBox::title { 
                color: #000000;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        self.lms_panel.setVisible(False)
        lms_layout = QVBoxLayout(self.lms_panel)
        
        # Enable checkbox and preset row
        header_row = QHBoxLayout()
        self.lms_enabled = QCheckBox("Enable LMS Styling")
        self.lms_enabled.setFont(self.std_font)
        self.lms_enabled.setStyleSheet("color: #000000;")
        self.lms_enabled.stateChanged.connect(self.toggle_lms_options)
        header_row.addWidget(self.lms_enabled)
        
        preset_label = QLabel("Preset:")
        preset_label.setFont(self.std_font)
        preset_label.setStyleSheet("color: #000000;")
        header_row.addWidget(preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.setFont(self.std_font)
        self.preset_combo.setMinimumWidth(180)
        self.preset_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #000000;
                padding: 4px 8px;
                border: 1px solid #b0d4e8;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                background-color: #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
                selection-background-color: #4a90e2;
                selection-color: #ffffff;
            }
        """)
        self.preset_combo.addItems(list(self.vtt_presets.keys()))
        self.preset_combo.setCurrentText("LMS Standard")
        self.preset_combo.currentTextChanged.connect(self.apply_vtt_preset)
        header_row.addWidget(self.preset_combo)
        header_row.addStretch()
        lms_layout.addLayout(header_row)
        
        # Options container (shown/hidden based on enable checkbox)
        self.lms_options = QWidget()
        options_layout = QVBoxLayout(self.lms_options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        
        # Cue settings row
        cue_row = QHBoxLayout()
        cue_label = QLabel("Cue settings:")
        cue_label.setFont(self.std_font)
        cue_label.setStyleSheet("color: #000000;")
        cue_row.addWidget(cue_label)
        self.cue_settings = QLineEdit("line:80%")
        self.cue_settings.setFont(QFont("Consolas", 16))
        self.cue_settings.setStyleSheet("color: #000000;")
        cue_row.addWidget(self.cue_settings)
        options_layout.addLayout(cue_row)
        
        # CSS block
        css_label = QLabel("STYLE block:")
        css_label.setFont(self.std_font)
        css_label.setStyleSheet("color: #000000;")
        options_layout.addWidget(css_label)
        self.css_text = QTextEdit()
        self.css_text.setFont(QFont("Consolas", 16))
        self.css_text.setStyleSheet("color: #000000;")
        self.css_text.setMaximumHeight(120)
        self.css_text.setPlainText("::cue {\n  background-color: rgb(0, 0, 0, 60%);\n  line-height: 1.5em;\n}")
        options_layout.addWidget(self.css_text)
        
        self.lms_options.setVisible(False)
        lms_layout.addWidget(self.lms_options)
        
        layout.addWidget(self.lms_panel)
        
        # Output location notice
        output_notice = QLabel("Files saved next to your source file")
        output_notice.setFont(self.std_font)
        output_notice.setStyleSheet("color: #000000;")
        output_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(output_notice)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(25)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("0%")
        self.progress_label.setFont(self.std_font_bold)
        self.progress_label.setStyleSheet("color: #000000;")
        self.progress_label.setFixedWidth(60)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)
        
        # Elapsed time
        self.elapsed_label = QLabel("")
        self.elapsed_label.setFont(self.std_font)
        self.elapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.elapsed_label.setStyleSheet("color: #000000;")
        layout.addWidget(self.elapsed_label)
        
        # Status label
        self.status_label = QLabel("Ready to transcribe")
        self.status_label.setFont(self.std_font)
        self.status_label.setStyleSheet("color: #000000;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        if not WHISPER_AVAILABLE:
            self.install_btn = QPushButton("Install Whisper")
            self.install_btn.setFont(self.std_font)
            self.install_btn.setStyleSheet("background-color: #E67E22; color: white; padding: 10px 20px; border-radius: 4px;")
            self.install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.install_btn.clicked.connect(self.install_whisper)
            button_layout.addWidget(self.install_btn)
        
        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setFont(self.std_font_bold)
        self.transcribe_btn.setStyleSheet("background-color: #4a90e2; color: white; padding: 10px 24px; border-radius: 4px;")
        self.transcribe_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(WHISPER_AVAILABLE)
        button_layout.addWidget(self.transcribe_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFont(self.std_font)
        self.clear_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 10px 20px; border-radius: 4px;")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_or_cancel)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        clear_action = QAction("Clear Selection", self)
        clear_action.setShortcut("Ctrl+K")
        clear_action.triggered.connect(self.clear_or_cancel)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        # Only show Exit on Windows/Linux
        import platform
        if platform.system() != "Darwin":
            exit_action = QAction("Exit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        clear_cache_action = QAction("Clear Model Cache...", self)
        clear_cache_action.triggered.connect(self.clear_cache)
        tools_menu.addAction(clear_cache_action)
        
        manage_models_action = QAction("Manage Models...", self)
        manage_models_action.triggered.connect(self.manage_models)
        tools_menu.addAction(manage_models_action)
        
        tools_menu.addSeparator()
        
        open_folder_action = QAction("Open Model Folder", self)
        open_folder_action.triggered.connect(self.open_model_folder)
        tools_menu.addAction(open_folder_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        getting_started_action = QAction("Getting Started", self)
        getting_started_action.triggered.connect(self.show_getting_started)
        help_menu.addAction(getting_started_action)
        
        speed_guide_action = QAction("Processing Speed Guide", self)
        speed_guide_action.triggered.connect(self.show_speed_guide)
        help_menu.addAction(speed_guide_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("About Voxtext", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        help_menu.addSeparator()
        
        website_action = QAction("Visit Website", self)
        website_action.triggered.connect(lambda: webbrowser.open("https://donburnside.com"))
        help_menu.addAction(website_action)
    
    # === Drag and Drop ===
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.selected_file = file_path
                self.update_file_display(file_path)
                self.status_label.setText("File loaded. Ready to transcribe.")
    
    # === File Selection ===
    def browse_file(self):
        if self.worker and self.worker.isRunning():
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or Video File",
            "",
            "Media Files (*.mp3 *.mp4 *.wav *.m4a *.flac *.ogg *.mov *.avi *.mkv *.webm);;Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*.*)"
        )
        
        if file_path:
            self.selected_file = file_path
            self.update_file_display(file_path)
            self.status_label.setText("File loaded. Ready to transcribe.")
    
    def update_file_display(self, filepath):
        ext = Path(filepath).suffix.lower()
        if ext in self.audio_extensions:
            emoji = "🎵"
        elif ext in self.video_extensions:
            emoji = "🎬"
        else:
            emoji = "📄"
        
        filename = os.path.basename(filepath)
        self.file_label.setText(f"{emoji} {filename}")
        self.file_label.setStyleSheet("color: #000000;")
    
    # === LMS Panel Toggle ===
    def on_vtt_toggle(self, state):
        is_visible = state == Qt.CheckState.Checked.value
        self.lms_panel.setVisible(is_visible)
        
        # Dynamically resize window
        if is_visible:
            # Expand window to accommodate LMS panel
            self.resize(950, 950)
        else:
            # Shrink window back to compact size
            self.resize(950, 700)
    
    def toggle_lms_options(self, state):
        is_visible = state == Qt.CheckState.Checked.value
        self.lms_options.setVisible(is_visible)
        
        # Adjust window height when options expand/collapse
        if is_visible:
            self.resize(950, 1000)
        else:
            self.resize(950, 850)
    
    def apply_vtt_preset(self, preset_name):
        if preset_name in self.vtt_presets:
            preset = self.vtt_presets[preset_name]
            self.cue_settings.setText(preset['cue_settings'])
            self.css_text.setPlainText(preset['css'])
    
    # === Transcription ===
    def start_transcription(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please select an audio or video file first.")
            return
        
        if not WHISPER_AVAILABLE:
            QMessageBox.warning(self, "Whisper Not Installed", "Please install Whisper first.")
            return
        
        output_formats = [k for k, cb in self.format_checks.items() if cb.isChecked()]
        if not output_formats:
            QMessageBox.warning(self, "No Format", "Please select at least one output format.")
            return
        
        # Get model name
        selected_btn = self.model_group.checkedButton()
        model_name = selected_btn.property("model_name") if selected_btn else "medium"
        
        # Get output directory
        output_dir = Path(self.selected_file).parent
        
        # Get LMS settings
        lms_settings = {
            'enabled': self.lms_enabled.isChecked(),
            'cue_settings': self.cue_settings.text(),
            'css': self.css_text.toPlainText()
        }
        
        # Update UI
        self.transcribe_btn.setEnabled(False)
        self.clear_btn.setText("Cancel")
        self.clear_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px 16px; border-radius: 4px;")
        
        self.start_time = time.time()
        self.timer_id = self.startTimer(1000)
        
        # Start worker thread
        self.worker = TranscriptionWorker(
            self.selected_file, model_name, output_formats, output_dir, lms_settings
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_transcription_complete)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.start()
    
    def on_progress(self, message, percentage):
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"{percentage}%")
    
    def on_transcription_complete(self, created_files):
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")
        self.status_label.setText("Transcription complete!")
        
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = None
        
        if self.start_time:
            total_time = int(time.time() - self.start_time)
            minutes = total_time // 60
            seconds = total_time % 60
            self.elapsed_label.setText(f"Completed in {minutes}m {seconds}s")
        
        self.reset_buttons()
        
        # Show success message
        files_list = "\n".join([f"• {Path(f).name}" for f in created_files])
        QMessageBox.information(
            self,
            "Success!",
            f"Transcription complete!\n\nCreated files:\n{files_list}\n\nSaved to:\n{Path(self.selected_file).parent}"
        )
    
    def on_transcription_error(self, error_msg):
        self.progress_bar.setValue(0)
        self.progress_label.setText("0%")
        self.status_label.setText("Error occurred. Please try again.")
        self.elapsed_label.setText("")
        
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = None
        
        self.reset_buttons()
        
        QMessageBox.critical(
            self,
            "Transcription Error",
            f"Transcription failed:\n\n{error_msg}\n\nPlease check the file and try again."
        )
    
    def clear_or_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            self.status_label.setText("Transcription cancelled")
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")
            self.elapsed_label.setText("")
            if self.timer_id:
                self.killTimer(self.timer_id)
                self.timer_id = None
            self.reset_buttons()
        else:
            self.selected_file = None
            self.file_label.setText("No file selected")
            self.file_label.setStyleSheet("color: #666666;")
            self.status_label.setText("Ready to transcribe")
            self.progress_bar.setValue(0)
            self.progress_label.setText("0%")
            self.elapsed_label.setText("")
    
    def reset_buttons(self):
        self.transcribe_btn.setEnabled(WHISPER_AVAILABLE)
        self.clear_btn.setText("Clear")
        self.clear_btn.setStyleSheet("background-color: #95a5a6; color: white; padding: 8px 16px; border-radius: 4px;")
    
    def timerEvent(self, event):
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.elapsed_label.setText(f"Elapsed: {minutes}m {seconds}s")
    
    # === Install Whisper ===
    def install_whisper(self):
        self.status_label.setText("Installing Whisper... This may take a minute.")
        
        def install():
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "openai-whisper", "--break-system-packages"],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0, result.stderr
            except Exception as e:
                return False, str(e)
        
        def on_complete(success, error):
            if success:
                global WHISPER_AVAILABLE
                WHISPER_AVAILABLE = True
                self.status_label.setText("Whisper installed successfully!")
                self.transcribe_btn.setEnabled(True)
                QMessageBox.information(
                    self, "Success",
                    "Whisper installed successfully!\n\nYou can now transcribe audio and video files."
                )
            else:
                self.status_label.setText("Installation failed. See error message.")
                QMessageBox.critical(
                    self, "Installation Error",
                    f"Failed to install Whisper:\n\n{error}\n\nTry running: pip install openai-whisper"
                )
        
        # Run in thread
        thread = threading.Thread(target=lambda: on_complete(*install()))
        thread.daemon = True
        thread.start()
    
    # === Menu Actions ===
    def clear_cache(self):
        import platform
        import shutil
        
        home = Path.home()
        if platform.system() == "Darwin":
            cache_dir = home / ".cache" / "whisper"
        elif platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"
        else:
            cache_dir = home / ".cache" / "whisper"
        
        if not cache_dir.exists():
            QMessageBox.information(self, "Cache Empty", "No model cache found. Models will be downloaded on first use.")
            return
        
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        result = QMessageBox.question(
            self, "Clear Cache",
            f"Model cache is using {size_mb:.1f} MB.\n\nClearing cache will delete all downloaded models. They will be re-downloaded when needed.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if result == QMessageBox.StandardButton.Yes:
            try:
                shutil.rmtree(cache_dir)
                QMessageBox.information(self, "Success", f"Cleared {size_mb:.1f} MB from cache.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache:\n\n{str(e)}")
    
    def manage_models(self):
        import platform
        
        home = Path.home()
        if platform.system() == "Darwin":
            cache_dir = home / ".cache" / "whisper"
        elif platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"
        else:
            cache_dir = home / ".cache" / "whisper"
        
        if not cache_dir.exists():
            QMessageBox.information(self, "No Models", "No models have been downloaded yet.")
            return
        
        models = list(cache_dir.glob("*.pt"))
        if not models:
            QMessageBox.information(self, "No Models", "No models found in cache.")
            return
        
        model_info = []
        for m in sorted(models):
            size_mb = m.stat().st_size / (1024 * 1024)
            model_info.append(f"• {m.stem} ({size_mb:.0f} MB)")
        
        QMessageBox.information(
            self, "Downloaded Models",
            f"Downloaded models:\n\n" + "\n".join(model_info) + f"\n\nLocation: {cache_dir}"
        )
    
    def open_model_folder(self):
        import platform
        import subprocess
        
        home = Path.home()
        if platform.system() == "Darwin":
            cache_dir = home / ".cache" / "whisper"
        elif platform.system() == "Windows":
            cache_dir = home / "AppData" / "Local" / "whisper"
        else:
            cache_dir = home / ".cache" / "whisper"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", str(cache_dir)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(cache_dir)])
            else:
                subprocess.run(["xdg-open", str(cache_dir)])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open folder:\n\n{str(e)}")
    
    def show_getting_started(self):
        QMessageBox.information(
            self, "Getting Started",
            """Getting Started with Voxtext

1. SELECT A FILE
   Click the drop zone or drag & drop an audio/video file
   Supported: MP3, MP4, WAV, M4A, MOV, and more

2. CHOOSE A MODEL
   Tiny/Base: Fast, good for older computers
   Small: Balanced quality and speed
   Medium: Professional quality (slower)
   Large: Maximum quality (very slow)

3. SELECT OUTPUT FORMATS
   Text (.txt): Plain transcript
   SRT (.srt): Subtitles with timestamps
   WebVTT (.vtt): Web video captions
   HTML (.html): Formatted for websites
   Markdown (.md): For documentation
   JSON: Full data with timestamps

4. CLICK TRANSCRIBE
   Processing time: 2-5x your audio length
   Files saved next to your source file

TIPS:
• Use Tiny/Base models for testing
• Your audio never leaves your computer
• Models download once, then cached forever"""
        )
    
    def show_speed_guide(self):
        QMessageBox.information(
            self, "Processing Speed Guide",
            """Processing Speed Guide

WHY IS IT SLOW?
Transcription happens locally on your computer.
Speed depends on model size, CPU/GPU, and audio length.

TYPICAL TIMES (for 10 min audio):

MODERN COMPUTER (2020+):
• Tiny: 30 seconds - 1 minute
• Base: 1-2 minutes
• Small: 2-3 minutes
• Medium: 3-5 minutes
• Large: 5-10 minutes

OLDER COMPUTER:
• Tiny: 2-3 minutes
• Base: 3-5 minutes
• Small: 5-10 minutes
• Medium/Large: Not recommended

RECOMMENDATIONS:
• Podcasts: Medium or Small
• Quick drafts: Tiny
• Maximum accuracy: Large (modern PC only)

REMEMBER:
• All models are 100% free
• Processing is 100% local (private!)
• Models cached forever after first download"""
        )
    
    def show_about(self):
        QMessageBox.about(
            self, "About Voxtext",
            """<h2>Voxtext</h2>
<p>Version 1.1</p>
<p>Local Audio & Video Transcription</p>

<p>✓ 100% Local Processing<br>
✓ No API Keys Required<br>
✓ No Subscriptions Ever<br>
✓ Your Privacy Matters</p>

<p>Powered by OpenAI Whisper<br>
Open source and completely free</p>

<p>Created by Don Burnside<br>
Part of the Vox Suite</p>

<p><a href="https://donburnside.com">donburnside.com</a></p>

<p><i>Also check out Voxsmith for AI narration</i></p>"""
        )


def main():
    # CRITICAL: Prevents PyInstaller from spawning duplicate windows
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    app.setApplicationName("Voxtext")
    app.setOrganizationName("Vox Suite")
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = VoxtextWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
