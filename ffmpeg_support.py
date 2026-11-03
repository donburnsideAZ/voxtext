"""
ffmpeg_support.py — FFmpeg discovery and diagnostics for the Vox Suite.

Why this module exists
----------------------
Whisper shells out to `ffmpeg` to decode audio. If ffmpeg isn't reachable,
Python raises a bare FileNotFoundError. On Windows that surfaces as:

    [WinError 2] The system cannot find the file specified

...with no mention of ffmpeg anywhere in the message. Any error handler that
substring-matches on "ffmpeg" will silently miss it, and the user gets an
opaque dialog and quietly gives up.

So: find ffmpeg deliberately, check for it BEFORE starting a transcription,
and detect the missing-binary condition by errno/winerror rather than by
sniffing the message text.

Drop this next to voxtext_pyqt.py and import it. See INTEGRATION notes at the
bottom of this file.
"""

import os
import platform
import subprocess
import sys

__all__ = [
    "EXE_NAME",
    "find_ffmpeg",
    "ensure_ffmpeg_on_path",
    "ffmpeg_version",
    "is_missing_binary_error",
    "missing_ffmpeg_message",
    "diagnostic_report",
    "no_window_kwargs",
]

_IS_WINDOWS = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"

EXE_NAME = "ffmpeg.exe" if _IS_WINDOWS else "ffmpeg"

# Windows: keep child processes from flashing a console window.
_CREATE_NO_WINDOW = 0x08000000


def no_window_kwargs():
    """subprocess kwargs that suppress the console flash on Windows.

    Safe to splat into any subprocess call on any platform:
        subprocess.run(cmd, **no_window_kwargs())
    """
    if not _IS_WINDOWS:
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return {"creationflags": _CREATE_NO_WINDOW, "startupinfo": si}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _bundle_dirs():
    """Directories to check for an ffmpeg we shipped ourselves.

    Checked first and in this order, so a bundled binary always wins over
    whatever version happens to be installed on the user's machine. We know
    ours works; we don't know theirs does.
    """
    dirs = []

    # PyInstaller: _MEIPASS is the extraction root (onedir -> the _internal
    # folder on Windows/Linux, Contents/Frameworks inside a Mac .app).
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.join(base, "ffmpeg"))
    dirs.append(base)

    # Also check next to the executable itself. This covers the case where
    # ffmpeg.exe is dropped in by the Inno Setup script rather than baked in
    # by PyInstaller's --add-binary.
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        dirs.append(os.path.join(exe_dir, "ffmpeg"))
        dirs.append(exe_dir)
        if _IS_MAC:
            # Contents/MacOS/../Resources/ffmpeg
            dirs.append(os.path.join(exe_dir, os.pardir, "Resources", "ffmpeg"))

    return dirs


def _system_dirs():
    """Common install locations that a stripped PATH won't include.

    macOS .app bundles launched from Finder inherit only
    /usr/bin:/bin:/usr/sbin:/sbin — no Homebrew. So `brew install ffmpeg`
    works in Terminal but the built .app can't see it. Probe explicitly.
    """
    if _IS_MAC:
        return [
            "/opt/homebrew/bin",   # Apple Silicon Homebrew
            "/usr/local/bin",      # Intel Homebrew
            "/opt/local/bin",      # MacPorts
        ]
    if _IS_WINDOWS:
        dirs = [
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
        ]
        # winget / scoop / chocolatey shims land in per-user locations.
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "WinGet", "Links"))
        home = os.path.expanduser("~")
        dirs.append(os.path.join(home, "scoop", "shims"))
        program_data = os.environ.get("ProgramData")
        if program_data:
            dirs.append(os.path.join(program_data, "chocolatey", "bin"))
        return dirs
    return ["/usr/local/bin", "/usr/bin", "/snap/bin"]


def _executable_in(directory):
    """Return the full path to ffmpeg in `directory`, or None."""
    if not directory:
        return None
    try:
        candidate = os.path.abspath(os.path.join(directory, EXE_NAME))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    except Exception:
        pass
    return None


def find_ffmpeg():
    """Locate an ffmpeg executable. Returns an absolute path, or None.

    Search order: bundled -> current PATH -> known system install dirs.
    """
    for d in _bundle_dirs():
        found = _executable_in(d)
        if found:
            return found

    try:
        import shutil
        on_path = shutil.which("ffmpeg")
        if on_path:
            return os.path.abspath(on_path)
    except Exception:
        pass

    for d in _system_dirs():
        found = _executable_in(d)
        if found:
            return found

    return None


def ensure_ffmpeg_on_path():
    """Find ffmpeg and make sure its directory is on PATH.

    Whisper invokes ffmpeg by bare name via subprocess, so PATH is what
    actually matters — setting an env var isn't enough.

    Returns the absolute path to ffmpeg, or None if it wasn't found.
    Call this once at import time, before anything touches Whisper.
    """
    found = find_ffmpeg()
    if not found:
        return None

    directory = os.path.dirname(found)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if directory not in parts:
        os.environ["PATH"] = directory + os.pathsep + current

    # Some libraries honor these; harmless if nothing reads them.
    os.environ.setdefault("FFMPEG_BINARY", found)
    return found


def ffmpeg_version(path=None):
    """Return the first line of `ffmpeg -version`, or None if it won't run.

    Catches the case where ffmpeg exists on disk but is broken — wrong
    architecture, missing DLLs (a shared build with its libav* DLLs left
    behind), or blocked by policy.
    """
    exe = path or find_ffmpeg()
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            **no_window_kwargs(),
        )
        if result.returncode != 0:
            return None
        first = (result.stdout or "").splitlines()
        return first[0].strip() if first else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
def is_missing_binary_error(exc):
    """True if `exc` looks like 'the OS could not find an executable'.

    Checks errno/winerror instead of substring-matching the message, because
    the Windows message for a missing executable never contains the word
    "ffmpeg" — which is exactly how ~80 users hit this silently.

    Note this can't distinguish "ffmpeg is missing" from "the input file is
    missing" on its own. Validate the input path before transcribing; then
    ENOENT during transcription means ffmpeg.
    """
    import errno as _errno

    if isinstance(exc, FileNotFoundError):
        return True

    winerror = getattr(exc, "winerror", None)
    if winerror in (2, 3):  # FILE_NOT_FOUND, PATH_NOT_FOUND
        return True

    if getattr(exc, "errno", None) == _errno.ENOENT:
        return True

    # Whisper wraps a non-zero ffmpeg exit in RuntimeError("Failed to load
    # audio: ..."). Different failure, but the same user-facing cause often
    # enough to be worth catching.
    text = str(exc).lower()
    return "failed to load audio" in text or "ffmpeg" in text


def missing_ffmpeg_message():
    """User-facing text for the 'no ffmpeg' condition.

    Written for someone who has never heard of ffmpeg and doesn't want to.
    """
    if _IS_WINDOWS:
        install_hint = (
            "The quickest fix is to update Voxtext — newer builds include\n"
            "everything needed.\n\n"
            "If you'd rather not update, open Command Prompt and run:\n"
            "    winget install \"FFmpeg (Essentials Build)\"\n"
            "then restart Voxtext."
        )
    elif _IS_MAC:
        install_hint = (
            "The quickest fix is to update Voxtext — newer builds include\n"
            "everything needed.\n\n"
            "If you'd rather not update, open Terminal and run:\n"
            "    brew install ffmpeg\n"
            "then restart Voxtext."
        )
    else:
        install_hint = (
            "Install it with your package manager, for example:\n"
            "    sudo apt install ffmpeg\n"
            "then restart Voxtext."
        )

    return (
        "Voxtext is missing a component it needs to read audio.\n\n"
        "Transcription can't start without it — nothing you did caused this.\n\n"
        + install_hint
    )


def diagnostic_report():
    """Multi-line environment summary, for a 'Report a problem' flow.

    Contains no file paths from the user's media and no personal content —
    just enough to tell whether ffmpeg is present and usable.
    """
    exe = find_ffmpeg()
    lines = [
        f"Platform:        {platform.platform()}",
        f"Python:          {sys.version.split()[0]}",
        f"Frozen build:    {bool(getattr(sys, 'frozen', False))}",
        f"ffmpeg found:    {exe or '(not found)'}",
        f"ffmpeg version:  {ffmpeg_version(exe) or '(could not run)'}",
    ]
    lines.append("Searched (bundled first):")
    for d in _bundle_dirs() + _system_dirs():
        marker = "  ->" if _executable_in(d) else "    "
        lines.append(f"{marker} {d}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# INTEGRATION — three edits in voxtext_pyqt.py
# ---------------------------------------------------------------------------
#
# 1. Replace the inline _ensure_ffmpeg_on_path() block near the top with:
#
#        from ffmpeg_support import (
#            ensure_ffmpeg_on_path, is_missing_binary_error,
#            missing_ffmpeg_message,
#        )
#        FFMPEG_PATH = ensure_ffmpeg_on_path()
#
#    Must run before anything imports whisper.
#
# 2. Preflight in start_transcription(), right after the "No File" check.
#    This is the fix that matters most — fail before the progress bar moves,
#    not four seconds into a model load:
#
#        if not os.path.isfile(self.selected_file):
#            QMessageBox.warning(self, "File Not Found",
#                "That file has moved or been renamed. Select it again.")
#            return
#
#        if FFMPEG_PATH is None:
#            QMessageBox.critical(self, "Missing Component",
#                                 missing_ffmpeg_message())
#            return
#
# 3. In TranscriptionWorker.run(), replace the
#    `except FileNotFoundError as e:` block with:
#
#        except Exception as e:
#            if self.cancelled:
#                return
#            if is_missing_binary_error(e):
#                self.error.emit(missing_ffmpeg_message())
#            else:
#                self.error.emit(str(e))
#
#    (Delete the old `if 'ffmpeg' in str(e).lower():` branch — that test is
#    what never fired on Windows.)
#
# ---------------------------------------------------------------------------
