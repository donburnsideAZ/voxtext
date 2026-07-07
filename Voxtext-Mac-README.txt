Voxtext for Mac — Internal Beta
================================

WHAT THIS IS
Local audio/video transcription. Drag a file in, get a transcript out.
100% local — your audio never leaves your computer.

INSTALL
1. Unzip this folder
2. Drag the entire Voxtext folder into your Applications folder
   KEEP Voxtext.app AND ffmpeg TOGETHER — don't separate them.
3. Double-click Voxtext.app to launch

⚠️  FIRST LAUNCH — READ THIS FIRST  ⚠️
Because this app isn't signed by Apple yet, macOS will refuse to open it
the first time with one of these messages:

  "Voxtext can't be opened because Apple cannot check it for malicious software"
  OR
  "Voxtext is damaged and can't be opened"

This is NORMAL for unsigned apps. The app isn't actually damaged.

TO FIX IT (one time only):

Method 1 — Right-click trick (easiest):
  1. Right-click (or Control-click) on Voxtext.app
  2. Choose "Open" from the menu
  3. Click "Open" in the dialog that appears
  4. After this first launch, double-click works normally

Method 2 — Terminal command (if Method 1 doesn't work):
  Open Terminal and paste this, then press Enter:

    xattr -cr /Applications/Voxtext

  Then double-click Voxtext.app normally.

FIRST TRANSCRIPTION
- Choose your model and run a test transcription
- First model download takes a few minutes — this only happens once
- After the model is cached, subsequent transcriptions are much faster

WHICH MODEL TO USE
- Tiny / Base: Fast, good for quick drafts
- Small: Balanced — recommended starting point
- Medium: Professional quality, slower
- Large: Maximum quality, slowest

OUTPUT FILES
Saved next to the source file. Example:
  ~/Recordings/meeting.mp3
  → ~/Recordings/meeting_transcript.txt
  → ~/Recordings/meeting_transcript.srt

KNOWN QUIRKS
- No paragraph breaks in transcripts yet
- No speaker labels yet
- Code signing coming in a future release (will eliminate the first-launch dance)

FEEDBACK
Found a bug? Have a feature request? Tell Don.
This is a beta — your feedback shapes v1.0.
