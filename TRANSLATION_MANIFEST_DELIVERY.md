# Translation Manifest Delivery

*An optional, off-by-default power-user feature in Voxtext.*

Most people will never need this page, and that's fine. Voxtext's job is still "drag a file in, get a transcript out." Translation Manifest Delivery is for the small group of people who already have an automated translation workflow and want Voxtext to hand it work, without anyone copying and pasting between apps.

## What it does

When the feature is on, Voxtext does everything it normally does (transcribes locally with Whisper, writes your transcript files) and then also writes a couple of small JSON files, called manifests, that describe the transcript in a form an automation can read:

- **One per-video manifest**, saved next to each source file. It holds the segment timings, the segment text pre-formatted for translation, the target language, and the folder the finished translation should land in.
- **One job manifest per run**, saved in a folder you choose. It lists the per-video manifests from that run. Because it's the only file that lands in that folder, it works as the "go" signal for a watch-folder automation.

Your workflow takes it from there. Voxtext doesn't translate anything, in this mode or any other. The manifests are written in the same language as the transcript, and Voxtext doesn't talk to your workflow directly either. It just prepares the work and drops the files where you told it to.

```
Voxtext transcribes locally
        |
        v
video01.manifest.json  (next to the video)
job manifest           (in your watched folder)  --->  your automation notices it,
                                                       translates the segments,
                                                       writes video01.fr.vtt back
```

## Who it's for

- Teams that produce captions in more than one language and already have (or are building) an automated flow to translate them.
- Anyone comfortable with a tool like Power Automate, Zapier, n8n, or a script that watches a folder.

It's **not** for you if you just want a translated transcript. Voxtext doesn't translate anything. It transcribes what was said, in the language it was said, and this feature hands that transcript to something else that does the translating.

## A word on privacy

Voxtext's promise is that your *audio* never leaves your computer, and that stays true here. But a manifest contains the transcript text, and you're pointing Voxtext at a folder that (in the typical setup) syncs to the cloud. So with this feature on, your transcript text travels wherever that folder syncs to, and whatever reads it (your translation service) sees it. Only turn it on if that's what you want.

## Turning it on

1. Open **Options** and find the **Translation Manifest Delivery** card.
2. Tick **Enable Translation Manifest Delivery**.
3. Fill in the three fields below.

| Setting | What it's for |
|---|---|
| **Job manifest folder** | The watched folder your automation triggers on. Voxtext drops one job manifest here per run. |
| **Base cloud folder** | The local folder that mirrors your SharePoint document library, for example your OneDrive-synced site folder. Use **Browse...** to pick it. |
| **Its SharePoint path** | What that same folder is called on SharePoint, typed by you, for example `/Shared Documents/Training`. |

The last two exist because the folder name your computer uses for a synced library often doesn't line up neatly with the real SharePoint path, and guessing wrong breaks the handoff. So Voxtext asks you for both ends and does the conversion for everything underneath. If you leave them blank, Voxtext still writes the manifests but leaves out `sourceFolder` (more on that below) and tells you in the status bar.

Once the feature is enabled, a **Translation Manifest** panel appears on the main screen. Untick the feature in Options and the panel disappears again, no restart needed.

## Using it

1. In the **Translation Manifest** panel, tick **Generate Translation Manifest**.
2. Pick the **target language**.
3. Optionally fill in **Course / batch name** (used to name the job manifest; defaults to the source folder's name) and a **Context hint** (see below).
4. Make sure **VTT** is ticked in your output formats. The manifest points at the `<name>_transcript.vtt` file, so your workflow expects that file to exist.
5. Transcribe as usual. Single files and batches both work.

**Context hint.** A short note like `compliance training video` or `casual podcast, first names only`. It nudges the translator toward the right register and terminology. It's added as a single line at the top of the text your workflow sends to its translator, and it isn't part of the translated output.

## The files

### Naming and location

| File | Name | Where it goes |
|---|---|---|
| Per-video manifest | `<video name>.manifest.json` | Next to the source file. Never in the watched folder, so it can't trigger your automation early. |
| Job manifest | `<course name>_<YYYYMMDD_HHMMSS>.job.manifest.json` | Your **Job manifest folder**. |

Voxtext deletes and recreates each manifest instead of overwriting it. Cloud folders often treat an overwrite as a *modification* rather than a *new file*, and many triggers only fire on new files. Delete-then-recreate means your automation gets a fresh "file created" event every time.

### Per-video manifest fields

| Field | Type | Notes |
|---|---|---|
| `courseName` | string | The course/batch name you entered, or the video's name if you left it blank. |
| `sourceLanguage` | string | Language code (`en`, `es`, ...) as detected by Whisper. |
| `targetLanguage` | string | Language code you picked in the panel. |
| `sourceFolder` | string | SharePoint library-relative path of the folder holding the source files. Your workflow uses it as the destination for the translated file. Left out if the two base-folder settings aren't filled in. |
| `sourceFileName` | string | The source-language transcript, always `<video name>_transcript.vtt`. |
| `translatePrompt` | string | The segment text, one line per segment, each starting with an id tag like `[[7]]`. If you set a context hint, a `Content context: ...` line and a blank line come first. It's in the source language, and despite the name it contains no instructions, only the text. This is the only thing your workflow needs to send to a translator. |
| `segments` | array | One entry per segment: `id` (whole number starting at 1), `start` and `end` (as `HH:MM:SS.mmm`). No text here, since the text is in `translatePrompt`. |

### Job manifest fields

| Field | Type | Notes |
|---|---|---|
| `courseName` | string | Same as the per-video manifests. |
| `sourceFolder` | string | Same path as above. Your workflow uses it to find each per-video manifest. Left out if the base-folder settings are blank. |
| `manifestFiles` | array of strings | The filenames (not paths) of the per-video manifests from this run. |

A machine-readable JSON Schema for the per-video manifest is available alongside this page as `manifest.schema.json`.

### Example

A per-video manifest:

```json
{
  "courseName": "Onboarding Basics - Module 3",
  "sourceLanguage": "en",
  "targetLanguage": "fr",
  "sourceFolder": "/Shared Documents/Training/Onboarding/Module 3",
  "sourceFileName": "01 - Welcome_transcript.vtt",
  "translatePrompt": "Content context: internal training video\n\n[[1]] Welcome to the team.\n[[2]] In this module we'll cover your first week.",
  "segments": [
    { "id": 1, "start": "00:00:00.000", "end": "00:00:02.800" },
    { "id": 2, "start": "00:00:02.800", "end": "00:00:06.350" }
  ]
}
```

The matching job manifest:

```json
{
  "courseName": "Onboarding Basics - Module 3",
  "sourceFolder": "/Shared Documents/Training/Onboarding/Module 3",
  "manifestFiles": [
    "01 - Welcome.manifest.json",
    "02 - Your First Week.manifest.json"
  ]
}
```

## What your workflow needs to do

Voxtext is only half of the handshake. If you're building the other half, these four rules keep translated captions from drifting out of sync with the video:

1. **One segment in, one segment out.** Never merge, split, add, or drop segments. The number of translated segments must equal the number in `segments`, matched by `id`.
2. **Never renumber ids.** `[[7]]` in means id `7` out.
3. **Take timing from the manifest, not the translator.** Copy `start` and `end` from `segments` exactly as written. The translation step should only touch the words after each `[[id]]` tag.
4. **Send only `translatePrompt` to the translator.** Nothing else in the manifest needs translating.

Then a typical flow is: trigger on a new `.job.manifest.json` file, loop over `manifestFiles`, read each per-video manifest, translate `translatePrompt`, check that the returned ids exactly match `segments`, build a `WEBVTT` file from the timings plus the translated text, and save it into `sourceFolder` (for example as `<video name>.<language>.vtt`). If the ids don't match, stop and flag it rather than writing a broken caption file. A captions file that's silently out of sync is worse than no captions file.

## Tips and gotchas

- **`sourceFolder` must be a SharePoint path, never a local one.** A Windows or Mac path like `C:\Users\...` will make SharePoint-based flows fail. That's what the two base-folder settings are for.
- **Nothing happens if the feature is off.** With the master checkbox off, Voxtext behaves exactly as it always has.
- **A manifest problem never sinks your transcript.** If a manifest can't be written, Voxtext keeps your normal transcript files and tells you what went wrong.
- **Check the status bar.** If the job manifest went out without a `sourceFolder`, Voxtext says so rather than failing silently.
- **Tidy up as you go.** Each run leaves manifest files behind. Your workflow can move processed ones into a subfolder so the source folders and watched folder stay tidy. Just let your team know what those files are.

## What's supported

This feature is deliberately kept out of the everyday app, and it's supported on a best-effort basis. It's built and tested against a SharePoint/OneDrive folder setup feeding a Power Automate flow. The manifest format is plain JSON, so you're welcome to build against it with other tools, but I can't promise to troubleshoot your particular setup.

If something in the manifest format doesn't work for your workflow, I'd like to hear about it.
