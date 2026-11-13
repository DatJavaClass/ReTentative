<p align="center">
  <img width="448" height="560" alt="Codeman, the Re:Tentative mascot" src="https://github.com/user-attachments/assets/99da5453-5993-4993-8aac-51c0a4b6510c" />
</p>

# Re:Tentative

**Stable Diffusion Model Manager** · Beta

<p align="center">
  <img alt="status: beta" src="https://img.shields.io/badge/status-beta-e8a33d">
  <img alt="python 3" src="https://img.shields.io/badge/python-3-3776ab">
  <img alt="dependencies: none" src="https://img.shields.io/badge/dependencies-none-9fb87a">
  <img alt="gui: tkinter" src="https://img.shields.io/badge/gui-tkinter-74b8b8">
  <img alt="domain: generative AI" src="https://img.shields.io/badge/generative%20AI-image%20models-b07ad6">
  <img alt="platforms" src="https://img.shields.io/badge/platforms-Windows%20%C2%B7%20Linux%20%C2%B7%20macOS-8aa6d6">
</p>

A small Python desktop app that keeps a local Stable Diffusion model
library tidy. It classifies your checkpoints and LoRAs by base model, renames
them to a consistent `[BaseModel]-Friendly Name` format, downloads a preview
image when one is missing, sorts freshly-downloaded files into the correct
folders, and includes a Civitai download queue that persists between sessions.

Zero third-party dependencies: standard-library Python only (`tkinter`,
`urllib`, `hashlib`, `json`, `threading`). No build step, clone and run.

## Requirements

- Python 3
- tkinter (bundled with most Python installs; on Debian/Ubuntu run
  `sudo apt install python3-tk` if it is missing)

## Running

```
python retentative.py
```

## First-time setup

1. Click **Configuration** and point the tool at your three model folders:
   Checkpoints, LoRA, and VAE.
2. Paste your Civitai API key and click **Save Key**. The key lets the tool
   read model metadata and download previews without rate-limit stalls.

A `Downloaded Content` folder is created next to the program; new downloads
land there before being sorted.

## The buttons

| Button | What it does |
| --- | --- |
| **Add Batch** | Load a `.txt` file of Civitai URLs, one per line, into the queue. Non-Civitai lines and duplicates are skipped. |
| **`+`** / **Ctrl+V** | Add a single Civitai URL from your clipboard. Click the `+` on the queue's top-right corner, or press Ctrl+V with the queue focused. |
| **Remove** | Remove the selected queue entry. |
| **Clear Queue** | Empty the queue and its saved copy. Asks first only if a download is running. |
| **Begin Download** | Download everything in the queue into `Downloaded Content`, then sort and label/skin the new files automatically. Each file shows a live progress bar. |
| **Cancel** | Stop after the current file finishes. |
| **Label & Skin** | Walk your whole library: classify each model, rename it to `[BaseModel]-Friendly Name`, and grab a preview if missing. Safe to run repeatedly; already-named files are skipped. |
| **Sort Only** | Move loose files out of `Downloaded Content` into the right model folder (VAEs, LoRAs, checkpoints). |
| **Repair Asset** | Fix one asset by hand. Search your library, pick a file, and correct its base model, name, keywords, or preview. |
| **Add Asset** | Register a file that did not come from Civitai (HuggingFace, TensorArt, a local merge). Pick it with **Load From Local**; it is labeled, skinned, and added to the exclusion list so the auto-pass leaves it alone. |
| **Save Queue** | Save the current queue so it reloads automatically on the next launch. |
| **Export Queue** | Save the queue to a `.txt` you choose (shareable; loads back in via Add Batch). |
| **Configuration** | Set the three model folder paths. |
| **Save Key** | Store your Civitai API key. |
| **Exit** | Close the program. |

## The download queue

The queue is a list of Civitai URLs to fetch. Add to it from a `.txt` file
(**Add Batch**) or one URL at a time from your clipboard (the `+` button on the
queue's top-right corner, or **Ctrl+V** with the queue focused). Both paths skip
non-Civitai lines and duplicates.

The queue survives restarts. **Save Queue** writes it next to the program and it
reloads automatically the next time you launch. **Export Queue** writes a
portable copy to a `.txt` you choose, which loads back in through Add Batch.
**Clear Queue** empties it, and its saved copy, asking first only if a download
is in progress.

While **Begin Download** runs, each file shows a live, terminal-style progress
bar in the log (`[####      47%]`) that advances in place. When a file finishes
it collapses to a single permanent record line (`done: name (147.3 MB)`), so the
log stays readable instead of filling with frozen bars. A download whose size
the server does not report falls back to a running byte count.

## How classification works

For each file the tool tries, in order:

1. Read the base model from the file's `.civitai.info` sidecar (LoRAs).
2. If there is no sidecar (checkpoints), hash the file and ask Civitai what it
   is by hash.
3. If that returns no base model, scrape the model page as a last resort.
4. If nothing works, the file is **quarantined**: left completely untouched and
   logged to `_unresolved.txt`. The tool never guesses a base model from a
   filename, so it will never write a wrong label silently.

## Previews

When a model has no preview image, the tool fetches one from its Civitai image
list. Roughly a third of Civitai previews are videos, so the fetcher prefers a
real still: it skips video entries and falls through to the next image, and as a
last resort tries to pull a still frame from a video. Every candidate is checked
to confirm it is actually an image before it is saved, so a video or an error
page is never written as a `.preview.png`. If no usable still exists, the model
is left without a preview rather than with a broken one.

## Files the tool keeps

All are created next to the program:

| File | Purpose |
| --- | --- |
| `_exclusion_list.txt` | Filename stems to skip entirely. Put non-Civitai files here so they are not re-checked every run. |
| `_unresolved.txt` | A fresh report each run of files that could not be resolved: your punch list for manual Repair. |
| `_skip_cache.json` | Remembers files that failed once, so big files are not re-hashed every run. Delete it to force a full retry. |
| `sd_mm_config.json` | Your API key and folder paths. |

## A note about renaming

Renaming model files changes the strings your SD UI stored in older image
metadata and in any saved styles or workflows. This is expected, and it does
**not** break Automatic1111, Forge, or similar web UIs: after a run, click the
refresh arrows on your UI's model lists so it re-reads the new names. The model
simply shows up under its new, tidier name.

## If something seems off

Re:Tentative is built to fail safe: when it is not certain about a file, it
leaves the file untouched and logs it rather than guessing. In practice it has
been hard to make it misbehave. The notes below are anticipated edge cases and
how you would handle them, not a list of known bugs.

**First move for anything that looks stuck: run it from a terminal.**

```
python retentative.py
```

The terminal shows the live log, so a long operation that looks frozen in the
GUI is usually visible there as real progress.

**"Label & Skin" looks like it is hanging.** Almost always it is busy, not
stuck. The likeliest cause is hashing a very large file that has no sidecar: a
SHA-256 of a multi-hundred-megabyte checkpoint is CPU-bound and can look frozen
for a bit. It only happens the first time a file is seen; the skip-cache records
it so it is not re-hashed next run. Adding persistent non-Civitai files to
`_exclusion_list.txt` skips them entirely. Any network wait is bounded by a
built-in timeout.

**A rename fails on Windows (for example `WinError 123`).** This comes from an
over-length file path, usually a Civitai name that is absurdly long (see
[About the name](#about-the-name)). Two guards handle it: friendly names are
capped in length, and each rename is wrapped so one bad file is logged to
`_unresolved.txt` and the sweep continues instead of stopping.

**A file landed in `_unresolved.txt`.** That is the tool working as designed: it
refused to guess. Usual causes:

- It is not a Civitai file (a local merge, HuggingFace, TensorArt). Register it
  with **Add Asset**, which labels it and adds it to the exclusion list, or list
  it in `_exclusion_list.txt`.
- It is a Civitai file whose hash changed (re-saved, pruned, converted). Use
  **Repair Asset** and paste the Civitai URL, or re-download a clean copy.
- No API key is set, so checkpoints (which have no sidecar) cannot be
  classified. Set the key with **Save Key**.

**A file classified fine but no preview downloaded.** Some Civitai posts are
video-only. The fetcher automatically skips videos, falls through to the next
still, and confirms that what it saves is really an image, so a missing preview
now means no usable still existed at all. Supply one yourself with **Repair
Asset** by URL, local file, or Browse.

**A file sorted into the wrong folder.** Loose files with no sidecar are sorted
with a size heuristic, so a rare edge case is possible. New Civitai downloads
carry sidecars and sort by their reported type, so this seldom happens. Use
**Repair Asset** or **Add Asset** to correct it.

**Nothing is created on first run.** On first launch the tool creates its working
files next to the script and walks you through Configuration (your folder paths)
and the API-key prompt. If that did not happen, launch from a terminal to see
any message, and confirm the script can write to its own folder.

**It will not launch at all.** The most common cause on a fresh machine is a
Python install without tkinter. On Debian/Ubuntu: `sudo apt install python3-tk`.
The official python.org installers for Windows and macOS include it.

## About the name

During development the software was, frankly, uncooperative about breaking.
Across more than ten iterations the worst defect that ever surfaced was a button
that had drifted about two inches to the left, and the developer was genuinely
happy to finally have a bug to point at. The one real failure came from outside
the code entirely: `Label & Skin` froze on a single Civitai upload whose
"friendly name" was a roughly 300-character joke. That path blew past the Windows
`MAX_PATH` limit and took down the whole run. Friendly names are now capped, and
every rename is wrapped so one bad file logs to `_unresolved.txt` and the sweep
keeps going. Ever since, the developer has been *Re:Tentative* about every detail
he downloads.

## Status

Beta. Developed by DatJavaClass.

---

## Changelog

### [0.3.0-beta] - 2026-11-12

Faster under the hood, but the gas station only pumps so fast.

The download path was rebuilt. Files are now pulled in pieces over several
connections, several files at a time, and a broken download picks up where it
stopped. The catch? In testing, Civitai handed over about 10 to 12 MB/s in
total no matter how many connections asked. So the defaults are a polite two
connections and two files. Other hosts may be more generous.

#### Added

- **Segmented downloads.** Each file is cut into pieces and pulled over
  several keep-alive connections at once. Set "Connections per file" (1 to 16)
  in Configuration.
- **Several files at once.** Set "Files at once" (1 to 6) in Configuration.
- **Resume.** A cancelled or dropped download keeps its part file and
  continues from there on the next run.
- **Integrity check.** A finished file is compared with the SHA-256 Civitai
  publishes. A mismatch is set aside as `.corrupt` and stays in the queue.
- **Staging folder setting.** Downloads can land on the same drive as your
  model folders, which makes sorting instant.
- **Active downloads table** with size, progress and speed per file, plus
  overall speed and time left in the status line.
- **Headless downloads:** `python retentative.py get <url> [more urls]`.
- **Downloads from any direct link**, with an optional preview image URL,
  keywords, name, base model and kind.
- **Duplicate finder** (exact, by SHA-256), **Update Database** under Repair,
  and a **Pruner** tab.
- A test suite (`python -m unittest discover -s tests`) and a small benchmark
  (`python tools/bench.py`).

#### Changed

- The single file became a package of small modules. `retentative.py` is
  still the way in.
- The window is organized into Download, Library, Repair and Pruner tabs, so
  the button table above describes the older layout.
- The Civitai API key is sent only to Civitai's own API host, never to the
  storage host a download is handed off to.
- File names can carry the version: `[BaseModel]-Friendly Name (Version)`.

---

### [0.2.0-beta] - 2026-07-09

First push since the initial release. Adds download-queue persistence and a
live progress display, rebuilds the Add Asset flow, and fixes two naming and
preview bugs.

#### Added

- **Download queue persistence and management.**
  - Save Queue writes the pending list to disk and it auto-reloads into the
    queue on the next startup.
  - Export Queue writes a portable `.txt` (one URL per line) that round-trips
    back in through Add Batch.
  - Clear Queue empties the list and wipes the saved copy so a restart does not
    restore a queue you cleared. It only asks for confirmation while a run is
    active.
  - Clipboard add is now bound to `Ctrl+V` on the queue and to a small `+`
    button on the queue frame; both share the same validated, de-duplicated
    append path.
  - Added an Exit button to the top row.
- **Terminal-style live download progress bar.** Each file shows a single
  in-place bar (`[####      47%]`) that advances as it downloads, then collapses
  to one permanent record line per file (`done: name (147.3 MB)`, or a
  cancelled / error line). Widget updates are marshalled to the main thread, and
  redraws are throttled to whole-percent changes so the UI is not flooded.
  - Falls back to a byte counter (`[downloading... 47.3 MB]`) when the server
    sends no Content-Length.
- **Add Asset: up-front local file picker.** A "Model file" row with a
  `Load From Local` button lets you choose the `.safetensors` deliberately,
  first. Its basename seeds the friendly-name field. The picker defaults to your
  Downloads folder, where non-Civitai files usually land.
- **Preview fetching now tries multiple candidates.** Instead of taking only the
  first entry in a Civitai `images` array, the fetcher walks the list, prefers
  real images, and as a last resort attempts a still frame derived from a video
  entry. Every downloaded candidate is checked by its magic bytes before it is
  kept.

#### Changed

- **Add Asset severed from the Repair flow.** Add and Repair had shared one form
  and Add had inherited Repair behavior that did not fit it. The Civitai URL row
  and Pull-from-URL button are now hidden in Add (they belong only to Repair,
  where the file is known and the metadata is corrected). Add is now built around
  the fact that its file is the unknown and its metadata is supplied by hand.
- Preview files are still saved as `<stem>.preview.png` for A1111
  compatibility; only the selection and validation logic around them changed.

#### Fixed

- **Double-prefix ("double-dip") naming bug.** Repairing or re-adding a file
  whose name already carried a base-model tag produced a stacked prefix such as
  `[Anima]-[Anima] Canopy The Digital Alchemist` or
  `[Anima]-[Illustrious] An Ban Illus`. The name builder is now idempotent: it
  strips any existing leading tag off the friendly text before composing, so a
  second tag can never be bolted on. A standalone repair tool
  (`fix_double_prefix.py`) for filenames already affected on disk ships
  alongside the project: run it with no arguments for a simple folder-picker
  window, or pass the Checkpoint and LoRA folder paths on the command line.
- **Preview fetch failing on video entries.** Roughly a third of Civitai
  previews are videos. The old fetcher grabbed the first entry unconditionally,
  saved the video bytes as `.preview.png`, and the result would not render, which
  showed up as a high preview-failure rate. Video entries are now skipped in
  favor of the next real image, and content validation guarantees a video or an
  error page is never saved as a preview.

#### Notes

- Previews that were already saved incorrectly (video bytes under a `.preview.png`
  name) are not replaced automatically, because the fetcher skips any file that
  already has a preview. Delete the bad preview files and re-run Label & Skin to
  re-pull them.

---

### [0.1.0-beta] - 2026-07-02

Initial public release. A single-file, zero-dependency, clone-and-run tool that
replaced four ad-hoc utilities (checkpoint and LoRA preview grabbers, a prompt
extractor, and a model sorter) with one manager.

#### Added

- **Classification ladder with a strict null case.** LoRAs resolve from their
  `.civitai.info` sidecar first (fast, local). Checkpoints, which have no
  sidecar, resolve by SHA-256 through Civitai's by-hash API. A page scrape of the
  `data-next-head` element is the last resort. When nothing authoritative
  resolves, the file is quarantined and logged to `_unresolved.txt` rather than
  guessed from its filename.
- **Label & Skin**, an idempotent mass normalizer that renames models to
  `[BaseModel]-Friendly Name` and grabs a Civitai preview image. The friendly
  name comes from Civitai's clean `model.name`, not the mangled download filename.
- **Sort Only**, which files models into their folders with VAE detection running
  before the checkpoint branch (fixing the old tool's VAE-into-checkpoints bug).
- **Download manager** with Add Batch (a `.txt` of URLs, de-duplicated, non-Civitai
  lines skipped), Begin, and Cancel, plus an exclusion list for non-Civitai files
  that would otherwise re-log every run.
- **Repair Asset and Add Asset** breakout forms sharing one backend, with a search
  bar, status panel, prefill from existing state, and keyword writing to the
  sidecar.
- **A base-model dropdown derived from the library** plus a permanent "Other"
  entry, so a brand-new base model needs zero code change. The tag map is
  formatting-only, with passthrough for unknown base models.
- **Config-driven release build** with first-run onboarding (welcome, then folder
  configuration, then API-key prompt). Working files sit next to the script.
- Project documentation: `CLAUDE.md`, `README.md`, and a Claude Code handoff
  guide.

#### Fixed

- **Long-name crash on rename (`WinError 123`).** A Civitai `model.name` of roughly
  300 characters pushed a path past the Windows MAX_PATH limit and crashed the
  sweep. Friendly names are now capped at 80 characters, and `rename_with_siblings`
  wraps each rename so one bad file logs and the sweep continues instead of
  aborting.

#### Changed

- Hardened the network path: request timeout cut from 30s to 8s, retry with
  backoff on HTTP 429, and every network exception (not just `HTTPError`) routes to
  quarantine-and-continue. A `_skip_cache.json` keeps files that 404 once from
  being re-hashed on every run.
- README updated with a mascot header and status badges.
