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

A single-file Python desktop app that keeps a local Stable Diffusion model
library tidy. It classifies your checkpoints and LoRAs by base model, renames
them to a consistent `[BaseModel]-Friendly Name` format, downloads a preview
image when one is missing, sorts freshly-downloaded files into the correct
folders, and includes a small Civitai download queue.

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
| **Add** (menu) | *Add Clipboard* adds a Civitai URL from your clipboard to the queue. *Add Batch* loads a `.txt` file of URLs, one per line. |
| **Remove** | Remove the selected queue entry. |
| **Begin Download** | Download everything in the queue into `Downloaded Content`, then sort and label/skin the new files automatically. |
| **Cancel** | Stop after the current file finishes. |
| **Label & Skin** | Walk your whole library: classify each model, rename it to `[BaseModel]-Friendly Name`, and grab a preview if missing. Safe to run repeatedly; already-named files are skipped. |
| **Sort Only** | Move loose files out of `Downloaded Content` into the right model folder (VAEs, LoRAs, checkpoints). |
| **Repair Asset** | Fix one asset by hand. Search your library, pick a file, and correct its base model, name, keywords, or preview. |
| **Add Asset** | Register a file that did not come from Civitai (HuggingFace, TensorArt, a local merge). It gets labeled and skinned, and added to the exclusion list so the auto-pass leaves it alone. |
| **Configuration** | Set the three model folder paths. |
| **Save Key** | Store your Civitai API key. |

## How classification works

For each file the tool tries, in order:

1. Read the base model from the file's `.civitai.info` sidecar (LoRAs).
2. If there is no sidecar (checkpoints), hash the file and ask Civitai what it
   is by hash.
3. If that returns no base model, scrape the model page as a last resort.
4. If nothing works, the file is **quarantined**: left completely untouched and
   logged to `_unresolved.txt`. The tool never guesses a base model from a
   filename, so it will never write a wrong label silently.

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

- It is not a Civitai file (a local merge, HuggingFace, TensorHub). Register it
  with **Add Asset**, which labels it and adds it to the exclusion list, or list
  it in `_exclusion_list.txt`.
- It is a Civitai file whose hash changed (re-saved, pruned, converted). Use
  **Repair Asset** and paste the Civitai URL, or re-download a clean copy.
- No API key is set, so checkpoints (which have no sidecar) cannot be
  classified. Set the key with **Save Key**.

**A file classified fine but no preview downloaded.** The model's image list was
empty, or its first entry was a video rather than a still. Fix it with **Repair
Asset** and supply a preview by URL, local file, or Browse.

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

Beta. Developed by Victor.
