#!/usr/bin/env python3
"""
Re:Tentative - Stable Diffusion Model Manager  [Beta]
=====================================================

A single-file tool to keep a local Stable Diffusion model library tidy:
it classifies checkpoints and LoRAs by their base model, renames them to a
consistent [BaseModel]-Friendly Name format, grabs a preview image when one
is missing, sorts freshly-downloaded files into the right folders, and offers
a small Civitai download manager.

FIRST RUN
  On first launch with no saved configuration, the tool opens the
  Configuration dialog so you can point it at your three model folders
  (Checkpoints, LoRA, VAE). After that it asks for your Civitai API key.
  It also creates, next to this script, empty working files:
      _exclusion_list.txt   _unresolved.txt   _skip_cache.json
      sd_mm_config.json     Downloaded Content/   (empty folder)
  (_saved_queue.txt is created on first Save Queue, not at first run.)

WHAT THE BUTTONS DO   (see the ? Help button in-app for the full README)
  Add Batch      load a .txt of Civitai URLs (one per line) into the queue
  + / Ctrl+V     add one clipboard URL (corner + button, or Ctrl+V in queue)
  Remove         drop the selected queue item
  Clear Queue    empty the queue (confirms only while a download is running)
  Begin Download fetch the queue (live progress bar), sort, then label + skin
  Cancel         stop after the current file
  Label & Skin   walk your library: classify, rename, grab missing previews
  Sort Only      route loose files from Downloaded Content into model folders
  Repair Asset   fix one asset by hand (search, edit fields, re-skin)
  Add Asset      register a non-Civitai file (HuggingFace/TensorHub); it is
                 labeled/skinned AND added to the exclusion list
  Save Queue     save the queue to _saved_queue.txt (auto-reloads next launch)
  Export Queue   save the queue to a .txt you pick (loads back via Add Batch)
  Configuration  set your model folder paths
  Save Key       store your Civitai API key
  Exit           close the program
  ?              open the in-app README

CLASSIFICATION (per file, in order)
  LoRAs        -> read baseModel from the .civitai.info sidecar (usually present)
  Checkpoints  -> no sidecar, so SHA-256 the file and ask Civitai by-hash
  Fallback     -> scrape the model page for baseModel
  None of these-> QUARANTINE: leave the file untouched, log to _unresolved.txt

The tool never guesses a base model from a filename. Anything it cannot resolve
from an authoritative source is left alone and logged, so a wrong label is never
written silently.

NOTE ON A1111 / FORGE ETC.
  Renaming model files changes the strings stored in prior image metadata and
  in any saved styles or workflows that reference those files. This is expected;
  refresh your UI's model lists after a run.
"""

import os
import sys
import re
import json
import time
import hashlib
import threading
import urllib.request
import urllib.error
import urllib.parse

# ----------------------------------------------------------------------------
# PATHS
#   Everything working (config, lists, Downloaded Content) lives next to THIS
#   script -- tight and predictable. Model folders are user-configured and
#   stored in sd_mm_config.json; they are EMPTY until the user sets them.
# ----------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH     = os.path.join(SCRIPT_DIR, "sd_mm_config.json")
EXCLUSION_PATH  = os.path.join(SCRIPT_DIR, "_exclusion_list.txt")
UNRESOLVED_PATH = os.path.join(SCRIPT_DIR, "_unresolved.txt")
SKIPCACHE_PATH  = os.path.join(SCRIPT_DIR, "_skip_cache.json")
SAVED_QUEUE_PATH = os.path.join(SCRIPT_DIR, "_saved_queue.txt")
DOWNLOAD_DIR    = os.path.join(SCRIPT_DIR, "Downloaded Content")


def set_console_title(title="Re:Tentative Worker"):
    """Give the console window a recognizable name so it isn't mistaken for
    junk and closed (closing it kills the app). Windows-only; a harmless
    no-op elsewhere. NOTE: the truly console-free launch is renaming this
    file to .pyw (runs under pythonw.exe, no console at all) -- see README.
    """
    try:
        if os.name == "nt":
            os.system(f"title {title}")
    except Exception:
        pass

# ----------------------------------------------------------------------------
# CIVITAI ENDPOINTS
# ----------------------------------------------------------------------------
BY_HASH    = "https://civitai.com/api/v1/model-versions/by-hash/{}"
MODEL_PAGE = "https://civitai.com/models/{}"

# ----------------------------------------------------------------------------
# BASE MODEL -> BRACKET TAG  (formatting only; NOT a gate on allowed values)
#   Maps known Civitai baseModel strings to a tidy bracket tag. Anything not
#   listed passes through verbatim as [Whatever] -- so a brand-new base model
#   is handled with no code change; it just isn't prettified.
# ----------------------------------------------------------------------------
TAG_MAP = {
    "illustrious":          "[Illustrious]",
    "pony":                 "[Pony]",
    "anima":                "[Anima]",
    "sdxl 1.0":             "[SDXL 1.0]",
    "sdxl":                 "[SDXL 1.0]",
    "stable diffusion xl":  "[SDXL 1.0]",
    "sd 1.5":               "[SD 1.5]",
    "stable diffusion 1.5": "[SD 1.5]",
    "noobai":               "[NoobAI]",
    "flux.1 d":             "[Flux]",
    "flux.1 s":             "[Flux]",
}

MODEL_EXTS = {".safetensors", ".ckpt", ".pt"}


# ============================================================================
# CONFIG
# ============================================================================
def default_config():
    return {"api_key": "", "ckpt_dir": "", "lora_dir": "", "vae_dir": ""}


def load_config():
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base = default_config()
            base.update(cfg)
            return base
        except Exception:
            pass
    return default_config()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def save_queue_to_disk(urls):
    """Persist the current queue so it survives closing the app (auto-reload)."""
    try:
        with open(SAVED_QUEUE_PATH, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(u + "\n")
        return True
    except Exception:
        return False


def load_saved_queue():
    """Read the auto-reload queue file, if present. Returns a list of URLs."""
    if not os.path.isfile(SAVED_QUEUE_PATH):
        return []
    try:
        with open(SAVED_QUEUE_PATH, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []


def paths_configured(cfg):
    """True only if all three model folders are set and exist."""
    for key in ("ckpt_dir", "lora_dir", "vae_dir"):
        p = cfg.get(key, "")
        if not p or not os.path.isdir(p):
            return False
    return True


# ============================================================================
# FIRST-RUN SCAFFOLDING
# ============================================================================
DEFAULT_EXCLUSIONS = []   # release ships with an EMPTY exclusion list


def ensure_working_files():
    """Create the empty working files and Downloaded Content folder if absent.
    Idempotent -- never overwrites an existing file."""
    if not os.path.isfile(EXCLUSION_PATH):
        with open(EXCLUSION_PATH, "w", encoding="utf-8") as f:
            f.write("# One filename stem per line. Lines starting with # are ignored.\n")
            f.write("# Files matching these stems are skipped entirely by the tool:\n")
            f.write("# no hashing, no renaming, no unresolved-log entries.\n")
            f.write("# Add non-Civitai files (local merges, hand-trained, HuggingFace,\n")
            f.write("# TensorHub) here so the auto-pass leaves them alone.\n")
            f.write("# Example (delete or replace):\n")
            f.write("# myLocalMerge_v1\n")
    if not os.path.isfile(UNRESOLVED_PATH):
        open(UNRESOLVED_PATH, "w", encoding="utf-8").close()
    if not os.path.isfile(SKIPCACHE_PATH):
        with open(SKIPCACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ============================================================================
# EXCLUSION LIST
# ============================================================================
def load_exclusions():
    """Return a set of lowercased, normalized stems to skip."""
    ensure_working_files()
    stems = set()
    with open(EXCLUSION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.add(normalize_stem(line))
    return stems


def add_to_exclusions(stem):
    """Append a stem to the exclusion list if not already present."""
    ensure_working_files()
    existing = set()
    with open(EXCLUSION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                existing.add(normalize_stem(s))
    if normalize_stem(stem) in existing:
        return False
    with open(EXCLUSION_PATH, "a", encoding="utf-8") as f:
        f.write(stem + "\n")
    return True


def normalize_stem(name):
    """Drop extension, drop trailing kohya step suffix (-000014), lowercase."""
    stem = os.path.splitext(name)[0]
    stem = re.sub(r"-0*\d{4,6}$", "", stem)
    return stem.strip().lower()


# ============================================================================
# HTTP HELPERS  (short timeouts; never hang a batch run)
# ============================================================================
def http_get_json(url, api_key=None, timeout=8, retries=2):
    req = urllib.request.Request(url, headers={"User-Agent": "retentative/1.0"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                last_err = e
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5)
                continue
            raise
    if last_err:
        raise last_err


def http_download(url, dest_path, api_key=None, timeout=120, cancel_flag=None,
                  progress=None):
    req = urllib.request.Request(url, headers={"User-Agent": "retentative/1.0"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        tmp = dest_path + ".part"
        with open(tmp, "wb") as out:
            while True:
                if cancel_flag is not None and cancel_flag.is_set():
                    out.close()
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return False
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress:
                    if total:
                        progress(done / total)
                    else:
                        # No Content-Length: no fraction to report. Signal the
                        # unknown-total case with a negative sentinel carrying
                        # bytes-done, so a caller can show a byte counter
                        # instead of a percent. Callers that only understand
                        # fractions simply clamp it to 0%.
                        progress(-(done + 1))
        os.replace(tmp, dest_path)
    return True


def sha256_file(path, cancel_flag=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if cancel_flag is not None and cancel_flag.is_set():
                return None
            h.update(chunk)
    return h.hexdigest()


def decamel(stem):
    s = stem.replace("_", " ").replace("-", " ")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def tag_for_base_model(base_model):
    if not base_model:
        return None
    key = base_model.strip().lower()
    if key in TAG_MAP:
        return TAG_MAP[key]
    return "[" + base_model.strip() + "]"


def log_unresolved(path, tried, reason):
    with open(UNRESOLVED_PATH, "a", encoding="utf-8") as f:
        f.write(f"{path}\n    tried: {tried}\n    reason: {reason}\n\n")


def load_skip_cache():
    if os.path.isfile(SKIPCACHE_PATH):
        try:
            with open(SKIPCACHE_PATH, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_skip_cache(stems):
    try:
        with open(SKIPCACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(stems), f, indent=2)
    except Exception:
        pass


# ============================================================================
# SIDECAR / CLASSIFICATION
# ============================================================================
def read_sidecar(model_path):
    sidecar = os.path.splitext(model_path)[0] + ".civitai.info"
    if not os.path.isfile(sidecar):
        return None, None, None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None
    base = data.get("baseModel")
    name = (data.get("model") or {}).get("name")
    images = data.get("images") or []
    return base, name, images


def classify_by_hash(model_path, api_key, cancel_flag=None):
    digest = sha256_file(model_path, cancel_flag=cancel_flag)
    if digest is None:
        return None
    try:
        data = http_get_json(BY_HASH.format(digest), api_key=api_key)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception:
        return None
    return {
        "base_model": data.get("baseModel"),
        "model_name": (data.get("model") or {}).get("name"),
        "images": data.get("images") or [],
        "model_id": data.get("modelId"),
    }


def scrape_data_next_head(model_id, api_key=None):
    if not model_id:
        return None
    try:
        url = MODEL_PAGE.format(model_id)
        req = urllib.request.Request(url, headers={"User-Agent": "retentative/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    for pat in (r'"baseModel"\s*:\s*"([^"]+)"',
                r'data-next-head[^>]*baseModel[^>]*?"([^"]+)"'):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


# Video previews are common on Civitai (roughly a third of entries). Prefer a
# real image and fall through the list: the still a user wants is usually the
# very next preview after a video. As a last resort, try to pull a still frame
# from a video URL. Every candidate is content-validated on download, so a
# video or an error page is never saved as a preview.
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv", ".gifv"}

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",   # png
    b"\xff\xd8\xff",         # jpeg
    b"GIF87a", b"GIF89a",    # gif
    b"BM",                    # bmp
)


def _looks_like_image(head):
    """True if the leading bytes are a known image signature (WEBP is RIFF....WEBP)."""
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return any(head.startswith(sig) for sig in _IMAGE_MAGIC)


def _video_to_poster(url):
    """Best-effort still-frame URL for a Civitai video: swap the trailing video
    extension for .jpeg. Civitai's CDN often returns a poster frame for that.
    A wrong guess is discarded by the content check on download."""
    base, ext = os.path.splitext(url)
    if ext.lower() in _VIDEO_EXTS:
        return base + ".jpeg"
    return None


def image_preview_candidates(images):
    """Ordered preview URLs to try: real images first (in list order), then
    still-frame attempts derived from any videos. Entries are judged by their
    'type' field and their URL extension."""
    if not images:
        return []
    image_urls, video_urls = [], []
    for img in images:
        url = img.get("url")
        if not url:
            continue
        itype = (img.get("type") or "").lower()
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if itype == "video" or ext in _VIDEO_EXTS:
            video_urls.append(url)
        else:
            image_urls.append(url)
    candidates = list(image_urls)
    for v in video_urls:
        poster = _video_to_poster(v)
        if poster:
            candidates.append(poster)
    return candidates


def pick_preview_url(images):
    """First usable image preview URL (video entries skipped), or None."""
    cands = image_preview_candidates(images)
    return cands[0] if cands else None


def resolve_model(model_path, api_key, cancel_flag=None):
    tried = []
    base, name, images = read_sidecar(model_path)
    if base:
        return {
            "base_model": base,
            "tag": tag_for_base_model(base),
            "model_name": name,
            "preview_url": pick_preview_url(images),
            "images": images,
            "source": "sidecar",
        }
    tried.append("sidecar(absent)")

    if not api_key:
        tried.append("hash(skipped: no API key)")
        log_unresolved(model_path, tried,
                       "no sidecar and no API key -> cannot classify")
        return None
    hit = classify_by_hash(model_path, api_key, cancel_flag=cancel_flag)
    if hit and hit.get("base_model"):
        return {
            "base_model": hit["base_model"],
            "tag": tag_for_base_model(hit["base_model"]),
            "model_name": hit.get("model_name"),
            "preview_url": pick_preview_url(hit.get("images") or []),
            "images": hit.get("images") or [],
            "source": "hash",
        }

    if hit and hit.get("model_id"):
        tried.append("hash(no baseModel)")
        scraped = scrape_data_next_head(hit["model_id"], api_key=api_key)
        if scraped:
            return {
                "base_model": scraped,
                "tag": tag_for_base_model(scraped),
                "model_name": hit.get("model_name"),
                "preview_url": pick_preview_url(hit.get("images") or []),
                "images": hit.get("images") or [],
                "source": "scrape",
            }
        tried.append("scrape(failed)")
    else:
        tried.append("hash(404)")

    hint = decamel(os.path.splitext(os.path.basename(model_path))[0])
    log_unresolved(
        model_path, tried,
        f"not resolvable from any authoritative source; "
        f"filename hint only (NOT applied): {hint!r}")
    return None


# ============================================================================
# RENAME / PREVIEW
# ============================================================================
def build_new_stem(tag, model_name, original_stem):
    friendly = model_name.strip() if model_name else decamel(original_stem)
    friendly = sanitize_filename(friendly)
    # Idempotency guard against the double-dip bug: if the friendly text already
    # begins with a [BaseModel] tag (from an earlier naming pass, or an already-
    # formatted filename fed back through decamel), strip it before composing so
    # we never produce "[Tag]-[Tag] Name". Handles a leading "[X] " or "[X]-"
    # and repeats to collapse any accidental multi-prefix.
    prev = None
    while prev != friendly:
        prev = friendly
        friendly = re.sub(r"^\[[^\[\]]*\]\s*-?\s*", "", friendly, count=1).strip()
    MAX_FRIENDLY = 80   # keep full path under Windows MAX_PATH (~260)
    if len(friendly) > MAX_FRIENDLY:
        friendly = friendly[:MAX_FRIENDLY].rstrip()
    return f"{tag}-{friendly}"


def already_formatted(stem):
    return bool(re.match(r"^\[[^\]]+\]-", stem))


def rename_with_siblings(model_path, new_stem):
    folder = os.path.dirname(model_path)
    old_stem = os.path.splitext(os.path.basename(model_path))[0]
    results = []
    for fname in os.listdir(folder):
        if fname == old_stem or fname.startswith(old_stem + "."):
            suffix = fname[len(old_stem):]
            src = os.path.join(folder, fname)
            dst = os.path.join(folder, new_stem + suffix)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if os.path.exists(dst):
                continue
            try:
                os.rename(src, dst)
                results.append((fname, new_stem + suffix))
            except OSError as e:
                log_unresolved(
                    src, ["rename"],
                    f"os.rename failed ({e.__class__.__name__}: {e}); "
                    f"target was {new_stem + suffix!r}")
    return results


def has_preview(model_path):
    stem = os.path.splitext(model_path)[0]
    for ext in (".preview.png", ".preview.jpg", ".preview.jpeg"):
        if os.path.isfile(stem + ext):
            return True
    return False


def _download_validated_image(url, dest, api_key=None, cancel_flag=None):
    """Download url to a temp file and keep it only if the bytes are actually an
    image. Returns True on success. A video, HTML error page, or empty body is
    rejected and removed, never saved as the preview."""
    tmp = dest + ".dl"
    try:
        ok = http_download(url, tmp, api_key=api_key, cancel_flag=cancel_flag)
    except Exception:
        ok = False
    if not ok:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    try:
        with open(tmp, "rb") as f:
            head = f.read(32)
    except Exception:
        head = b""
    if not _looks_like_image(head):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    try:
        os.replace(tmp, dest)
        return True
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def grab_preview(model_path, images, api_key, cancel_flag=None):
    """Save <stem>.preview.png from the first candidate that downloads AND is a
    real image. Tries every image in the list, then still-frame attempts from
    videos, before giving up. Skips if a preview already exists."""
    if has_preview(model_path):
        return False
    dest = os.path.splitext(model_path)[0] + ".preview.png"
    for url in image_preview_candidates(images):
        if cancel_flag is not None and cancel_flag.is_set():
            return False
        if _download_validated_image(url, dest, api_key=api_key,
                                     cancel_flag=cancel_flag):
            return True
    return False


# ============================================================================
# LABEL & SKIN  (the library normalizer)
# ============================================================================
def iter_model_files(folder):
    if not folder or not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext in MODEL_EXTS:
            yield os.path.join(folder, fname)


def label_and_skin(cfg, log=print, cancel_flag=None):
    if not paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return None
    api_key = cfg.get("api_key", "")
    exclusions = load_exclusions()
    skip_cache = load_skip_cache()
    open(UNRESOLVED_PATH, "w", encoding="utf-8").close()

    counts = {"renamed": 0, "previewed": 0, "skipped_excluded": 0,
              "skipped_formatted": 0, "skipped_cached": 0,
              "unresolved": 0, "errors": 0}

    folders = ((cfg["ckpt_dir"], "checkpoint"), (cfg["lora_dir"], "lora"))
    for folder, label in folders:
        log(f"\n=== {label.upper()}S in {folder} ===")
        for model_path in iter_model_files(folder):
            if cancel_flag is not None and cancel_flag.is_set():
                log("Cancelled.")
                return counts
            fname = os.path.basename(model_path)
            stem = os.path.splitext(fname)[0]
            nstem = normalize_stem(fname)

            if nstem in exclusions:
                counts["skipped_excluded"] += 1
                log(f"  skip (excluded): {fname}")
                continue
            if already_formatted(stem):
                counts["skipped_formatted"] += 1
                continue
            if nstem in skip_cache and read_sidecar(model_path)[0] is None:
                counts["skipped_cached"] += 1
                log(f"  skip (cached-miss): {fname}")
                continue

            try:
                info = resolve_model(model_path, api_key, cancel_flag=cancel_flag)
            except Exception as e:
                counts["errors"] += 1
                log(f"  ERROR {fname}: {e}")
                continue

            if info is None:
                counts["unresolved"] += 1
                skip_cache.add(nstem)
                log(f"  UNRESOLVED (quarantined): {fname}")
                continue

            new_stem = build_new_stem(info["tag"], info["model_name"], stem)
            if grab_preview(model_path, info["images"], api_key,
                            cancel_flag=cancel_flag):
                counts["previewed"] += 1
                log(f"  preview grabbed: {fname}")
            moved = rename_with_siblings(model_path, new_stem)
            if moved:
                counts["renamed"] += 1
                log(f"  {info['source']:>7} | {fname}  ->  {new_stem}")

    save_skip_cache(skip_cache)
    log("\n=== SUMMARY ===")
    for k, v in counts.items():
        log(f"  {k:20s}: {v}")
    log(f"\nUnresolved (if any): {UNRESOLVED_PATH}")
    return counts


# ============================================================================
# SORTER  (VAE checked before checkpoint)
# ============================================================================
KNOWN_VAE_NAMES = {"sdxl.vae", "qwen_image_vae", "vae-ft-mse-840000-ema-pruned"}


def is_vae(fname):
    stem = os.path.splitext(fname)[0].lower()
    if stem in KNOWN_VAE_NAMES:
        return True
    return "vae" in stem


def is_lora(model_path):
    sidecar = os.path.splitext(model_path)[0] + ".civitai.info"
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            t = (data.get("model") or {}).get("type", "")
            if t.upper() == "LORA":
                return True
            if t.upper() == "CHECKPOINT":
                return False
        except Exception:
            pass
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return size_mb < 500
    except OSError:
        return False


def destination_for(model_path, cfg):
    fname = os.path.basename(model_path)
    if is_vae(fname):
        return cfg["vae_dir"]
    if is_lora(model_path):
        return cfg["lora_dir"]
    return cfg["ckpt_dir"]


def sort_downloads(cfg, log=print, cancel_flag=None):
    if not paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return 0
    if not os.path.isdir(DOWNLOAD_DIR):
        log(f"No download dir: {DOWNLOAD_DIR}")
        return 0
    moved = 0
    for fname in list(os.listdir(DOWNLOAD_DIR)):
        if cancel_flag is not None and cancel_flag.is_set():
            log("Sort cancelled.")
            break
        ext = os.path.splitext(fname)[1].lower()
        if ext not in MODEL_EXTS:
            continue
        src = os.path.join(DOWNLOAD_DIR, fname)
        dest_dir = destination_for(src, cfg)
        os.makedirs(dest_dir, exist_ok=True)
        stem = os.path.splitext(fname)[0]
        for sib in list(os.listdir(DOWNLOAD_DIR)):
            sstem = os.path.splitext(sib)[0]
            if sib == fname or sstem == stem or sib.startswith(stem + "."):
                s = os.path.join(DOWNLOAD_DIR, sib)
                d = os.path.join(dest_dir, sib)
                if not os.path.exists(d):
                    os.rename(s, d)
        moved += 1
        log(f"  sorted -> {os.path.basename(dest_dir)}: {fname}")
    return moved


# ============================================================================
# CIVITAI URL / DOWNLOAD
# ============================================================================
def civitai_version_id_from_url(url):
    url = url.strip()
    q = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(q.query)
    if "modelVersionId" in params:
        return ("version", params["modelVersionId"][0])
    m = re.search(r"/api/download/models/(\d+)", url)
    if m:
        return ("version", m.group(1))
    m = re.search(r"/models/(\d+)", url)
    if m:
        return ("model", m.group(1))
    return (None, None)


def resolve_download_url(kind, cid, api_key):
    if kind == "version":
        data = http_get_json(
            f"https://civitai.com/api/v1/model-versions/{cid}", api_key=api_key)
    else:
        data = http_get_json(
            f"https://civitai.com/api/v1/models/{cid}", api_key=api_key)
        versions = data.get("modelVersions") or []
        if not versions:
            return None, None
        data = versions[0]
    files = data.get("files") or []
    primary = next((f for f in files if f.get("primary")),
                   files[0] if files else None)
    if not primary:
        return None, None
    return primary.get("downloadUrl"), primary.get("name")


def fetch_metadata_by_url(url, api_key):
    kind, cid = civitai_version_id_from_url(url)
    if not cid:
        return None
    try:
        if kind == "version":
            data = http_get_json(
                f"https://civitai.com/api/v1/model-versions/{cid}", api_key=api_key)
        else:
            mdata = http_get_json(
                f"https://civitai.com/api/v1/models/{cid}", api_key=api_key)
            versions = mdata.get("modelVersions") or []
            if not versions:
                return None
            data = versions[0]
            data.setdefault("model", {})["name"] = mdata.get("name")
    except Exception:
        return None
    return {
        "base_model": data.get("baseModel"),
        "model_name": (data.get("model") or {}).get("name"),
        "preview_url": pick_preview_url(data.get("images") or []),
        "keywords": data.get("trainedWords") or [],
    }


# ============================================================================
# MANUAL ASSET HELPERS  (Repair / Add forms)
# ============================================================================
def scan_known_base_models(cfg):
    """Derive the base-model dropdown from what the library actually contains,
    plus common defaults. A brand-new base model appears automatically once a
    file carrying it exists; 'Other...' covers the not-yet-present case."""
    found = set()
    for key in ("ckpt_dir", "lora_dir"):
        folder = cfg.get(key, "")
        if not folder or not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not fname.endswith(".civitai.info"):
                continue
            try:
                with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                bm = data.get("baseModel")
                if bm:
                    found.add(bm.strip())
            except Exception:
                continue
    for pretty in ("Illustrious", "Pony", "Anima", "SDXL 1.0",
                   "SD 1.5", "NoobAI", "Flux"):
        found.add(pretty)
    return sorted(found, key=str.lower)


def find_model_file(stem_or_name, folder):
    if not folder or not os.path.isdir(folder):
        return None
    for ext in MODEL_EXTS:
        cand = os.path.join(folder, stem_or_name + ext)
        if os.path.isfile(cand):
            return cand
    target = os.path.splitext(stem_or_name)[0]
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext in MODEL_EXTS and os.path.splitext(fname)[0] == target:
            return os.path.join(folder, fname)
    return None


def list_model_stems(folder):
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for fname in os.listdir(folder):
        ext = os.path.splitext(fname)[1].lower()
        if ext in MODEL_EXTS:
            out.append(os.path.splitext(fname)[0])
    return sorted(out, key=str.lower)


def asset_status(model_path):
    base, name, images = read_sidecar(model_path)
    sidecar = os.path.splitext(model_path)[0] + ".civitai.info"
    keywords = []
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            keywords = data.get("trainedWords") or []
        except Exception:
            pass
    return {
        "has_sidecar": os.path.isfile(sidecar),
        "has_preview": has_preview(model_path),
        "base_model": base,
        "model_name": name,
        "keywords": keywords,
    }


def write_keywords_to_sidecar(model_path, keywords):
    sidecar = os.path.splitext(model_path)[0] + ".civitai.info"
    data = {}
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["trainedWords"] = keywords
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def apply_manual_asset(model_path, tag, friendly, keywords, preview_source,
                       api_key=None, make_exclusion=False, log=print):
    if not os.path.isfile(model_path):
        return False, f"Model file not found: {model_path}"
    if keywords:
        write_keywords_to_sidecar(model_path, keywords)
    if preview_source:
        dest = os.path.splitext(model_path)[0] + ".preview.png"
        src = preview_source.strip().strip('"')
        try:
            if src.lower().startswith(("http://", "https://")):
                http_download(src, dest, api_key=api_key)
                log("  preview downloaded from URL")
            elif os.path.isfile(src):
                import shutil
                shutil.copyfile(src, dest)
                log("  preview copied from local file")
            else:
                log(f"  ! preview source not a URL or existing file: {src}")
        except Exception as e:
            log(f"  ! preview failed: {e}")
    new_stem = build_new_stem(tag, friendly,
                              os.path.splitext(os.path.basename(model_path))[0])
    moved = rename_with_siblings(model_path, new_stem)
    if make_exclusion and add_to_exclusions(new_stem):
        log(f"  added to exclusion list: {new_stem}")
    if moved:
        return True, f"Applied: {new_stem}"
    return True, f"Metadata applied; filename unchanged: {new_stem}"


# ============================================================================
# GUI
# ============================================================================
README_TEXT = """\
Re:Tentative - Stable Diffusion Model Manager  [Beta]

WHAT IT DOES
  Keeps a local Stable Diffusion model library tidy. It classifies your
  checkpoints and LoRAs by base model, renames them to a consistent
  [BaseModel]-Friendly Name format, downloads a preview image when one is
  missing, sorts freshly-downloaded files into the correct folders, and
  includes a small Civitai download queue.

FIRST-TIME SETUP
  1. Click Configuration and point the tool at your three model folders:
     Checkpoints, LoRA, and VAE.
  2. Paste your Civitai API key and click Save Key. The key lets the tool
     read model metadata and download previews without rate-limit stalls.
  A "Downloaded Content" folder is created next to this program; new
  downloads land there before being sorted.

THE BUTTONS
  Add Batch      Load a .txt file of Civitai URLs (one per line) into the queue.
  + / Ctrl+V     Add a single Civitai URL from your clipboard to the queue.
                 Click the small + on the queue's top-right corner, or press
                 Ctrl+V with the queue focused.
  Remove         Remove the selected queue entry.
  Clear Queue    Empty the whole queue. Asks first only if a download is
                 running; otherwise clears immediately.
  Begin Download Download everything in the queue into Downloaded Content,
                 then sort and label/skin the new files automatically. Each
                 file shows a live progress bar that collapses to one record
                 line when it finishes.
  Cancel         Stop after the current file finishes.
  Label & Skin   Walk your whole library: classify each model, rename it to
                 [BaseModel]-Friendly Name, and grab a preview if missing.
                 Safe to run repeatedly -- already-named files are skipped.
  Sort Only      Move loose files out of Downloaded Content into the right
                 model folder (VAEs, LoRAs, checkpoints).
  Repair Asset   Fix one asset by hand. Search your library, pick a file, and
                 correct its base model, name, keywords, or preview.
  Add Asset      Register a file that did NOT come from Civitai (HuggingFace,
                 TensorHub, a local merge). Pick it with Load From Local; it is
                 labeled and skinned, and added to the exclusion list so the
                 auto-pass leaves it alone.
  Save Queue     Save the current queue so it reloads automatically next launch.
  Export Queue   Save the queue to a .txt file you choose (shareable; loads
                 back in via Add Batch).
  Configuration  Set the three model folder paths.
  Save Key       Store your Civitai API key.
  Exit           Close the program.

HOW CLASSIFICATION WORKS
  For each file the tool tries, in order:
    1. Read the base model from the file's .civitai.info sidecar (LoRAs).
    2. If there's no sidecar (checkpoints), hash the file and ask Civitai
       what it is by hash.
    3. If that returns no base model, scrape the model page as a last resort.
    4. If nothing works, the file is QUARANTINED: left completely untouched
       and logged to _unresolved.txt. The tool never guesses a base model
       from a filename, so it will never write a wrong label silently.

PREVIEWS
  When a model has no preview, one is fetched from its Civitai images. Many
  Civitai previews are videos, so the tool prefers a real still: it skips
  videos and falls through to the next image, and validates that what it saves
  is actually an image. If no usable still exists, the model is left without a
  preview rather than with a broken one.

FILES THE TOOL KEEPS  (all next to this program)
  _exclusion_list.txt  Filename stems to skip entirely. Put non-Civitai files
                       here so they aren't re-checked every run.
  _unresolved.txt      A fresh report each run of files that couldn't be
                       resolved -- your punch list for manual Repair.
  _skip_cache.json     Remembers files that failed once, so big files aren't
                       re-hashed every run. Delete it to force a full retry.
  _saved_queue.txt     Your saved download queue. Written by Save Queue and
                       reloaded automatically on the next launch.
  sd_mm_config.json    Your API key and folder paths.

A NOTE ABOUT RENAMING
  Renaming model files changes the strings your SD UI stored in older image
  metadata and in any saved styles or workflows. This is expected. After a
  run, click the refresh arrows on your UI's model lists so it re-reads the
  new names.
"""


def open_config_dialog(parent, cfg, on_saved=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    win = tk.Toplevel(parent)
    win.title("Configuration")
    win.geometry("640x220")
    win.resizable(False, False)
    win.transient(parent)
    win.grab_set()

    pad = {"padx": 8, "pady": 8}
    rows = [("Checkpoints folder:", "ckpt_dir"),
            ("LoRA folder:", "lora_dir"),
            ("VAE folder:", "vae_dir")]
    vars_ = {}

    for i, (label, key) in enumerate(rows):
        ttk.Label(win, text=label).grid(row=i, column=0, sticky="w", **pad)
        v = tk.StringVar(value=cfg.get(key, ""))
        vars_[key] = v
        ttk.Entry(win, textvariable=v, width=52).grid(row=i, column=1, sticky="we", **pad)

        def make_browse(var=v, lbl=label):
            def browse():
                d = filedialog.askdirectory(title=f"Select {lbl}",
                                            initialdir=var.get() or SCRIPT_DIR)
                if d:
                    var.set(d)
            return browse
        ttk.Button(win, text="Browse", command=make_browse()).grid(
            row=i, column=2, **pad)

    win.columnconfigure(1, weight=1)

    def save():
        for key in ("ckpt_dir", "lora_dir", "vae_dir"):
            cfg[key] = vars_[key].get().strip()
        save_config(cfg)
        missing = [k for k in ("ckpt_dir", "lora_dir", "vae_dir")
                   if not os.path.isdir(cfg.get(k, ""))]
        if missing:
            messagebox.showwarning(
                "Saved with warnings",
                "Saved, but one or more folders don't exist yet:\n" +
                "\n".join(cfg[k] or "(blank)" for k in missing))
        else:
            messagebox.showinfo("Saved", "Folder paths saved.")
        win.destroy()
        if on_saved:
            on_saved()

    btns = ttk.Frame(win)
    btns.grid(row=len(rows), column=0, columnspan=3, pady=8)
    ttk.Button(btns, text="Save", command=save).pack(side="left", padx=4)
    ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=4)


def open_help_dialog(parent):
    import tkinter as tk
    from tkinter import ttk

    win = tk.Toplevel(parent)
    win.title("Re:Tentative - Help / README")
    win.geometry("680x640")
    win.transient(parent)

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill="both", expand=True)
    txt = tk.Text(frame, wrap="word")
    txt.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    sb.pack(side="right", fill="y")
    txt.config(yscrollcommand=sb.set)
    txt.insert("1.0", README_TEXT)
    txt.config(state="disabled")


def open_asset_form(parent, mode, cfg, log):
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    api_key = cfg.get("api_key", "")
    win = tk.Toplevel(parent)
    win.title("Repair Asset" if mode == "repair" else "Add Asset")
    win.geometry("620x600")
    win.resizable(False, False)
    win.transient(parent)

    pad = {"padx": 8, "pady": 4}
    OTHER = "Other... (type new)"

    type_frame = ttk.Frame(win, padding=(8, 8, 8, 0))
    type_frame.pack(fill="x")
    ttk.Label(type_frame, text="Asset type:").pack(side="left")
    type_var = tk.StringVar(value="Lora")
    ttk.Radiobutton(type_frame, text="Lora", variable=type_var,
                    value="Lora").pack(side="left", padx=6)
    ttk.Radiobutton(type_frame, text="Checkpoint", variable=type_var,
                    value="Checkpoint").pack(side="left", padx=6)

    def folder_for_type():
        return cfg["lora_dir"] if type_var.get() == "Lora" else cfg["ckpt_dir"]

    body = ttk.Frame(win, padding=8)
    body.pack(fill="both", expand=True)
    selected_path = {"path": None}

    if mode == "repair":
        ttk.Label(body, text="Search asset:").grid(row=0, column=0, sticky="w", **pad)
        search_var = tk.StringVar()
        ttk.Entry(body, textvariable=search_var, width=48).grid(
            row=0, column=1, sticky="we", **pad)
        results_list = tk.Listbox(body, height=5)
        results_list.grid(row=1, column=0, columnspan=2, sticky="we", **pad)

        def refresh_results(*_):
            q = search_var.get().strip().lower()
            results_list.delete(0, "end")
            for s in list_model_stems(folder_for_type()):
                if q in s.lower():
                    results_list.insert("end", s)
        search_var.trace_add("write", refresh_results)
        type_var.trace_add("write", refresh_results)
        refresh_results()

    # Add mode: pick the local .safetensors up front so the file is chosen
    # deliberately, not sprung as an Open dialog at commit time. This row sits
    # where Repair's search list would be. Selecting a file fills the friendly
    # name from its stem as a starting point.
    add_file_var = tk.StringVar()
    if mode == "add":
        ttk.Label(body, text="Model file:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(body, textvariable=add_file_var, width=42, state="readonly").grid(
            row=0, column=1, sticky="we", **pad)

        def browse_model():
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            init_dir = downloads if os.path.isdir(downloads) else (
                folder_for_type() or SCRIPT_DIR)
            f = filedialog.askopenfilename(
                title="Load model file from local disk",
                initialdir=init_dir,
                filetypes=[("Models", "*.safetensors *.ckpt *.pt"),
                           ("All", "*.*")])
            if not f:
                return
            selected_path["path"] = f
            add_file_var.set(os.path.basename(f))
            if not friendly_var.get().strip():
                friendly_var.set(decamel(os.path.splitext(os.path.basename(f))[0]))
            update_preview_name()
        ttk.Button(body, text="Load From Local", command=browse_model).grid(
            row=0, column=2, sticky="w", padx=(0, 8))

    status_var = tk.StringVar(value="")
    if mode == "repair":
        ttk.Label(body, textvariable=status_var, foreground="#555",
                  justify="left", wraplength=560).grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)

    base_row = 3 if mode == "repair" else 0

    # Civitai URL + Pull-from-URL only make sense in Repair, where the asset
    # is a Civitai file whose metadata we can re-fetch. Add operates on a file
    # that came from somewhere else (HuggingFace/TensorHub/local merge) and is
    # already on disk -- there is nothing to pull -- so both are hidden in Add.
    url_var = tk.StringVar()
    if mode == "repair":
        ttk.Label(body, text="Civitai URL:").grid(row=base_row, column=0, sticky="w", **pad)
        ttk.Entry(body, textvariable=url_var, width=42).grid(
            row=base_row, column=1, sticky="we", **pad)

    ttk.Label(body, text="Base model:").grid(row=base_row + 1, column=0, sticky="w", **pad)
    base_choices = scan_known_base_models(cfg) + [OTHER]
    base_var = tk.StringVar(value=base_choices[0])
    ttk.Combobox(body, textvariable=base_var, values=base_choices,
                 state="readonly", width=39).grid(
        row=base_row + 1, column=1, sticky="we", **pad)
    other_var = tk.StringVar()
    other_entry = ttk.Entry(body, textvariable=other_var, width=42)

    def on_base_change(*_):
        if base_var.get() == OTHER:
            other_entry.grid(row=base_row + 2, column=1, sticky="we", **pad)
            ttk.Label(body, text="New base name:").grid(
                row=base_row + 2, column=0, sticky="w", **pad)
        else:
            other_entry.grid_remove()
        update_preview_name()
    base_var.trace_add("write", on_base_change)

    ttk.Label(body, text="Friendly name:").grid(row=base_row + 3, column=0, sticky="w", **pad)
    friendly_var = tk.StringVar()
    ttk.Entry(body, textvariable=friendly_var, width=42).grid(
        row=base_row + 3, column=1, sticky="we", **pad)
    friendly_var.trace_add("write", lambda *_: update_preview_name())

    ttk.Label(body, text="Keywords (comma):").grid(row=base_row + 4, column=0, sticky="w", **pad)
    kw_var = tk.StringVar()
    ttk.Entry(body, textvariable=kw_var, width=42).grid(
        row=base_row + 4, column=1, sticky="we", **pad)

    prev_label = "Preview image (file):" if mode == "add" else "Preview (URL/file):"
    ttk.Label(body, text=prev_label).grid(row=base_row + 5, column=0, sticky="w", **pad)
    prev_var = tk.StringVar()
    ttk.Entry(body, textvariable=prev_var, width=32).grid(
        row=base_row + 5, column=1, sticky="we", **pad)

    def browse_preview():
        # Add works on a file already on disk, so default the image picker to
        # that file's own folder, not the script dir -- the preview usually
        # sits right next to the model it belongs to.
        init_dir = SCRIPT_DIR
        if mode == "add":
            existing = selected_path.get("path")
            if existing:
                init_dir = os.path.dirname(existing)
            elif os.path.isdir(folder_for_type()):
                init_dir = folder_for_type()
        f = filedialog.askopenfilename(
            title="Select preview image", initialdir=init_dir,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")])
        if f:
            prev_var.set(f)
    ttk.Button(body, text="Browse", command=browse_preview).grid(
        row=base_row + 5, column=2, sticky="w", padx=(0, 8))

    ttk.Label(body, text="Result filename:").grid(row=base_row + 6, column=0, sticky="w", **pad)
    preview_name_var = tk.StringVar(value="")
    ttk.Label(body, textvariable=preview_name_var, foreground="#0a6",
              wraplength=420, justify="left").grid(
        row=base_row + 6, column=1, columnspan=2, sticky="w", **pad)

    def current_tag():
        bm = other_var.get().strip() if base_var.get() == OTHER else base_var.get()
        return tag_for_base_model(bm) if bm else "[?]"

    def update_preview_name(*_):
        friendly = friendly_var.get().strip()
        if not friendly:
            preview_name_var.set("(enter a friendly name)")
            return
        preview_name_var.set(build_new_stem(current_tag(), friendly, friendly)
                             + ".safetensors")

    def load_selected(*_):
        if mode != "repair":
            return
        sel = results_list.curselection()
        if not sel:
            return
        stem = results_list.get(sel[0])
        path = find_model_file(stem, folder_for_type())
        if not path:
            return
        selected_path["path"] = path
        st = asset_status(path)
        status_var.set(
            f"File: {os.path.basename(path)}\n"
            f"Sidecar: {'yes' if st['has_sidecar'] else 'NO'}   "
            f"Preview: {'yes' if st['has_preview'] else 'NO'}   "
            f"Base: {st['base_model'] or 'unknown'}")
        if st["base_model"] and st["base_model"] in base_choices:
            base_var.set(st["base_model"])
        if st["model_name"]:
            friendly_var.set(st["model_name"])
        elif not friendly_var.get():
            friendly_var.set(decamel(stem))
        if st["keywords"]:
            kw_var.set(", ".join(st["keywords"]))
        update_preview_name()
    if mode == "repair":
        results_list.bind("<<ListboxSelect>>", load_selected)

    def pull_from_url():
        u = url_var.get().strip()
        if not u:
            messagebox.showinfo("No URL", "Paste a Civitai URL first.")
            return
        meta = fetch_metadata_by_url(u, api_key)
        if not meta:
            messagebox.showwarning("Pull failed",
                                   "Couldn't fetch metadata from that URL.")
            return
        if meta.get("base_model") and meta["base_model"] in base_choices:
            base_var.set(meta["base_model"])
        elif meta.get("base_model"):
            base_var.set(OTHER)
            other_var.set(meta["base_model"])
        if meta.get("model_name"):
            friendly_var.set(meta["model_name"])
        if meta.get("keywords"):
            kw_var.set(", ".join(meta["keywords"]))
        if meta.get("preview_url") and not prev_var.get().strip():
            prev_var.set(meta["preview_url"])
        update_preview_name()
    if mode == "repair":
        ttk.Button(body, text="Pull from URL", command=pull_from_url).grid(
            row=base_row, column=2, sticky="w", padx=(0, 8))

    body.columnconfigure(1, weight=1)

    action = ttk.Frame(win, padding=8)
    action.pack(fill="x", side="bottom")

    def do_apply():
        if mode == "repair":
            path = selected_path["path"]
            if not path:
                messagebox.showwarning("No asset", "Select an asset to repair.")
                return
        else:
            # Add: the file was chosen up front via Load From Local. No guessing,
            # no commit-time picker. If nothing's loaded, say so plainly.
            path = selected_path["path"]
            if not path:
                messagebox.showwarning(
                    "No file loaded",
                    "Click Load From Local and choose the model file to add.")
                return
        bm = other_var.get().strip() if base_var.get() == OTHER else base_var.get()
        if not bm:
            messagebox.showwarning("No base model", "Choose or type a base model.")
            return
        friendly = friendly_var.get().strip()
        if not friendly:
            messagebox.showwarning("No name", "Enter a friendly name.")
            return
        keywords = [k.strip() for k in kw_var.get().split(",") if k.strip()]
        tag = tag_for_base_model(bm)
        ok, msg = apply_manual_asset(
            path, tag, friendly, keywords, prev_var.get().strip(),
            api_key=api_key, make_exclusion=(mode == "add"), log=log)
        log(f"[{mode}] {msg}")
        messagebox.showinfo("Done" if ok else "Failed", msg)
        if ok:
            win.destroy()

    apply_text = "Attempt Repair" if mode == "repair" else "Attempt Add"
    cancel_text = "Cancel Repair" if mode == "repair" else "Cancel Add"
    ttk.Button(action, text=apply_text, command=do_apply).pack(side="left", padx=4)
    ttk.Button(action, text=cancel_text, command=win.destroy).pack(side="left", padx=4)

    on_base_change()
    update_preview_name()


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    ensure_working_files()
    cfg = load_config()
    cancel_flag = threading.Event()

    root = tk.Tk()
    root.title("Re:Tentative-Stable Diffusion Model Manager [Beta]")
    root.geometry("840x600")
    root.resizable(False, False)

    # Live-download flag: Clear Queue confirms only while a run is active.
    download_active = threading.Event()

    # --- top row: queue-file actions (left), API key + Save Key + ? (right) ---
    key_frame = ttk.Frame(root, padding=8)
    key_frame.pack(fill="x")
    key_var = tk.StringVar(value=cfg.get("api_key", ""))

    def save_key():
        cfg["api_key"] = key_var.get().strip()
        save_config(cfg)
        messagebox.showinfo("Saved", "API key saved.")

    # LEFT group (cols 2-5): queue-file actions + Exit + Configuration.
    ttk.Button(key_frame, text="Save Queue",
               command=lambda: save_queue()).grid(row=0, column=2, padx=3)
    ttk.Button(key_frame, text="Export Queue",
               command=lambda: export_queue()).grid(row=0, column=3, padx=3)
    ttk.Button(key_frame, text="Exit",
               command=lambda: root.destroy()).grid(row=0, column=4, padx=3)
    ttk.Button(key_frame, text="Configuration",
               command=lambda: open_config_dialog(root, cfg)).grid(row=0, column=5, padx=3)

    key_frame.columnconfigure(6, weight=1)   # spacer eats the slack

    # RIGHT group (cols 7-10): API Key label + field, then Save Key, then ?.
    ttk.Label(key_frame, text="Civitai API Key:").grid(row=0, column=7, sticky="e", padx=(0, 4))
    ttk.Entry(key_frame, textvariable=key_var, show="*", width=32).grid(
        row=0, column=8, padx=(0, 6))
    ttk.Button(key_frame, text="Save Key",
               command=save_key).grid(row=0, column=9, padx=3)
    ttk.Button(key_frame, text="?", width=3,
               command=lambda: open_help_dialog(root)).grid(row=0, column=10, padx=(3, 0))

    # --- download queue ---
    list_frame = ttk.LabelFrame(root, text="Download Queue (Civitai .com / .red)",
                                padding=8)
    list_frame.pack(fill="both", expand=False, padx=8, pady=4)
    listbox = tk.Listbox(list_frame, height=8)
    listbox.pack(fill="both", expand=True, side="left")
    sb = ttk.Scrollbar(list_frame, command=listbox.yview)
    sb.pack(side="right", fill="y")
    listbox.config(yscrollcommand=sb.set)

    btn_frame = ttk.Frame(root, padding=8)
    btn_frame.pack(fill="x")

    def add_from_clipboard(silent=False):
        """Validate the clipboard as a Civitai URL and append if new.
        Shared by the '+' button and the Ctrl+V binding on the queue."""
        try:
            clip = root.clipboard_get().strip()
        except tk.TclError:
            if not silent:
                messagebox.showwarning("Clipboard empty", "Nothing in clipboard.")
            return
        if "civitai." not in clip:
            if not silent:
                messagebox.showwarning(
                    "Not a Civitai URL",
                    f"Clipboard doesn't look like a Civitai URL:\n{clip}")
            return
        if clip in set(listbox.get(0, "end")):
            if not silent:
                messagebox.showinfo("Duplicate", "That URL is already in the queue.")
            return
        listbox.insert("end", clip)

    # Ctrl+V on the queue area pastes-and-validates (silent: no popup on miss).
    def _paste_into_queue(_event=None):
        add_from_clipboard(silent=True)
        return "break"
    listbox.bind("<Control-v>", _paste_into_queue)
    list_frame.bind("<Control-v>", _paste_into_queue)

    # '+' on the queue frame's top-right border line, lifted up onto the border
    # itself (y negative) so it rides above the scrollbar's up-arrow.
    plus_btn = ttk.Button(list_frame, text="+", width=2,
                          command=lambda: add_from_clipboard(silent=False))
    plus_btn.place(relx=1.0, rely=0.0, x=8, y=-1, anchor="se")

    def add_batch():
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select a .txt of Civitai URLs (one per line)",
            initialdir=os.getcwd(),
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            messagebox.showerror("Read failed", str(e))
            return
        existing = set(listbox.get(0, "end"))
        added = skipped = dupes = 0
        for raw in lines:
            u = raw.strip()
            if not u:
                continue
            if "civitai." not in u:
                skipped += 1
                continue
            if u in existing:
                dupes += 1
                continue
            listbox.insert("end", u)
            existing.add(u)
            added += 1
        log(f"Batch load: added {added}, skipped {skipped} non-Civitai, "
            f"{dupes} duplicate(s).")

    def remove_selected():
        for i in reversed(listbox.curselection()):
            listbox.delete(i)

    def save_queue():
        """Silent auto-reload save to _saved_queue.txt (loaded on next launch)."""
        urls = list(listbox.get(0, "end"))
        if save_queue_to_disk(urls):
            log(f"Queue saved ({len(urls)} item(s)) -- will auto-reload on next launch.")
        else:
            log("! Queue save failed.")

    def export_queue():
        """Write the queue to a user-chosen portable .txt (round-trips via Add Batch)."""
        from tkinter import filedialog
        urls = list(listbox.get(0, "end"))
        if not urls:
            messagebox.showinfo("Empty", "Queue is empty -- nothing to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export queue to a .txt of URLs",
            initialdir=os.getcwd(), defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for u in urls:
                    f.write(u + "\n")
            log(f"Queue exported ({len(urls)} item(s)) -> {path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def clear_queue():
        """Empty the queue. Confirm only if a download run is active; else silent.
        Also clears the auto-reload file so it doesn't repopulate on restart."""
        if download_active.is_set():
            if not messagebox.askyesno(
                "Clear queue?",
                "A download is currently running. Clear the queue anyway?\n"
                "(The in-progress file finishes; remaining items are removed.)"):
                return
        listbox.delete(0, "end")
        save_queue_to_disk([])   # wipe auto-reload so restart doesn't restore it
        log("Queue cleared.")

    # --- log ---
    log_frame = ttk.LabelFrame(root, text="Log", padding=8)
    log_frame.pack(fill="both", expand=True, padx=8, pady=4)
    log_text = tk.Text(log_frame, height=12, wrap="word")
    log_text.pack(fill="both", expand=True)

    def log(msg):
        log_text.insert("end", str(msg) + "\n")
        log_text.see("end")
        root.update_idletasks()

    def _render_bar(frac, label, width=24):
        """Build a terminal-style bar: [#######         42%] name."""
        frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
        filled = int(round(frac * width))
        bar = "#" * filled + " " * (width - filled)
        return f"  [{bar} {int(frac * 100):3d}%] {label}"

    def _bar_bytes(done, label):
        """Fallback when the server sent no Content-Length: show MB pulled."""
        return f"  [downloading... {done / (1024 * 1024):6.1f} MB] {label}"

    def log_progress(text):
        """Rewrite the LAST line of the log in place (the live progress bar).
        Marshaled to the main thread; safe to call from the download worker.
        The bar owns the last line during a download; log_finalize replaces it
        with a permanent record when the file completes."""
        def _do():
            # 'end-1c linestart' is the start of the final line; delete to end,
            # then write the new bar with no trailing newline so it stays put.
            log_text.delete("end-1c linestart", "end-1c")
            log_text.insert("end-1c", text)
            log_text.see("end")
        root.after(0, _do)

    def log_finalize(text):
        """Replace the live bar line with a permanent one-line record, then
        open a fresh line so the next log() append lands cleanly."""
        def _do():
            log_text.delete("end-1c linestart", "end-1c")
            log_text.insert("end-1c", text + "\n")
            log_text.see("end")
        root.after(0, _do)

    def log_open_bar_line():
        """Append an empty line that the bar will own and rewrite in place."""
        def _do():
            log_text.insert("end", "")
            log_text.see("end")
        root.after(0, _do)

    def in_thread(fn):
        threading.Thread(target=fn, daemon=True).start()

    def require_paths():
        if not paths_configured(cfg):
            messagebox.showinfo(
                "Configuration needed",
                "Set your Checkpoints, LoRA, and VAE folders in Configuration first.")
            open_config_dialog(root, cfg)
            return False
        return True

    def begin_download():
        if not require_paths():
            return
        cancel_flag.clear()
        urls = list(listbox.get(0, "end"))
        if not urls:
            messagebox.showinfo("Empty", "Queue is empty.")
            return
        api_key = key_var.get().strip()
        if not api_key:
            messagebox.showwarning("No API key",
                                   "Enter and save your Civitai API key first.")
            return
        cfg["api_key"] = api_key
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        def work():
            download_active.set()
            try:
                log(f"Starting {len(urls)} download(s)...")
                for url in urls:
                    if cancel_flag.is_set():
                        log("Cancelled by user.")
                        return
                    kind, cid = civitai_version_id_from_url(url)
                    if not cid:
                        log(f"  ! could not parse id from: {url}")
                        continue
                    try:
                        dl_url, fname = resolve_download_url(kind, cid, api_key)
                    except Exception as e:
                        log(f"  ! resolve failed for {url}: {e}")
                        continue
                    if not dl_url:
                        log(f"  ! no downloadable file for {url}")
                        continue
                    dest = os.path.join(DOWNLOAD_DIR, fname or f"{cid}.safetensors")
                    shown = fname or os.path.basename(dest)

                    # Per-file progress bar. http_download calls back with the
                    # fraction done; we throttle to integer-percent changes so
                    # tkinter's event queue isn't flooded (the callback can fire
                    # hundreds of times a second otherwise). The bar owns the
                    # last log line during the download, then log_finalize
                    # replaces it with one permanent record line.
                    log_open_bar_line()
                    log_progress(_render_bar(0.0, shown))
                    _last_pct = {"v": -1}

                    def _on_progress(frac, _label=shown, _state=_last_pct):
                        # Positive: a real fraction (0..1) -> percent bar.
                        # Negative sentinel -(bytes+1): no server total, so
                        # show a MB counter that at least proves movement.
                        # Both branches throttle (integer percent / whole MB)
                        # so tkinter's event queue isn't flooded per chunk.
                        if frac < 0:
                            done_mb = int(((-frac) - 1) / (1024 * 1024))
                            if done_mb != _state["v"]:
                                _state["v"] = done_mb
                                log_progress(_bar_bytes((-frac) - 1, _label))
                            return
                        pct = int(frac * 100)
                        if pct != _state["v"]:
                            _state["v"] = pct
                            log_progress(_render_bar(frac, _label))

                    try:
                        ok = http_download(dl_url, dest, api_key=api_key,
                                           cancel_flag=cancel_flag,
                                           progress=_on_progress)
                        if ok:
                            try:
                                mb = os.path.getsize(dest) / (1024 * 1024)
                                size_str = f" ({mb:.1f} MB)"
                            except OSError:
                                size_str = ""
                            log_finalize(f"  done: {shown}{size_str}")
                        else:
                            log_finalize(f"  cancelled: {shown}")
                    except Exception as e:
                        log_finalize(f"  ! download error: {shown} -- {e}")
                if cancel_flag.is_set():
                    return
                log("\nSorting downloads into model folders...")
                sort_downloads(cfg, log=log, cancel_flag=cancel_flag)
                log("\nLabel & Skin pass on new files...")
                label_and_skin(cfg, log=log, cancel_flag=cancel_flag)
                log("\nAll done.")
            finally:
                download_active.clear()
        in_thread(work)

    def label_skin_only():
        if not require_paths():
            return
        cancel_flag.clear()
        cfg["api_key"] = key_var.get().strip()
        if not cfg["api_key"]:
            if not messagebox.askyesno(
                "No API key",
                "No API key set. LoRAs with sidecars still process, but "
                "checkpoints (no sidecar) can't be classified and will be "
                "quarantined. Continue anyway?"):
                return
        in_thread(lambda: label_and_skin(cfg, log=log, cancel_flag=cancel_flag))

    def sort_only():
        if not require_paths():
            return
        cancel_flag.clear()
        in_thread(lambda: sort_downloads(cfg, log=log, cancel_flag=cancel_flag))

    def cancel_all():
        cancel_flag.set()
        log("Cancel requested -- finishing current file then stopping.")

    # Button row on grid. Every column gets equal weight and each button
    # stretches to fill (sticky ew) with roomy padding, so the buttons spread
    # across the whole width instead of clustering with a gap in the middle.
    _mid_btns = [
        ("Add Batch", add_batch),
        ("Remove", remove_selected),
        ("Begin Download", begin_download),
        ("Cancel", cancel_all),
        ("Clear Queue", clear_queue),
        ("Label & Skin", label_skin_only),
        ("Sort Only", sort_only),
        ("Repair Asset", lambda: open_asset_form(root, "repair", cfg, log)),
        ("Add Asset", lambda: open_asset_form(root, "add", cfg, log)),
    ]
    for _c, (_label, _cmd) in enumerate(_mid_btns):
        ttk.Button(btn_frame, text=_label, command=_cmd).grid(
            row=0, column=_c, padx=6, sticky="ew")
        btn_frame.columnconfigure(_c, weight=1)

    log("Re:Tentative ready.")
    log(f"Working folder: {SCRIPT_DIR}")
    log("Click ? for the README. First time: set folders in Configuration, "
        "then Save your Civitai API key.")

    # Auto-reload a previously saved queue, if one exists next to the script.
    _restored = load_saved_queue()
    if _restored:
        for _u in _restored:
            listbox.insert("end", _u)
        log(f"Restored {len(_restored)} queued URL(s) from _saved_queue.txt.")

    # --- first-run onboarding: Configuration, then API key prompt ---
    def first_run_flow():
        if not paths_configured(cfg):
            messagebox.showinfo(
                "Welcome to Re:Tentative",
                "First, tell the tool where your model folders are.")
            open_config_dialog(root, cfg, on_saved=prompt_api_key)
        elif not cfg.get("api_key"):
            prompt_api_key()

    def prompt_api_key():
        if cfg.get("api_key"):
            return
        from tkinter import simpledialog
        key = simpledialog.askstring(
            "Civitai API Key",
            "Paste your Civitai API key (leave blank to skip for now):",
            parent=root, show="*")
        if key:
            cfg["api_key"] = key.strip()
            key_var.set(cfg["api_key"])
            save_config(cfg)

    root.after(300, first_run_flow)
    root.mainloop()


def main():
    ensure_working_files()
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        cfg = load_config()
        if cmd == "labelskin":
            label_and_skin(cfg)
        elif cmd == "sort":
            sort_downloads(cfg)
        elif cmd == "gui":
            launch_gui()
        else:
            print("Usage: retentative.py [gui|labelskin|sort]")
    else:
        launch_gui()


if __name__ == "__main__":
    main()
