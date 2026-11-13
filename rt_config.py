## File Name: rt_config.py
## Creator: DatJavaClass
## Associated Files: rt_download.py, rt_library.py, rt_gui.py, rt_forms.py, retentative.py
## Exposed Vars: CONFIG_PATH, LIBRARY_PATH, QUEUE_PATH, DOWNLOAD_DIR, DEFAULT_DOWNLOAD_DIR
## Exposed Funcs: load_config, save_config, apply_download_dir, clamp_speed, add_to_exclusions, log_unresolved
## Desc: Paths, config load and save, staging folder, download speed limits, working files, exclusions and the unresolved log.
import os, re, json, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_p = lambda name: os.path.join(SCRIPT_DIR, name) ## script adjacent path

CONFIG_PATH, LIBRARY_PATH, EXCLUSION_PATH = _p("sd_mm_config.json"), _p("_library.json"), _p("_exclusion_list.txt")
UNRESOLVED_PATH, QUEUE_PATH, LEGACY_QUEUE_PATH = _p("_unresolved.txt"), _p("_saved_queue.json"), _p("_saved_queue.txt")
DEFAULT_DOWNLOAD_DIR = _p("Downloaded Content")
DOWNLOAD_DIR = DEFAULT_DOWNLOAD_DIR
MODEL_EXTS = {".safetensors", ".ckpt", ".pt"}
DIR_KINDS = (("ckpt_dir", "checkpoint"), ("lora_dir", "lora"), ("vae_dir", "vae"))
SPEED_LIMITS = (("connections", 1, 16), ("parallel_files", 1, 6))
EXCLUSION_HEADER = ("# One filename stem per line. Lines starting with # are ignored.\n"
                    "# Matching files are skipped: no hash, no rename, no unresolved entry.\n"
                    "# Add non Civitai files here (local merges, HuggingFace, TensorHub).\n")

def default_config():
    return {"ckpt_dir": "", "lora_dir": "", "vae_dir": "", "api_key": "", "add_version": True, "window": "",
            "connections": 2, "parallel_files": 2, "download_dir": ""}

def apply_download_dir(cfg):
    global DOWNLOAD_DIR
    folder = str((cfg or {}).get("download_dir") or "").strip()
    DOWNLOAD_DIR = os.path.abspath(os.path.expanduser(folder)) if folder else DEFAULT_DOWNLOAD_DIR ## blank means beside the program
    return DOWNLOAD_DIR

def _clamp(value, low, high, fallback):
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return fallback ## junk in the config file

def clamp_speed(cfg):
    base = default_config()
    return tuple(_clamp((cfg or {}).get(key), low, high, base[key]) for key, low, high in SPEED_LIMITS)

def load_config():
    cfg = default_config()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass ## missing or corrupt, defaults win

    apply_download_dir(cfg)
    return cfg

def save_config(cfg):
    apply_download_dir(cfg)
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CONFIG_PATH)
        return True
    except OSError:
        return False

def folders(cfg):
    return [(cfg.get(key) or "", kind) for key, kind in DIR_KINDS]

def paths_configured(cfg):
    return all(p and os.path.isdir(p) for p, _ in folders(cfg))

def ensure_working_files():
    try:
        if not os.path.isfile(EXCLUSION_PATH):
            with open(EXCLUSION_PATH, "w", encoding="utf-8") as f:
                f.write(EXCLUSION_HEADER)
        if not os.path.isfile(UNRESOLVED_PATH):
            open(UNRESOLVED_PATH, "w", encoding="utf-8").close()
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except OSError:
        pass ## read only folder, callers degrade

def normalize_stem(name):
    root, ext = os.path.splitext(name.strip())
    stem = root if ext.lower() in MODEL_EXTS else name.strip() ## keep dotted versions like v1.5
    return re.sub(r"-0*\d{4,6}$", "", stem).strip().lower() ## drop kohya step suffix

def load_exclusions():
    ensure_working_files()
    try:
        with open(EXCLUSION_PATH, "r", encoding="utf-8") as f:
            return {normalize_stem(s) for s in (ln.strip() for ln in f) if s and not s.startswith("#")}
    except OSError:
        return set()

def add_to_exclusions(stem):
    if normalize_stem(stem) in load_exclusions():
        return False
    try:
        with open(EXCLUSION_PATH, "a", encoding="utf-8") as f:
            f.write(stem + "\n")
        return True
    except OSError:
        return False

def log_unresolved(path, tried, reason):
    tried = ", ".join(tried) if isinstance(tried, (list, tuple)) else tried
    try:
        with open(UNRESOLVED_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {path}\n    tried: {tried}\n    reason: {reason}\n\n")
    except OSError:
        pass ## logging must never break a run
