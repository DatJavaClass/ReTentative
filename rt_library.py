## Library database, sidecars, naming, rename, previews, sort.
import datetime, json, os, re, shutil
import rt_civitai, rt_config, rt_net

MAX_FRIENDLY, MAX_VERSION = 80, 40 ## keeps paths under MAX_PATH
PREVIEW_EXTS = (".preview.png", ".preview.jpg", ".preview.jpeg")
KNOWN_VAE_NAMES = {"sdxl.vae", "qwen_image_vae", "vae-ft-mse-840000-ema-pruned"}
KIND_BY_TYPE = {"checkpoint": "checkpoint", "lora": "lora", "locon": "lora", "dora": "lora", "vae": "vae"}
ENTRY_FIELDS = ("model_id", "version_id", "model_name", "version_name", "base_model")
TAG_RX, VER_RX = re.compile(r"^\[[^\[\]]*\]\s*-?\s*"), re.compile(r"\s*\(([^()]*)\)\s*$")
LORA_MAX_BYTES = 500 * 1024 * 1024 ## size fallback for LoRA guess

def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")

def _norm(path):
    return os.path.normcase(os.path.abspath(path))

def _same(a, b):
    return bool(a) and bool(b) and _norm(a) == _norm(b)

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def _cancelled(cancel_flag):
    return cancel_flag is not None and cancel_flag.is_set()

def _is_sib(name, stem):
    return name == stem or name.startswith(stem + ".")

def _is_model(name):
    return os.path.splitext(name)[1].lower() in rt_config.MODEL_EXTS

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

def _write_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError):
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False

def _summary(title, counts, log):
    log(f"\n{title}")
    for k, v in counts.items():
        log(f"  {k:20s}: {v}")

def _reset_unresolved():
    try:
        open(rt_config.UNRESOLVED_PATH, "w", encoding="utf-8").close()
    except OSError:
        pass

def iter_model_files(folder):
    if not folder or not os.path.isdir(folder):
        return []
    try:
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if _is_model(f)]
    except OSError:
        return []

def load_library():
    data = _read_json(rt_config.LIBRARY_PATH)
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return {"version": 1, "files": {}}
    data["version"] = 1
    return data

def save_library(db):
    return _write_json(rt_config.LIBRARY_PATH, {"version": 1, "files": db.get("files", {})})

def file_key(path):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime
    except OSError:
        return None, None

def _index(db):
    files = db.setdefault("files", {})
    if db.get("_idx") is None or db.get("_idx_n") != len(files): ## stale after outside edits
        idx = {}
        for sha, e in files.items():
            for r in [e] + e.get("alts", []):
                if r.get("path"):
                    idx[_norm(r["path"])] = sha
        db["_idx"], db["_idx_n"] = idx, len(files)
    return db["_idx"]

def _record_for(entry, path):
    return next((r for r in [entry] + entry.get("alts", []) if _same(r.get("path"), path)), None)

def find_by_path(db, path):
    sha = _index(db).get(_norm(path))
    entry = db["files"].get(sha) if sha else None
    if entry is None or _record_for(entry, path) is None:
        return None, None
    return sha, entry

def hash_for(db, path, cancel_flag=None):
    size, mtime = file_key(path)
    if size is None:
        return None
    sha, entry = find_by_path(db, path)
    rec = _record_for(entry, path) if entry else None
    if rec and rec.get("size") == size and rec.get("mtime") == mtime:
        return sha
    try:
        return rt_net.sha256_file(path, cancel_flag=cancel_flag)
    except Exception:
        return None

def _blank_entry():
    return {"path": "", "size": 0, "mtime": 0.0, "kind": "", "model_id": None, "version_id": None,
            "model_name": None, "version_name": None, "base_model": None, "tag": None, "tags": [],
            "source_url": None, "image_url": None, "updated": ""}

def _page_url(entry):
    if not entry.get("model_id"):
        return None
    url = f"https://civitai.com/models/{entry['model_id']}"
    return url + (f"?modelVersionId={entry['version_id']}" if entry.get("version_id") else "")

def _fill(entry, rec):
    for k in ENTRY_FIELDS:
        if rec.get(k) is not None:
            entry[k] = rec[k]
    cands = rt_civitai.preview_candidates(rec.get("images") or [])
    entry["tag"] = rt_civitai.tag_for_base_model(entry.get("base_model"))
    entry["tags"] = list(rec.get("tags") or entry.get("tags") or [])
    entry["source_url"] = rec.get("source_url") or _page_url(entry) or entry.get("source_url")
    entry["image_url"] = cands[0] if cands else entry.get("image_url")
    entry["updated"] = _now()
    entry.pop("miss", None)

def _drop_path(db, sha, path):
    entry = db["files"].get(sha)
    if entry is None:
        return
    alts = [a for a in entry.get("alts", []) if not _same(a.get("path"), path)]
    if _same(entry.get("path"), path) and not alts: ## last copy gone
        db["files"].pop(sha)
    elif _same(entry.get("path"), path): ## promote first duplicate
        first = alts.pop(0)
        entry.update(path=first["path"], size=first["size"], mtime=first["mtime"])
    if sha in db["files"]:
        entry["alts"] = alts
    db.pop("_idx", None)

def _register(db, path, kind, rec=None, cancel_flag=None, sha=None):
    sha = sha or hash_for(db, path, cancel_flag) ## known sha skips the rehash
    if not sha:
        return None, None
    old_sha = _index(db).get(_norm(path))
    if old_sha and old_sha != sha: ## content changed under same path
        _drop_path(db, old_sha, path)
    size, mtime = file_key(path)
    files, here = db["files"], os.path.abspath(path)
    entry = files.setdefault(sha, _blank_entry())
    main = entry.get("path")
    if main and not _same(main, path) and os.path.exists(main): ## duplicate content, keep as alt
        alt = _record_for(entry, path)
        if alt is None:
            alt = {}
            entry.setdefault("alts", []).append(alt)
        alt.update(path=here, size=size, mtime=mtime)
    else:
        entry["alts"] = [a for a in entry.get("alts", []) if not _same(a.get("path"), path)]
        entry.update(path=here, size=size, mtime=mtime, kind=kind or entry.get("kind", ""))
    if rec:
        _fill(entry, rec)
    entry["updated"] = _now()
    _index(db)[_norm(path)] = sha
    db["_idx_n"] = len(files)
    return sha, entry

def register(db, path, kind, rec=None, sha=None):
    return _register(db, path, kind, rec, sha=sha)[1]

def _move_path(db, old, new):
    sha, entry = find_by_path(db, old)
    if entry is None:
        return
    size, mtime = file_key(new)
    _record_for(entry, old).update(path=os.path.abspath(new), size=size, mtime=mtime)
    idx = _index(db)
    idx.pop(_norm(old), None)
    idx[_norm(new)] = sha

def _prune(db):
    gone = [(sha, r["path"]) for sha, e in db["files"].items() for r in [e] + e.get("alts", []) if not os.path.exists(r.get("path") or "")]
    for sha, path in gone:
        _drop_path(db, sha, path)
    return len(gone)

def _is_orphan(entry):
    return not entry.get("model_id") or not entry.get("version_id")

def orphans(db):
    return [e for e in db.get("files", {}).values() if _is_orphan(e)]

def _cached_miss(db, path):
    sha, entry = find_by_path(db, path)
    rec = _record_for(entry, path) if entry else None
    return bool(rec and entry.get("miss") and (rec.get("size"), rec.get("mtime")) == file_key(path))

def _kind_of(path, cfg, rec=None):
    for folder, kind in rt_config.folders(cfg):
        if _same(os.path.dirname(os.path.abspath(path)), folder):
            return kind
    return KIND_BY_TYPE.get(((rec or {}).get("type") or "").lower(), "")

def sidecar_path(model_path):
    return os.path.splitext(model_path)[0] + ".civitai.info"

def read_sidecar(model_path):
    data = _read_json(sidecar_path(model_path))
    if not isinstance(data, dict):
        return None
    try:
        return rt_civitai.normalize(data, source="sidecar")
    except Exception:
        return None

def _sidecar_json(rec):
    raw = rec.get("raw")
    if not isinstance(raw, dict) or not raw: ## no Civitai JSON, build minimal
        return {"baseModel": rec.get("base_model"), "model": {"name": rec.get("model_name"), "type": rec.get("type")},
                "name": rec.get("version_name"), "trainedWords": list(rec.get("tags") or []),
                "images": list(rec.get("images") or []), "modelId": rec.get("model_id"), "id": rec.get("version_id")}
    data = dict(raw)
    if rec.get("base_model"):
        data["baseModel"] = rec["base_model"]
    if rec.get("model_name"):
        data["model"] = dict(data.get("model") or {}, name=rec["model_name"])
    if rec.get("version_name"):
        data["name"] = rec["version_name"]
    if rec.get("tags"):
        data["trainedWords"] = list(rec["tags"])
    if not data.get("images") and rec.get("images"):
        data["images"] = list(rec["images"])
    return data

def write_sidecar(model_path, rec):
    return _write_json(sidecar_path(model_path), _sidecar_json(rec or {}))

def write_a1111_json(model_path, rec):
    path = os.path.splitext(model_path)[0] + ".json"
    data = _read_json(path)
    data = data if isinstance(data, dict) else {}
    data.update({"activation text": ", ".join(rec.get("tags") or []), "sd version": rec.get("base_model") or ""})
    data.setdefault("description", "")
    return _write_json(path, data)

def decamel(stem):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem.replace("_", " ").replace("-", " "))
    return re.sub(r"\s+", " ", re.sub(r"(?<=[A-Za-z])(?=\d)", " ", s)).strip()

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "", name or "").strip()

def _strip_tags(text):
    prev = None
    while prev != text: ## collapse any repeated [Tag]- prefixes
        prev, text = text, TAG_RX.sub("", text, count=1).strip()
    return text

def _strip_version(text, version):
    m = VER_RX.search(text)
    while m and m.group(1).strip().lower() == version.lower(): ## drop repeated trailing (Version)
        text = text[:m.start()].rstrip()
        m = VER_RX.search(text)
    return text

def build_new_stem(tag, model_name, version_name, original_stem, add_version=True):
    if model_name and model_name.strip():
        friendly = _strip_tags(sanitize_filename(model_name))
    else: ## formatted stems are already friendly
        bare = _strip_tags(original_stem)
        friendly = sanitize_filename(bare if already_formatted(original_stem) else decamel(bare))
    version = re.sub(r"[()]", "", sanitize_filename(version_name))[:MAX_VERSION].strip()
    show = bool(add_version and version)
    friendly = _strip_version(friendly, version) if show else friendly
    show = show and version.lower() != friendly.lower()
    friendly = friendly[:MAX_FRIENDLY].rstrip() or "Unnamed"
    tag = sanitize_filename(tag)
    stem = f"{tag}-{friendly}" if tag else friendly
    return f"{stem} ({version})" if show else stem

def already_formatted(stem):
    return bool(re.match(r"^\[[^\]]+\]-", stem))

def has_version(stem):
    return bool(VER_RX.search(stem))

def _siblings(model_path):
    folder, own, stem = os.path.dirname(model_path) or ".", os.path.basename(model_path), _stem(model_path)
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    longer = [_stem(f) for f in names if _is_model(f) and _stem(f).startswith(stem + ".")]
    return [f for f in names if f != own and _is_sib(f, stem) and not _is_model(f) and not any(_is_sib(f, o) for o in longer)]

def _relocate(model_path, dest_dir, new_stem):
    old_stem, ext = os.path.splitext(os.path.basename(model_path))
    dst = os.path.join(dest_dir, new_stem + ext)
    if _same(model_path, dst) and os.path.basename(model_path) == os.path.basename(dst):
        return []
    if os.path.exists(dst) and not _same(model_path, dst): ## never overwrite another model
        rt_config.log_unresolved(model_path, ["rename"], f"target exists: {dst}")
        return []
    sibs = _siblings(model_path)
    try:
        shutil.move(model_path, dst)
    except OSError as e:
        rt_config.log_unresolved(model_path, ["rename"], f"move failed ({e.__class__.__name__}: {e}); target {dst}")
        return []
    results = [(model_path, dst)]
    for fname in sibs: ## sidecars, previews, json follow the model
        src, sdst = os.path.join(os.path.dirname(model_path), fname), os.path.join(dest_dir, new_stem + fname[len(old_stem):])
        if os.path.exists(sdst) and not _same(src, sdst):
            continue
        try:
            shutil.move(src, sdst)
            results.append((src, sdst))
        except OSError as e:
            rt_config.log_unresolved(src, ["rename"], f"move failed ({e.__class__.__name__}: {e}); target {sdst}")
    return results

def _relocate_db(model_path, dest_dir, new_stem, db=None):
    moved = _relocate(model_path, dest_dir, new_stem)
    if moved and db is not None: ## entry follows the file
        _move_path(db, model_path, moved[0][1])
    return moved

def rename_with_siblings(model_path, new_stem, db=None):
    return _relocate_db(model_path, os.path.dirname(model_path) or ".", new_stem, db)

def has_preview(model_path):
    stem = os.path.splitext(model_path)[0]
    return any(os.path.isfile(stem + ext) for ext in PREVIEW_EXTS)

def _key_for(url, api_key):
    return api_key if api_key and rt_civitai.is_civitai(url) else None

def grab_preview(model_path, images, api_key, cancel_flag=None):
    if has_preview(model_path):
        return False
    dest = os.path.splitext(model_path)[0] + ".preview.png"
    for url in rt_civitai.preview_candidates(images or []): ## first real image wins
        if _cancelled(cancel_flag):
            return False
        try:
            if rt_net.download_image(url, dest, api_key=_key_for(url, api_key), cancel_flag=cancel_flag):
                return True
        except Exception:
            continue
    return False

def _skin(model_path, rec, api_key, cancel_flag=None):
    if has_preview(model_path):
        return False
    if not rec.get("images") and (rec.get("version_id") or rec.get("model_id")): ## empty list, ask Civitai again
        try:
            rec["images"] = rt_civitai.fetch_images_fallback(rec, api_key) or []
        except Exception:
            rec["images"] = []
    return grab_preview(model_path, rec.get("images"), api_key, cancel_flag)

def _scrape_into(rec, tried):
    tried.append("hash(no baseModel)")
    try:
        scraped = rt_civitai.scrape_base_model(rec["model_id"])
    except Exception:
        scraped = None
    if scraped:
        rec["base_model"], rec["source"] = scraped, "scrape"
    else:
        tried.append("scrape(failed)")

def _by_hash(sha, api_key, tried):
    try:
        rec = rt_civitai.fetch_by_hash(sha, api_key)
    except Exception:
        rec = None
    if not rec:
        tried.append("hash(no match)")
        return None
    rec["source"], rec["sha256"] = "hash", sha
    if not rec.get("base_model") and rec.get("model_id"):
        _scrape_into(rec, tried)
    return rec

def resolve_model(model_path, api_key, db, cancel_flag=None):
    rec = read_sidecar(model_path)
    if rec and rec.get("base_model"):
        return rec
    tried = ["sidecar(no baseModel)" if rec else "sidecar(absent)"]
    sha = hash_for(db, model_path, cancel_flag)
    rec = _by_hash(sha, api_key, tried) if sha else None
    if rec is None or not rec.get("base_model"): ## quarantine, never guess from filename
        if sha:
            _register(db, model_path, "")[1]["miss"] = True
        hint = decamel(_stem(model_path))
        rt_config.log_unresolved(model_path, tried, f"not resolvable; filename hint only (NOT applied): {hint!r}")
        return None
    write_sidecar(model_path, rec)
    write_a1111_json(model_path, rec)
    _register(db, model_path, KIND_BY_TYPE.get((rec.get("type") or "").lower(), ""), rec)
    return rec

def label_and_skin(cfg, log=print, cancel_flag=None, relabel=False):
    if not rt_config.paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return None
    api_key, add_version = cfg.get("api_key") or None, cfg.get("add_version", True)
    exclusions, db = rt_config.load_exclusions(), load_library()
    counts = dict.fromkeys(("renamed", "previewed", "skipped_excluded", "skipped_formatted", "skipped_cached", "unresolved", "errors"), 0)
    _reset_unresolved()
    try:
        for folder, kind in rt_config.folders(cfg):
            if kind == "vae": ## VAEs keep their names
                continue
            log(f"\n{kind.upper()} in {folder}")
            for model_path in iter_model_files(folder):
                if _cancelled(cancel_flag):
                    log("Cancelled.")
                    return counts
                fname, stem = os.path.basename(model_path), _stem(model_path)
                if rt_config.normalize_stem(fname) in exclusions:
                    counts["skipped_excluded"] += 1
                    log(f"  skip (excluded): {fname}")
                    continue
                if already_formatted(stem) and not (relabel and add_version and not has_version(stem)):
                    counts["skipped_formatted"] += 1
                    continue
                if not os.path.isfile(sidecar_path(model_path)) and _cached_miss(db, model_path):
                    counts["skipped_cached"] += 1
                    log(f"  skip (cached miss): {fname}")
                    continue
                try:
                    rec = resolve_model(model_path, api_key, db, cancel_flag)
                except Exception as e:
                    counts["errors"] += 1
                    log(f"  ERROR {fname}: {e}")
                    continue
                if rec is None:
                    counts["unresolved"] += 1
                    log(f"  UNRESOLVED (quarantined): {fname}")
                    continue
                if not os.path.isfile(os.path.splitext(model_path)[0] + ".json"):
                    write_a1111_json(model_path, rec)
                if _skin(model_path, rec, api_key, cancel_flag):
                    counts["previewed"] += 1
                    log(f"  preview grabbed: {fname}")
                new_stem = build_new_stem(rt_civitai.tag_for_base_model(rec.get("base_model")), rec.get("model_name"), rec.get("version_name"), stem, add_version)
                if new_stem != stem and rename_with_siblings(model_path, new_stem, db):
                    counts["renamed"] += 1
                    log(f"  {rec.get('source', ''):>7} | {fname}  ->  {new_stem}")
    finally:
        save_library(db)
    _summary("SUMMARY", counts, log)
    log(f"\nUnresolved (if any): {rt_config.UNRESOLVED_PATH}")
    return counts

def is_vae(fname):
    stem = os.path.splitext(fname)[0].lower()
    return stem in KNOWN_VAE_NAMES or "vae" in stem

def is_lora(model_path):
    kind = KIND_BY_TYPE.get(((read_sidecar(model_path) or {}).get("type") or "").lower())
    if kind:
        return kind == "lora"
    size = file_key(model_path)[0]
    return size is not None and size < LORA_MAX_BYTES

def destination_for(model_path, cfg):
    if is_vae(os.path.basename(model_path)):
        return cfg["vae_dir"]
    return cfg["lora_dir"] if is_lora(model_path) else cfg["ckpt_dir"]

def sort_downloads(cfg, log=print, cancel_flag=None, db=None):
    if not rt_config.paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return 0
    if not os.path.isdir(rt_config.DOWNLOAD_DIR):
        log(f"No download dir: {rt_config.DOWNLOAD_DIR}")
        return 0
    own, moved = db is None, 0
    db = load_library() if own else db
    for model_path in iter_model_files(rt_config.DOWNLOAD_DIR):
        if _cancelled(cancel_flag):
            log("Sort cancelled.")
            break
        dest_dir = destination_for(model_path, cfg)
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as e:
            log(f"  ! cannot create {dest_dir}: {e}")
            continue
        if _relocate_db(model_path, dest_dir, _stem(model_path), db):
            moved += 1
            log(f"  sorted -> {os.path.basename(dest_dir)}: {os.path.basename(model_path)}")
        else:
            log(f"  ! not sorted (target exists or move failed): {os.path.basename(model_path)}")
    if own and moved: ## caller holds no db, persist here
        save_library(db)
    return moved

def find_duplicates(cfg, log=print, cancel_flag=None):
    if not rt_config.paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return []
    db, groups, seen = load_library(), {}, set()
    try:
        for folder, kind in rt_config.folders(cfg):
            log(f"Hashing {kind} in {folder}")
            for path in iter_model_files(folder):
                if _cancelled(cancel_flag):
                    log("Cancelled.")
                    return []
                if _norm(path) in seen: ## same folder configured twice
                    continue
                seen.add(_norm(path))
                sha = _register(db, path, kind, cancel_flag=cancel_flag)[0]
                if sha:
                    groups.setdefault(sha, []).append(path)
    finally:
        save_library(db)
    dupes = [g for g in groups.values() if len(g) > 1]
    log(f"{len(dupes)} duplicate group(s) found.")
    for g in dupes:
        log("  " + " | ".join(g))
    return dupes

def _update_one(sha, entry, cfg, db, counts, log, cancel_flag):
    api_key, tried = cfg.get("api_key") or None, []
    rec = _by_hash(sha, api_key, tried)
    if rec is None:
        entry["miss"] = True
        counts["unresolved"] += 1
        rt_config.log_unresolved(entry["path"], tried, "update database: by-hash returned nothing")
        return
    _fill(entry, rec)
    counts["resolved"] += 1
    for path in [entry["path"]] + [a["path"] for a in entry.get("alts", [])]: ## every copy gets sidecars
        write_sidecar(path, rec)
        write_a1111_json(path, rec)
        if _skin(path, rec, api_key, cancel_flag):
            counts["previewed"] += 1
        stem = _stem(path)
        new_stem = build_new_stem(entry.get("tag"), rec.get("model_name"), rec.get("version_name"), stem, True)
        if cfg.get("add_version", True) and new_stem != stem and rename_with_siblings(path, new_stem, db):
            counts["renamed"] += 1
            log(f"  renamed: {stem}  ->  {new_stem}")
    log(f"  resolved: {entry.get('model_name')} ({entry.get('version_name')})")

def update_database(cfg, log=print, cancel_flag=None):
    if not rt_config.paths_configured(cfg):
        log("Model folders are not configured. Open Configuration first.")
        return None
    exclusions, db = rt_config.load_exclusions(), load_library()
    counts = dict.fromkeys(("registered", "pruned", "queried", "resolved", "unresolved", "excluded", "previewed", "renamed", "errors"), 0)
    _reset_unresolved()
    try:
        for folder, kind in rt_config.folders(cfg):
            log(f"Registering {kind} in {folder}")
            for path in iter_model_files(folder):
                if _cancelled(cancel_flag):
                    log("Cancelled.")
                    return counts
                counts["registered" if _register(db, path, kind, cancel_flag=cancel_flag)[0] else "errors"] += 1
                if counts["registered"] % 50 == 0: ## first run hashes every file
                    log(f"  registered {counts['registered']} files...")
        counts["pruned"] = _prune(db)
        save_library(db)
        for sha, entry in [(s, e) for s, e in db["files"].items() if _is_orphan(e)]:
            if _cancelled(cancel_flag):
                log("Cancelled.")
                return counts
            if rt_config.normalize_stem(os.path.basename(entry["path"])) in exclusions:
                counts["excluded"] += 1
                continue
            counts["queried"] += 1
            try:
                _update_one(sha, entry, cfg, db, counts, log, cancel_flag)
            except Exception as e:
                counts["errors"] += 1
                log(f"  ERROR {entry.get('path')}: {e}")
            if counts["queried"] % 25 == 0: ## checkpoint progress on big libraries
                save_library(db)
    finally:
        save_library(db)
    _summary("UPDATE DATABASE", counts, log)
    return counts

def _manual_preview(model_path, source, api_key, log):
    dest, src = os.path.splitext(model_path)[0] + ".preview.png", source.strip().strip('"')
    try:
        if src.lower().startswith(("http://", "https://")): ## URL, validated as an image
            ok = rt_net.download_image(src, dest, api_key=_key_for(src, api_key))
            log("  preview downloaded from URL" if ok else f"  ! preview URL was not an image: {src}")
        elif os.path.isfile(src):
            shutil.copyfile(src, dest)
            log("  preview copied from local file")
        else:
            log(f"  ! preview source not a URL or existing file: {src}")
    except Exception as e:
        log(f"  ! preview failed: {e}")

def apply_manual_asset(model_path, rec, preview_source, cfg, log=print, make_exclusion=False):
    if not os.path.isfile(model_path):
        return False, f"Model file not found: {model_path}"
    rec, stem = dict(rec or {}), _stem(model_path)
    if not write_sidecar(model_path, rec) or not write_a1111_json(model_path, rec):
        log("  ! sidecar or a1111 json write failed")
    if preview_source:
        _manual_preview(model_path, preview_source, cfg.get("api_key"), log)
    new_stem = build_new_stem(rt_civitai.tag_for_base_model(rec.get("base_model")), rec.get("model_name"), rec.get("version_name"), stem, cfg.get("add_version", True))
    db = load_library()
    _register(db, model_path, _kind_of(model_path, cfg, rec), rec)
    moved = rename_with_siblings(model_path, new_stem, db) if new_stem != stem else []
    save_library(db)
    final = new_stem if moved else stem
    if make_exclusion and rt_config.add_to_exclusions(final):
        log(f"  added to exclusion list: {final}")
    return True, (f"Applied: {final}" if moved else f"Metadata applied; filename unchanged: {final}")
