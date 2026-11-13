## File Name: rt_download.py
## Creator: DatJavaClass
## Associated Files: rt_turbo.py, rt_config.py, rt_library.py, rt_civitai.py, rt_gui.py
## Exposed Vars: KINDS
## Exposed Funcs: entry_from_url, load_queue, save_queue, import_file, describe, run_queue
## Desc: Download queue entries and their persistence, plus the parallel queue run: resolve, segmented fetch through rt_turbo, sha256 check, register, sidecars, previews, then sort and label. Also export_queue.
import concurrent.futures
import json
import os
import threading

import rt_civitai
import rt_config
import rt_library
import rt_net
import rt_turbo

KINDS = {"checkpoint": "Checkpoint", "lora": "LORA", "vae": "VAE"}
TICK = 0.25 ## seconds between progress snapshots

def _tag_list(tags):
    if isinstance(tags, str):
        tags = tags.split(",")
    return [str(t).strip() for t in (tags or []) if str(t).strip()]

def entry_from_url(url, image_url=None, tags=None, name=None, base_model=None, kind=None):
    kind = (kind or "").strip().lower() or None
    return {"url": (url or "").strip(), "image_url": (image_url or "").strip() or None,
            "tags": _tag_list(tags), "name": (name or "").strip() or None,
            "base_model": (base_model or "").strip() or None,
            "kind": kind if kind in KINDS else None}

def _clean(item):
    if isinstance(item, str):
        return entry_from_url(item)
    if isinstance(item, dict) and item.get("url"):
        return entry_from_url(item.get("url"), item.get("image_url"), item.get("tags"),
                              item.get("name"), item.get("base_model"), item.get("kind"))
    return None

def _clean_all(items):
    return [e for e in (_clean(i) for i in items) if e and e["url"]]

def _read_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [entry_from_url(ln) for ln in f if ln.strip().lower().startswith("http")]
    except Exception:
        return []

def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _clean_all(data if isinstance(data, list) else [])
    except Exception:
        return []

def _write(path, entries, as_text=False):
    try:
        with open(path, "w", encoding="utf-8") as f:
            if as_text:
                f.writelines(e["url"] + "\n" for e in entries)
            else:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def load_queue():
    if os.path.isfile(rt_config.QUEUE_PATH):
        return _read_json(rt_config.QUEUE_PATH)
    return _read_lines(rt_config.LEGACY_QUEUE_PATH) if os.path.isfile(rt_config.LEGACY_QUEUE_PATH) else []

def save_queue(entries):
    return _write(rt_config.QUEUE_PATH, _clean_all(entries))

def export_queue(entries, path):
    return _write(path, _clean_all(entries), path.lower().endswith(".txt"))

def import_file(path):
    return _read_lines(path) if path.lower().endswith(".txt") else _read_json(path)

def describe(entry):
    bits = [entry.get("name") or entry["url"]]
    if entry.get("kind"):
        bits.append(entry["kind"])
    if entry.get("base_model"):
        bits.append(f"[{entry['base_model']}]")
    if entry.get("tags"):
        bits.append(f"{len(entry['tags'])} tags")
    if entry.get("name"):
        bits.append(entry["url"])
    return "  ".join(bits)

def _manual_rec(entry, url):
    images = [{"url": entry["image_url"], "type": "image"}] if entry.get("image_url") else []
    mtype = KINDS.get(entry.get("kind") or "")
    raw = {"name": None, "baseModel": entry.get("base_model"), "trainedWords": entry["tags"],
           "model": {"name": entry.get("name"), "type": mtype}, "images": images, "downloadUrl": url}
    return {"model_id": None, "version_id": None, "model_name": entry.get("name"),
            "version_name": None, "base_model": entry.get("base_model"), "type": mtype,
            "tags": list(entry["tags"]), "images": images, "download_url": url,
            "file_name": None, "sha256": None, "source": "manual", "raw": raw}

def _kind_of(entry, rec):
    if entry.get("kind"):
        return entry["kind"]
    return {"LORA": "lora", "LOCON": "lora", "VAE": "vae"}.get(str(rec.get("type") or "").upper(), "checkpoint")

class _Run:
    """State one queue run shares across workers."""
    def __init__(self, cfg, log, cancel_flag):
        self.cfg, self.log, self.flag = cfg, log, cancel_flag
        self.api_key = cfg.get("api_key") or None
        self.conns, self.files = rt_config.clamp_speed(cfg)
        self.meter, self.lock, self.db = rt_turbo.Meter(), threading.Lock(), rt_library.load_library()

    def cancelled(self):
        return self.flag is not None and self.flag.is_set()

    def work(self, entry):
        """One entry start to finish; never raises."""
        if self.cancelled(): ## cancel means no new entries
            return None

        try:
            got = _resolve(entry, self)
        except Exception as e:
            self.log(f"  ! resolve failed for {entry['url']}: {e}")
            return None

        try:
            path = _fetch_one(entry, *got, self) if got else None
            path = _finish(path, got[0], entry, self, got[3]) if path else None
        except Exception as e:
            self.log(f"  ! {entry['url']}: {e}")
            return None
        return {"entry": entry, "path": path} if path else None

def _resolve(entry, run):
    """Returns (rec, download_url, dest, key) or None."""
    url, dl_dir = entry["url"], rt_config.DOWNLOAD_DIR
    if not rt_civitai.is_civitai(url):
        return _manual_rec(entry, url), url, dl_dir, None

    rec = rt_civitai.fetch_by_url(url, run.api_key)
    if not rec or not rec.get("download_url"):
        run.log(f"  ! no downloadable file for {url}")
        with run.lock:
            rt_config.log_unresolved(url, "url", "fetch_by_url returned nothing")
        return None

    if entry.get("image_url"): ## fetch_by_url already ran the fallback
        rec["images"] = [{"url": entry["image_url"], "type": "image"}] + (rec.get("images") or [])
    rec["tags"] = rec.get("tags") or list(entry["tags"])
    rec["base_model"] = rec.get("base_model") or entry.get("base_model")
    fname = os.path.basename(rec.get("file_name") or "")
    return rec, rec["download_url"], os.path.join(dl_dir, fname) if fname else dl_dir, run.api_key

def _manual_rename(path, rec, cfg):
    folder, fname = os.path.split(path)
    stem, ext = os.path.splitext(fname)
    new_stem = rt_library.build_new_stem(rt_civitai.tag_for_base_model(rec["base_model"]),
                                          rec["model_name"], None, stem, cfg.get("add_version", True))
    if new_stem == stem:
        return path
    rt_library.rename_with_siblings(path, new_stem)
    new_path = os.path.join(folder, new_stem + ext)
    return new_path if os.path.isfile(new_path) else path

def _quarantine(path, expect, got, log):
    log(f"  ! sha256 mismatch: {os.path.basename(path)} expected {expect[:12]}, got {got[:12]}")
    try:
        os.replace(path, path + ".corrupt")
        log(f"  kept as {os.path.basename(path)}.corrupt, entry stays queued")
    except OSError as e:
        log(f"  ! could not mark corrupt file: {e}")
    return None

def _finish(path, rec, entry, run, api_key):
    """Returns the final path, or None when the file failed its check."""
    expect, got = (rec.get("sha256") or "").lower(), rt_net.sha256_file(path, run.flag)
    if got is None and run.cancelled(): ## next run finds it already downloaded
        return None
    if expect and got and got != expect: ## Civitai says this file is damaged
        return _quarantine(path, expect, got, run.log)
    rec["sha256"] = got or expect or None
    civitai = rec.get("source") != "manual"

    with run.lock: ## one thread owns db and sidecars
        if not civitai and rec.get("model_name") and rec.get("base_model"): ## named manual entry gets labelled
            path = _manual_rename(path, rec, run.cfg)
        try:
            rt_library.register(run.db, path, _kind_of(entry, rec), rec, got)
            rt_library.write_sidecar(path, rec)
            rt_library.write_a1111_json(path, rec)
        except Exception as e:
            run.log(f"  ! metadata write failed: {e}")

    try:
        if rec.get("images") and not rt_library.has_preview(path) and rt_library.grab_preview(path, rec["images"], api_key, run.flag):
            run.log("  preview grabbed")
    except Exception as e:
        run.log(f"  ! preview failed: {e}")

    with run.lock:
        if not civitai and rt_config.add_to_exclusions(os.path.splitext(os.path.basename(path))[0]):
            run.log("  added to exclusion list")
    return path

def _size(path):
    try:
        return f" ({os.path.getsize(path) / 1048576:.1f} MB)"
    except OSError:
        return ""

def _fetch_one(entry, rec, dl_url, dest, key, run):
    """Returns the downloaded path or None."""
    if os.path.isfile(dest):
        run.log(f"  already downloaded: {os.path.basename(dest)}")
        return dest

    try:
        ok, path = rt_turbo.fetch(dl_url, dest, api_key=key, connections=run.conns, cancel_flag=run.flag,
                                  meter=run.meter, key=entry["url"], log=run.log)
    except Exception as e:
        _meter_failed(run.meter, entry["url"])
        run.log(f"  ! download error: {entry['url']}: {e}")
        return None

    if not ok or not path:
        state = "cancelled" if run.cancelled() else "! download failed"
        run.log(f"  {state}: {entry['url']}")
        return None
    run.log(f"  done: {os.path.basename(path)}{_size(path)}")
    return path

def _meter_failed(meter, key):
    try:
        meter.finish(key, "failed")
    except Exception:
        pass ## engine died before it registered key

def _report(meter, progress):
    try:
        progress(meter.snapshot())
    except Exception:
        pass ## a display fault never stops downloads

def _tick(meter, progress, stop):
    while not stop.wait(TICK): ## exits when the run ends
        _report(meter, progress)
    _report(meter, progress) ## final snapshot after the drain

def run_queue(entries, cfg, log=print, cancel_flag=None, progress=None, on_done=None):
    """Download every entry in parallel, then sort and label."""
    finished, stop, ticker = [], threading.Event(), None
    try:
        os.makedirs(rt_config.DOWNLOAD_DIR, exist_ok=True)
        run, todo = _Run(cfg, log, cancel_flag), _clean_all(entries)
        log(f"Starting {len(todo)} download(s), {run.files} at once, {run.conns} connection(s) each...")
        if progress:
            ticker = threading.Thread(target=_tick, args=(run.meter, progress, stop), name="rt_tick", daemon=True)
            ticker.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=run.files, thread_name_prefix="rt_dl") as pool:
            finished = [r for r in pool.map(run.work, todo) if r] ## map keeps queue order
        rt_library.save_library(run.db)

        if run.cancelled(): ## pool drained, workers stopped on the flag
            log("Cancelled by user. Partial files are kept for resume.")
        elif finished: ## sort and label only a clean finish
            log("\nSorting downloads into model folders...")
            rt_library.sort_downloads(cfg, log=log, cancel_flag=cancel_flag)
            log("\nLabel & Skin pass on new files...")
            rt_library.label_and_skin(cfg, log=log, cancel_flag=cancel_flag)
            log("\nAll done.")
    except Exception as e:
        log(f"  ! queue run failed: {e}")
    finally:
        stop.set()
        if ticker:
            ticker.join()
        if on_done:
            on_done(finished)
    return finished
