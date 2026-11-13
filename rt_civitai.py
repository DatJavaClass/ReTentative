## Civitai API: URLs, metadata fetch, record normalize
import os, re, copy, urllib.error, urllib.parse
import rt_net

API = "https://civitai.com/api/v1"
MODEL_PAGE = "https://civitai.com/models/{}"
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv", ".gifv"}
BASE_TAGS = {"illustrious": "[Illustrious]", "pony": "[Pony]", "anima": "[Anima]", "noobai": "[NoobAI]",
             "sdxl 1.0": "[SDXL 1.0]", "sdxl": "[SDXL 1.0]", "stable diffusion xl": "[SDXL 1.0]",
             "sd 1.5": "[SD 1.5]", "stable diffusion 1.5": "[SD 1.5]", "flux.1 d": "[Flux]", "flux.1 s": "[Flux]"}
LOG = print ## callers may rebind to the GUI log
key_rejected, last_error = False, ""

def tag_for_base_model(base):
    if not base or not str(base).strip():
        return None
    return BASE_TAGS.get(base.strip().lower(), "[" + base.strip() + "]") ## unknown bases pass through

def is_civitai(url):
    return bool(url) and rt_net.is_civitai_host(url.strip())

def parse_url(url):
    if not is_civitai(url):
        return None, None
    url = url.strip()
    vid = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("modelVersionId", [""])[0]
    if vid.isdigit():
        return "version", int(vid)
    for kind, pat in (("version", r"/api/download/models/(\d+)"), ("version", r"/model-versions/(\d+)"), ("model", r"/models/(\d+)")):
        m = re.search(pat, url)
        if m:
            return kind, int(m.group(1))
    return None, None

def _get(path, api_key):
    global key_rejected, last_error
    key = None if key_rejected else api_key
    try:
        return rt_net.http_get_json(API + path, api_key=key)
    except urllib.error.HTTPError as e:
        e.close() ## release the error body
        if e.code in (401, 403) and key:
            key_rejected = True ## once per run, then keyless
            LOG("API key rejected, continuing without key")
            return _get(path, None)
        last_error = f"HTTP {e.code} on {path}"
    except Exception as e:
        last_error = f"{type(e).__name__}: {e} on {path}"
    return None

def _is_video(img):
    ext = os.path.splitext(urllib.parse.urlparse(img.get("url") or "").path)[1].lower()
    return (img.get("type") or "").lower() == "video" or ext in VIDEO_EXTS

def _images(raw_images):
    return [{"url": i["url"], "type": "video" if _is_video(i) else "image"} for i in raw_images or [] if isinstance(i, dict) and i.get("url")]

def normalize(version_json, model_json=None, source=""):
    v, m = copy.deepcopy(version_json or {}), model_json or {}
    if m:
        v["modelId"], v["model"] = v.get("modelId") or m.get("id"), v.get("model") or {}
        for key in ("name", "type"): ## fill gaps only, never overwrite
            v["model"][key] = v["model"].get(key) or m.get(key)
    model, files = v.get("model") or {}, v.get("files") or []
    primary = next((f for f in files if f.get("primary")), files[0] if files else {})
    tags = [str(t).strip() for t in v.get("trainedWords") or [] if str(t).strip()]
    return {"model_id": v.get("modelId"), "version_id": v.get("id"),
            "model_name": model.get("name"), "version_name": v.get("name"),
            "base_model": v.get("baseModel"), "type": model.get("type"),
            "tags": list(dict.fromkeys(tags)), "images": _images(v.get("images")),
            "download_url": primary.get("downloadUrl") or v.get("downloadUrl"), "file_name": primary.get("name"),
            "sha256": ((primary.get("hashes") or {}).get("SHA256") or "").lower() or None,
            "source": source, "raw": v}

def fetch_images_fallback(rec, api_key):
    vid, mid = rec.get("version_id"), rec.get("model_id")
    if vid:
        data = _get(f"/images?modelVersionId={vid}&limit=20", api_key)
        imgs = _images((data or {}).get("items"))
        if imgs:
            return imgs
    if mid:
        data = _get(f"/models/{mid}", api_key) or {}
        for ver in data.get("modelVersions") or []: ## exits on matching version
            if ver.get("id") == vid or not vid:
                return _images(ver.get("images"))
    return []

def _finish(rec, api_key):
    if rec and not rec["images"]:
        rec["images"] = fetch_images_fallback(rec, api_key)
        if rec["images"]:
            rec["raw"]["images"] = rec["images"] ## sidecar gets previews too
    return rec

def fetch_version(vid, api_key):
    data = _get(f"/model-versions/{vid}", api_key)
    return _finish(normalize(data, source="url"), api_key) if data else None

def fetch_model(mid, api_key):
    data = _get(f"/models/{mid}", api_key)
    versions = (data or {}).get("modelVersions") or []
    return _finish(normalize(versions[0], data, source="url"), api_key) if versions else None

def fetch_by_hash(sha256, api_key):
    data = _get(f"/model-versions/by-hash/{sha256}", api_key) if sha256 else None
    return _finish(normalize(data, source="hash"), api_key) if data else None

def fetch_by_url(url, api_key):
    kind, cid = parse_url(url)
    if kind == "version":
        return fetch_version(cid, api_key)
    return fetch_model(cid, api_key) if kind == "model" else None

def preview_candidates(images):
    stills, videos = [], []
    for img in images or []: ## images first, videos after
        img = img if isinstance(img, dict) else {"url": img}
        if img.get("url"):
            (videos if _is_video(img) else stills).append(img["url"])
    for url in videos:
        if "/width=" in url:
            stills.append(url.replace("/width=", "/anim=false,width=", 1)) ## Civitai still frame
        root, ext = os.path.splitext(url)
        if ext.lower() in VIDEO_EXTS:
            stills.append(root + ".jpeg")
    return list(dict.fromkeys(stills))

def scrape_base_model(model_id):
    global last_error
    if not model_id:
        return None
    try:
        with rt_net.OPENER.open(rt_net._request(MODEL_PAGE.format(model_id)), timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        last_error = f"scrape {type(e).__name__}: {e}"
        return None
    m = re.search(r'"baseModel"\s*:\s*"([^"]+)"', html)
    return m.group(1) if m else None
