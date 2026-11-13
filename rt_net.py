## HTTP opener, JSON GET, downloads, hashing
import os, re, json, time, hashlib, http.client, urllib.error, urllib.parse, urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
CHUNK, IMAGE_CAP = 256 * 1024, 40 * 1024 * 1024
BAD_CHARS = '<>:"/|?*' + chr(92) ## chr(92) is backslash
CIVITAI_HOSTS = ("civitai.com", "civitai.red", "civitai.green")
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM")

def is_civitai_host(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in CIVITAI_HOSTS) ## .red and .green mirrors too

class RedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not is_civitai_host(newurl):
            new.remove_header("Authorization") ## presigned CDN rejects double auth
        return new

OPENER = urllib.request.build_opener(RedirectHandler())

def _request(url, api_key=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    if api_key and is_civitai_host(url):
        req.add_header("Authorization", f"Bearer {api_key}") ## key never leaves civitai.com
    return req

def http_get_json(url, api_key=None, timeout=15, retries=3):
    req, attempts = _request(url, api_key), max(1, retries)
    for attempt in range(attempts): ## exits on success or final raise
        try:
            with OPENER.open(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == attempts - 1:
                raise ## 401 403 404 fail fast
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if attempt == attempts - 1:
                raise
        time.sleep(min(2.0, 0.5 * (attempt + 1))) ## short backoff, bounded total

def _cd_name(cd):
    m = re.search(r"filename\*\s*=\s*([^']*)''([^;]+)", cd or "", re.I)
    if m:
        return urllib.parse.unquote(m.group(2).strip().strip('"'), encoding=m.group(1) or "utf-8")
    m = re.search(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;]+)', cd or "", re.I)
    return (m.group(1) or m.group(2)).strip() if m else ""

def filename_from_response(resp, url):
    name = _cd_name(resp.headers.get("Content-Disposition", "")) if resp is not None else ""
    name = name or urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
    name = "".join(c for c in name if c not in BAD_CHARS and ord(c) > 31).strip().strip(".")
    return name or "download"

def http_download(url, dest, api_key=None, timeout=120, cancel_flag=None, progress=None):
    tmp = None
    try:
        with OPENER.open(_request(url, api_key), timeout=timeout) as resp:
            final = os.path.join(dest, filename_from_response(resp, url)) if os.path.isdir(dest) else dest
            total, done, tmp = int(resp.headers.get("Content-Length") or 0), 0, final + ".part"
            with open(tmp, "wb") as out:
                while chunk := resp.read(CHUNK): ## exits at EOF or cancel
                    if cancel_flag is not None and cancel_flag.is_set():
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done / total if total else -(done + 1))
            if cancel_flag is not None and cancel_flag.is_set():
                os.remove(tmp)
                return False, None
            os.replace(tmp, final)
            return True, final
    except BaseException:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass ## locked part file, leave it
        raise ## caller logs the reason

def looks_like_image(head):
    return (head[:4] == b"RIFF" and head[8:12] == b"WEBP") or any(head.startswith(s) for s in IMAGE_MAGIC)

def download_image(url, dest, api_key=None, cancel_flag=None):
    try:
        with OPENER.open(_request(url, api_key), timeout=30) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype.startswith(("text/", "video/", "application/json")):
                return False ## html error page or video
            data = resp.read(IMAGE_CAP)
        if (cancel_flag is not None and cancel_flag.is_set()) or not looks_like_image(data[:16]):
            return False
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
        return True
    except (OSError, ValueError, http.client.HTTPException):
        return False

def sha256_file(path, cancel_flag=None):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""): ## exits at EOF or cancel
                if cancel_flag is not None and cancel_flag.is_set():
                    return None
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()
