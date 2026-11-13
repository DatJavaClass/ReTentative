## File Name: rt_turbo.py
## Creator: DatJavaClass
## Associated Files: rt_net.py, rt_download.py
## Exposed Vars: PIECE, MAX_CONNECTIONS
## Exposed Funcs: probe, fetch, Meter
## Desc: Segmented multi connection downloader with resume.
import io, os, re, ssl, json, time, heapq, threading, collections, http.client, urllib.error, urllib.parse
from types import SimpleNamespace

import rt_net

try:
    import ctypes, msvcrt
except ImportError:
    ctypes = msvcrt = None ## not Windows, no sparse marking

PIECE = 16 * 1024 * 1024 ## bytes per piece
READ = 1024 * 1024 ## socket read size
MAX_CONNECTIONS = 16
RETRIES = 6 ## per piece
HOPS, TIMEOUT, WINDOW, SAVE_EVERY, FSCTL_SET_SPARSE = 8, 60, 3.0, 1.0, 0x900C4
REDIRECTS = (301, 302, 303, 307, 308)
RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)
SAFE = "/?&=%:;@!$'()*+,~[]" ## keep existing escapes and query marks
CTX = ssl.create_default_context()

class Meter:
    def __init__(self):
        self.lock, self.files = threading.Lock(), {}

    def credit(self, key, n):
        with self.lock:
            if key in self.files:
                self.files[key]["done"] += n ## resumed bytes, not speed

    def start(self, key, name, total, done=0, state="active"):
        with self.lock:
            self.files[key] ={"key": key, "name": name, "done": done, "total": total or 0, "state": state, "since": time.monotonic(), "ticks": collections.deque()}

    def add(self, key, n):
        now = time.monotonic()
        with self.lock:
            f = self.files.get(key)
            if f is None:
                return ## unregistered key, nothing to count
            if f["state"] == "queued":
                f["state"], f["since"] = "active", now
            f["done"] += n
            f["ticks"].append((now, n))
            self._trim(f, now)

    def retitle(self, key, name, total):
        with self.lock:
            f = self.files.get(key)
            if f is not None:
                f["name"], f["total"] = name or f["name"], total or f["total"]

    def finish(self, key, state):
        with self.lock:
            f = self.files.get(key)
            if f is not None:
                f["state"] = state
                f["ticks"].clear()

    def _trim(self, f, now):
        while f["ticks"] and f["ticks"][0][0] < now - WINDOW: ## exits once the window is current
            f["ticks"].popleft()

    def _speed(self, f, now):
        if f["state"] != "active":
            return 0.0
        self._trim(f, now)
        return max(0.0, sum(n for _, n in f["ticks"]) / max(0.5, min(WINDOW, now - f["since"])))

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            rows = [{"key": f["key"], "name": f["name"], "done": f["done"], "total": f["total"], "speed": self._speed(f, now), "state": f["state"]} for f in self.files.values()]
        speed, live = sum(r["speed"] for r in rows), [r for r in rows if r["state"] in ("queued", "active")]
        unknown, left = any(r["state"] == "active" and not r["total"] for r in rows), sum(max(0, r["total"] - r["done"]) for r in live)
        return {"done": sum(r["done"] for r in rows), "total": sum(r["total"] for r in rows), "speed": speed,
                "eta": None if unknown or speed <= 0 else left / speed, "active": sum(r["state"] == "active" for r in rows), "files": rows}

class _Halt(Exception): pass ## cancel or stop seen mid piece

def _halted(flag):
    return flag is not None and flag.is_set()

def _clean(name):
    return "".join(c for c in str(name) if c not in rt_net.BAD_CHARS and ord(c) > 31).strip().strip(".")

def _target(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.quote((p.path or "/") + ("?" + p.query if p.query else ""), safe=SAFE)

def _api_host(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower().removeprefix("www.")
    return host in rt_net.CIVITAI_HOSTS ## b2.civitai.com storage refuses bearer

def _headers(url, api_key, rng=None):
    h = {"User-Agent": rt_net.UA, "Accept": "*/*"}
    if rng:
        h["Range"] = rng
    if api_key and _api_host(url):
        h["Authorization"] = f"Bearer {api_key}" ## key never leaves civitai hosts
    return h

def _connect(url, timeout):
    p = urllib.parse.urlsplit(url)
    match p.scheme.lower():
        case "https": ## entry tls, exit pooled keep alive
            return http.client.HTTPSConnection(p.hostname, p.port, timeout=timeout, context=CTX)
        case "http":
            return http.client.HTTPConnection(p.hostname, p.port, timeout=timeout)
        case _:
            raise ValueError(f"unsupported scheme in {url}")

def _close(conn):
    if conn is not None:
        conn.close()
    return None

def _drop(path):
    try:
        os.remove(path)
    except OSError:
        pass ## gone already or locked, leave it

def _http_error(url, resp):
    return urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, io.BytesIO())

def _retry_after(err):
    v = (err.headers.get("Retry-After") or "").strip() if err.headers is not None else ""
    return min(60.0, float(v)) if v.isdigit() else 0.0

def _write(f, mv):
    while mv: ## exits when every byte is down
        mv = mv[f.write(mv):]

def _sparse(f):
    try:
        ret = ctypes.c_ulong()
        ctypes.windll.kernel32.DeviceIoControl(ctypes.c_void_p(msvcrt.get_osfhandle(f.fileno())), FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(ret), None)
    except (AttributeError, OSError):
        pass ## not Windows or not NTFS, plain file

def probe(url, api_key=None, timeout=30):
    cur, cd = url, ""
    for _ in range(HOPS + 1): ## exits on a final answer or raise
        conn = _connect(cur, timeout)
        try:
            conn.request("GET", _target(cur), headers=_headers(cur, api_key, "bytes=0-0"))
            resp = conn.getresponse()
            hop_cd, loc = resp.getheader("Content-Disposition") or "", resp.getheader("Location")
            cd = hop_cd if rt_net._cd_name(hop_cd) else cd ## last hop with a name wins
            if resp.status in REDIRECTS and loc:
                cur = urllib.parse.urljoin(cur, loc)
                continue
            if resp.status >= 300:
                raise _http_error(cur, resp)
            m = RANGE_RE.fullmatch((resp.getheader("Content-Range") or "").strip())
            ranged = resp.status == 206 and m is not None and m[1] == "0" and m[2] == "0" and m[3].isdigit()
            total = int(m[3]) if ranged else 0 if resp.status == 206 else int(resp.getheader("Content-Length") or 0)
            name = rt_net.filename_from_response(SimpleNamespace(headers={"Content-Disposition": cd}), cur)
            return {"url": cur, "total": total, "ranged": ranged, "name": name, "etag": resp.getheader("ETag") or ""}
        finally:
            conn.close() ## body never read, socket dropped
    raise urllib.error.HTTPError(cur, 310, "too many redirects", None, io.BytesIO())

class _Job:
    def __init__(self, url, info, final, api_key, cancel_flag, meter, key, say):
        self.url, self.api_key, self.cancel, self.meter, self.key, self.say = url, api_key, cancel_flag, meter, key, say
        self.total, self.etag, self.piece, self.part, self.state = info["total"], info["etag"], PIECE, final + ".part", final + ".part.json"
        self.pieces, self.where = -(-self.total // self.piece), (0, info["url"]) ## where is (generation, address)
        self.lock, self.state_lock, self.probe_lock, self.stop = threading.Lock(), threading.Lock(), threading.Lock(), threading.Event()
        self.tries, self.running, self.error, self.seen206, self.saved = {}, 0, None, True, 0.0
        self.done = self._resume(os.path.basename(final))
        self.free = [i for i in range(self.pieces) if i not in self.done] ## ascending list is already a heap

    def _resume(self, name):
        try:
            with open(self.state, "r", encoding="utf-8") as f:
                st = json.load(f)
            if (st.get("total"), st.get("etag"), st.get("piece")) == (self.total, self.etag, self.piece) and os.path.getsize(self.part) == self.total:
                done = {i for i in st.get("done", []) if type(i) is int and 0 <= i < self.pieces}
                self.say(f"resuming {name}: {len(done)} of {self.pieces} pieces kept")
                return done
        except (OSError, ValueError, AttributeError, TypeError):
            pass ## no state or bad state, start clean
        if os.path.exists(self.state):
            os.remove(self.state) ## stale state never outlives a clean part
        with open(self.part, "wb") as f:
            _sparse(f)
            f.truncate(self.total)
        return set()

    def have(self):
        return sum(min(self.total, (i + 1) * self.piece) - i * self.piece for i in self.done)

    def halted(self):
        return self.stop.is_set() or _halted(self.cancel)

    def take(self):
        with self.lock:
            if self.halted() or not self.free:
                self.running -= 1
                return None
            return heapq.heappop(self.free)

    def give_back(self, i):
        with self.lock:
            heapq.heappush(self.free, i)

    def retire(self, i):
        with self.lock:
            if self.running <= 1:
                return False ## last worker never retires
            heapq.heappush(self.free, i)
            self.running -= 1
            return True

    def complete(self, i):
        with self.lock:
            self.done.add(i)
        self.save()

    def fail(self, err):
        with self.lock:
            self.error = self.error or err
        self.stop.set()

    def miss(self, i, err, wait=0.0):
        with self.lock:
            tries = self.tries[i] = self.tries.get(i, 0) + 1
        if tries >= RETRIES:
            return self.fail(err) ## piece is out of chances
        _nap(self, max(wait, min(8.0, 0.25 * 2 ** (tries - 1))))
        self.give_back(i)

    def refresh(self, used):
        with self.probe_lock:
            if self.where[0] != used:
                return False ## another worker already re probed
            counted = not self.seen206 ## fresh address failed with no success
            try:
                info = probe(self.url, self.api_key)
            except (OSError, http.client.HTTPException):
                return True
            if info["total"] != self.total or not info["ranged"] or (self.etag and info["etag"] and info["etag"] != self.etag):
                self.fail(ValueError(f"file changed on server during download: {self.url}"))
                return True
            self.where, self.seen206 = (used + 1, info["url"]), False
        self.say("download address expired, re probed")
        return counted

    def save(self, force=False):
        if not self.state_lock.acquire(blocking=force):
            return ## another worker is saving
        try:
            now = time.monotonic()
            if not force and now - self.saved < SAVE_EVERY:
                return
            self.saved = now
            with self.lock:
                done = sorted(self.done)
            with open(self.state + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"url": self.url, "total": self.total, "etag": self.etag, "piece": self.piece, "done": done}, f)
            os.replace(self.state + ".tmp", self.state)
        except OSError:
            pass ## locked by a scanner, next save retries
        finally:
            self.state_lock.release()

def _nap(job, secs):
    end = time.monotonic() + secs
    while not job.halted() and (left := end - time.monotonic()) > 0: ## exits on time up or halt
        job.stop.wait(min(0.2, left))

def _piece(job, conn, addr, f, view, i):
    a, b = i * job.piece, min(job.total, (i + 1) * job.piece) - 1
    size, got = b - a + 1, 0
    try:
        conn.request("GET", _target(addr), headers=_headers(addr, job.api_key, f"bytes={a}-{b}"))
        resp = conn.getresponse()
        if resp.status != 206:
            raise _http_error(addr, resp)
        cr = (resp.getheader("Content-Range") or "").strip()
        m = RANGE_RE.fullmatch(cr)
        if m is None or (int(m[1]), int(m[2]), m[3]) != (a, b, str(job.total)) or int(resp.getheader("Content-Length") or size) != size:
            raise ConnectionError(f"piece {i}: wanted bytes {a}-{b}/{job.total}, got {cr!r}")
        job.seen206 = True
        f.seek(a)
        while got < size: ## exits when the piece is whole
            if job.halted():
                raise _Halt()
            n = resp.readinto(view[:min(len(view), size - got)])
            if not n:
                raise ConnectionError(f"piece {i}: cut off at {got} of {size} bytes")
            _write(f, view[:n])
            got += n
            job.meter.add(job.key, n)
        if resp.read(1):
            raise ConnectionError(f"piece {i}: body longer than its range")
    except BaseException:
        if got:
            job.meter.add(job.key, -got) ## roll back the partial piece
        raise
    job.complete(i)

def _setback(job, i, err, gen):
    match getattr(err, "code", 0):
        case 401 | 403: ## entry expired address, exit after re probe
            if not job.refresh(gen):
                job.give_back(i)
                return True
            job.miss(i, err)
        case 429 | 503:
            if job.retire(i):
                job.say("server asked for fewer connections, one retired")
                return False
            job.miss(i, err, _retry_after(err))
        case _:
            job.miss(i, err)
    return True

def _worker(job):
    conn, gen, addr, view = None, -1, "", memoryview(bytearray(READ))
    try:
        with open(job.part, "r+b", buffering=0) as f:
            while (i := job.take()) is not None: ## exits when the pool drains or halts
                try:
                    if conn is None or gen != job.where[0]:
                        _close(conn)
                        gen, addr = job.where
                        conn = _connect(addr, TIMEOUT)
                    _piece(job, conn, addr, f, view, i)
                except _Halt:
                    conn = _close(conn)
                    job.give_back(i)
                except (OSError, http.client.HTTPException) as e:
                    conn = _close(conn) ## reconnect fresh after any failure
                    if not _setback(job, i, e, gen):
                        return
    except Exception as e:
        job.fail(e) ## unexpected, stop the whole file
    finally:
        _close(conn)

def _segmented(url, info, final, api_key, connections, cancel_flag, meter, key, say):
    job = _Job(url, info, final, api_key, cancel_flag, meter, key, say)
    if job.done:
        meter.credit(key, job.have()) ## resumed bytes count toward percent
    count = min(max(1, int(connections or 1)), MAX_CONNECTIONS, len(job.free))
    workers = [threading.Thread(target=_worker, args=(job,), name=f"turbo-{n}", daemon=True) for n in range(count)]
    job.running = count
    try:
        for w in workers: ## exits once every worker runs
            w.start()
        for w in workers: ## exits when the pool drains or halts
            w.join()
    finally:
        job.stop.set()
        job.save(True)
    if _halted(cancel_flag):
        return False ## part and state stay for resume
    if job.error is not None:
        raise job.error
    if len(job.done) != job.pieces or os.path.getsize(job.part) != job.total:
        raise ConnectionError(f"incomplete: {len(job.done)} of {job.pieces} pieces, {os.path.getsize(job.part)} of {job.total} bytes")
    os.replace(job.part, final)
    _drop(job.state)
    return True

def _single(info, final, api_key, cancel_flag, meter, key):
    part, view, done = final + ".part", memoryview(bytearray(READ)), 0
    conn = _connect(info["url"], TIMEOUT)
    try:
        conn.request("GET", _target(info["url"]), headers=_headers(info["url"], api_key))
        resp = conn.getresponse()
        if resp.status != 200:
            raise _http_error(info["url"], resp)
        length = int(resp.getheader("Content-Length") or 0)
        meter.retitle(key, None, length)
        with open(part, "wb", buffering=0) as out:
            while not _halted(cancel_flag) and (n := resp.readinto(view)): ## exits at EOF or cancel
                _write(out, view[:n])
                done += n
                meter.add(key, n)
        if _halted(cancel_flag):
            _drop(part) ## no ranges, nothing to resume
            return False
        if length and done != length:
            raise ConnectionError(f"short body: {done} of {length} bytes")
        os.replace(part, final)
        return True
    except BaseException:
        _drop(part)
        raise ## caller logs the reason
    finally:
        conn.close()

def fetch(url, dest, api_key=None, connections=8, cancel_flag=None, meter=None, key=None, log=None, name_hint=None):
    meter, key, say, hint = meter or Meter(), key or url, log or (lambda msg: None), _clean(name_hint or "")
    try:
        info = probe(url, api_key)
        final = os.path.join(dest, hint or info["name"]) if os.path.isdir(dest) else dest
        meter.start(key, os.path.basename(final), info["total"])
        if _halted(cancel_flag):
            ok = False
        elif info["ranged"] and info["total"] >= 2 * PIECE:
            ok = _segmented(url, info, final, api_key, connections, cancel_flag, meter, key, say)
        else:
            ok = _single(info, final, api_key, cancel_flag, meter, key)
    except BaseException:
        meter.finish(key, "failed")
        raise ## caller logs the reason
    meter.finish(key, "done" if ok else "cancelled")
    return (True, final) if ok else (False, None)
