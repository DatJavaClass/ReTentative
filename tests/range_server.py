## File Name: range_server.py
## Creator: DatJavaClass
## Associated Files: test_turbo.py, bench.py
## Exposed Vars: None
## Exposed Funcs: RangeServer
## Desc: Local ranged HTTP fixture with faults.
import http.server, re, socket, threading, time, urllib.parse

class RangeServer:
    def __init__(self, files, rate=0, ranged=True, redirect=False, fail_every=0, expire_after=0):
        self.files, self.rate, self.ranged = dict(files), rate, ranged
        self.redirect, self.fail_every, self.expire_after = redirect, fail_every, expire_after
        self.hits, self.peak_connections, self.bytes_sent = 0, 0, 0
        self.auth_seen, self.ranges_seen = [], []
        self._active, self._ranged_hits, self._serial = 0, 0, 0
        self._signatures, self._lock, self._stopping = {}, threading.Lock(), threading.Event()
        self._server, self._thread = None, None

    def start(self):
        if self._server is not None: ## Running server returns existing address.
            return self.url

        fixture = self
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(1)

            def log_message(self, format, *args):
                pass

            def do_GET(self):
                try:
                    fixture._serve(self)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    self.close_connection = True

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._stopping.clear()
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        return self.url

    def stop(self):
        if self._server is None: ## Stopped server needs no cleanup.
            return

        self._stopping.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        self._server, self._thread = None, None

    def _headers(self, handler, status, length=0, headers=None):
        handler.send_response(status)
        handler.send_header("Content-Length", str(length))
        for name, value in (headers or {}).items(): ## Emit headers until exhausted.
            handler.send_header(name, value)
        handler.end_headers()

    def _serve(self, handler):
        parsed = urllib.parse.urlsplit(handler.path)
        route, _, name = parsed.path.lstrip("/").partition("/")
        name = urllib.parse.unquote(name)
        with self._lock:
            self.hits += 1
            self.auth_seen.append((handler.path, handler.headers.get("Authorization")))

        if name not in self.files or route not in ("f", "dl"): ## Unknown route exits with 404.
            self._headers(handler, 404)
            return
        if route == "dl": ## Redirect issues a fresh signature.
            if not self.redirect: ## Disabled redirects exit with 404.
                self._headers(handler, 404)
                return
            with self._lock:
                self._serial += 1
                signature = str(self._serial)
                self._signatures[signature] = 0
            location = f"/f/{urllib.parse.quote(name, safe='')}?sig={signature}"
            disposition = "attachment; filename*=UTF-8''" + urllib.parse.quote(name, safe="")
            self._headers(handler, 302, headers={"Location": location, "Content-Disposition": disposition})
            return

        signature = urllib.parse.parse_qs(parsed.query).get("sig", [None])[0]
        with self._lock:
            expired = self.expire_after and signature is not None and (signature not in self._signatures or self._signatures[signature] >= self.expire_after)
            if signature in self._signatures and not expired: ## Valid signature consumes one request.
                self._signatures[signature] += 1
        if expired: ## Exhausted signature exits with 403.
            self._headers(handler, 403)
            return

        data, status, headers = self.files[name], 200, {"ETag": '"range-fixture-v1"'}
        start, end, cut = 0, len(data) - 1, False
        requested = handler.headers.get("Range")
        if self.ranged and requested: ## Valid ranges select partial responses.
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()): ## Malformed ranges exit with 416.
                self._headers(handler, 416, headers={"Content-Range": f"bytes */{len(data)}"})
                return
            left, right = match.groups()
            start = int(left) if left else max(0, len(data) - int(right))
            end = min(end, int(right)) if left and right else end
            if start > end or start >= len(data): ## Unsatisfiable ranges exit with 416.
                self._headers(handler, 416, headers={"Content-Range": f"bytes */{len(data)}"})
                return
            status = 206
            headers.update({"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"})
            with self._lock:
                self._ranged_hits += 1
                self.ranges_seen.append((start, end))
                cut = bool(self.fail_every and self._ranged_hits % self.fail_every == 0)

        length, sent, began = end - start + 1, 0, time.monotonic()
        limit = length // 2 if cut else length
        with self._lock:
            self._active += 1
            self.peak_connections = max(self.peak_connections, self._active)
        try:
            self._headers(handler, status, length, headers)
            while sent < limit and not self._stopping.is_set(): ## Send chunks until complete or stopped.
                size = min(4096, limit - sent)
                if self.rate: ## Rate cap waits before each write.
                    delay = (sent + size) / self.rate - (time.monotonic() - began)
                    if delay > 0 and self._stopping.wait(delay): ## Shutdown interrupts throttled responses.
                        break
                handler.wfile.write(data[start + sent:start + sent + size])
                handler.wfile.flush()
                sent += size
                with self._lock:
                    self.bytes_sent += size
            if cut or sent < length: ## Incomplete bodies close their connection.
                handler.close_connection = True
                try:
                    handler.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        finally:
            with self._lock:
                self._active -= 1
