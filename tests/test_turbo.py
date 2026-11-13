## File Name: test_turbo.py
## Creator: DatJavaClass
## Associated Files: rt_turbo.py, range_server.py
## Exposed Vars: None
## Exposed Funcs: TurboTests, MeterTests
## Desc: Contract tests for downloads and progress.
import json, pathlib, tempfile, threading, time, unittest, urllib.error
from unittest.mock import patch
import rt_turbo
from tests.range_server import RangeServer

PIECE = 64 * 1024 ## Test piece bytes.
PAYLOAD = bytes(range(256)) * 2048 ## Eight deterministic pieces.

class TurboTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = pathlib.Path(folder.name)
        self.addCleanup(patch.stopall)
        patch.object(rt_turbo, "PIECE", PIECE).start()
        patch.object(rt_turbo, "READ", 16 * 1024).start()

    def server(self, data=PAYLOAD, **options):
        server = RangeServer({"model.bin": data, "odd:name.bin": data}, **options)
        self.addCleanup(server.stop)
        base = server.start()
        return server, base

    def fetch(self, url, dest=None, **options):
        dest = dest or self.folder / "output.bin"
        result = rt_turbo.fetch(url, str(dest), **options)
        self.assertEqual(result, (True, str(dest)))
        self.assertEqual(dest.read_bytes(), PAYLOAD)
        self.assertFalse(pathlib.Path(str(dest) + ".part").exists())
        self.assertFalse(pathlib.Path(str(dest) + ".part.json").exists())
        return result

    def test_probe_ranged(self):
        server, base = self.server()
        info = rt_turbo.probe(base + "/f/model.bin")
        self.assertEqual(set(info), {"url", "total", "ranged", "name", "etag"})
        self.assertEqual(info["url"], base + "/f/model.bin")
        self.assertEqual(info["total"], len(PAYLOAD))
        self.assertTrue(info["ranged"])
        self.assertEqual(info["name"], "model.bin")
        self.assertEqual(info["etag"], '"range-fixture-v1"')
        self.assertEqual(server.ranges_seen, [(0, 0)])

    def test_probe_not_ranged(self):
        _, base = self.server(ranged=False)
        info = rt_turbo.probe(base + "/f/model.bin")
        self.assertFalse(info["ranged"])
        self.assertEqual(info["total"], len(PAYLOAD))

    def test_probe_redirect_name_and_local_auth(self):
        server, base = self.server(redirect=True)
        original = server._headers
        def named_headers(handler, status, length=0, headers=None):
            headers = dict(headers or {})
            if status == 302: ## Redirect supplies a distinct display name.
                headers["Content-Disposition"] = 'attachment; filename="display:name.bin"'
            original(handler, status, length, headers)

        with patch.object(server, "_headers", side_effect=named_headers):
            info = rt_turbo.probe(base + "/dl/odd%3Aname.bin", api_key="local-test-key")
        self.assertTrue(info["url"].startswith(base + "/f/odd%3Aname.bin?sig="))
        self.assertEqual(info["name"], "displayname.bin")
        self.assertEqual(info["total"], len(PAYLOAD))
        self.assertTrue(info["ranged"])
        self.assertEqual(server.hits, 2)
        self.assertTrue(all(auth is None for _, auth in server.auth_seen))

    def test_segmented_fetch_and_parallel_connections(self):
        server, base = self.server(rate=1024 * 1024)
        self.fetch(base + "/f/model.bin", connections=4)
        self.assertGreater(server.peak_connections, 1)
        self.assertLessEqual(server.peak_connections, 4)
        pieces = {(start, end) for start, end in server.ranges_seen if end > start}
        self.assertEqual(pieces, {(start, start + PIECE - 1) for start in range(0, len(PAYLOAD), PIECE)})

    def test_single_stream_fallback(self):
        server, base = self.server(ranged=False)
        self.fetch(base + "/f/model.bin", connections=4)
        self.assertEqual(server.ranges_seen, [])

    def test_small_file_single_stream(self):
        small = PAYLOAD[:PIECE + 7]
        _, base = self.server(data=small)
        dest = self.folder / "small.bin"
        self.assertEqual(rt_turbo.fetch(base + "/f/model.bin", str(dest), connections=4), (True, str(dest)))
        self.assertEqual(dest.read_bytes(), small)

    def test_folder_destination_and_name_hint(self):
        _, base = self.server(redirect=True)
        result = rt_turbo.fetch(base + "/dl/model.bin", str(self.folder), name_hint="hint.bin")
        self.assertEqual(result, (True, str(self.folder / "hint.bin")))
        self.assertEqual((self.folder / "hint.bin").read_bytes(), PAYLOAD)

    def test_folder_destination_probed_name(self):
        _, base = self.server(redirect=True)
        result = rt_turbo.fetch(base + "/dl/model.bin", str(self.folder))
        self.assertEqual(result, (True, str(self.folder / "model.bin")))
        self.assertEqual((self.folder / "model.bin").read_bytes(), PAYLOAD)

    def test_flaky_pieces_recover(self):
        server, base = self.server(fail_every=4)
        meter = rt_turbo.Meter()
        self.fetch(base + "/f/model.bin", connections=2, meter=meter, key="flaky")
        self.assertGreater(len(server.ranges_seen), len(PAYLOAD) // PIECE + 1)
        self.assertEqual(meter.snapshot()["done"], len(PAYLOAD))

    def test_expired_signature_reprobes(self):
        server, base = self.server(redirect=True, expire_after=3)
        self.fetch(base + "/dl/model.bin", connections=2)
        redirects = [path for path, _ in server.auth_seen if path.startswith("/dl/")]
        self.assertGreater(len(redirects), 1)

    def test_cancel_keeps_parts_and_resume_transfers_less(self):
        server, base = self.server(rate=2 * 1024 * 1024)
        flag, dest = threading.Event(), self.folder / "resume.bin"
        class CancellingMeter(rt_turbo.Meter):
            def add(inner, key, n):
                super().add(key, n)
                if inner.snapshot()["done"] >= PIECE + 16 * 1024: ## Cancel after one complete piece.
                    flag.set()

        result = rt_turbo.fetch(base + "/f/model.bin", str(dest), connections=1, cancel_flag=flag, meter=CancellingMeter(), key="resume")
        self.assertEqual(result, (False, None))
        self.assertTrue(flag.is_set())
        self.assertFalse(dest.exists())
        self.assertTrue(pathlib.Path(str(dest) + ".part").exists())
        state = json.loads(pathlib.Path(str(dest) + ".part.json").read_text())
        self.assertEqual(state["url"], base + "/f/model.bin")
        self.assertEqual(state["total"], len(PAYLOAD))
        self.assertEqual(state["piece"], PIECE)
        self.assertEqual(state["etag"], '"range-fixture-v1"')
        self.assertTrue(state["done"])

        before, first_range = server.bytes_sent, len(server.ranges_seen)
        self.fetch(base + "/f/model.bin", dest, connections=2)
        self.assertLess(server.bytes_sent - before, len(PAYLOAD))
        resumed = {start // PIECE for start, end in server.ranges_seen[first_range:] if end > start}
        self.assertTrue(set(state["done"]).isdisjoint(resumed))

    def test_404_raises_http_error(self):
        _, base = self.server()
        for operation in (lambda: rt_turbo.probe(base + "/f/missing"), lambda: rt_turbo.fetch(base + "/f/missing", str(self.folder))): ## Both operations exit through HTTPError.
            with self.assertRaises(urllib.error.HTTPError) as caught:
                operation()
            self.assertEqual(caught.exception.code, 404)

    def test_mismatched_resume_state_starts_clean(self):
        server, base = self.server()
        dest, url = self.folder / "stale.bin", base + "/f/model.bin"
        for field, value in (("total", len(PAYLOAD) + 1), ("etag", '"stale"'), ("piece", PIECE * 2)): ## Each mismatch must discard saved pieces.
            with self.subTest(field=field):
                state = {"url": url, "total": len(PAYLOAD), "etag": '"range-fixture-v1"', "piece": PIECE, "done": [0]}
                state[field] = value
                pathlib.Path(str(dest) + ".part").write_bytes(b"x" * len(PAYLOAD))
                pathlib.Path(str(dest) + ".part.json").write_text(json.dumps(state))
                first_range = len(server.ranges_seen)
                self.fetch(url, dest, connections=2)
                self.assertIn((0, PIECE - 1), server.ranges_seen[first_range:])

class MeterTests(unittest.TestCase):
    def test_math_retitle_negative_add_and_finish(self):
        meter = rt_turbo.Meter()
        meter.start("a", "first", 100)
        meter.start("b", "unknown", 0)
        meter.add("a", 40)
        meter.add("a", -10)
        meter.add("b", 20)
        snap = meter.snapshot()
        self.assertEqual(set(snap), {"done", "total", "speed", "eta", "active", "files"})
        self.assertEqual((snap["done"], snap["total"], snap["active"]), (50, 100, 2))
        self.assertIsNone(snap["eta"])
        self.assertIsInstance(snap["speed"], float)
        self.assertGreaterEqual(snap["speed"], 0)
        files = {row["key"]: row for row in snap["files"]}
        self.assertEqual(set(files["a"]), {"key", "name", "done", "total", "speed", "state"})
        self.assertEqual(files["a"]["done"], 30)

        meter.retitle("b", "second", 200)
        meter.add("a", 70)
        meter.finish("a", "done")
        snap = meter.snapshot()
        self.assertEqual((snap["done"], snap["total"], snap["active"]), (120, 300, 1))
        files = {row["key"]: row for row in snap["files"]}
        self.assertEqual((files["b"]["name"], files["b"]["total"]), ("second", 200))
        self.assertEqual(files["a"]["state"], "done")
        self.assertIsNotNone(snap["eta"])
        meter.finish("b", "cancelled")
        self.assertEqual(meter.snapshot()["active"], 0)

    def test_failed_state(self):
        meter = rt_turbo.Meter()
        meter.start("failed", "failure", 100)
        meter.finish("failed", "failed")
        snap = meter.snapshot()
        self.assertEqual(snap["active"], 0)
        self.assertEqual(snap["files"][0]["state"], "failed")

    def test_concurrent_adds(self):
        meter = rt_turbo.Meter()
        meter.start("a", "parallel", 4000)
        def add_bytes():
            for _ in range(1000): ## Add until each worker exhausts iterations.
                meter.add("a", 1)

        workers = [threading.Thread(target=add_bytes) for _ in range(4)]
        for worker in workers: ## Start every counter worker.
            worker.start()
        for worker in workers: ## Join until every worker exits.
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(meter.snapshot()["done"], 4000)

    def test_speed_window_expires(self):
        meter = rt_turbo.Meter()
        with patch.object(time, "monotonic", return_value=100.0), patch.object(time, "time", return_value=100.0):
            meter.start("a", "speed", 1000)
        with patch.object(time, "monotonic", return_value=101.0), patch.object(time, "time", return_value=101.0):
            meter.add("a", 300)
            self.assertGreater(meter.snapshot()["speed"], 0)
        with patch.object(time, "monotonic", return_value=111.0), patch.object(time, "time", return_value=111.0):
            self.assertEqual(meter.snapshot()["speed"], 0)

if __name__ == "__main__": ## Direct invocation runs the unittest suite.
    unittest.main()
