## File Name: test_queue_parallel.py
## Creator: DatJavaClass
## Associated Files: rt_download.py, rt_turbo.py, rt_library.py
## Exposed Vars: None
## Exposed Funcs: QueueParallelTests
## Desc: Verify parallel queue behavior with fake downloads.
import hashlib
import os
import sys
import tempfile
import threading
import unittest
import urllib.parse
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_civitai
import rt_config
import rt_download
import rt_library
import rt_net
import rt_turbo

PATH_NAMES = ("CONFIG_PATH", "LIBRARY_PATH", "EXCLUSION_PATH", "UNRESOLVED_PATH", "QUEUE_PATH", "LEGACY_QUEUE_PATH", "DOWNLOAD_DIR")

class QueueParallelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory(prefix="rt_parallel_"))
        self.calls, self.logs, self.snaps, self.callbacks = [], [], [], []
        self.lock, self.active, self.peak = threading.Lock(), 0, 0
        self.gate, self.cancel_on_fetch, self.delay = None, False, 0
        for mod in (rt_config, rt_library): ## redirect paths before any queue writes
            for name in PATH_NAMES: ## exits after every persistence path
                if hasattr(mod, name): ## present paths receive temporary replacements
                    self.replace(mod, name, os.path.join(self.tmp, os.path.basename(getattr(mod, name))))
        self.cfg = {"api_key": "KEY", "add_version": True, "connections": 4, "parallel_files": 3, "download_dir": rt_config.DOWNLOAD_DIR}
        for key in ("ckpt_dir", "lora_dir", "vae_dir"): ## temporary model directories only
            self.cfg[key] = os.path.join(self.tmp, key)
            os.makedirs(self.cfg[key])
        self.fetch = self.replace(rt_turbo, "fetch", self.fake_fetch)
        self.replace(rt_net, "http_download", unittest.mock.Mock(side_effect=AssertionError("unexpected V2 download")))
        self.replace(rt_net.OPENER, "open", unittest.mock.Mock(side_effect=AssertionError("network forbidden")))
        self.replace(rt_civitai, "fetch_by_url", unittest.mock.Mock(side_effect=AssertionError("unexpected Civitai lookup")))
        self.replace(rt_library, "grab_preview", unittest.mock.Mock(return_value=False))
        self.sort = self.replace(rt_library, "sort_downloads", unittest.mock.Mock())
        self.label = self.replace(rt_library, "label_and_skin", unittest.mock.Mock())

    def replace(self, mod, name, value):
        return self.enterContext(patch.object(mod, name, value))

    def entries(self, count):
        return [rt_download.entry_from_url(f"https://example.invalid/file-{i}.safetensors") for i in range(count)]

    def fake_fetch(self, url, dest, api_key=None, connections=8, cancel_flag=None, meter=None, key=None, log=None, name_hint=None):
        payload = url.encode("utf-8")
        path = os.path.join(dest, name_hint or os.path.basename(urllib.parse.urlparse(url).path)) if os.path.isdir(dest) else dest
        with self.lock:
            self.calls.append((url, connections, key, meter))
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if meter is not None: ## fake engine registers its progress
                meter.start(key, os.path.basename(path), len(payload))
            if self.gate is not None: ## synchronized workers exit together
                self.gate.wait(timeout=3)
            if self.cancel_on_fetch: ## first worker cancels remaining entries
                with open(path + ".part", "wb") as stream:
                    stream.write(payload[:4])
                cancel_flag.set()
                if meter is not None: ## cancelled engine records terminal state
                    meter.finish(key, "cancelled")
                return False, None
            threading.Event().wait(self.delay)
            with open(path, "wb") as stream:
                stream.write(payload)
            if meter is not None: ## completed engine publishes byte count
                meter.add(key, len(payload))
                meter.finish(key, "done")
            return True, path
        finally:
            with self.lock:
                self.active -= 1

    def run_queue(self, entries, cancel_flag=None):
        result = rt_download.run_queue(entries, self.cfg, log=self.logs.append, cancel_flag=cancel_flag,
                                       progress=self.snaps.append, on_done=self.callbacks.append)
        self.assertEqual(self.callbacks, [result])
        return result

    def test_parallel_limit_and_database_keeps_every_file(self):
        for width in (1, 3, 6): ## each width uses distinct payloads
            with self.subTest(parallel_files=width):
                self.cfg["parallel_files"] = width
                self.calls.clear()
                self.callbacks.clear()
                self.peak, self.gate = 0, threading.Barrier(width)
                entries = [rt_download.entry_from_url(f"https://example.invalid/group-{width}-{i}.safetensors") for i in range(width * 2)]
                result = self.run_queue(entries)
                self.assertEqual(len(result), len(entries), self.logs)
                self.assertEqual(self.peak, width)
                self.assertEqual({r["entry"]["url"] for r in result}, {e["url"] for e in entries})
                db = rt_library.load_library()
                for item in result: ## every finished path must survive saving
                    self.assertIsNotNone(rt_library.find_by_path(db, item["path"])[1])
                self.assertEqual({call[1] for call in self.calls}, {4})
                self.assertEqual({call[2] for call in self.calls}, {e["url"] for e in entries})
                self.assertEqual(len({id(call[3]) for call in self.calls}), 1)
                self.assertIsNotNone(self.calls[0][3])

    def civitai_record(self, sha):
        return {"model_id": 1, "version_id": 2, "model_name": "Test", "version_name": "v1", "base_model": "Pony",
                "type": "LORA", "tags": [], "images": [], "download_url": "https://example.invalid/verified.safetensors",
                "file_name": "verified.safetensors", "sha256": sha, "source": "url", "raw": {}}

    def test_hash_mismatch_is_quarantined_and_not_registered(self):
        self.replace(rt_civitai, "fetch_by_url", lambda url, key: self.civitai_record("0" * 64))
        result = self.run_queue([rt_download.entry_from_url("https://civitai.com/models/1")])
        path = os.path.join(rt_config.DOWNLOAD_DIR, "verified.safetensors")
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(path))
        self.assertTrue(os.path.isfile(path + ".corrupt"))
        self.assertEqual(rt_library.load_library()["files"], {})
        self.assertTrue(any("sha256" in line.lower() or "mismatch" in line.lower() for line in self.logs), self.logs)

    def test_matching_hash_is_registered(self):
        sha = hashlib.sha256(b"https://example.invalid/verified.safetensors").hexdigest()
        self.replace(rt_civitai, "fetch_by_url", lambda url, key: self.civitai_record(sha))
        result = self.run_queue([rt_download.entry_from_url("https://civitai.com/models/1")])
        self.assertEqual(len(result), 1, self.logs)
        self.assertIn(sha, rt_library.load_library()["files"])

    def test_cancel_stops_new_entries_and_keeps_partial(self):
        self.cfg["parallel_files"], self.cancel_on_fetch = 1, True
        result = self.run_queue(self.entries(8), threading.Event())
        self.assertEqual(result, [])
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(os.path.isfile(os.path.join(rt_config.DOWNLOAD_DIR, "file-0.safetensors.part")))
        self.assertEqual(sum("resume" in line.lower() and "part" in line.lower() for line in self.logs), 1, self.logs)
        self.sort.assert_not_called()
        self.label.assert_not_called()

    def test_already_cancelled_starts_nothing(self):
        flag = threading.Event()
        flag.set()
        self.assertEqual(self.run_queue(self.entries(4), flag), [])
        self.assertEqual(self.calls, [])

    def test_progress_receives_live_and_final_snapshots(self):
        self.delay = 0.65
        result = self.run_queue(self.entries(2))
        self.assertEqual(len(result), 2, self.logs)
        self.assertGreaterEqual(len(self.snaps), 2)
        for snap in self.snaps: ## every callback uses the public schema
            self.assertIsInstance(snap, dict)
            self.assertTrue({"done", "total", "speed", "eta", "active", "files"}.issubset(snap))
        self.assertTrue(any(snap["active"] > 0 for snap in self.snaps))
        self.assertEqual(self.snaps[-1]["active"], 0)
        self.assertEqual(self.snaps[-1]["done"], sum(len(e["url"].encode()) for e in self.entries(2)))

    def test_existing_completed_file_skips_fetch(self):
        self.replace(rt_civitai, "fetch_by_url", lambda url, key: self.civitai_record(None))
        os.makedirs(rt_config.DOWNLOAD_DIR, exist_ok=True)
        path = os.path.join(rt_config.DOWNLOAD_DIR, "verified.safetensors")
        with open(path, "wb") as stream:
            stream.write(b"existing file")
        result = self.run_queue([rt_download.entry_from_url("https://civitai.com/models/1")])
        self.assertEqual([r["path"] for r in result], [path])
        self.assertEqual(self.calls, [])

if __name__ == "__main__": ## direct execution runs unittest
    unittest.main()
