## File Name: test_download.py
## Creator: DatJavaClass
## Associated Files: rt_download.py, rt_turbo.py, rt_config.py, rt_library.py
## Exposed Vars: CIVITAI_URL, OTHER_URL
## Exposed Funcs: QueueBase, TestEntries, TestPersistence, TestRunQueue, TestConfig
## Desc: rt_download and rt_config tests: queue entries, persistence, the queue run over a fake rt_turbo.fetch, speed clamps and the staging folder.
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_civitai
import rt_config
import rt_download
import rt_library
import rt_net
import rt_turbo

PATH_NAMES = ("CONFIG_PATH", "LIBRARY_PATH", "EXCLUSION_PATH", "UNRESOLVED_PATH", "QUEUE_PATH", "LEGACY_QUEUE_PATH", "DOWNLOAD_DIR", "DEFAULT_DOWNLOAD_DIR")
CIVITAI_URL = "https://civitai.com/models/1?modelVersionId=2"
OTHER_URL = "https://example.com/files/thing.safetensors?dl=1"

def civitai_rec():
    raw = {"id": 2, "modelId": 1, "name": "v1", "baseModel": "Pony", "trainedWords": ["foo", "bar"],
           "model": {"name": "Foo Model", "type": "LORA"}, "images": [{"url": "https://image.civitai.com/x/width=450/a.jpeg", "type": "image"}]}
    return {"model_id": 1, "version_id": 2, "model_name": "Foo Model", "version_name": "v1", "base_model": "Pony",
            "type": "LORA", "tags": ["foo", "bar"], "images": list(raw["images"]),
            "download_url": "https://civitai.com/api/download/models/2", "file_name": "fooModel_v1.safetensors",
            "sha256": None, "source": "url", "raw": raw}

class QueueBase(unittest.TestCase):
    def setUp(self):
        self.tmp, self.saved, self.calls = tempfile.mkdtemp(prefix="rt_dl_"), [], {"dl": [], "preview": [], "sort": 0, "label": 0}
        for mod in (rt_config, rt_library): ## redirect every working path
            for name in PATH_NAMES:
                if hasattr(mod, name):
                    self.patch(mod, name, os.path.join(self.tmp, os.path.basename(getattr(mod, name))))
        self.cfg = {"api_key": "KEY", "add_version": True, "window": ""}
        for key in ("ckpt_dir", "lora_dir", "vae_dir"):
            self.cfg[key] = os.path.join(self.tmp, key)
            os.makedirs(self.cfg[key])
        self.patch(rt_turbo, "fetch", self.fake_fetch)
        self.patch(rt_net, "http_download", self.forbidden)
        self.patch(rt_civitai, "fetch_by_url", lambda url, key: civitai_rec())
        self.patch(rt_library, "grab_preview", lambda path, images, key, cancel_flag=None: self.calls["preview"].append(images) or True)
        self.patch(rt_library, "sort_downloads", lambda cfg, log=print, cancel_flag=None: self.bump("sort"))
        self.patch(rt_library, "label_and_skin", lambda cfg, log=print, cancel_flag=None, relabel=False: self.bump("label"))

    def tearDown(self):
        for mod, name, value in reversed(self.saved):
            setattr(mod, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch(self, mod, name, value):
        self.saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, value)

    def bump(self, key):
        self.calls[key] += 1

    def forbidden(self, *args, **kwargs):
        raise AssertionError("V2 http_download called")

    def fake_fetch(self, url, dest, api_key=None, connections=8, cancel_flag=None, meter=None, key=None, log=None, name_hint=None):
        final = os.path.join(dest, name_hint or os.path.basename(urllib.parse.urlparse(url).path)) if os.path.isdir(dest) else dest
        with open(final, "wb") as f:
            f.write(os.urandom(4096))
        self.calls["dl"].append((url, api_key))
        if meter is not None: ## fake engine reports like the real one
            meter.start(key, os.path.basename(final), 4096)
            meter.add(key, 4096)
            meter.finish(key, "done")
        return True, final

    def run_queue(self, entries, cancel_flag=None):
        lines, done = [], []
        result = rt_download.run_queue(entries, self.cfg, log=lines.append, cancel_flag=cancel_flag, progress=lambda f: None, on_done=done.append)
        self.assertEqual(done, [result])
        return result, lines

    def entry_for(self, path):
        db = rt_library.load_library()
        return next((e for e in db["files"].values() if os.path.normcase(e["path"]) == os.path.normcase(path)), None)

class TestEntries(QueueBase):
    def test_entry_from_url_defaults(self):
        e = rt_download.entry_from_url("  " + CIVITAI_URL + " ")
        self.assertEqual(e, {"url": CIVITAI_URL, "image_url": None, "tags": [], "name": None, "base_model": None, "kind": None})

    def test_entry_from_url_fields(self):
        e = rt_download.entry_from_url(OTHER_URL, "https://example.com/p.png", "a, b,,c", "Thing", "SDXL 1.0", "LoRA")
        self.assertEqual((e["tags"], e["kind"], e["name"], e["base_model"]), (["a", "b", "c"], "lora", "Thing", "SDXL 1.0"))
        self.assertIsNone(rt_download.entry_from_url(OTHER_URL, kind="embedding")["kind"])

    def test_describe(self):
        self.assertEqual(rt_download.describe(rt_download.entry_from_url(CIVITAI_URL)), CIVITAI_URL)
        line = rt_download.describe(rt_download.entry_from_url(OTHER_URL, None, ["x"], "Thing", "Pony", "lora"))
        self.assertTrue(line.startswith("Thing") and "[Pony]" in line and OTHER_URL in line)

class TestPersistence(QueueBase):
    def test_save_load_round_trip(self):
        entries = [rt_download.entry_from_url(CIVITAI_URL), rt_download.entry_from_url(OTHER_URL, None, ["t"], "N", "Pony", "vae")]
        self.assertTrue(rt_download.save_queue(entries))
        self.assertEqual(rt_download.load_queue(), entries)

    def test_legacy_txt_import(self):
        with open(rt_config.LEGACY_QUEUE_PATH, "w", encoding="utf-8") as f:
            f.write(CIVITAI_URL + "\n\nnot a url\n" + OTHER_URL + "\n")
        self.assertEqual([e["url"] for e in rt_download.load_queue()], [CIVITAI_URL, OTHER_URL])

    def test_json_wins_over_legacy(self):
        with open(rt_config.LEGACY_QUEUE_PATH, "w", encoding="utf-8") as f:
            f.write(OTHER_URL + "\n")
        rt_download.save_queue([CIVITAI_URL])
        self.assertEqual([e["url"] for e in rt_download.load_queue()], [CIVITAI_URL])

    def test_export_and_import(self):
        entries = [rt_download.entry_from_url(OTHER_URL, "https://example.com/p.png", ["t"], "Ünïcode", "Pony", "lora")]
        jpath, tpath = os.path.join(self.tmp, "q.json"), os.path.join(self.tmp, "q.txt")
        self.assertTrue(rt_download.export_queue(entries, jpath) and rt_download.export_queue(entries, tpath))
        self.assertEqual(rt_download.import_file(jpath), entries)
        self.assertEqual(rt_download.import_file(tpath), [rt_download.entry_from_url(OTHER_URL)])
        self.assertEqual(rt_download.import_file(os.path.join(self.tmp, "missing.json")), [])

class TestRunQueue(QueueBase):
    def test_civitai_entry(self):
        result, _ = self.run_queue([{"url": CIVITAI_URL}])
        path = os.path.join(rt_config.DOWNLOAD_DIR, "fooModel_v1.safetensors")
        self.assertEqual([r["path"] for r in result], [path])
        self.assertEqual(self.calls["dl"], [("https://civitai.com/api/download/models/2", "KEY")])
        self.assertEqual(rt_library.read_sidecar(path)["tags"], ["foo", "bar"])
        self.assertTrue(os.path.isfile(os.path.splitext(path)[0] + ".json"))
        entry = self.entry_for(path)
        self.assertEqual((entry["model_id"], entry["version_id"]), (1, 2))
        self.assertEqual(len(self.calls["preview"]), 1)
        self.assertNotIn("foomodel_v1", rt_config.load_exclusions())
        self.assertEqual((self.calls["sort"], self.calls["label"]), (1, 1))

    def test_generic_entry(self):
        entry = rt_download.entry_from_url(OTHER_URL, "https://example.com/p.png", ["kw1", "kw2"], kind="lora")
        result, _ = self.run_queue([entry])
        path = os.path.join(rt_config.DOWNLOAD_DIR, "thing.safetensors")
        self.assertEqual([r["path"] for r in result], [path])
        self.assertEqual(self.calls["dl"], [(OTHER_URL, None)])
        self.assertEqual(rt_library.read_sidecar(path)["tags"], ["kw1", "kw2"])
        self.assertEqual(self.calls["preview"][0][0]["url"], "https://example.com/p.png")
        self.assertIn("thing", rt_config.load_exclusions())
        self.assertIsNotNone(self.entry_for(path))

    def test_generic_entry_named_is_renamed(self):
        entry = rt_download.entry_from_url(OTHER_URL, None, ["kw"], "Nice Thing", "Pony", "lora")
        result, _ = self.run_queue([entry])
        stem = os.path.splitext(os.path.basename(result[0]["path"]))[0]
        self.assertTrue(stem.startswith(rt_civitai.tag_for_base_model("Pony")))
        self.assertTrue(os.path.isfile(result[0]["path"]))
        self.assertIn(rt_config.normalize_stem(stem), rt_config.load_exclusions())

    def test_unresolved_civitai_is_skipped(self):
        self.patch(rt_civitai, "fetch_by_url", lambda url, key: None)
        result, lines = self.run_queue([CIVITAI_URL])
        self.assertEqual((result, self.calls["dl"], self.calls["sort"]), ([], [], 0))
        self.assertTrue(any("no downloadable file" in ln for ln in lines))

    def test_cancel_stops_before_download(self):
        flag = threading.Event()
        flag.set()
        result, _ = self.run_queue([CIVITAI_URL, OTHER_URL], cancel_flag=flag)
        self.assertEqual((result, self.calls["dl"], self.calls["sort"]), ([], [], 0))

class TestConfig(QueueBase):
    def test_new_keys_default(self):
        cfg = rt_config.default_config()
        self.assertEqual((cfg["connections"], cfg["parallel_files"], cfg["download_dir"]), (2, 2, ""))

    def test_clamp_speed(self):
        self.assertEqual(rt_config.clamp_speed({}), (2, 2))
        self.assertEqual(rt_config.clamp_speed({"connections": 99, "parallel_files": 0}), (16, 1))
        self.assertEqual(rt_config.clamp_speed({"connections": "junk", "parallel_files": "5"}), (2, 5))

    def test_download_dir_applies_and_blank_restores(self):
        stage = os.path.join(self.tmp, "stage")
        self.assertEqual(rt_config.apply_download_dir({"download_dir": stage}), stage)
        self.assertEqual(rt_config.DOWNLOAD_DIR, stage)
        self.assertTrue(rt_config.save_config(dict(rt_config.default_config(), download_dir="")))
        self.assertEqual(rt_config.DOWNLOAD_DIR, rt_config.DEFAULT_DOWNLOAD_DIR)
        with open(rt_config.CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"download_dir": stage}, f)
        rt_config.load_config()
        self.assertEqual(rt_config.DOWNLOAD_DIR, stage)

if __name__ == "__main__":
    unittest.main()
