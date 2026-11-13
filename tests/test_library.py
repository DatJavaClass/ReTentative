## rt_library tests, temp fixtures only, no network.
import json, os, shutil, sys, tempfile, unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_civitai, rt_config, rt_library, rt_net

def _fake(path, size=4096):
    with open(path, "wb") as f:
        f.write(os.urandom(size))
    return path

def _rec(**kw):
    rec = {"model_id": 11, "version_id": 22, "model_name": "Nova Furry XL", "version_name": "v1.0",
           "base_model": "Pony", "type": "LORA", "tags": ["nova", "furry"], "images": [],
           "download_url": None, "file_name": None, "sha256": None, "source": "hash", "raw": {}}
    rec.update(kw)
    return rec

class LibraryBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patches = [mock.patch.object(rt_config, k, os.path.join(self.tmp, v)) for k, v in (("LIBRARY_PATH", "_library.json"),
                        ("EXCLUSION_PATH", "_exclusion_list.txt"), ("UNRESOLVED_PATH", "_unresolved.txt"), ("DOWNLOAD_DIR", "dl"))]
        self.patches += [mock.patch.object(rt_net, "download_image", return_value=False),
                         mock.patch.object(rt_civitai, "fetch_images_fallback", return_value=[]),
                         mock.patch.object(rt_civitai, "scrape_base_model", return_value=None)]
        for p in self.patches:
            p.start()
        self.dirs = {k: os.path.join(self.tmp, k) for k in ("ckpt", "lora", "vae", "dl")}
        for d in self.dirs.values():
            os.makedirs(d)
        self.cfg = dict(rt_config.default_config(), ckpt_dir=self.dirs["ckpt"], lora_dir=self.dirs["lora"], vae_dir=self.dirs["vae"])

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

class TestNaming(unittest.TestCase):
    def test_full_format(self):
        self.assertEqual(rt_library.build_new_stem("[Pony]", "Nova Furry XL", "v1.0", "novaFurryXL_v10"), "[Pony]-Nova Furry XL (v1.0)")

    def test_version_rules(self):
        b = rt_library.build_new_stem
        self.assertEqual(b("[Pony]", "Nova", "v1.0", "x", add_version=False), "[Pony]-Nova")
        self.assertEqual(b("[Pony]", "Nova", "", "x"), "[Pony]-Nova")
        self.assertEqual(b("[Pony]", "Nova", None, "x"), "[Pony]-Nova")
        self.assertEqual(b("[Pony]", "Nova", "NOVA", "x"), "[Pony]-Nova")

    def test_idempotent(self):
        b, first = rt_library.build_new_stem, "[Pony]-Nova Furry XL (v1.0)"
        self.assertEqual(b("[Pony]", None, "v1.0", first), first)
        self.assertEqual(b("[Pony]", first, "v1.0", "x"), first)
        self.assertEqual(b("[Pony]", "[Pony]-[Pony] Nova", "", "x"), "[Pony]-Nova")
        self.assertEqual(b("[Pony]", None, "", "[Pony]-Nova Furry XL"), "[Pony]-Nova Furry XL")

    def test_cap_sanitize_decamel(self):
        b = rt_library.build_new_stem
        self.assertEqual(len(b("[Pony]", "a" * 300, "", "x")), len("[Pony]-") + rt_library.MAX_FRIENDLY)
        self.assertEqual(b("[Pony]", 'a/b:c?"d', "", "x"), "[Pony]-abcd")
        self.assertEqual(b("[Pony]", None, "", "novaFurryXL"), "[Pony]-nova Furry XL")

    def test_predicates(self):
        self.assertTrue(rt_library.already_formatted("[Pony]-Nova"))
        self.assertFalse(rt_library.already_formatted("Nova"))
        self.assertTrue(rt_library.has_version("[Pony]-Nova (v1.0)"))
        self.assertFalse(rt_library.has_version("[Pony]-Nova"))

class TestRename(LibraryBase):
    def test_siblings_follow_and_neighbors_stay(self):
        d = self.dirs["lora"]
        for n in ("Foo.safetensors", "Foo.civitai.info", "Foo.preview.png", "Foo.json", "Foo.v2.safetensors", "Foo.v2.civitai.info", "Foobar.txt"):
            _fake(os.path.join(d, n), 64)
        db = rt_library.load_library()
        rt_library.register(db, os.path.join(d, "Foo.safetensors"), "lora")
        moved = rt_library.rename_with_siblings(os.path.join(d, "Foo.safetensors"), "[X]-Bar", db)
        self.assertEqual(len(moved), 4)
        self.assertEqual(sorted(os.listdir(d)), sorted(["[X]-Bar.safetensors", "[X]-Bar.civitai.info", "[X]-Bar.preview.png", "[X]-Bar.json",
                                                        "Foo.v2.safetensors", "Foo.v2.civitai.info", "Foobar.txt"]))
        sha, entry = rt_library.find_by_path(db, os.path.join(d, "[X]-Bar.safetensors"))
        self.assertIsNotNone(sha)
        self.assertEqual(len(db["files"]), 1)

    def test_no_overwrite(self):
        d = self.dirs["lora"]
        a, b = _fake(os.path.join(d, "A.safetensors")), _fake(os.path.join(d, "B.safetensors"))
        self.assertEqual(rt_library.rename_with_siblings(a, "B"), [])
        self.assertTrue(os.path.isfile(a) and os.path.isfile(b))

class TestDuplicates(LibraryBase):
    def test_groups_across_folders(self):
        a = _fake(os.path.join(self.dirs["lora"], "a.safetensors"))
        shutil.copyfile(a, os.path.join(self.dirs["lora"], "b.safetensors"))
        shutil.copyfile(a, os.path.join(self.dirs["ckpt"], "c.safetensors"))
        _fake(os.path.join(self.dirs["vae"], "d.safetensors"))
        groups = rt_library.find_duplicates(self.cfg, log=lambda s: None)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(os.path.basename(p) for p in groups[0]), ["a.safetensors", "b.safetensors", "c.safetensors"])
        self.assertEqual(len(rt_library.load_library()["files"]), 2)

class TestHashCache(LibraryBase):
    def test_cache_hit_and_miss(self):
        path = _fake(os.path.join(self.dirs["lora"], "m.safetensors"))
        db = rt_library.load_library()
        entry = rt_library.register(db, path, "lora")
        rt_library.save_library(db)
        db = rt_library.load_library()
        with mock.patch.object(rt_net, "sha256_file", side_effect=AssertionError("rehashed")):
            self.assertIn(rt_library.hash_for(db, path), db["files"])
        os.utime(path, (1, 1))
        with mock.patch.object(rt_net, "sha256_file", return_value="f" * 64) as h:
            self.assertEqual(rt_library.hash_for(db, path), "f" * 64)
            h.assert_called_once()
        self.assertEqual(entry["kind"], "lora")

    def test_register_merges_one_entry(self):
        path = _fake(os.path.join(self.dirs["lora"], "m.safetensors"))
        db = rt_library.load_library()
        rt_library.register(db, path, "lora")
        entry = rt_library.register(db, path, "lora", _rec())
        self.assertEqual(len(db["files"]), 1)
        self.assertEqual((entry["model_id"], entry["version_id"], entry["tags"]), (11, 22, ["nova", "furry"]))
        self.assertEqual(entry["tag"], rt_civitai.tag_for_base_model("Pony"))
        self.assertEqual(rt_library.orphans(db), [])

class TestSidecar(LibraryBase):
    def test_raw_round_trip(self):
        path = _fake(os.path.join(self.dirs["lora"], "m.safetensors"))
        raw = {"id": 22, "modelId": 11, "name": "v1.0", "baseModel": "Pony", "trainedWords": ["nova"],
               "model": {"name": "Nova", "type": "LORA"}, "images": [{"url": "https://image.civitai.com/x.png", "type": "image"}], "files": []}
        rec = rt_civitai.normalize(raw, source="hash")
        self.assertTrue(rt_library.write_sidecar(path, rec))
        with open(rt_library.sidecar_path(path), encoding="utf-8") as f:
            self.assertEqual(json.load(f), raw)
        back = rt_library.read_sidecar(path)
        self.assertEqual((back["model_name"], back["base_model"], back["tags"], back["version_id"]), ("Nova", "Pony", ["nova"], 22))

    def test_minimal_and_a1111(self):
        path = _fake(os.path.join(self.dirs["lora"], "m.safetensors"))
        rt_library.write_sidecar(path, _rec(raw=None))
        with open(rt_library.sidecar_path(path), encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual((data["baseModel"], data["model"]["name"], data["name"], data["trainedWords"], data["modelId"], data["id"]),
                         ("Pony", "Nova Furry XL", "v1.0", ["nova", "furry"], 11, 22))
        self.assertEqual(rt_library.read_sidecar(path)["tags"], ["nova", "furry"])
        rt_library.write_a1111_json(path, _rec())
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"activation text": "nova, furry", "description": "", "sd version": "Pony"})

    def test_missing_sidecar(self):
        self.assertIsNone(rt_library.read_sidecar(os.path.join(self.dirs["lora"], "none.safetensors")))

class TestUpdateDatabase(LibraryBase):
    def test_orphans_filled_renamed_excluded(self):
        hit = _fake(os.path.join(self.dirs["ckpt"], "novaFurryXL.safetensors"))
        miss = _fake(os.path.join(self.dirs["lora"], "local_merge.safetensors"))
        skip = _fake(os.path.join(self.dirs["lora"], "my_own.safetensors"))
        rt_config.add_to_exclusions("my_own")
        hit_sha = rt_net.sha256_file(hit)
        calls = []
        def fake_fetch(sha, key):
            calls.append(sha)
            return _rec(type="Checkpoint", raw={"id": 22, "modelId": 11, "name": "v1.0", "baseModel": "Pony", "trainedWords": ["nova"],
                                                "model": {"name": "Nova Furry XL", "type": "Checkpoint"}}) if sha == hit_sha else None
        with mock.patch.object(rt_civitai, "fetch_by_hash", side_effect=fake_fetch):
            counts = rt_library.update_database(self.cfg, log=lambda s: None)
        self.assertEqual((counts["registered"], counts["queried"], counts["resolved"], counts["unresolved"], counts["excluded"], counts["renamed"]), (3, 2, 1, 1, 1, 1))
        self.assertNotIn(rt_net.sha256_file(skip), calls)
        new = os.path.join(self.dirs["ckpt"], rt_civitai.tag_for_base_model("Pony") + "-Nova Furry XL (v1.0).safetensors")
        self.assertTrue(os.path.isfile(new))
        self.assertTrue(os.path.isfile(rt_library.sidecar_path(new)))
        self.assertTrue(os.path.isfile(os.path.splitext(new)[0] + ".json"))
        self.assertTrue(os.path.isfile(miss))
        db = rt_library.load_library()
        self.assertEqual(db["files"][hit_sha]["path"], os.path.abspath(new))
        self.assertEqual(db["files"][hit_sha]["tags"], ["nova", "furry"])
        self.assertEqual(len(rt_library.orphans(db)), 2)

    def test_resolve_model_writes_sidecar_on_hash(self):
        path = _fake(os.path.join(self.dirs["ckpt"], "ckpt.safetensors"))
        db = rt_library.load_library()
        with mock.patch.object(rt_civitai, "fetch_by_hash", return_value=_rec(raw=None)):
            rec = rt_library.resolve_model(path, None, db)
        self.assertEqual(rec["source"], "hash")
        self.assertEqual(rt_library.read_sidecar(path)["tags"], ["nova", "furry"])
        self.assertEqual(rt_library.find_by_path(db, path)[1]["version_id"], 22)

    def test_sort_moves_entry(self):
        path = _fake(os.path.join(self.dirs["dl"], "tiny.safetensors"))
        _fake(os.path.join(self.dirs["dl"], "tiny.civitai.info"), 8)
        db = rt_library.load_library()
        sha = rt_library._register(db, path, "")[0]
        rt_library.save_library(db)
        self.assertEqual(rt_library.sort_downloads(self.cfg, log=lambda s: None), 1)
        moved = os.path.join(self.dirs["lora"], "tiny.safetensors")
        self.assertTrue(os.path.isfile(moved) and os.path.isfile(os.path.join(self.dirs["lora"], "tiny.civitai.info")))
        self.assertEqual(rt_library.load_library()["files"][sha]["path"], os.path.abspath(moved))

if __name__ == "__main__":
    unittest.main()
