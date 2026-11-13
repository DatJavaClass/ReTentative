## File Name: test_gui_headless.py
## Creator: DatJavaClass
## Associated Files: rt_gui.py, rt_forms.py, rt_config.py, rt_download.py, rt_library.py
## Exposed Vars: PATH_NAMES, URL_A, URL_B, MB
## Exposed Funcs: PureLogicTest, GuiTest
## Desc: Headless GUI tests in a temp working folder: build, queue, dupes, dialogs, worker marshalling, the download meter and the configuration save.
import json, os, re, shutil, sys, tempfile, unittest
import tkinter as tk
from tkinter import messagebox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rt_civitai, rt_config, rt_download, rt_forms, rt_gui, rt_library, rt_pruner

PATH_NAMES = ("CONFIG_PATH", "LIBRARY_PATH", "EXCLUSION_PATH", "UNRESOLVED_PATH", "QUEUE_PATH", "LEGACY_QUEUE_PATH", "DOWNLOAD_DIR", "DEFAULT_DOWNLOAD_DIR")
URL_A, URL_B = "https://civitai.com/models/1?modelVersionId=2", "https://example.com/files/thing.safetensors"
MB = 1048576

def snap(files, done, total, speed, eta):
    return {"done": done, "total": total, "speed": speed, "eta": eta, "active": sum(f["state"] == "active" for f in files), "files": files}

def row(key, name, done, total, speed, state="active"):
    return {"key": key, "name": name, "done": done, "total": total, "speed": speed, "state": state}

def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return path

class PureLogicTest(unittest.TestCase):
    def test_dupe_targets_keeps_one(self):
        paths, blocked = rt_gui.dupe_targets([["a1", "a2"], ["b1", "b2", "b3"]], {"a1", "b1", "b2"})
        self.assertEqual((paths, blocked), (["a1", "b1", "b2"], []))

    def test_status_text_shapes(self):
        files = [row("a", "a", 0, 0, 0)] * 3
        self.assertEqual(rt_gui.status_text(snap(files, 41, 100, 84.2 * MB, 130)), "3 active  84.2 MB/s  41%  2:10 left")
        self.assertEqual(rt_gui.status_text(snap(files[:1], 5 * MB, 0, 0, None)), "1 active  0.0 MB/s  5.0 MB")
        self.assertEqual(rt_gui.status_text(snap([], 1, 2, MB, 3725)), "0 active  1.0 MB/s  50%  1:02:05 left")

    def test_dupe_targets_blocks_full_group(self):
        paths, blocked = rt_gui.dupe_targets([["a1", "a2"], ["b1", "b2"]], {"a1", "a2", "b1"})
        self.assertEqual(blocked, [["a1", "a2"]])

    def test_with_siblings_skips_longer_models(self):
        tmp = tempfile.mkdtemp(prefix="rt_sib_")
        try:
            for name in ("foo.safetensors", "foo.civitai.info", "foo.preview.png", "foo.v2.safetensors", "foo.v2.preview.png", "food.png"):
                write_bytes(os.path.join(tmp, name), b"x")
            got = sorted(os.path.basename(p) for p in rt_gui.with_siblings([os.path.join(tmp, "foo.safetensors")]))
            self.assertEqual(got, ["foo.civitai.info", "foo.preview.png", "foo.safetensors"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

class GuiTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            self.skipTest(f"no display: {e}")
        self.tmp, self.saved = tempfile.mkdtemp(prefix="rt_gui_"), []
        for name in PATH_NAMES: ## working files into the temp dir
            self.patch(rt_config, name, os.path.join(self.tmp, os.path.basename(getattr(rt_config, name))))
        self.cfg = rt_config.default_config()
        for key in ("ckpt_dir", "lora_dir", "vae_dir"): ## fixture folders, never real ones
            self.cfg[key] = os.path.join(self.tmp, key)
            os.makedirs(self.cfg[key])
        self.cfg["api_key"] = "TEST"
        for name in ("showinfo", "showwarning", "showerror", "askyesno"):
            self.patch(messagebox, name, lambda *a, **k: True)
        self.patch(rt_civitai, "LOG", rt_civitai.LOG)
        self.app = rt_gui.build_window(self.root, self.cfg, rt_library.load_library())
        self.root.update()

    def tearDown(self):
        self.app.destroy()
        for mod, name, old in reversed(self.saved):
            setattr(mod, name, old)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def patch(self, mod, name, value):
        self.saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, value)

    def log_text(self):
        return self.app.text.get("1.0", "end")

    def test_tabs_built(self):
        names = [self.app.nb.tab(t, "text") for t in self.app.nb.tabs()]
        self.assertEqual(names, ["Download", "Library", "Repair (0)", "Pruner"])
        self.assertTrue(self.app.actions)
        self.assertIs(rt_civitai.LOG.__func__, rt_gui.App.log)

    def test_queue_add_remove_clear(self):
        self.assertEqual(self.app.add_entries([rt_download.entry_from_url(URL_A), rt_download.entry_from_url(URL_B), rt_download.entry_from_url(URL_A)]), 2)
        self.assertEqual(self.app.listbox.size(), 2)
        self.app.listbox.selection_set(0)
        self.app.remove_selected()
        self.assertEqual([e["url"] for e in self.app.queue], [URL_B])
        self.app.clear_queue()
        self.assertEqual((self.app.listbox.size(), rt_download.load_queue()), (0, []))

    def test_paste_reads_every_url(self):
        self.root.clipboard_get = lambda: f"{URL_A}\nnot a url\n{URL_B}\n{URL_A}"
        self.assertEqual(self.app.paste(), "break")
        self.assertEqual([e["url"] for e in self.app.queue], [URL_A, URL_B])

    def test_finished_entries_drop_and_save(self):
        a, b = rt_download.entry_from_url(URL_A), rt_download.entry_from_url(URL_B)
        self.app.add_entries([a, b])
        self.app._downloads_done([{"entry": dict(a), "path": "x.safetensors"}])
        self.assertEqual([e["url"] for e in self.app.queue], [URL_B])
        self.assertEqual([e["url"] for e in rt_download.load_queue()], [URL_B])

    def test_worker_marshals_and_reenables(self):
        got = []

        def job(log, cancel_flag, progress):
            log("hello from the worker")
            progress(snap([row(URL_A, "a.safetensors", MB, 2 * MB, MB)], MB, 2 * MB, MB, 1))
            progress(0.5) ## V2 fraction is ignored
            return 7

        self.assertTrue(self.app.run_worker("Test job", job, got.append))
        self.assertTrue(all(str(b.cget("state")) == "disabled" for b in self.app.actions))
        self.assertFalse(self.app.run_worker("Second", job))
        self.app.worker.join(5)
        self.app.drain()
        self.assertIn("hello from the worker", self.log_text())
        self.assertEqual(got, [7])
        self.assertTrue(all(str(b.cget("state")) == "normal" for b in self.app.actions))
        self.assertEqual((self.app.busy, self.app.status_var.get()), (False, "Ready"))
        self.assertEqual(self.app.active.get_children(), ()) ## cleared when the job ends

    def tree_rows(self):
        return {iid: (self.app.active.item(iid, "text"), tuple(self.app.active.item(iid, "values"))) for iid in self.app.active.get_children()}

    def test_progress_snapshot_status_and_rows(self):
        a, b, c = row(URL_A, "a.safetensors", 3 * MB, 4 * MB, 2 * MB), row(URL_B, "b.safetensors", 0, 0, 0, "queued"), row("k3", "c.safetensors", 1 * MB, 4 * MB, 0, "done")
        self.app.progress(snap([a, b, c], 4 * MB, 8 * MB, 2 * MB, 2))
        self.app.drain()
        self.assertEqual(self.app.status_var.get(), "1 active  2.0 MB/s  50%  0:02 left")
        self.assertEqual(float(self.app.bar.cget("value")), 50.0)
        self.assertEqual(self.tree_rows(), {URL_A: ("a.safetensors", ("4.0 MB", "75%", "2.0 MB/s")),
                                            URL_B: ("b.safetensors", ("?", "0.0 MB", "queued")),
                                            "k3": ("c.safetensors", ("4.0 MB", "25%", "done"))})

        stale = snap([dict(a, done=MB)], MB, 4 * MB, MB, 3)
        fresh = snap([dict(a, done=4 * MB, name="a2.safetensors", speed=0)], 4 * MB, 4 * MB, 0, None)
        self.app.progress(stale)
        self.app.progress(fresh)
        self.assertEqual(self.app.events.qsize(), 1) ## stale snapshot dropped, not queued
        self.app.drain()
        self.assertEqual(self.app.status_var.get(), "1 active  0.0 MB/s  100%")
        self.assertEqual(self.tree_rows(), {URL_A: ("a2.safetensors", ("4.0 MB", "100%", "0.0 MB/s"))})

        self.app.progress(fresh)
        self.app._job_done("Downloading", None, None)
        self.app.drain()
        self.assertEqual((self.app.active.get_children(), self.app.status_var.get()), ((), "Ready"))
        self.assertEqual(str(self.app.bar.cget("mode")), "determinate")

    def test_config_dialog_saves_download_keys(self):
        self.cfg.update(download_dir="", connections=99, parallel_files="junk")
        win = rt_forms.open_config_dialog(self.root, self.cfg)
        self.root.update()
        spins = [w for w in win.winfo_children()[0].winfo_children() if isinstance(w, rt_forms.ttk.Spinbox)]
        self.assertEqual([s.get() for s in spins], ["16", "2"]) ## clamped on open
        spins[0].set("4")
        spins[1].set("9")
        save = [w for w in win.winfo_children()[1].winfo_children() if w.cget("text") == "Save"][0]
        save.invoke()
        with open(rt_config.CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual((saved["download_dir"], saved["connections"], saved["parallel_files"]), ("", 4, 6))

    def test_duplicates_view_and_delete(self):
        data = os.urandom(4096)
        p1 = write_bytes(os.path.join(self.cfg["ckpt_dir"], "same_a.safetensors"), data)
        p2 = write_bytes(os.path.join(self.cfg["lora_dir"], "same_b.safetensors"), data)
        write_bytes(os.path.join(self.cfg["lora_dir"], "same_b.civitai.info"), b"{}")
        write_bytes(os.path.join(self.cfg["vae_dir"], "other.safetensors"), os.urandom(4096))
        groups = rt_library.find_duplicates(self.cfg, log=lambda m: None)
        self.app.db = rt_library.load_library()
        self.app.show_duplicates(groups)
        rows = {path: iid for iid, path in self.app.dupe_rows.items()}
        self.assertEqual(sorted(rows), sorted([p1, p2]))
        deleted = []

        def fake_delete(paths, log=print, db=None):
            deleted.extend(paths)
            for p in paths:
                os.remove(p)
            return []

        self.patch(rt_pruner, "delete_paths", fake_delete)
        self.app.toggle(rows[p1])
        self.app.toggle(rows[p2])
        self.app.delete_dupes()
        self.assertIsNone(self.app.worker) ## every copy marked, blocked
        self.app.toggle(rows[p1])
        self.assertTrue(self.app.tree.item(rows[p1], "text").startswith(rt_gui.MARK_OFF))
        self.app.delete_dupes()
        self.app.worker.join(5)
        self.app.drain()
        self.assertEqual(sorted(deleted), sorted([p2, os.path.join(self.cfg["lora_dir"], "same_b.civitai.info")]))
        self.assertTrue(os.path.exists(p1))
        self.assertEqual(self.app.tree.get_children(), ())

    def test_orphan_count_on_tab(self):
        db = rt_library.load_library()
        rt_library.register(db, write_bytes(os.path.join(self.cfg["ckpt_dir"], "loose.safetensors"), os.urandom(2048)), "checkpoint")
        rt_library.save_library(db)
        self.app.after_asset()
        self.assertEqual(self.app.nb.tab(self.app.repair_tab, "text"), "Repair (1)")
        self.assertIn("1 orphan", self.app.orphan_var.get())

    def test_dialogs_open_and_close(self):
        write_bytes(os.path.join(self.cfg["lora_dir"], "someLora_v1.safetensors"), b"x" * 64)
        db = rt_library.load_library()
        wins = [rt_forms.open_config_dialog(self.root, self.cfg), rt_forms.open_key_dialog(self.root, self.cfg),
                rt_forms.open_url_dialog(self.root, lambda e: None), rt_forms.open_help_dialog(self.root, rt_gui.README_TEXT),
                rt_forms.open_asset_form(self.root, "repair", self.cfg, db, self.app.log), rt_forms.open_asset_form(self.root, "add", self.cfg, db, self.app.log)]
        self.root.update()
        for win in wins: ## each dialog is a live Toplevel
            self.assertIsInstance(win, tk.Toplevel)
            self.assertTrue(win.winfo_exists())
            win.destroy()

    def test_close_saves_geometry(self):
        self.app.on_close()
        with open(rt_config.CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertRegex(saved["window"], r"^\d+x\d+")
        self.assertTrue(re.match(r"^\d+x\d+", self.cfg["window"]))

if __name__ == "__main__":
    unittest.main()
