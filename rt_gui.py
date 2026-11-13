## File Name: rt_gui.py
## Creator: DatJavaClass
## Associated Files: rt_forms.py, rt_download.py, rt_library.py, rt_pruner.py, rt_config.py
## Exposed Vars: TITLE, README_TEXT, MARK_ON, MARK_OFF
## Exposed Funcs: App, build_window, launch_gui, dupe_targets, with_siblings, status_text
## Desc: Main window: menu, notebook tabs, shared log, job workers, and the live download meter with its Active downloads list.
import os, queue, re, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import rt_civitai, rt_config, rt_download, rt_forms, rt_library, rt_pruner

TITLE = "Re:Tentative V3: Stable Diffusion Model Manager"
MIN_W, MIN_H, PUMP_MS = 900, 640, 50
MARK_ON, MARK_OFF = "[x]", "[ ]"
DUPE_COLS = (("#0", "File", 360), ("folder", "Folder", 360), ("size", "Size", 90))
ACTIVE_COLS = (("#0", "File", 420), ("size", "Size", 90), ("done", "Done", 80), ("speed", "Speed", 100))
QUEUE_TYPES = [("Queue files", "*.json *.txt"), ("JSON queue", "*.json"), ("Text, one URL per line", "*.txt"), ("All", "*.*")]
PRUNER_TABS = (("Checkpoints", "ckpt_dir"), ("LoRA", "lora_dir"), ("VAE", "vae_dir"))
REPAIR_NOTE = ("Orphans are library files with no Civitai model and version id. Update Database asks Civitai about "
               "each one by hash. Repair Asset fixes one file by hand; Add Asset labels a file that did not come from Civitai.")

README_TEXT = """\
Re:Tentative V3, Stable Diffusion Model Manager

WHAT IT DOES
  Keeps a local Stable Diffusion model library tidy. It identifies your
  checkpoints and LoRAs on Civitai, renames them to a consistent
  [BaseModel]-Friendly Name (Version) format, downloads a preview image
  when one is missing, sorts new downloads into the right folders, finds
  duplicate files, and prunes models you no longer want.

FIRST TIME SETUP
  1. Tools, Configuration: point the tool at your Checkpoints, LoRA and
     VAE folders, and choose whether file names carry the version.
     The same dialog sets the Staging folder, Connections per file and
     Files at once.
  2. Tools, API Key: paste your Civitai API key. It lets the tool read
     metadata and download gated files and previews.
  New downloads land in the staging folder before they are sorted. Left
  blank it is "Downloaded Content" next to this program. Staging on the
  same drive as the model folders makes sorting instant and keeps big
  files out of synced folders.

DOWNLOAD TAB
  Add URL         Add one URL with optional image URL, tags, name, base
                  model and kind. Civitai URLs need only the URL. Other
                  hosts (HuggingFace, TensorHub) use the extra fields.
  Add Batch       Load a .txt of URLs (one per line) or a saved .json queue.
  Paste, Ctrl+V   Add every http URL on the clipboard to the queue.
                  Ctrl+V works while the queue list has focus.
  Remove          Remove the selected entries (the Delete key works too).
                  Shift and Ctrl click select several.
  Clear           Empty the queue. Asks first only while a download runs.
  Begin Download  Download everything into the staging folder, several
                  files at once, each over several connections. Then
                  write the sidecar and preview for each file, sort, and
                  run Label & Skin. Finished entries drop off the queue
                  and the queue is saved. A cancelled or broken download
                  keeps its partial file and resumes on the next run.
  Active list     The Active downloads list under the queue shows one
                  row per file of the running batch: size, percent done
                  and speed. It clears when the job ends.

LIBRARY TAB
  Label & Skin     Walk the library: identify each model, rename it, and
                   grab a missing preview. Already named files are skipped.
                   Relabel versions also revisits named files that lack a
                   (Version) suffix.
  Sort Only        Move loose files out of Downloaded Content into the
                   right model folder (VAEs, LoRAs, checkpoints).
  Find Duplicates  Hash every model and list files with identical contents,
                   grouped by hash. Click a file to mark it, then Delete
                   Selected. At least one copy in each group must stay.

REPAIR TAB
  The tab shows how many library entries are orphans: files with no
  Civitai model and version id.
  Repair Asset     Fix one file by hand. Search, pick, optionally Pull the
                   metadata from a Civitai URL, then correct the base model,
                   name, version, keywords or preview.
  Add Asset        Label a file that did NOT come from Civitai. Pick it with
                   Load From Local. It goes on the exclusion list so the
                   automatic passes leave it alone.
  Update Database  Register every file, then ask Civitai about each orphan
                   by hash, write its sidecars, grab a preview and rename.

PRUNER TAB
  Browse each folder with a preview and the library details of the
  selected model. Delete removes the model and its sidecars: Recycle Bin
  when send2trash is installed, otherwise permanent.

LOG, PROGRESS, CANCEL
  Every job writes to the shared log. The status label names the running
  job. During a download the bar shows overall progress and the label
  reads active files, speed, percent and time left. Cancel stops the job
  at the next safe point. Only one job runs at a time.

MENUS
  File     Save Queue (reloads next launch), Export Queue (.json keeps every
           field, .txt keeps URLs only), Import, Exit.
  Tools    Configuration, API Key, Open Downloaded Content.
  Help     This text, About.

HOW IDENTIFICATION WORKS
  For each file the tool tries, in order:
    1. The .civitai.info sidecar next to the file.
    2. Civitai, by the hash of the file.
    3. The model page, scraped for a base model, as a last resort.
    4. If nothing works the file is QUARANTINED: left untouched and logged
       to _unresolved.txt. The tool never guesses a base model from a
       filename, so it never writes a wrong label silently.

PREVIEWS
  Many Civitai previews are videos. The tool prefers real stills, asks
  Civitai for the still frame of a video, and checks that what it saves is
  an image. No usable still means no preview rather than a broken one.

FILES THE TOOL KEEPS (next to this program)
  _library.json        Hash database of every model the tool has seen.
  _exclusion_list.txt  Filename stems to skip entirely. Put files that did
                       not come from Civitai here.
  _unresolved.txt      A fresh report each run of files that could not be
                       resolved. Your punch list for Repair.
  _saved_queue.json    The saved download queue.
  sd_mm_config.json    Folders, API key, options and window size.

COMMAND LINE
  python retentative.py labelskin | sort | dupes | update
  runs one job without the window and prints to the console.
  python retentative.py get <url> [more urls]
  downloads without the window, with a one line progress print.

A NOTE ABOUT RENAMING
  Renaming model files changes the names your SD UI stored in older image
  metadata and in saved styles or workflows. After a run, refresh the
  model lists in your UI so it reads the new names.
"""

def dupe_targets(groups, checked):
    """Returns (marked paths, groups with every copy marked)."""
    paths, blocked = [], []
    for group in groups: ## exits once every group is judged
        picked = [p for p in group if p in checked]
        if picked and len(picked) == len(group):
            blocked.append(group)
        paths.extend(picked)
    return paths, blocked

def with_siblings(paths):
    out = []
    for path in paths: ## model first, then its sidecars
        folder, fname = os.path.split(path)
        names = rt_pruner.siblings_of(folder, fname)
        models = {n for n in names if os.path.splitext(n)[1].lower() in rt_config.MODEL_EXTS} - {fname}
        longer = [os.path.splitext(n)[0] for n in models] ## foo.v2 beside foo is another model
        out.append(path)
        out.extend(os.path.join(folder, n) for n in names if n != fname and n not in models and not any(n.startswith(s + ".") for s in longer))
    return list(dict.fromkeys(out))

def _bytes_text(n):
    return f"{n / 1073741824:.2f} GB" if n >= 1073741824 else f"{n / 1048576:.1f} MB"

def _size_text(path):
    try:
        return _bytes_text(os.path.getsize(path))
    except OSError:
        return "?"

def _rate_text(bps):
    return f"{(bps or 0) / 1048576:.1f} MB/s"

def _eta_text(sec):
    m, s = divmod(int(sec), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"

def _done_text(done, total):
    return f"{int(done * 100 / total)}%" if total else _bytes_text(done) ## size unknown, show bytes

def status_text(snap):
    """One line summary of a Meter snapshot."""
    done, total, eta = snap.get("done", 0), snap.get("total", 0), snap.get("eta")
    parts = [f"{snap.get('active', 0)} active", _rate_text(snap.get("speed")), _done_text(done, total)]

    if eta is not None: ## None while any size is unknown
        parts.append(f"{_eta_text(eta)} left")
    return "  ".join(parts)

def _file_values(f):
    done, total, state = f.get("done", 0), f.get("total", 0), f.get("state", "")
    return (_bytes_text(total) if total else "?", _done_text(done, total), _rate_text(f.get("speed")) if state == "active" else state)

def _heads(tree, cols, fixed):
    for col, text, width in cols: ## fixed columns keep their width
        tree.heading(col, text=text, anchor="w")
        tree.column(col, width=width, stretch=col not in fixed)

class App:
    def __init__(self, root, cfg, db):
        self.root, self.cfg, self.db = root, cfg, db
        self.cancel_flag, self.downloading = threading.Event(), threading.Event()
        self.events = queue.SimpleQueue() ## worker to main thread
        self.queue, self.actions, self.dupe_groups = [], [], []
        self.dupe_rows, self.checked = {}, set()
        self.worker, self.job, self._snap = None, "", None
        self.busy, self._closed, self._snap_lock = False, False, threading.Lock()
        root.title(TITLE)
        root.minsize(MIN_W, MIN_H)
        try:
            root.geometry(cfg.get("window") or f"{MIN_W + 100}x{MIN_H + 80}")
        except tk.TclError:
            root.geometry(f"{MIN_W + 100}x{MIN_H + 80}") ## stale or corrupt saved geometry
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_menu()

        panes = ttk.PanedWindow(root, orient="vertical")
        panes.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self.nb = ttk.Notebook(panes)
        panes.add(self.nb, weight=3)
        self._build_download()
        self._build_library()
        self._build_repair()
        self._build_pruner()
        panes.add(self._build_log(panes), weight=2)
        self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._on_tab())

        self.log("Re:Tentative ready.")
        self.log(f"Working folder: {rt_config.SCRIPT_DIR}")
        restored = self.add_entries(rt_download.load_queue())
        if restored:
            self.log(f"Restored {restored} queued entr{'y' if restored == 1 else 'ies'}.")
        self.refresh()
        self._pump_id = root.after(PUMP_MS, self._pump)

    def _build_menu(self):
        bar = tk.Menu(self.root)
        menus = (("File", (("Save Queue", self.save_queue), ("Export Queue...", self.export_queue), ("Import...", self.add_batch), None, ("Exit", self.on_close))),
                 ("Tools", (("Configuration...", self.open_config), ("API Key...", self.open_key), ("Open Downloaded Content", self.open_downloads))),
                 ("Help", (("Help", lambda: rt_forms.open_help_dialog(self.root, README_TEXT)), ("About", self.about))))
        for title, items in menus: ## one cascade per menu
            menu = tk.Menu(bar, tearoff=False)
            for item in items: ## None draws a separator
                if item is None:
                    menu.add_separator()
                else:
                    menu.add_command(label=item[0], command=item[1])
            bar.add_cascade(label=title, menu=menu)
        self.root.config(menu=bar)

    def _tab(self, text):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text=text)
        return tab

    def _button(self, parent, text, cmd, action=True, side="left"):
        b = ttk.Button(parent, text=text, command=cmd)
        b.pack(side=side, padx=3, pady=3)
        if action: ## disabled while a job runs
            self.actions.append(b)
        return b

    def _scrolled(self, parent, widget):
        widget.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(parent, command=widget.yview)
        sb.pack(side="right", fill="y")
        widget.config(yscrollcommand=sb.set)
        return widget

    def _build_download(self):
        tab = self._tab("Download")
        wrap = ttk.LabelFrame(tab, text="Download Queue (Civitai or any direct link)", padding=6)
        wrap.pack(fill="both", expand=True)
        self.listbox = self._scrolled(wrap, tk.Listbox(wrap, selectmode="extended", exportselection=False, activestyle="dotbox"))
        for seq in ("<Control-v>", "<Control-V>"): ## caps lock sends the capital
            self.listbox.bind(seq, self.paste)
        self.listbox.bind("<Delete>", lambda _e: self.remove_selected())
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(6, 0))
        for text, cmd in (("Add URL", self.add_url), ("Add Batch", self.add_batch), ("Paste", self.paste), ("Remove", self.remove_selected), ("Clear", self.clear_queue)):
            self._button(row, text, cmd, action=False)
        self._button(row, "Begin Download", self.begin_download, side="right")

        box = ttk.LabelFrame(tab, text="Active downloads", padding=6)
        box.pack(fill="x", pady=(6, 0))
        self.active = self._scrolled(box, ttk.Treeview(box, columns=("size", "done", "speed"), height=5, selectmode="none"))
        _heads(self.active, ACTIVE_COLS, ("size", "done", "speed"))

    def _build_library(self):
        tab = self._tab("Library")
        row = ttk.Frame(tab)
        row.pack(fill="x")
        self.relabel_var = tk.BooleanVar(value=False)
        self._button(row, "Label & Skin", self.label_skin)
        ttk.Checkbutton(row, text="Relabel versions", variable=self.relabel_var).pack(side="left", padx=(0, 16))
        self._button(row, "Sort Only", self.sort_only)
        self._button(row, "Find Duplicates", self.find_dupes)
        box = ttk.LabelFrame(tab, text="Duplicates: click a file to mark it for deletion", padding=6)
        box.pack(fill="both", expand=True, pady=(8, 0))
        self.tree = self._scrolled(box, ttk.Treeview(box, columns=("folder", "size"), selectmode="browse"))
        _heads(self.tree, DUPE_COLS, ("size",))
        self.tree.bind("<Button-1>", self.on_tree_click)
        bottom = ttk.Frame(tab)
        bottom.pack(fill="x", pady=(6, 0))
        self.dupe_var = tk.StringVar(value="Run Find Duplicates to list files with identical contents.")
        ttk.Label(bottom, textvariable=self.dupe_var, foreground="#555").pack(side="left")
        self._button(bottom, "Delete Selected", self.delete_dupes, side="right")

    def _build_repair(self):
        self.repair_tab = self._tab("Repair")
        self.orphan_var = tk.StringVar()
        ttk.Label(self.repair_tab, textvariable=self.orphan_var).pack(anchor="w")
        ttk.Label(self.repair_tab, text=REPAIR_NOTE, foreground="#555", wraplength=820, justify="left").pack(anchor="w", pady=(4, 8))
        row = ttk.Frame(self.repair_tab)
        row.pack(fill="x")
        for text, cmd in (("Repair Asset", lambda: self.asset_form("repair")), ("Add Asset", lambda: self.asset_form("add")), ("Update Database", self.update_db)):
            self._button(row, text, cmd)

    def _build_pruner(self):
        self.pruner_tab = self._tab("Pruner")
        inner = ttk.Notebook(self.pruner_tab)
        inner.pack(fill="both", expand=True)
        self.pruners = [rt_pruner.PrunerTab(inner, title, lambda k=key: self.cfg.get(k) or "", lambda: self.db, self.log) for title, key in PRUNER_TABS]

    def _build_log(self, parent):
        frame = ttk.LabelFrame(parent, text="Log", padding=6)
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        self.text = self._scrolled(body, tk.Text(body, height=10, wrap="word", state="disabled"))
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(6, 0))
        self.bar = ttk.Progressbar(row, mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(row, textvariable=self.status_var, width=44).pack(side="left", padx=8)
        ttk.Button(row, text="Cancel", command=self.cancel).pack(side="left")
        return frame

    def _write(self, msg):
        self.text.config(state="normal")
        self.text.insert("end", str(msg) + "\n")
        self.text.see("end")
        self.text.config(state="disabled")

    def log(self, msg):
        if threading.current_thread() is threading.main_thread():
            return self._write(msg)
        self.events.put((self._write, (msg,))) ## Tk only on the main thread

    def progress(self, snap):
        if not isinstance(snap, dict):
            return ## only Meter snapshots drive the bar

        with self._snap_lock: ## newest wins, a stale one is dropped
            pending, self._snap = self._snap is not None, snap
        if not pending:
            self.events.put((self._show_progress, ()))

    def _take_snap(self):
        with self._snap_lock:
            snap, self._snap = self._snap, None
        return snap

    def _show_progress(self):
        snap = self._take_snap()
        if not snap:
            return ## job ended first

        done, total = snap.get("done", 0), snap.get("total", 0)
        if total > 0: ## overall percent of known sizes
            self.bar.stop()
            self.bar.config(mode="determinate", value=min(100.0, done * 100 / total))
        elif str(self.bar.cget("mode")) != "indeterminate": ## no size known yet
            self.bar.config(mode="indeterminate")
            self.bar.start(20)

        self.status_var.set(status_text(snap))
        self._show_files(snap.get("files") or [])

    def _show_files(self, files):
        keys = {str(f.get("key")) for f in files}
        self.active.delete(*[iid for iid in self.active.get_children() if iid not in keys])

        for f in files: ## exits after every file row is current
            iid, name, values = str(f.get("key")), f.get("name") or "?", _file_values(f)
            if self.active.exists(iid):
                self.active.item(iid, text=name, values=values)
            else:
                self.active.insert("", "end", iid=iid, text=name, values=values)

    def drain(self, limit=500):
        for _ in range(limit): ## bounded so the UI stays live
            try:
                fn, args = self.events.get_nowait()
            except queue.Empty:
                return
            try:
                fn(*args)
            except Exception as e:
                self._write(f"! UI update failed: {type(e).__name__}: {e}")

    def _pump(self):
        self.drain()
        self._pump_id = self.root.after(PUMP_MS, self._pump)

    def _set_actions(self, state):
        for b in self.actions: ## every job starter button
            b.config(state=state)

    def run_worker(self, name, fn, on_result=None):
        if self.busy:
            messagebox.showinfo("Busy", f"{self.job} is still running. Wait for it or press Cancel.")
            return False
        self.busy, self.job = True, name
        self.cancel_flag.clear()
        self._set_actions("disabled")
        self.status_var.set(name)
        self.bar.config(mode="indeterminate", value=0)
        self.bar.start(20)
        self.log(f"\n{name}...")

        def work():
            result = None
            try:
                result = fn(self.log, self.cancel_flag, self.progress)
            except Exception as e:
                self.log(f"! {name} failed: {type(e).__name__}: {e}")
            finally:
                self.events.put((self._job_done, (name, result, on_result)))

        self.worker = threading.Thread(target=work, name=name, daemon=True)
        self.worker.start()
        return True

    def _job_done(self, name, result, on_result):
        self.busy = False
        self._set_actions("normal")
        self.bar.stop()
        self.bar.config(mode="determinate", value=0)
        self._take_snap() ## a late snapshot must not repaint
        self.active.delete(*self.active.get_children())
        self.status_var.set(f"Cancelled: {name}" if self.cancel_flag.is_set() else "Ready")
        self.db = rt_library.load_library()
        self.refresh()
        if on_result:
            on_result(result)

    def cancel(self):
        if not self.busy:
            return self.log("Nothing to cancel.")
        self.cancel_flag.set()
        self.log("Cancel requested, stopping at the next safe point.")

    def refresh(self):
        files = self.db.get("files", {})
        n = len(rt_library.orphans(self.db))
        self.orphan_var.set(f"Library: {len(files)} file(s), {n} orphan(s).")
        self.nb.tab(self.repair_tab, text=f"Repair ({n})")
        if self.nb.select() == str(self.pruner_tab):
            self._on_tab()

    def _on_tab(self):
        if self.nb.select() == str(self.pruner_tab): ## folder listing only when visible
            for tab in self.pruners:
                tab.rescan()

    def require_paths(self):
        if rt_config.paths_configured(self.cfg):
            return True
        messagebox.showinfo("Configuration needed", "Set your Checkpoints, LoRA and VAE folders in Configuration first.")
        self.open_config()
        return False

    def add_entries(self, entries):
        urls, added = {e["url"] for e in self.queue}, 0
        for e in entries or []: ## skip blanks and duplicates
            if e and e.get("url") and e["url"] not in urls:
                self.queue.append(e)
                urls.add(e["url"])
                added += 1
        self._render_queue()
        return added

    def _render_queue(self):
        self.listbox.delete(0, "end")
        self.listbox.insert("end", *(rt_download.describe(e) for e in self.queue))

    def add_url(self):
        def on_add(entry):
            if self.add_entries([entry]):
                return self.log(f"Queued: {rt_download.describe(entry)}")
            messagebox.showinfo("Duplicate", "That URL is already in the queue.")
        return rt_forms.open_url_dialog(self.root, on_add)

    def add_batch(self):
        path = filedialog.askopenfilename(title="Import a queue: .txt of URLs or .json", initialdir=rt_config.SCRIPT_DIR, filetypes=QUEUE_TYPES)
        if not path:
            return
        entries = rt_download.import_file(path)
        if not entries:
            return messagebox.showwarning("Nothing imported", f"No URLs found in:\n{path}")
        added = self.add_entries(entries)
        self.log(f"Import: added {added}, skipped {len(entries) - added} duplicate(s) from {os.path.basename(path)}.")

    def paste(self, _event=None):
        try:
            clip = self.root.clipboard_get()
        except tk.TclError:
            clip = ""
        urls = [ln.strip() for ln in clip.splitlines() if ln.strip().lower().startswith(("http://", "https://"))]
        if not urls:
            self.log("Paste: the clipboard holds no http URL.")
        else:
            added = self.add_entries([rt_download.entry_from_url(u) for u in urls])
            self.log(f"Paste: added {added}, skipped {len(urls) - added} duplicate(s).")
        return "break" ## stop the default Listbox paste

    def remove_selected(self):
        for i in sorted(self.listbox.curselection(), reverse=True): ## back to front keeps indexes valid
            del self.queue[i]
        self._render_queue()

    def clear_queue(self):
        if self.downloading.is_set() and not messagebox.askyesno(
                "Clear queue?", "A download is running. Clear the queue anyway?\nThe running batch is not stopped; use Cancel for that."):
            return
        self.queue = []
        self._render_queue()
        rt_download.save_queue([])
        self.log("Queue cleared.")

    def save_queue(self):
        ok = rt_download.save_queue(self.queue)
        self.log(f"Queue saved ({len(self.queue)} entries), reloads next launch." if ok else "! Queue save failed.")

    def export_queue(self):
        if not self.queue:
            return messagebox.showinfo("Empty", "The queue is empty, nothing to export.")
        path = filedialog.asksaveasfilename(title="Export queue", initialdir=rt_config.SCRIPT_DIR, defaultextension=".json", filetypes=QUEUE_TYPES[1:])
        if path:
            ok = rt_download.export_queue(self.queue, path)
            self.log(f"Queue exported ({len(self.queue)} entries) to {path}" if ok else f"! Export failed: {path}")

    def begin_download(self):
        if not self.require_paths():
            return
        if not self.queue:
            return messagebox.showinfo("Empty", "The queue is empty.")
        civitai = any(rt_civitai.is_civitai(e["url"]) for e in self.queue)
        if civitai and not self.cfg.get("api_key") and not messagebox.askyesno(
                "No API key", "No Civitai API key is set. Many Civitai downloads need one. Continue anyway?"):
            return
        entries = [dict(e) for e in self.queue]

        def job(log, cancel_flag, progress):
            self.downloading.set()
            try:
                return rt_download.run_queue(entries, self.cfg, log, cancel_flag, progress, on_done=lambda done: self.events.put((self._downloads_done, (done,))))
            finally:
                self.downloading.clear()

        self.run_worker("Downloading", job)

    def _downloads_done(self, finished):
        urls = {f["entry"]["url"] for f in finished or []}
        if not urls:
            return
        self.queue = [e for e in self.queue if e["url"] not in urls]
        self._render_queue()
        saved = rt_download.save_queue(self.queue)
        self.log(f"Queue: {len(urls)} finished entr{'y' if len(urls) == 1 else 'ies'} removed{'' if saved else ', save failed'}.")

    def label_skin(self):
        if not self.require_paths():
            return
        if not self.cfg.get("api_key") and not messagebox.askyesno(
                "No API key", "No API key is set. Files with sidecars still process, but checkpoints without one are quarantined. Continue anyway?"):
            return
        relabel = self.relabel_var.get()
        self.run_worker("Label & Skin", lambda log, cf, pr: rt_library.label_and_skin(self.cfg, log, cf, relabel))

    def sort_only(self):
        if self.require_paths():
            self.run_worker("Sort", lambda log, cf, pr: rt_library.sort_downloads(self.cfg, log, cf))

    def find_dupes(self):
        if self.require_paths():
            self.run_worker("Find Duplicates", lambda log, cf, pr: rt_library.find_duplicates(self.cfg, log, cf), self.show_duplicates)

    def update_db(self):
        if self.require_paths():
            self.run_worker("Update Database", lambda log, cf, pr: rt_library.update_database(self.cfg, log, cf))

    def show_duplicates(self, groups):
        self.tree.delete(*self.tree.get_children())
        self.dupe_groups, self.dupe_rows, self.checked = [list(g) for g in groups or []], {}, set()
        for n, group in enumerate(self.dupe_groups, 1): ## one parent row per hash
            sha, size = rt_library.find_by_path(self.db, group[0])[0] or "unknown hash", _size_text(group[0])
            gid = self.tree.insert("", "end", text=f"Group {n}: {len(group)} copies, sha256 {sha[:12]}", values=("", size), open=True)
            for path in group: ## file rows start unmarked
                iid = self.tree.insert(gid, "end", text=f"{MARK_OFF} {os.path.basename(path)}", values=(os.path.dirname(path), size))
                self.dupe_rows[iid] = path
        self.dupe_var.set(f"{len(self.dupe_groups)} duplicate group(s). Click files to mark them." if self.dupe_groups else "No duplicates found.")

    def on_tree_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid in self.dupe_rows:
            self.toggle(iid)

    def toggle(self, iid):
        path = self.dupe_rows[iid]
        self.checked ^= {path}
        self.tree.item(iid, text=f"{MARK_ON if path in self.checked else MARK_OFF} {os.path.basename(path)}")

    def delete_dupes(self):
        paths, blocked = dupe_targets(self.dupe_groups, self.checked)
        if not paths:
            return messagebox.showinfo("Nothing marked", "Click files in the list to mark them for deletion.")
        if blocked:
            names = "\n".join(os.path.basename(g[0]) for g in blocked)
            return messagebox.showwarning("Keep one copy", f"Every copy is marked in these groups. Unmark at least one file in each:\n\n{names}")
        files = with_siblings(paths)
        mode = "move to the Recycle Bin" if rt_pruner.HAVE_TRASH else "PERMANENTLY delete"
        if not messagebox.askyesno("Confirm delete", f"This will {mode} {len(files)} file(s) for {len(paths)} model(s):\n\n" + "\n".join(files[:20]) + "\n\nProceed?", icon="warning", default="no"):
            return
        self.run_worker("Deleting duplicates", lambda log, cf, pr: rt_pruner.delete_paths(files, log), self._dupes_deleted)

    def _dupes_deleted(self, errors):
        self.show_duplicates([g for g in ([p for p in group if os.path.exists(p)] for group in self.dupe_groups) if len(g) > 1])
        if errors:
            messagebox.showwarning("Some files were not removed", "\n".join(errors))

    def asset_form(self, mode):
        if self.require_paths():
            return rt_forms.open_asset_form(self.root, mode, self.cfg, self.db, self.log, on_done=self.after_asset)

    def after_asset(self):
        self.db = rt_library.load_library()
        self.refresh()

    def open_config(self):
        return rt_forms.open_config_dialog(self.root, self.cfg, on_saved=self.refresh)

    def open_key(self):
        return rt_forms.open_key_dialog(self.root, self.cfg, on_saved=lambda: self.log("API key saved."))

    def open_downloads(self):
        rt_config.ensure_working_files()
        try:
            os.startfile(rt_config.DOWNLOAD_DIR)
        except (AttributeError, OSError):
            messagebox.showinfo("Downloaded Content", rt_config.DOWNLOAD_DIR) ## no shell open here

    def about(self):
        yes = lambda b: "yes" if b else "no"
        messagebox.showinfo("About", f"Re:Tentative V3\nStable Diffusion model manager.\n\nWorking folder:\n{rt_config.SCRIPT_DIR}\n\n"
                                     f"Pillow previews: {yes(rt_pruner.HAVE_PIL)}   Recycle Bin: {yes(rt_pruner.HAVE_TRASH)}")

    def first_run(self):
        if not rt_config.paths_configured(self.cfg):
            messagebox.showinfo("Welcome to Re:Tentative", "First, tell the tool where your model folders are.")
            rt_forms.open_config_dialog(self.root, self.cfg, on_saved=self._after_first_config)
        elif not self.cfg.get("api_key"):
            self.open_key()

    def _after_first_config(self):
        self.refresh()
        if not self.cfg.get("api_key"):
            self.open_key()

    def on_close(self):
        if self.busy and not messagebox.askyesno("Job running", f"{self.job} is still running. Cancel it and exit?", icon="warning"):
            return
        self.cancel_flag.set()
        try:
            geo = self.root.geometry()
            if self.root.state() == "normal" and re.match(r"^\d{3,}x\d{3,}", geo): ## skip maximized and unmapped
                self.cfg["window"] = geo
        except tk.TclError:
            pass
        rt_config.save_config(self.cfg)
        self.destroy()

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.root.after_cancel(self._pump_id)
            self.root.destroy()
        except tk.TclError:
            pass ## already gone

def build_window(root, cfg, db):
    app = App(root, cfg, db)
    rt_civitai.LOG = app.log ## Civitai notices land in the log
    return app

def launch_gui():
    rt_config.ensure_working_files()
    root = tk.Tk()
    app = build_window(root, rt_config.load_config(), rt_library.load_library())
    root.after(300, app.first_run)
    root.mainloop()
    return app
