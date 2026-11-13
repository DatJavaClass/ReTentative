## File Name: rt_forms.py
## Creator: DatJavaClass
## Associated Files: rt_gui.py, rt_config.py, rt_civitai.py, rt_download.py, rt_library.py
## Exposed Vars: PAD, DIR_ROWS, SPIN_ROWS, STAGE_NOTE
## Exposed Funcs: open_config_dialog, open_key_dialog, open_url_dialog, open_help_dialog, open_asset_form
## Desc: The dialogs: configuration (model folders, staging folder, download speed), API key, add URL, help, and the repair and add asset forms.
import os, re, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import rt_civitai, rt_config, rt_download, rt_library

PAD = {"padx": 8, "pady": 4}
DIR_ROWS = (("Checkpoints folder:", "ckpt_dir"), ("LoRA folder:", "lora_dir"), ("VAE folder:", "vae_dir"))
SPIN_ROWS = (("Connections per file:", "connections", 1, 16), ("Files at once:", "parallel_files", 1, 6)) ## label, key, low, high
STAGE_NOTE = "Staging on the same drive as the model folders makes sorting instant, and keeps big files out of synced folders."
KIND_CHOICES = ("", "checkpoint", "lora", "vae")
DEFAULT_BASES = ("Illustrious", "Pony", "NoobAI", "SDXL 1.0", "SD 1.5", "Flux.1 D", "Anima")
MODEL_TYPES = [("Models", "*.safetensors *.ckpt *.pt"), ("All", "*.*")]
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")]
CIVITAI_HINT = "Civitai URL: only the download URL is needed, the rest is pulled from Civitai."
OTHER_HINT = "Other host: give a name and base model so the file is labeled; kind picks the folder."

def _dialog(parent, title):
    win = tk.Toplevel(parent)
    win.title(title)
    win.resizable(False, False)
    win.transient(parent)
    try:
        win.grab_set()
    except tk.TclError:
        pass ## not viewable yet, stays modeless
    return win

def _body(win):
    body = ttk.Frame(win, padding=8)
    body.pack(fill="both", expand=True)
    body.columnconfigure(1, weight=1)
    return body

def _row(frame, r, label, var, width=52, show=""):
    ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", **PAD)
    ent = ttk.Entry(frame, textvariable=var, width=width, show=show)
    ent.grid(row=r, column=1, sticky="we", **PAD)
    return ent

def _buttons(win, pairs):
    bar = ttk.Frame(win, padding=(8, 0, 8, 8))
    bar.pack(side="bottom", fill="x")
    return [_pack_button(bar, text, cmd) for text, cmd in pairs]

def _pack_button(bar, text, cmd):
    b = ttk.Button(bar, text=text, command=cmd)
    b.pack(side="left", padx=4)
    return b

def _pick_dir(win, var, title):
    d = filedialog.askdirectory(parent=win, title=f"Select {title}", initialdir=var.get() or rt_config.SCRIPT_DIR)
    if d:
        var.set(os.path.normpath(d))

def _save_or_warn(win, cfg):
    if rt_config.save_config(cfg):
        return True
    messagebox.showerror("Save failed", f"Could not write {rt_config.CONFIG_PATH}", parent=win)
    return False

def _bg(win, fn, done):
    """Run fn off the UI thread, then done(result, error)."""
    box = {}

    def work():
        try:
            box["result"] = fn()
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()

    def poll():
        try:
            if t.is_alive():
                return win.after(100, poll)
            if win.winfo_exists():
                done(box.get("result"), box.get("error"))
        except tk.TclError:
            pass ## dialog closed mid job

    win.after(100, poll)

def open_config_dialog(parent, cfg, on_saved=None):
    win = _dialog(parent, "Configuration")
    body = _body(win)
    dirs = {key: tk.StringVar(value=cfg.get(key, "")) for _, key in DIR_ROWS}
    for r, (label, key) in enumerate(DIR_ROWS): ## one row per model folder
        _row(body, r, label, dirs[key], 60)
        ttk.Button(body, text="Browse", command=lambda v=dirs[key], t=label[:-1]: _pick_dir(win, v, t)).grid(row=r, column=2, **PAD)
    ver_var = tk.BooleanVar(value=bool(cfg.get("add_version", True)))
    ttk.Checkbutton(body, text="Add the version to file names, as [Pony]-Name (v2.0)", variable=ver_var).grid(row=3, column=0, columnspan=3, sticky="w", **PAD)

    dirs["download_dir"] = tk.StringVar(value=cfg.get("download_dir", "")) ## blank means beside the program
    _row(body, 4, "Staging folder:", dirs["download_dir"], 60)
    ttk.Button(body, text="Browse", command=lambda: _pick_dir(win, dirs["download_dir"], "Staging folder")).grid(row=4, column=2, **PAD)
    ttk.Label(body, text=STAGE_NOTE, foreground="#555").grid(row=5, column=0, columnspan=3, sticky="w", **PAD)

    spins = {row[1]: tk.StringVar(value=str(n)) for row, n in zip(SPIN_ROWS, rt_config.clamp_speed(cfg))}
    for r, (label, key, lo, hi) in enumerate(SPIN_ROWS, 6): ## one Spinbox row per speed knob
        ttk.Label(body, text=label).grid(row=r, column=0, sticky="w", **PAD)
        ttk.Spinbox(body, from_=lo, to=hi, increment=1, textvariable=spins[key], width=6).grid(row=r, column=1, sticky="w", **PAD)

    def save():
        for key, var in dirs.items(): ## quotes from Explorer copy paste
            cfg[key] = var.get().strip().strip('"')
        cfg.update({key: var.get() for key, var in spins.items()})
        cfg["connections"], cfg["parallel_files"] = rt_config.clamp_speed(cfg) ## typed junk falls back to defaults
        cfg["add_version"] = ver_var.get()
        if not _save_or_warn(win, cfg):
            return
        missing = [cfg[key] or "(blank)" for _, key in DIR_ROWS if not os.path.isdir(cfg[key])]
        if missing:
            messagebox.showwarning("Saved with warnings", "Saved, but these folders do not exist:\n" + "\n".join(missing), parent=win)
        win.destroy()
        if on_saved:
            on_saved()

    _buttons(win, (("Save", save), ("Cancel", win.destroy)))
    return win

def open_key_dialog(parent, cfg, on_saved=None):
    win = _dialog(parent, "Civitai API Key")
    body = _body(win)
    key_var, show_var = tk.StringVar(value=cfg.get("api_key", "")), tk.BooleanVar(value=False)
    ent = _row(body, 0, "API key:", key_var, 52, "*")
    ttk.Checkbutton(body, text="Show", variable=show_var, command=lambda: ent.config(show="" if show_var.get() else "*")).grid(row=0, column=2, **PAD)
    ttk.Label(body, text="civitai.com, Account Settings, API Keys. Leave blank to skip for now.", foreground="#555").grid(row=1, column=0, columnspan=3, sticky="w", **PAD)
    ent.focus_set()

    def save():
        cfg["api_key"] = key_var.get().strip()
        if not _save_or_warn(win, cfg):
            return
        rt_civitai.key_rejected = False ## a new key gets a fresh try
        win.destroy()
        if on_saved:
            on_saved()

    _buttons(win, (("Save", save), ("Cancel", win.destroy)))
    win.bind("<Return>", lambda _e: save())
    return win

def open_url_dialog(parent, on_add):
    win = _dialog(parent, "Add URL")
    body = _body(win)
    fields = (("Download URL:", "url"), ("Image URL:", "image_url"), ("Tags (comma):", "tags"), ("Name:", "name"), ("Base model:", "base_model"))
    v = {key: tk.StringVar() for _, key in fields}
    for r, (label, key) in enumerate(fields): ## one entry row per field
        _row(body, r, label, v[key], 64)
    kind_var, hint = tk.StringVar(), tk.StringVar(value=CIVITAI_HINT)
    ttk.Label(body, text="Kind:").grid(row=5, column=0, sticky="w", **PAD)
    ttk.Combobox(body, textvariable=kind_var, values=KIND_CHOICES, state="readonly", width=16).grid(row=5, column=1, sticky="w", **PAD)
    ttk.Label(body, textvariable=hint, foreground="#555", wraplength=560, justify="left").grid(row=6, column=0, columnspan=2, sticky="w", **PAD)
    v["url"].trace_add("write", lambda *_: hint.set(CIVITAI_HINT if rt_civitai.is_civitai(v["url"].get()) or not v["url"].get().strip() else OTHER_HINT))
    try:
        clip = win.clipboard_get().strip()
    except tk.TclError:
        clip = ""
    if clip.lower().startswith(("http://", "https://")) and "\n" not in clip:
        v["url"].set(clip) ## prefill from the clipboard

    def add():
        url = v["url"].get().strip()
        if not url.lower().startswith(("http://", "https://")):
            return messagebox.showwarning("No URL", "Enter a download URL starting with http.", parent=win)
        bare = not (v["name"].get().strip() and v["base_model"].get().strip())
        if not rt_civitai.is_civitai(url) and bare and not messagebox.askyesno(
                "Missing details", "Without a name and base model the file keeps its download name and gets no label. Add anyway?", parent=win):
            return
        on_add(rt_download.entry_from_url(url, v["image_url"].get(), v["tags"].get(), v["name"].get(), v["base_model"].get(), kind_var.get()))
        win.destroy()

    _buttons(win, (("Add", add), ("Cancel", win.destroy)))
    return win

def open_help_dialog(parent, text):
    win = tk.Toplevel(parent)
    win.title("Re:Tentative Help")
    win.geometry("760x680")
    win.transient(parent)
    frame = ttk.Frame(win, padding=8)
    frame.pack(fill="both", expand=True)
    txt = tk.Text(frame, wrap="word", font=("Consolas", 10))
    txt.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    sb.pack(side="right", fill="y")
    txt.config(yscrollcommand=sb.set)
    txt.insert("1.0", text)
    txt.config(state="disabled")
    return win

def _base_choices(db):
    found = {e.get("base_model") for e in (db or {}).get("files", {}).values() if e.get("base_model")}
    return sorted(found | set(DEFAULT_BASES), key=str.lower)

def _library_files(cfg):
    return [(p, kind) for folder, kind in rt_config.folders(cfg) for p in rt_library.iter_model_files(folder)]

def _entry_for(db, path):
    try:
        return rt_library.find_by_path(db, path)[1] if db else None
    except Exception:
        return None

def _is_orphan(db, path):
    entry = _entry_for(db, path)
    return entry is None or not entry.get("model_id") or not entry.get("version_id")

def _friendly_from_stem(stem):
    bare = re.sub(r"\s*\([^()]*\)\s*$", "", re.sub(r"^\[[^\]]*\]-?", "", stem)).strip()
    return bare if rt_library.already_formatted(stem) else rt_library.decamel(bare)

def _status_text(path, db):
    entry, yes = _entry_for(db, path), lambda b: "yes" if b else "NO"
    state = "not registered" if entry is None else ("orphan" if _is_orphan(db, path) else "resolved")
    return (f"File: {os.path.basename(path)}\nSidecar: {yes(os.path.isfile(rt_library.sidecar_path(path)))}   "
            f"Preview: {yes(rt_library.has_preview(path))}   Library: {state}")

def _rec_from_entry(entry):
    keys = ("model_id", "version_id", "model_name", "version_name", "base_model")
    rec = {k: (entry or {}).get(k) for k in keys}
    rec.update(tags=list((entry or {}).get("tags") or []), images=[], source="library", raw={})
    return rec

def _first_dir(cfg, keys):
    return next((cfg[k] for k in keys if cfg.get(k) and os.path.isdir(cfg[k])), rt_config.SCRIPT_DIR)

def open_asset_form(parent, mode, cfg, db, log, on_done=None):
    repair = mode == "repair"
    win = _dialog(parent, "Repair Asset" if repair else "Add Asset")
    body = _body(win)
    v = {k: tk.StringVar() for k in ("search", "file", "url", "base", "friendly", "version", "keywords", "preview", "status", "result")}
    orphan_var, excl_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
    state = {"path": None, "rec": None, "shown": [], "files": _library_files(cfg) if repair else []}
    widgets = {}

    def update_result(*_):
        friendly = v["friendly"].get().strip()
        if not friendly:
            return v["result"].set("(enter a friendly name)")
        ext = os.path.splitext(state["path"] or "x.safetensors")[1]
        tag = rt_civitai.tag_for_base_model(v["base"].get())
        v["result"].set(rt_library.build_new_stem(tag, friendly, v["version"].get(), "", cfg.get("add_version", True)) + ext)

    def fill(rec):
        for key, field in (("base", "base_model"), ("friendly", "model_name"), ("version", "version_name")):
            if rec.get(field):
                v[key].set(rec[field])
        if rec.get("tags"):
            v["keywords"].set(", ".join(rec["tags"]))
        update_result()

    def refresh(*_):
        q, only = v["search"].get().strip().lower(), orphan_var.get()
        state["shown"] = [(p, k) for p, k in state["files"] if q in os.path.basename(p).lower() and (not only or _is_orphan(db, p))]
        widgets["list"].delete(0, "end")
        widgets["list"].insert("end", *(f"{k:<11}{os.path.basename(p)}" for p, k in state["shown"]))

    def on_pick(_e=None):
        sel = widgets["list"].curselection()
        if not sel:
            return
        path = state["shown"][sel[0]][0]
        entry, sidecar = _entry_for(db, path), rt_library.read_sidecar(path)
        state["path"], state["rec"] = path, sidecar or _rec_from_entry(entry)
        for key in ("base", "friendly", "version", "keywords", "preview", "url"): ## clear the last pick
            v[key].set("")
        v["status"].set(_status_text(path, db))
        v["url"].set((entry or {}).get("source_url") or "")
        fill(state["rec"])
        if not v["friendly"].get():
            v["friendly"].set(_friendly_from_stem(os.path.splitext(os.path.basename(path))[0]))

    def pulled(rec, err):
        if err or not rec:
            v["status"].set("Pull failed.")
            return messagebox.showwarning("Pull failed", f"Could not fetch metadata from that URL.\n{err or rt_civitai.last_error}", parent=win)
        state["rec"] = rec
        fill(rec)
        cands = rt_civitai.preview_candidates(rec.get("images"))
        if cands and not v["preview"].get().strip():
            v["preview"].set(cands[0])
        v["status"].set(f"Pulled: {rec.get('model_name')} ({rec.get('version_name')}) [{rec.get('base_model')}]")

    def pull():
        url = v["url"].get().strip()
        if rt_civitai.parse_url(url)[0] is None:
            return messagebox.showinfo("No URL", "Paste a Civitai model or version URL first.", parent=win)
        v["status"].set("Pulling metadata...")
        _bg(win, lambda: rt_civitai.fetch_by_url(url, cfg.get("api_key") or None), pulled)

    def load_local():
        f = filedialog.askopenfilename(parent=win, title="Load model file from local disk", initialdir=_first_dir(cfg, ("lora_dir", "ckpt_dir")), filetypes=MODEL_TYPES)
        if not f:
            return
        state["path"] = os.path.normpath(f)
        v["file"].set(os.path.basename(f))
        if not v["friendly"].get().strip():
            v["friendly"].set(_friendly_from_stem(os.path.splitext(os.path.basename(f))[0]))
        update_result()

    def browse_preview():
        start = os.path.dirname(state["path"]) if state["path"] else rt_config.SCRIPT_DIR
        f = filedialog.askopenfilename(parent=win, title="Select preview image", initialdir=start, filetypes=IMAGE_TYPES)
        if f:
            v["preview"].set(os.path.normpath(f))

    def applied(result, err):
        widgets["apply"].config(state="normal")
        ok, msg = result if result else (False, f"{type(err).__name__}: {err}")
        log(f"[{mode}] {msg}")
        if not ok:
            return messagebox.showerror("Failed", msg, parent=win)
        messagebox.showinfo("Done", msg, parent=win)
        win.destroy()
        if on_done:
            on_done()

    def apply():
        path, base, friendly = state["path"], v["base"].get().strip(), v["friendly"].get().strip()
        if not path:
            return messagebox.showwarning("No asset", "Select an asset to repair." if repair else "Click Load From Local and choose the model file.", parent=win)
        if not base or not friendly:
            return messagebox.showwarning("Missing details", "Enter a base model and a friendly name.", parent=win)
        rec = dict(state["rec"] or {})
        rec.update(base_model=base, model_name=friendly, version_name=v["version"].get().strip() or None,
                   tags=[t.strip() for t in v["keywords"].get().split(",") if t.strip()], source=rec.get("source") or "manual")
        rec.setdefault("images", [])
        preview, exclude = v["preview"].get().strip(), excl_var.get() or not repair ## add always excludes
        widgets["apply"].config(state="disabled")
        _bg(win, lambda: rt_library.apply_manual_asset(path, rec, preview, cfg, log, exclude), applied)

    if repair:
        _row(body, 0, "Search asset:", v["search"])
        ttk.Checkbutton(body, text="Orphans only", variable=orphan_var, command=refresh).grid(row=0, column=2, sticky="w", **PAD)
        wrap = ttk.Frame(body)
        wrap.grid(row=1, column=0, columnspan=3, sticky="nsew", **PAD)
        widgets["list"] = tk.Listbox(wrap, height=9, exportselection=False, font=("Consolas", 9))
        widgets["list"].pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(wrap, command=widgets["list"].yview)
        sb.pack(side="right", fill="y")
        widgets["list"].config(yscrollcommand=sb.set)
        widgets["list"].bind("<<ListboxSelect>>", on_pick)
        ttk.Label(body, textvariable=v["status"], foreground="#555", justify="left", wraplength=560).grid(row=2, column=0, columnspan=3, sticky="w", **PAD)
        _row(body, 3, "Civitai URL:", v["url"])
        ttk.Button(body, text="Pull", command=pull).grid(row=3, column=2, sticky="w", **PAD)
        v["search"].trace_add("write", refresh)
        refresh()
    else:
        _row(body, 0, "Model file:", v["file"]).config(state="readonly")
        ttk.Button(body, text="Load From Local", command=load_local).grid(row=0, column=2, sticky="w", **PAD)
    top = 4 if repair else 1

    ttk.Label(body, text="Base model:").grid(row=top, column=0, sticky="w", **PAD)
    ttk.Combobox(body, textvariable=v["base"], values=_base_choices(db), width=50).grid(row=top, column=1, sticky="we", **PAD)
    for r, (label, key) in enumerate((("Friendly name:", "friendly"), ("Version:", "version"), ("Keywords (comma):", "keywords")), top + 1):
        _row(body, r, label, v[key])
    _row(body, top + 4, "Preview (URL or file):" if repair else "Preview image (file):", v["preview"])
    ttk.Button(body, text="Browse", command=browse_preview).grid(row=top + 4, column=2, sticky="w", **PAD)
    if repair:
        ttk.Checkbutton(body, text="Add to the exclusion list (automatic passes skip it)", variable=excl_var).grid(row=top + 5, column=0, columnspan=3, sticky="w", **PAD)
    ttk.Label(body, text="Result filename:").grid(row=top + 6, column=0, sticky="w", **PAD)
    ttk.Label(body, textvariable=v["result"], foreground="#0a6", wraplength=460, justify="left").grid(row=top + 6, column=1, columnspan=2, sticky="w", **PAD)
    for key in ("base", "friendly", "version"): ## live result name
        v[key].trace_add("write", update_result)

    widgets["apply"] = _buttons(win, (("Attempt Repair" if repair else "Attempt Add", apply), ("Cancel", win.destroy)))[0]
    update_result()
    return win
