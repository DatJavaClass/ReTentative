## PrunerTab: browse, preview, delete with siblings
import os
import tkinter as tk
from tkinter import messagebox, ttk

import rt_config
import rt_library

try:
    from PIL import Image, ImageTk
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

try:
    from send2trash import send2trash
    HAVE_TRASH = True
except Exception:
    HAVE_TRASH = False

PREVIEW_EXTS = [".preview.png", ".preview.jpg", ".preview.jpeg", ".preview.webp",
                ".png", ".jpg", ".jpeg", ".webp", ".gif"]
PREVIEW_MAX = 360 ## longest preview edge, pixels
NO_META = "Not in library"

def list_model_files(folder):
    try:
        return sorted((f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in rt_config.MODEL_EXTS), key=str.lower)
    except OSError:
        return []

def find_preview(folder, model_fname):
    stem = os.path.splitext(model_fname)[0]
    return next((p for p in (os.path.join(folder, stem + e) for e in PREVIEW_EXTS) if os.path.isfile(p)), None)

def siblings_of(folder, model_fname): ## model plus sidecars, never other models
    return sorted([model_fname] + rt_library._siblings(os.path.join(folder, model_fname)), key=str.lower)

def _drop_entries(paths, db, log):
    models = [p for p in paths if os.path.splitext(p)[1].lower() in rt_config.MODEL_EXTS]
    if not models:
        return
    try:
        db = db if db is not None else rt_library.load_library()
        dropped = 0
        for p in models: ## one library entry per model
            sha, _ = rt_library.find_by_path(db, p)
            if sha and db.get("files", {}).pop(sha, None) is not None:
                dropped += 1
        if dropped:
            rt_library.save_library(db)
            log(f"  library: dropped {dropped} entr{'y' if dropped == 1 else 'ies'}")
    except Exception as e:
        log(f"  ! library update failed: {e}")

def delete_paths(paths, log=print, db=None):
    """Trash or remove paths; returns error strings."""
    errors, gone = [], []
    for p in paths:
        try:
            send2trash(p) if HAVE_TRASH else os.remove(p)
            gone.append(p)
            log(f"  {'trashed' if HAVE_TRASH else 'deleted'}: {os.path.basename(p)}")
        except Exception as e:
            errors.append(f"{os.path.basename(p)}: {e}")
            log(f"  ! delete failed: {os.path.basename(p)}: {e}")
    _drop_entries(gone, db, log)
    return errors

class PrunerTab:
    def __init__(self, notebook, title, folder_getter, db_getter, log):
        self.folder_getter, self.db_getter, self.log = folder_getter, db_getter, log
        self.all_files, self.shown = [], []
        self._preview_img = None ## keep a ref against GC
        self.frame = ttk.Frame(notebook, padding=8)
        notebook.add(self.frame, text=title)

        left = ttk.Frame(self.frame)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.LabelFrame(self.frame, text="Preview", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.frame.columnconfigure(0, weight=3)
        self.frame.columnconfigure(1, weight=2)
        self.frame.rowconfigure(0, weight=1)

        search_row = ttk.Frame(left)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(search_row, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.search_var.trace_add("write", lambda *_: self.apply_filter())

        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True, pady=6)
        self.listbox = tk.Listbox(list_wrap, activestyle="dotbox", exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_wrap, command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.on_select())

        self.status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_var, foreground="#555").pack(fill="x")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        for text, cmd in (("Previous", lambda: self.nav(-1)), ("Delete", self.delete_selected), ("Next", lambda: self.nav(1))):
            ttk.Button(btns, text=text, command=cmd).pack(side="left", expand=True, fill="x", padx=2)

        self.preview_img_lbl = ttk.Label(right, anchor="center")
        self.preview_img_lbl.pack(fill="both", expand=True)
        self.preview_cap = ttk.Label(right, text="Preview N/A", anchor="center", wraplength=PREVIEW_MAX)
        self.preview_cap.pack(fill="x")
        meta = ttk.LabelFrame(right, text="Library", padding=6)
        meta.pack(fill="x", pady=(8, 0))
        self.meta_var = tk.StringVar(value=NO_META)
        ttk.Label(meta, textvariable=self.meta_var, justify="left", wraplength=PREVIEW_MAX).pack(fill="x")

    def rescan(self):
        self.all_files = list_model_files(self.folder_getter())
        self.apply_filter()

    def apply_filter(self):
        q, folder = self.search_var.get().strip().lower(), self.folder_getter()
        self.shown = [f for f in self.all_files if q in f.lower()]
        self.listbox.delete(0, "end")
        self.listbox.insert("end", *self.shown)
        if not folder:
            self.status_var.set("No folder set. Open Configuration.")
        elif not os.path.isdir(folder):
            self.status_var.set(f"Folder not found: {folder}")
        else:
            mode = "Recycle Bin" if HAVE_TRASH else "PERMANENT"
            self.status_var.set(f"{len(self.shown)} shown / {len(self.all_files)} total   Delete: {mode}")
        self.clear_preview()

    def selected_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def select_index(self, idx):
        if not self.shown:
            return
        idx = max(0, min(idx, len(self.shown) - 1))
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.listbox.see(idx)
        self.on_select()

    def nav(self, delta):
        cur = self.selected_index()
        self.select_index(0 if cur is None else cur + delta)

    def on_select(self):
        idx = self.selected_index()
        if idx is None or idx >= len(self.shown):
            return self.clear_preview()
        folder, fname = self.folder_getter(), self.shown[idx]
        self.show_preview(find_preview(folder, fname), fname)
        self.show_meta(os.path.join(folder, fname))

    def show_meta(self, path):
        try:
            db = self.db_getter()
            _, entry = rt_library.find_by_path(db, path) if db else (None, None)
        except Exception:
            entry = None
        if not entry:
            return self.meta_var.set(NO_META)
        rows = (("Model", entry.get("model_name")), ("Version", entry.get("version_name")),
                ("Base", entry.get("base_model")), ("Tags", ", ".join(entry.get("tags") or [])))
        self.meta_var.set("\n".join(f"{k}: {v or '-'}" for k, v in rows))

    def clear_preview(self):
        self._preview_img = None
        self.preview_img_lbl.config(image="")
        self.preview_cap.config(text="Preview N/A")
        self.meta_var.set(NO_META)

    def show_preview(self, path, model_fname):
        img = self._load_image(path) if path else None
        self._preview_img = img
        self.preview_img_lbl.config(image=img or "")
        if img or not path:
            return self.preview_cap.config(text=model_fname if img else "Preview N/A")
        ext = os.path.splitext(path)[1].lower()
        self.preview_cap.config(text=f"{model_fname}\n(preview is {ext}; install Pillow to view)")

    def _load_image(self, path):
        if HAVE_PIL:
            try:
                im = Image.open(path)
                im.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
                return ImageTk.PhotoImage(im)
            except Exception:
                return None
        if os.path.splitext(path)[1].lower() not in (".png", ".gif"):
            return None ## Tk alone reads png and gif
        try:
            img = tk.PhotoImage(file=path)
            factor = max(1, img.width() // PREVIEW_MAX, img.height() // PREVIEW_MAX)
            return img.subsample(factor, factor) if factor > 1 else img
        except Exception:
            return None

    def delete_selected(self):
        idx = self.selected_index()
        if idx is None or idx >= len(self.shown):
            return messagebox.showinfo("Nothing selected", "Select a model to delete.")
        folder, fname = self.folder_getter(), self.shown[idx]
        sibs = siblings_of(folder, fname)
        mode = "move to the Recycle Bin" if HAVE_TRASH else "PERMANENTLY delete"
        listing = "\n".join("    " + s for s in sibs)
        if not messagebox.askyesno("Confirm delete", f"This will {mode} {len(sibs)} file(s) for:\n\n  {fname}\n\n{listing}\n\nProceed?",
                                   icon="warning", default="no"):
            return
        self.log(f"Pruning {fname}")
        errors = delete_paths([os.path.join(folder, s) for s in sibs], self.log, self.db_getter())
        self.rescan()
        if self.shown:
            self.select_index(min(idx, len(self.shown) - 1)) ## neighbor at the old spot
        if errors:
            messagebox.showwarning("Some files were not removed", "\n".join(errors))
