#!/usr/bin/env python3
"""One-shot repair for Re:Tentative's double-prefix naming bug. See repo README.

Bare launch = GUI folder picker; folder args run headless (dry-run, --apply to
rename). Peels a leading "[X]-" tag off the model file and siblings, never clobbers.
"""

import os
import re
import sys

MODEL_EXTS = {".safetensors", ".ckpt", ".pt"}

## Bug signature: leading "[tag]-" dash-joined to another "[".
## True names space after "]", never match.
_DOUBLE_PREFIX = re.compile(r"^\[[^\[\]]*\]-(?=\[)")


def corrected_stem(stem):
    """Peel every leading "[X]-" prefix; loops so triples collapse too."""
    new = stem
    while _DOUBLE_PREFIX.match(new):
        new = _DOUBLE_PREFIX.sub("", new, count=1)
    return new


def siblings_of(folder, old_stem):
    """Model file plus its sidecars (old_stem or old_stem + ".")."""
    for fname in os.listdir(folder):
        if fname == old_stem or fname.startswith(old_stem + "."):
            yield fname


def plan_folder(folder, log=print):
    """Return (renames, collisions) for one folder; collisions = target exists."""
    renames, collisions = [], []
    if not folder:
        return renames, collisions
    if not os.path.isdir(folder):
        log(f"  (skip, not a directory: {folder})")
        return renames, collisions

    ## Key off model files; sidecars follow their model.
    model_stems = set()
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in MODEL_EXTS:
            model_stems.add(stem)

    for old_stem in sorted(model_stems, key=str.lower):
        new_stem = corrected_stem(old_stem)
        if new_stem == old_stem:
            continue ## not double-prefixed, leave it alone
        for fname in siblings_of(folder, old_stem):
            suffix = fname[len(old_stem):] ## ".safetensors", ".preview.png", ...
            dst = new_stem + suffix
            if os.path.exists(os.path.join(folder, dst)):
                collisions.append((fname, dst))
            else:
                renames.append((fname, dst))
    return renames, collisions


def scan(dirs, log=print):
    """Dry-run scan; logs a preview, returns per-folder results."""
    results = []
    for folder in dirs:
        if not folder:
            continue
        log(f"[{folder}]")
        renames, collisions = plan_folder(folder, log)
        for src, dst in renames:
            log(f"  {src}")
            log(f"    -> {dst}")
        for src, dst in collisions:
            log(f"  SKIP (target exists): {src}  ->  {dst}")
        if not renames and not collisions:
            log("  nothing to fix.")
        log("")
        results.append({"folder": folder, "renames": renames, "collisions": collisions})
    return results


def apply_results(results, log=print):
    """Perform the renames from scan(); returns (done, failed)."""
    done = failed = 0
    for r in results:
        folder = r["folder"]
        for src, dst in r["renames"]:
            try:
                os.rename(os.path.join(folder, src), os.path.join(folder, dst))
                done += 1
            except OSError as e:
                log(f"  ! failed: {src} -> {dst}: {e.__class__.__name__}: {e}")
                failed += 1
    return done, failed


def _counts(results):
    renamed = sum(len(r["renames"]) for r in results)
    collisions = sum(len(r["collisions"]) for r in results)
    return renamed, collisions


## cli

def main_cli(argv):
    apply = "--apply" in argv
    dirs = [a for a in argv if not a.startswith("-")]
    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== fix_double_prefix ({mode}) ===\n")

    results = scan(dirs)
    renamed, collisions = _counts(results)

    print("=== SUMMARY ===")
    if apply:
        done, failed = apply_results(results)
        print(f"  files renamed      : {done}")
        print(f"  renames failed     : {failed}")
        print(f"  collisions skipped : {collisions}")
        if collisions:
            print("\nSome items were skipped because a correctly-named file already "
                  "exists at the target. Resolve those by hand.")
    else:
        print(f"  files to rename    : {renamed}")
        print(f"  collisions skipped : {collisions}")
        print("\nDry run only. Re-run with --apply (or launch with no arguments "
              "for the window) to perform these renames.")


## gui

def main_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("Re:Tentative  -  Double-Prefix Fix")
    root.minsize(660, 480)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(3, weight=1) ## the log pane grows

    intro = (
        "Repairs the old double-prefix naming bug: names like "
        "\"[Anima]-[Anima] Canopy\" that picked up a second base-model tag.\n"
        "Pick your Checkpoint and LoRA folders, then click Run Fix. You will see "
        "exactly what will be renamed and confirm before anything changes."
    )
    tk.Label(root, text=intro, justify="left", anchor="w", wraplength=620).grid(
        row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    ckpt_var, lora_var = tk.StringVar(), tk.StringVar()

    def _folder_row(parent, label, var):
        frame = tk.Frame(parent)
        frame.columnconfigure(1, weight=1)
        tk.Label(frame, text=label, width=16, anchor="w").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=var).grid(row=0, column=1, sticky="ew", padx=(0, 6))

        def browse():
            start = var.get().strip() or None
            chosen = filedialog.askdirectory(title="Choose " + label.rstrip(":"),
                                             initialdir=start if start and os.path.isdir(start) else None)
            if chosen:
                var.set(chosen)

        tk.Button(frame, text="Browse...", command=browse).grid(row=0, column=2)
        return frame

    _folder_row(root, "Checkpoint folder:", ckpt_var).grid(
        row=1, column=0, sticky="ew", padx=12, pady=2)
    _folder_row(root, "LoRA folder:", lora_var).grid(
        row=2, column=0, sticky="ew", padx=12, pady=2)

    log = scrolledtext.ScrolledText(root, height=16, wrap="none", state="disabled")
    log.grid(row=3, column=0, sticky="nsew", padx=12, pady=(8, 4))

    def log_write(msg=""):
        log.configure(state="normal")
        log.insert("end", str(msg) + "\n")
        log.see("end")
        log.configure(state="disabled")
        root.update_idletasks()

    def clear_log():
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")

    def run_fix():
        ckpt, lora = ckpt_var.get().strip(), lora_var.get().strip()
        dirs = [d for d in (ckpt, lora) if d]
        if not dirs:
            messagebox.showerror("No folders",
                                 "Pick at least one folder (Checkpoint or LoRA).")
            return
        bad = [d for d in dirs if not os.path.isdir(d)]
        if bad:
            messagebox.showerror("Not a folder",
                                 "These paths are not folders:\n" + "\n".join(bad))
            return

        clear_log()
        log_write("=== Preview (nothing has been renamed yet) ===\n")
        results = scan(dirs, log_write)
        renamed, collisions = _counts(results)

        if renamed == 0:
            note = "No double-prefixed files found."
            if collisions:
                note += f"\n({collisions} name collision(s) noted above.)"
            messagebox.showinfo("Nothing to fix", note)
            return

        prompt = f"Rename {renamed} file(s)?"
        if collisions:
            prompt += (f"\n\n{collisions} will be skipped because a correctly-named "
                       "file already exists at the target.")
        if not messagebox.askyesno("Confirm Run Fix", prompt):
            log_write("Cancelled. Nothing was renamed.")
            return

        done, failed = apply_results(results, log_write)
        log_write("")
        log_write("=== DONE ===")
        log_write(f"  renamed: {done}    failed: {failed}    "
                  f"skipped (collision): {collisions}")
        summary = f"Renamed {done} file(s)."
        if failed:
            summary += f"\n{failed} failed (see the log)."
        if collisions:
            summary += f"\n{collisions} skipped (name already existed)."
        messagebox.showinfo("Run Fix complete", summary)

    btns = tk.Frame(root)
    btns.grid(row=4, column=0, sticky="e", padx=12, pady=(0, 12))
    tk.Button(btns, text="Run Fix", command=run_fix, width=12,
              default="active").grid(row=0, column=0)

    root.mainloop()


if __name__ == "__main__":
    ## Folder args -> CLI. No args -> GUI.
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    if positional:
        main_cli(sys.argv[1:])
    else:
        main_gui()
