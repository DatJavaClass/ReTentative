## File Name: retentative.py
## Creator: DatJavaClass
## Associated Files: rt_config.py, rt_library.py, rt_download.py, rt_gui.py
## Exposed Vars: USAGE, VERBS
## Exposed Funcs: main
## Desc: Entry point. Opens the GUI, or runs a headless verb: labelskin, sort, dupes, update, or get to download urls with a one line progress readout.
import sys, threading
import rt_config, rt_library

USAGE = "Usage: python retentative.py [gui | labelskin | sort | dupes | update | get <url> [more urls]]"
WIDTH = 79 ## console line the progress overwrites

def _counts_ok(counts):
    return counts is not None and not counts.get("errors")

def _dupes(cfg, rest):
    groups = rt_library.find_duplicates(cfg)
    print(f"Result: {len(groups)} duplicate group(s), {sum(len(g) - 1 for g in groups)} redundant file(s).")
    return 0

def _sort(cfg, rest):
    print(f"Result: {rt_library.sort_downloads(cfg)} file(s) sorted.")
    return 0

def _line(text, end="\n"):
    print("\r" + text.ljust(WIDTH), end=end, flush=True)

def _clock(seconds):
    if seconds is None:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"

def _bar(snap):
    pct = f"{snap['done'] * 100 // snap['total']}%" if snap.get("total") else "?%"
    _line(f"{snap['active']} active  {snap['speed'] / 1048576:.1f} MB/s  {pct}  {_clock(snap.get('eta'))} left", "")

def _get(cfg, urls):
    import rt_download ## engine loads only for downloads
    if not urls:
        print(USAGE)
        return 2

    flag, done = threading.Event(), []
    job = threading.Thread(target=lambda: done.extend(rt_download.run_queue(urls, cfg, log=_line, cancel_flag=flag, progress=_bar)), daemon=True)
    job.start()
    try:
        while job.is_alive(): ## exits when the queue drains
            job.join(0.5)
    except KeyboardInterrupt:
        flag.set() ## workers stop, part files stay
        _line("Cancelling, partial files are kept for resume...")
        job.join()
        return 130

    _line(f"Result: {len(done)} of {len(urls)} downloaded.")
    return 0 if len(done) == len(urls) else 1

VERBS = {"labelskin": lambda cfg, rest: 0 if _counts_ok(rt_library.label_and_skin(cfg)) else 1,
         "sort": _sort,
         "dupes": _dupes,
         "update": lambda cfg, rest: 0 if _counts_ok(rt_library.update_database(cfg)) else 1,
         "get": _get}

def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    verb = args[0].lower() if args else "gui"
    cfg = rt_config.load_config() ## sets the staging folder first
    rt_config.ensure_working_files()
    if verb == "gui":
        import rt_gui ## tkinter loads only for the window
        rt_gui.launch_gui()
        return 0

    if verb not in VERBS:
        print(USAGE)
        return 0 if verb in ("-h", "--help", "help") else 2
    try:
        sys.stdout.reconfigure(errors="backslashreplace") ## odd paths on a cp1252 pipe
    except (AttributeError, ValueError):
        pass
    if not rt_config.paths_configured(cfg):
        print(f"Model folders are not configured or missing. Run the GUI or edit {rt_config.CONFIG_PATH}")
        return 2

    try:
        return VERBS[verb](cfg, args[1:])
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"! {verb} failed: {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
