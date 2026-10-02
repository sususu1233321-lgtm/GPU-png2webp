import sys, os, time, threading

if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ["VERIFY_MAXFLIGHT"] = "64"
    os.environ["VERIFY_RECYCLE"] = "300"
    sys.path.insert(0, ".")
    from gpuwebp.pipeline import collect_pngs, run_batch_files_fast

    SRC = r"L:\图片备份8"
    DST = r"D:\gpu压图\_L8out"
    _lock = threading.Lock()
    LOG = open(r"D:\gpu压图\_L8run.log", "a", buffering=1, encoding="utf-8")

    def log(msg):
        with _lock:
            LOG.write(str(msg).rstrip() + chr(10))

    files = sorted(collect_pngs(SRC, recursive=True))
    base = os.path.commonpath(files)
    CHUNK = 2500
    todo = [f for f in files
            if not os.path.exists(os.path.join(
                DST, os.path.splitext(os.path.relpath(f, base))[0] + ".webp"))]
    log(f"==== chunk run: {len(todo)} remaining of {len(files)} ====")
    t00 = time.time()
    ndone = 0
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        st = run_batch_files_fast(
            chunk, DST, base=base, quality=90, device=1,
            skip_existing=True, min_psnr=34.0, verify_meta=True,
            batch=64, decode_workers=8, finish_workers=10, nvp=8,
            log_cb=log, progress_cb=None)
        ndone += st.done + st.skipped
        log(f"---- chunk {i // CHUNK}: done={st.done} skip={st.skipped} "
            f"fail={st.failed} fb={st.fallbacks} "
            f"{ndone / (time.time() - t00):.1f}/s cumulative ----")
    log(f"==== ALL DONE in {time.time() - t00:.0f}s ====")
    LOG.close()
