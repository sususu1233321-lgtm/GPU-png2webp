"""Subprocess worker for the multi-process fast batch mode.

Invoked (via the exe's ``-m`` proxy or ``python -m gpuwebp.worker``) as:
    python -m gpuwebp.worker <job.json> <result.json>

job.json:   {"files": [...], "dst": ..., "quality": ..., "device": ...,
             "batch": ..., "decode_workers": ..., "finish_workers": ...}
result.json written at the end: {"done":..,"failed":..,"skipped":..,
             "fallbacks":..,"src_bytes":..,"dst_bytes":..,"elapsed":..,
             "errors": [...]}
Progress is appended to result.json + ".prog" as plain text "done failed
skipped" so the parent can poll.
"""
import json
import sys


def main():
    import os
    import time

    # Pin this process to exactly one physical GPU before anything imports
    # cupy: module-level device arrays then live on the right card (avoids
    # "peer access unavailable" when the batch driver assigns device 1+).
    if os.environ.get("GPUPIC_DEVICE") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPUPIC_DEVICE"]
        dev = 0
    else:
        dev = int(os.environ.get("GPUPIC_DEVICE_ID", "0"))

    import numpy as np
    from .pipeline import run_batch_fast
    from .closed_loop_jit import closed_loop_full
    from .pipeline import _psnr_rgba

    # warm numba caches before the timers start
    _ = closed_loop_full(np.zeros((16, 16), np.int16),
                         np.zeros((8, 8), np.int16),
                         np.zeros((8, 8), np.int16),
                         np.zeros(1, bool), np.zeros(1, np.uint8),
                         np.zeros(1, np.uint8), np.zeros((1, 16), np.uint8),
                         *([np.ones(16, np.int64)] * 5) * 3,
                         np.ones(2, np.int64), np.ones(2, np.int64),
                         np.ones(2, np.int64))
    _ = _psnr_rgba(np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8))

    job_file, result_file = sys.argv[1], sys.argv[2]
    with open(job_file, "r", encoding="utf-8") as f:
        job = json.load(f)
    if os.environ.get("GPUPIC_DEVICE") is None:
        dev = job.get("device", 0)

    # build a temp source dir listing? run_batch_fast walks a folder; instead
    # we give it the parent folder + accept processing every file, but the
    # parent splits by giving each worker its own file list through a
    # virtual "src": we create a job-specific approach — write list files.
    # Simplest robust contract: worker processes job["files"] directly via
    # a tiny inline driver replicating run_batch_fast with a file list.
    errors = []

    def log(msg):
        if "失败" in msg or "异常" in msg:
            errors.append(msg)

    prog_file = result_file + ".prog"
    last = [-1]

    def progress(stats):
        key = stats.done + stats.failed + stats.skipped
        if key != last[0]:
            last[0] = key
            try:
                with open(prog_file, "w", encoding="utf-8") as f:
                    f.write(f"{stats.done} {stats.failed} {stats.skipped} "
                            f"{stats.fallbacks} {stats.src_bytes} {stats.dst_bytes}")
            except OSError:
                pass

    from .pipeline import run_batch_files_fast
    t0 = time.time()
    stats = run_batch_files_fast(
        job["files"], job["dst"], base=job.get("base"),
        quality=job.get("quality", 90),
        device=job.get("device", 0), batch=job.get("batch", 24),
        decode_workers=job.get("decode_workers", 6),
        finish_workers=job.get("finish_workers", 14),
        log_cb=log, progress_cb=progress)
    out = dict(done=stats.done, failed=stats.failed, skipped=stats.skipped,
               fallbacks=stats.fallbacks, src_bytes=stats.src_bytes,
               dst_bytes=stats.dst_bytes, elapsed=time.time() - t0,
               errors=errors[:50])
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
