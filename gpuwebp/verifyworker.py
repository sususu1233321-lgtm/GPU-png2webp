"""Subprocess verification worker (true parallelism, no GIL contention).

Each worker process receives (src_path, webp_bytes, verify_meta, min_psnr,
meta) for one already-encoded image, re-decodes the source PNG and the WebP
with imagecodecs (C speed, GIL-free inside this process), and checks
dimensions / alpha / metadata round-trip / PSNR. Passing only the path (not
the decoded array) keeps the main-process pickle cost near zero; the worker
re-reads the file from the OS cache.
"""
import io

import numpy as np
from PIL import Image


def _decode_webp(webp):
    """imagecodecs (libdeflate-class C) webp decode; Pillow fallback."""
    try:
        import imagecodecs
        arr = imagecodecs.webp_decode(webp)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        return arr
    except Exception:                                   # noqa: BLE001
        return np.asarray(Image.open(io.BytesIO(webp)).convert("RGBA"))


def _decode_png(path):
    try:
        import imagecodecs
        arr = imagecodecs.png_decode(open(path, "rb").read())
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        return arr
    except Exception:                                   # noqa: BLE001
        return np.asarray(Image.open(path).convert("RGBA"))


def verify_payload(payload):
    import os as _os
    if _os.environ.get("VSKIP"):        # perf isolation: skip all work
        return True, None
    if _os.environ.get("VTIME"):
        import time as _t
        _t0 = _t.time()
        _r = _verify_payload_impl(payload)
        _f = open(f"_vrate_{_os.getpid()}.txt", "a", buffering=1)
        _f.write(f"{_t.time():.3f} {(_t.time()-_t0)*1000:.1f}{chr(10)}")
        _f.close()
        return _r
    return _verify_payload_impl(payload)


def _verify_payload_impl(payload):
    """payload = ("shm", ring_name, count, cap, slot, h, w, webp, vmeta,
    psnr, meta) or the fallback (src_path, webp, vmeta, psnr, meta).
    Returns (ok, note) — note None means all checks passed."""
    if payload[0] == "shm":
        _, rname, count, cap, slot, h, w, webp, verify_meta, min_psnr, meta = payload
        from .shmr import read_slot
        arr = read_slot(rname, count, cap, slot, h, w)
    else:
        src, webp, verify_meta, min_psnr, meta = payload
        arr = _decode_png(src)
    darr = _decode_webp(webp)
    H, W = arr.shape[:2]
    if darr.shape[:2] != (H, W):
        return False, f"尺寸不符 {darr.shape[:2]}"
    if not np.array_equal(darr[..., 3], arr[..., 3]):
        return False, "alpha不一致"
    if verify_meta:
        from .png_meta import verify_pre
        ok, problems = verify_pre(meta, webp)
        if not ok:
            return False, "元数据校验失败: " + ";".join(problems)
    # PSNR over RGB only: alpha is equality-checked above, so it adds zero
    # to the SSE — summing 3 channels but dividing by the full 4-channel
    # element count keeps the verdict bit-identical to the old 4-channel
    # pass while cutting ~60% of its memory traffic
    d = np.subtract(arr[..., :3], darr[..., :3], dtype=np.int32)
    # einsum with int64 accumulation: exact (int32 vdot overflows SSE), and
    # no 16MB squared temp
    mse = float(np.einsum("i,i->", d.reshape(-1), d.reshape(-1),
                          dtype=np.int64)) / arr.size
    psnr = 99.0 if mse == 0 else 10 * np.log10(255 * 255 / mse)
    if psnr < min_psnr:
        return False, f"PSNR {psnr:.2f} < {min_psnr}"
    import os as _os2
    if _os2.environ.get("VPSNR"):   # calibration: report the true PSNR
        return True, f"PSNR {psnr:.2f}"
    return True, None
