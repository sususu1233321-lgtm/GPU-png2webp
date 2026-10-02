"""Subprocess verification worker (true parallelism, no GIL contention).

Each worker process receives (arr, webp_bytes, meta) for one already-encoded
image, re-decodes the WebP with Pillow, and checks dimensions / alpha /
metadata round-trip / PSNR — everything the in-process `check()` did, but in
its own interpreter so Pillow/numpy glue cannot contend for the main GIL.
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


def verify_payload(payload):
    """payload = (arr, webp, verify_meta, min_psnr, meta)
    Returns (ok, note) — note None means all checks passed."""
    arr, webp, verify_meta, min_psnr, meta = payload
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
    d = np.subtract(arr, darr, dtype=np.int32)
    mse = float((d * d).mean())
    psnr = 99.0 if mse == 0 else 10 * np.log10(255 * 255 / mse)
    if psnr < min_psnr:
        return False, f"PSNR {psnr:.2f} < {min_psnr}"
    return True, None
