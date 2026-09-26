"""Comprehensive encoder validation: PSNR and size vs Pillow(libwebp) reference."""
import io
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from gpuwebp import vp8_encode
from gpuwebp.webp_container import make_simple
from test_m1 import rgb_to_yuv420, psnr, make_test_image


def images():
    yield "M1 synthetic", make_test_image()
    rng = np.random.default_rng(42)
    yield "noise64", (rng.normal(0.5, 0.18, (64, 64, 3)) * 255).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(1)
    yield "noise256", (rng.normal(0.5, 0.15, (256, 256, 3)) * 255).clip(0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    img = np.zeros((256, 256, 3), np.uint8)
    img[..., 0] = (60 + yy * 0.7 + np.sin(xx * 0.1) * 20).clip(0, 255).astype(np.uint8)
    img[..., 1] = (90 + xx * 0.5 + np.cos(yy * 0.08) * 15).clip(0, 255).astype(np.uint8)
    img[..., 2] = (150 - yy * 0.3).clip(0, 255).astype(np.uint8)
    yield "gradient256", img
    img = np.full((512, 384, 3), 128, np.uint8)
    img[128:384, 128:256] = [220, 40, 60]
    img[:, 190:194] = [255, 255, 255]
    img[200:300, 300:350] = 30
    yield "shapes512", img
    rng = np.random.default_rng(9)
    base = np.zeros((832, 1216, 3), np.float32)
    yy, xx = np.mgrid[0:832, 0:1216]
    base[..., 0] += np.sin(xx * 0.02) * 40 + np.cos(yy * 0.03) * 40 + 128
    base[..., 1] += np.cos(xx * 0.015) * 30 + 110
    base[..., 2] += np.sin(yy * 0.025) * 35 + 140
    base += rng.normal(0, 6, base.shape)
    for _ in range(12):
        cx, cy, r = rng.integers(0, 1216), rng.integers(0, 832), rng.integers(20, 90)
        color = rng.integers(0, 255, 3)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        base[mask] = color
    yield "synthetic832", base.clip(0, 255).astype(np.uint8)


def main():
    print(f"{'image':<15} {'q':>3} {'mine PSNR':>10} {'Pillow PSNR':>11} {'mine B':>8} {'pil B':>8} {'ratio':>6} {'t(s)':>6}")
    for name, rgb in images():
        for q in (75, 90, 95):
            H, W = rgb.shape[:2]
            rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], 2)
            y, u, v = rgb_to_yuv420(rgba)
            t0 = time.time()
            vp8 = vp8_encode.encode(y, u, v, q)
            t1 = time.time()
            mine = make_simple(vp8)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "WEBP", quality=q)
            pil = buf.getvalue()
            dec_m = np.asarray(Image.open(io.BytesIO(mine)).convert("RGB"))
            dec_p = np.asarray(Image.open(io.BytesIO(pil)).convert("RGB"))
            print(f"{name:<15} {q:>3} {psnr(rgb, dec_m):>10.2f} {psnr(rgb, dec_p):>11.2f} "
                  f"{len(mine):>8} {len(pil):>8} {len(mine)/max(1,len(pil)):>6.2f} {t1-t0:>6.2f}")


if __name__ == "__main__":
    main()
