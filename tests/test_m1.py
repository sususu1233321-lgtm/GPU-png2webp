"""M1 milestone test: encode a synthetic image with our VP8 encoder, wrap in
WebP, decode with Pillow (libwebp decoder), and check PSNR + dimensions."""
import struct
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from gpuwebp import vp8_encode
from gpuwebp.webp_container import make_simple, parse


def rgb_to_yuv420(rgba):
    """libwebp BT.601 fixed-point conversion (no dithering)."""
    r = rgba[..., 0].astype(np.int64)
    g = rgba[..., 1].astype(np.int64)
    b = rgba[..., 2].astype(np.int64)
    YFIX = 16
    HALF = 1 << (YFIX - 1)

    def clip_uv(v):
        # v is 4x the per-pixel value (2x2 sum): /4 then >> YFIX
        x = (v + (HALF << 2) + (128 << YFIX << 2)) >> (YFIX + 2)
        return np.clip(x, 0, 255)

    y = (16839 * r + 33059 * g + 6420 * b + HALF + (16 << YFIX)) >> YFIX
    H, W = y.shape
    y = y.reshape(H // 2, 2, W // 2, 2)
    r2 = r.reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    g2 = g.reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    b2 = b.reshape(H // 2, 2, W // 2, 2).sum((1, 3))
    u = clip_uv(-9719 * r2 - 19081 * g2 + 28800 * b2)
    v = clip_uv(28800 * r2 - 24116 * g2 - 4684 * b2)
    return (np.clip(y, 0, 255).reshape(H, W).astype(np.uint8),
            u.astype(np.uint8), v.astype(np.uint8))


def make_test_image(w=96, h=64):
    rng = np.random.default_rng(42)
    x = np.linspace(0, 255, w, dtype=np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img += x[None, :, None]                       # horizontal gradient
    img += np.linspace(0, 255, h, dtype=np.float32)[:, None, None] * 0.3
    # add some structure: circle + lines
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.3, h * 0.4
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) < (h * 0.3) ** 2
    img[circle] = [220, 40, 60]
    img[:, w // 2:w // 2 + 2] = [0, 0, 0]
    img[h // 2:h // 2 + 2, :] = [255, 255, 255]
    # light noise (like AI art, but small for M1)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    rgb = make_test_image()
    H, W = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], axis=2)
    y, u, v = rgb_to_yuv420(rgba)

    t0 = time.time()
    vp8 = vp8_encode.encode(y, u, v, quality=90, num_parts=1)
    t1 = time.time()
    print(f"encoded VP8: {len(vp8)} bytes in {t1 - t0:.3f}s  ({W}x{H})")

    webp = make_simple(vp8)
    open("test_m1.webp", "wb").write(webp)
    print(f"webp file: {len(webp)} bytes -> test_m1.webp")

    im = Image.open("test_m1.webp")
    print("Pillow decode:", im.format, im.size, im.mode)
    dec = np.asarray(im.convert("RGB"))
    if dec.shape[0] != H or dec.shape[1] != W:
        print("DIMENSION MISMATCH!", dec.shape, (H, W))
        return
    print(f"PSNR: {psnr(rgb, dec):.2f} dB")
    if psnr(rgb, dec) > 30:
        print("M1 PASS")
    else:
        print("M1 FAIL - decode succeeded but quality too low / corrupted")
    # save side by side for visual check
    comp = np.concatenate([rgb, dec], axis=0)
    Image.fromarray(comp.astype(np.uint8)).save("test_m1_compare.png")


if __name__ == "__main__":
    main()
