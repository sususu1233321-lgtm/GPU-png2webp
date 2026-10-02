"""Bisect debugging: flat image (all MBs should skip, zero residual)."""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from gpuwebp import vp8_encode
from gpuwebp.webp_container import make_simple
sys.path.insert(0, "tests")
from test_m1 import rgb_to_yuv420, psnr


def run(name, rgb):
    H, W = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], axis=2)
    y, u, v = rgb_to_yuv420(rgba)
    vp8 = vp8_encode.encode(y, u, v, quality=90)
    open(f"dbg_{name}.webp", "wb").write(make_simple(vp8))
    dec = np.asarray(Image.open(f"dbg_{name}.webp").convert("RGB"))
    p = psnr(rgb, dec)
    print(f"{name}: {len(vp8)}B  PSNR={p:.2f}dB  dec[0,0]={dec[0,0]} rgb[0,0]={rgb[0,0]}")
    return p


# 1. perfectly flat gray
flat = np.full((64, 64, 3), 128, np.uint8)
run("flat128", flat)

# 2. flat but different colors per quadrant (still zero residual after chroma pred?)
q = np.full((64, 64, 3), 128, np.uint8)
q[:, 32:] = [200, 50, 50]
run("flat2", q)
