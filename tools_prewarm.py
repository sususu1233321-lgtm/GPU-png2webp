"""Prewarm the cupy kernel cache for the packaged exe.

Run AFTER building the exe:
    python tools_prewarm.py dist/GPU压图/cupy_cache

Encodes sample images through the GPU engine so every cupy kernel variant
(elementwise/reduction/indexing, all used dtypes) lands in the cache; the
frozen exe then loads cubins from disk and never invokes NVRTC.
"""
import glob
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cache_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("dist", "GPU压图", "cupy_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["CUPY_CACHE_DIR"] = cache_dir

import numpy as np                      # noqa: E402
from PIL import Image                   # noqa: E402

from gpuwebp.encoder import encode_rgba  # noqa: E402
from gpuwebp.closed_loop_jit import closed_loop_full  # noqa: E402
from gpuwebp.alpha_enc import make_alph_chunk          # noqa: E402

# warm numba too (its cache lives next to sources and ships automatically)
_ = closed_loop_full(np.zeros((16, 16), np.int16), np.zeros((8, 8), np.int16),
                     np.zeros((8, 8), np.int16), np.zeros(1, bool),
                     np.zeros(1, np.uint8), np.zeros(1, np.uint8),
                     np.zeros((1, 16), np.uint8), *([np.ones(16, np.int64)] * 5) * 3,
                     np.ones(2, np.int64), np.ones(2, np.int64), np.ones(2, np.int64))
_ = make_alph_chunk(np.full((8, 8), 255, np.uint8))
_ = make_alph_chunk((np.arange(64).reshape(8, 8) % 7).astype(np.uint8))

samples = sorted(glob.glob(os.path.join("test_batch", "in50", "*.png")))[:3]
if not samples:
    samples = sorted(glob.glob(r"L:/图片备份8/nai3_240531/*.png"))[:3]

rng = np.random.default_rng(0)
cases = []
for p in samples[:2]:
    png = open(p, "rb").read()
    img = Image.open(io.BytesIO(png))
    arr = np.asarray(img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB"))
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
    cases.append((arr, png))
# synthetic variants: transposed shape, gradient alpha, tiny
h, w = cases[0][0].shape[:2] if cases else (832, 1216)
g = np.zeros((w, h, 4), np.uint8)
g[..., 0] = (np.add.outer(np.arange(w), np.arange(h)) % 256).astype(np.uint8)
g[..., 3] = np.broadcast_to(
    np.linspace(200, 255, h, dtype=np.uint8)[None, :], (w, h))
cases.append((g, None))
t = rng.integers(0, 256, (64, 96, 4), np.uint8)
cases.append((t, None))

for arr, png in cases:
    webp = encode_rgba(arr, 90, engine="gpu", device=0, png_data=png)
    print(f"prewarmed {arr.shape[1]}x{arr.shape[0]} -> {len(webp)}B")

n = len([f for f in os.listdir(cache_dir)])
print(f"cupy cache entries: {n} in {cache_dir}")
