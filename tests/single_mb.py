"""Single-macroblock micro test: isolate token path bugs."""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from gpuwebp import vp8_encode as E
from gpuwebp.bool_coder import bool_encode
from token_check import get_coeffs, BoolDec

# 16x16 gradient: one MB, i16 with a few coefficients
img = np.zeros((16, 16, 3), np.uint8)
img[..., 0] = np.linspace(80, 180, 16, dtype=np.uint8)[:, None]
img[..., 1] = 100
img[..., 2] = 120

rgba = np.concatenate([img, np.full((16, 16, 1), 255, np.uint8)], 2)
from test_m1 import rgb_to_yuv420
y, u, v = rgb_to_yuv420(rgba)

base_quant, y1, y2, uv_m, filter_level = E.setup_quant(90)
Y = E.pad_to_mb(y, 1, 1)
U = E.pad_to_mb(u, 1, 1, half=True)
V = E.pad_to_mb(v, 1, 1, half=True)
(mb_w, mb_h, is_i4, i16_mode, uv_mode, i4_modes,
 y_dc, y_ac, uv_levels, skip) = E.analyze(Y, U, V, y1, y2, uv_m)

print("is_i4:", is_i4, "i16_mode:", i16_mode, "uv_mode:", uv_mode, "skip:", skip)
print("y_dc:", y_dc[0])
print("y_ac nonzero blocks:", np.flatnonzero(y_ac[0].any(-1)))
for b in np.flatnonzero(y_ac[0].any(-1)):
    print(f"  y_ac[{b}]:", y_ac[0][b])
print("uv nonzero blocks:", np.flatnonzero(uv_levels[0].any(-1)))
for b in np.flatnonzero(uv_levels[0].any(-1)):
    print(f"  uv[{b}]:", uv_levels[0][b])

# emit token partition ops for this single MB
ops = np.empty(8200 + 64, np.int32)
pos = E.write_token_partition(1, 1, 0, 1, False, skip, is_i4, y_dc, y_ac, uv_levels, ops)
buf = np.empty(pos + 16, np.uint8)
nb = bool_encode(ops[:pos], buf)
data = buf[:nb].tobytes()
print(f"\ntoken ops: {pos}, bytes: {nb}, hex: {data.hex()}")

# decode back following the encoder's actual mode sequence
br = BoolDec(data)
ok = True
if not is_i4[0]:
    nz, dc = get_coeffs(br, 1, 0, 0)
    match = (dc == list(y_dc[0]))
    print("y_dc decoded:", dc, "nz:", nz, "match:", match)
    ok &= match
first = 0 if is_i4[0] else 1
ctype = 3 if is_i4[0] else 0
nz_map = np.zeros((4, 4), int)
for yb in range(4):
    for xb in range(4):
        top = nz_map[yb - 1, xb] if yb > 0 else 0
        left = nz_map[yb, xb - 1] if xb > 0 else 0
        ctx2 = top + left
        nz, blk = get_coeffs(br, ctype, ctx2, first)
        match = (blk == list(y_ac[0][xb + yb * 4]))
        if not match:
            print(f"y_ac({xb},{yb}) MISMATCH ctx={ctx2}")
            print("  decoded:", blk)
            print("  encoded:", list(y_ac[0][xb + yb * 4]))
            ok = False
        nz_map[yb, xb] = 1 if nz > first else 0
uv_nz = {}
for n in range(8):
    ch = n >> 2
    x = n & 1
    y = (n >> 1) & 1
    top = uv_nz.get((ch, y - 1, x), 0) if y > 0 else 0
    left = uv_nz.get((ch, y, x - 1), 0) if x > 0 else 0
    ctx = top + left
    nz, blk = get_coeffs(br, 2, ctx, 0)
    match = (blk == list(uv_levels[0][n]))
    if not match:
        print(f"uv[{n}] MISMATCH ctx={ctx}")
        print("  decoded:", blk)
        print("  encoded:", list(uv_levels[0][n]))
        ok = False
    uv_nz[(ch, y, x)] = 1 if nz > 0 else 0
print("TOKEN ROUNDTRIP", "OK" if ok else "FAILED")
