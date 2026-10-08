# -*- coding: utf-8 -*-
"""对抗性图片生成器: 结构/内容/元数据/损坏 四类边界."""
import os, struct, zlib, random
import numpy as np
os.makedirs("_adv/in", exist_ok=True)

def chunk(t, d):
    c = t + d
    return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))

def pack_rows(vals, bd):
    bits = []
    for v in vals:
        for k in range(bd - 1, -1, -1):
            bits.append((v >> k) & 1)
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for x in bits[i:i + 8]:
            b = (b << 1) | x
        out.append(b)
    return bytes(out) or b"\0"

def apply_filter(recon, prev, f, bpp):
    n = len(recon)
    out = bytearray(n)
    for x in range(n):
        a = recon[x - bpp] if x >= bpp else 0
        b = prev[x] if prev is not None else 0
        c = prev[x - bpp] if (prev is not None and x >= bpp) else 0
        if f == 0: v = recon[x]
        elif f == 1: v = (recon[x] - a) & 0xFF
        elif f == 2: v = (recon[x] - b) & 0xFF
        elif f == 3: v = (recon[x] - ((a + b) >> 1)) & 0xFF
        else:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            v = (recon[x] - pr) & 0xFF
        out[x] = v
    return bytes(out)

ADAM = [(0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2)]

def build_raw(w, h, bd, ct, px, inter, fmode, zlib_style):
    """px(y,x,ch)->int 索引或样本值"""
    C = {0:1, 2:3, 3:1, 4:2, 6:4}[ct]
    def samples(x, y):
        return [px(y, x, c) for c in range(C)]
    passes = [(0,0,1,1,w,h)] if not inter else [
        (x0,y0,dx,dy,(w-x0+dx-1)//dx,(h-y0+dy-1)//dy) for x0,y0,dx,dy in ADAM]
    rows = []
    for x0,y0,dx,dy,wp,hp in passes:
        if wp <= 0 or hp <= 0: continue
        prev = None  # passes are independent
        rb = (wp*C*bd + 7) >> 3
        for j in range(hp):
            line = []
            for i in range(wp):
                line.extend(samples(x0+i*dx, y0+j*dy))
            rowb = pack_rows(line, bd) if bd < 8 else (
                b"".join(struct.pack(">H", v) for v in line) if bd == 16
                else bytes(line))
            f = fmode if fmode >= 0 else (len(rows) % 5)
            rows.append((apply_filter(rowb, prev, f, 1 if bd < 8 else C*(bd//8)), f))
            prev = rowb
    raw = b"".join(bytes([r[1]]) + r[0] for r in rows)
    if zlib_style == 0:
        z = zlib.compress(raw, 0)
    elif zlib_style == 2:
        co = zlib.compressobj(6, zlib.DEFLATED, 15, 8, zlib.Z_FIXED)
        z = co.compress(raw) + co.flush()
    else:
        z = zlib.compress(raw, 9)
    return z

def write_png(name, w, h, bd, ct, px, inter=0, fmode=-1, zlib_style=1,
              plte=None, trns=None, idat_split=1, extra_chunks=(),
              corrupt=None):
    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bd, ct, 0, 0, inter))
    for t, d in extra_chunks:
        out += chunk(t, d)
    if plte is not None:
        out += chunk(b"PLTE", plte)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    z = build_raw(w, h, bd, ct, px, inter, fmode, zlib_style)
    if corrupt == "trunc_idat":
        z = z[:max(2, len(z)//2)]
    if corrupt == "bad_adler":
        z = z[:-4] + b"\0\0\0\0"
    if idat_split > 1:
        step = max(1, len(z)//idat_split)
        for i in range(0, len(z), step):
            out += chunk(b"IDAT", z[i:i+step])
    else:
        out += chunk(b"IDAT", z)
    out += chunk(b"IEND", b"")
    if corrupt == "bad_crc":
        i = out.find(b"IDAT")
        ln = int.from_bytes(out[i-4:i], "big")
        off = i-4+8+ln
        out = out[:off] + b"\xde\xad\xbe\xef" + out[off+4:]
    path = f"_adv/in/{name}.png"
    open(path, "wb").write(out)
    return path

rng = random.Random(42)
made = []

def R(y, x, c): return rng.randrange(256)
def Z(y, x, c): return 0
def F(y, x, c): return 255

# ---- 结构边界: 尺寸 x 变体 x 隔行 x 滤波 ----
sizes = [(1,1),(2,2),(3,3),(5,5),(7,7),(8,8),(15,15),(16,16),(17,17),
         (31,33),(16,64),(64,16),(1,64),(64,1),(256,80)]
def _mkpal(n):
    rng2 = random.Random(7)
    return bytes(v for t in ((rng2.randrange(256), rng2.randrange(256),
                              rng2.randrange(256)) for _ in range(n))
                for v in t)

PAL16, PAL256 = _mkpal(16), _mkpal(256)
variant_sets = [
    ("g8", 8, 0, R), ("rgb8", 8, 2, R), ("rgba8", 8, 6, R),
    ("pal4", 4, 3, lambda y,x,c: (x+y) % 16),
    ("pal8", 8, 3, lambda y,x,c: (x*3+y*7) % 256),
    ("g16", 16, 0, lambda y,x,c: (x*997+y*31) % 65536),
    ("rgb16", 16, 2, lambda y,x,c: (x*(500+c*1000)+y*77) % 65536),
    ("rgba16", 16, 6, lambda y,x,c: (x*(300+c*900)+y*13) % 65536),
    ("ga8", 8, 4, lambda y,x,c: (x*11+y*3+c*60) % 256),
    ("ga16", 16, 4, lambda y,x,c: (x*101+y*7+c*3000) % 65536),
    ("g1", 1, 0, lambda y,x,c: (x+y) % 2),
    ("g2", 2, 0, lambda y,x,c: (x+y) % 4),
    ("pal1", 1, 3, lambda y,x,c: (x//2+y) % 2),
    ("pal2", 2, 3, lambda y,x,c: (x+y) % 3),
    ("pal4b", 4, 3, lambda y,x,c: (x*y) % 16),
    ("pal8b", 8, 3, lambda y,x,c: (x*5+y*3) % 256),
]
for w, h in sizes:
    for vn, bd, ct, fn in variant_sets:
        for inter in (0, 1):
            for fm, ztag in ((-1, ""), (4, "f4"), (0, "s0")):
                nm = f"{w}x{h}_{vn}_i{inter}{ztag}{fm if fm>=0 else ''}"
                plte = ((PAL256 if vn in ("pal8", "pal8b") else PAL16)
                        if ct == 3 else None)
                write_png(nm, w, h, bd, ct, fn, inter, fm,
                          0 if ztag == "s0" else 1, plte=plte)
                made.append(nm)

# ---- tRNS 矩阵 ----
write_png("trns_pal_partial", 64, 32, 8, 3,
          lambda y,x,c: (x+y) % 4,
          plte=bytes([255,0,0, 0,255,0, 0,0,255, 9,9,9]),
          trns=bytes([0, 128]))
write_png("trns_pal_full_len", 64, 32, 4, 3,
          lambda y,x,c: (x+y) % 15,
          plte=bytes([255,0,0]*15 + [1,2,3]),
          trns=bytes(range(16)))
write_png("trns_rgb8_key", 64, 32, 8, 2,
          lambda y,x,c: (10,20,30)[c] if x < 8 else (x*3+y+c*40) % 256,
          trns=struct.pack(">HHH", 10, 20, 30))
# 嫌疑: 16bit color-key, 低字节不同 -> libpng精确16位比较 vs 我们>>8
write_png("trns_rgb16_lowbyte", 64, 32, 16, 2,
          lambda y,x,c: (100*256+50, 200*256+60, 30*256+70)[c]
          if x < 8 else ((x*33+c*7000+y*11) % 65536),
          trns=struct.pack(">HHH", 100*256, 200*256, 30*256))
write_png("trns_g16_key", 64, 32, 16, 0,
          lambda y,x,c: 5000 if x < 8 else (x*97+y*3) % 65536,
          trns=struct.pack(">H", 4999))
# 调色板越界索引(非法文件, 观察双方行为)
write_png("pal_oob_idx", 64, 32, 8, 3,
          lambda y,x,c: 200 if x == 5 else (x+y) % 3,
          plte=bytes([255,0,0, 0,255,0, 0,0,255]))

# ---- 内容边界 ----
write_png("all_black", 256, 256, 8, 6, Z)
write_png("all_white", 256, 256, 8, 6, F)
write_png("one_dot", 256, 256, 8, 6,
          lambda y,x,c: 255 if (x, y, c) == (128, 128, 0) else 0)
write_png("checker1", 256, 256, 8, 6, lambda y,x,c: (x ^ y) & 1)
write_png("checker_ch", 256, 256, 8, 2,
          lambda y,x,c: (x + c) & 1)
write_png("v_edge", 256, 256, 8, 6, lambda y,x,c: 255 if x == 128 else 0)
write_png("h_edge", 256, 256, 8, 6, lambda y,x,c: 255 if y == 128 else 0)
write_png("alpha_all0", 256, 256, 8, 6, lambda y,x,c: 0 if c == 3 else R(y,x,c))
write_png("alpha_all255", 256, 256, 8, 6,
          lambda y,x,c: 255 if c == 3 else (x*y) % 256)
write_png("alpha_1px", 256, 256, 8, 6,
          lambda y,x,c: 0 if (c == 3 and x == 7 and y == 9) else
          (255 if c == 3 else (x*2+y) % 256))
write_png("alpha_grad", 320, 192, 8, 6, lambda y,x,c: (y*255//191 if c==3 else (x+y)%256))
write_png("max_freq", 320, 192, 8, 6,
          lambda y,x,c: ((x*37 + y*91 + c*53) ^ (x*13)) % 256)
write_png("noise_2mp", 1600, 1200, 8, 6, R)
write_png("grad16_max", 128, 128, 16, 6,
          lambda y,x,c: 65535 if (x+y) % 7 == 0 else ((x*511+y*131+c*311) % 65536))

# ---- 元数据 ----
big_text = b"v" * 300000
write_png("meta_bigtext", 64, 32, 8, 6, R,
          extra_chunks=[(b"tEXt", b"Comment\x00" + big_text),
                        (b"tEXt", b"Title\x00hello"),
                        (b"iTXt", b"Desc\x00\x00\x00en\x00desc here"),
                        (b"pHYs", struct.pack(">IIB", 2835, 2835, 1))])
exif = b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x01\x00\x04\x00\x00\x00\x01\x00\x00\x00\x00" + b"\x00"*8
write_png("meta_exif", 64, 32, 8, 6, R, extra_chunks=[(b"eXIf", exif)])

# ---- zlib/IDAT 结构 ----
write_png("trns_g16_key", 64, 32, 16, 0,
          lambda y,x,c: 5000 if x < 8 else (x*97+y*3) % 65536,
          trns=struct.pack(">H", 4999))
write_png("trns_pal_partial", 64, 32, 8, 3,
          lambda y,x,c: (x+y) % 4,
          plte=bytes([255,0,0, 0,255,0, 0,0,255, 9,9,9]),
          trns=bytes([0, 128]))
write_png("trns_pal_full_len", 64, 32, 4, 3,
          lambda y,x,c: (x+y) % 15,
          plte=bytes([255,0,0]*15 + [1,2,3]),
          trns=bytes(range(16)))
write_png("trns_rgb8_key", 64, 32, 8, 2,
          lambda y,x,c: (10,20,30)[c] if x < 8 else (x*3+y+c*40) % 256,
          trns=struct.pack(">HHH", 10, 20, 30))
write_png("trns_rgb16_lowbyte", 64, 32, 16, 2,
          lambda y,x,c: (100*256+50, 200*256+60, 30*256+70)[c]
          if x < 8 else ((x*33+c*7000+y*11) % 65536),
          trns=struct.pack(">HHH", 100*256, 200*256, 30*256))
write_png("idat_many", 128, 64, 8, 6, R, idat_split=64)
write_png("stored_big", 128, 128, 8, 6, R, zlib_style=0)
write_png("zfixed", 128, 128, 8, 6, R, zlib_style=2)
write_png("bad_trunc", 64, 32, 8, 6, R, corrupt="trunc_idat")
write_png("bad_adler", 64, 32, 8, 6, R, corrupt="bad_adler")
write_png("bad_crc", 64, 32, 8, 6, R, corrupt="bad_crc")

bomb = (b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x11" * 16))
        + chunk(b"IEND", b""))
open("_adv/in/bomb_header.png", "wb").write(bomb)

import imagecodecs
import numpy as np
arr = np.zeros((64, 64, 4), np.uint8)
arr[..., 0] = np.arange(64 * 64).reshape(64, 64) % 256
open("_adv/in/recomp_lossy.webp", "wb").write(
    imagecodecs.webp_encode(arr, level=90, lossless=False))
open("_adv/in/recomp_ll.webp", "wb").write(
    imagecodecs.webp_encode(arr, lossless=True))

print("生成", len(os.listdir("_adv/in")))
