"""Fast self-contained PNG decoder: zlib (C, GIL-released) + numba defilter.

Replaces Pillow for the common AI-tool/PIL-saved formats — 8-bit non-interlaced
RGB(A) — cutting decode from ~41ms to ~8ms per 832x1216 image.  Anything
else (palette, 16-bit, interlaced, grey...) returns None and the caller
falls back to Pillow.
"""
import struct
import zlib

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _paeth(a, b, c):
    ai = np.int32(a)
    bi = np.int32(b)
    ci = np.int32(c)
    p = ai + bi - ci
    pa = p - ai
    if pa < 0:
        pa = -pa
    pb = p - bi
    if pb < 0:
        pb = -pb
    pc = p - ci
    if pc < 0:
        pc = -pc
    if pa <= pb and pa <= pc:
        return ai
    if pb <= pc:
        return bi
    return ci


@njit(cache=True, nogil=True)
def _defilter(raw, recon, h, stride, bpp):
    """raw: (h, 1+stride) filter-prefixed rows; recon: flat (h*stride) out."""
    for y in range(h):
        f = np.int32(raw[y, 0])
        r = y * stride
        if f == 0:
            for x in range(stride):
                recon[r + x] = raw[y, 1 + x]
        elif f == 1:
            for x in range(bpp):
                recon[r + x] = raw[y, 1 + x]
            for x in range(bpp, stride):
                v = np.int32(raw[y, 1 + x]) + np.int32(recon[r + x - bpp])
                recon[r + x] = np.uint8(v)
        elif f == 2:
            for x in range(stride):
                if y == 0:
                    recon[r + x] = raw[y, 1 + x]
                else:
                    v = np.int32(raw[y, 1 + x]) + np.int32(recon[r + x - stride])
                    recon[r + x] = np.uint8(v)
        elif f == 3:
            for x in range(bpp):
                if y == 0:
                    recon[r + x] = raw[y, 1 + x]
                else:
                    v = np.int32(raw[y, 1 + x]) + (np.int32(recon[r + x - stride]) >> 1)
                    recon[r + x] = np.uint8(v)
            for x in range(bpp, stride):
                up = np.int32(recon[r + x - stride]) if y > 0 else np.int32(0)
                v = np.int32(raw[y, 1 + x]) + ((np.int32(recon[r + x - bpp]) + up) >> 1)
                recon[r + x] = np.uint8(v)
        else:  # 4 Paeth
            for x in range(bpp):
                up = np.int32(recon[r + x - stride]) if y > 0 else np.int32(0)
                v = np.int32(raw[y, 1 + x]) + _paeth(np.int32(0), up, np.int32(0))
                recon[r + x] = np.uint8(v)
            for x in range(bpp, stride):
                left = np.int32(recon[r + x - bpp])
                up = np.int32(recon[r + x - stride]) if y > 0 else np.int32(0)
                ul = (np.int32(recon[r + x - stride - bpp]) if y > 0
                      else np.int32(0))
                v = np.int32(raw[y, 1 + x]) + _paeth(left, up, ul)
                recon[r + x] = np.uint8(v)
    return recon


def png_decode_fast(data):
    """data: PNG bytes.  Returns (H, W, 4) uint8 array or None if the
    format is not supported (caller should fall back to Pillow)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    idat = bytearray()
    w = h = bd = ct = interlace = None
    n = len(data)
    while pos + 8 <= n:
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, bd, ct, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
        elif typ == b"IDAT":
            idat += body
        elif typ == b"IEND":
            break
        pos += 12 + ln
    if w is None or not idat:
        return None
    if bd != 8 or ct not in (2, 6) or interlace != 0:
        return None
    bpp = 4 if ct == 6 else 3
    stride = w * bpp
    try:
        raw = zlib.decompress(bytes(idat), 15, (h * (stride + 1)) + 64)
    except zlib.error:
        return None
    if len(raw) < h * (stride + 1):
        return None
    raw = np.frombuffer(raw[:h * (stride + 1)], np.uint8).reshape(h, stride + 1)
    if ct == 6:
        out = np.empty(h * stride, np.uint8)
        _defilter(raw, out, h, stride, bpp)
        return out.reshape(h, w, 4)
    out = np.empty(h * stride, np.uint8)
    _defilter(raw, out, h, stride, 3)
    rgb = out.reshape(h, w, 3)
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    return rgba


_PNG_MAGIC = bytes((137, 80, 78, 71, 13, 10, 26, 10))


def png_inflate_fast(data):
    """Parse chunks and inflate IDAT.  Returns (raw uint8 (h, stride+1),
    w, h, bpp) for supported 8-bit non-interlaced RGB/RGBA, else None."""
    if data[:8] != _PNG_MAGIC:
        return None
    pos = 8
    idat = bytearray()
    w = h = ct = None
    n = len(data)
    while pos + 8 <= n:
        ln = int.from_bytes(data[pos:pos + 4], "big")
        typ = data[pos + 4:pos + 8]
        if typ == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", data[pos + 8:pos + 18])
            if bd != 8 or ct not in (2, 6):
                return None
            if data[pos + 20] != 0:      # interlace byte in IHDR
                return None
        elif typ == b"IDAT":
            idat += data[pos + 8:pos + 8 + ln]
        elif typ == b"IEND":
            break
        pos += 12 + ln
    if w is None or not idat:
        return None
    bpp = 4 if ct == 6 else 3
    rstride = 1 + w * bpp
    try:
        raw = zlib.decompress(bytes(idat), 15, h * rstride + 64)
    except zlib.error:
        return None
    if len(raw) < h * rstride:
        return None
    return (np.frombuffer(raw[:h * rstride], np.uint8).reshape(h, rstride),
            w, h, bpp)
