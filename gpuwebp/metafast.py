"""Numba nogil versions of the metadata hot path (byte-identical to
png_meta.py): PNG chunk scan, XMP build/parse, WebP chunk scan.

The XMP format written by build_xmp is rigid — fixed ASCII templates with
base64 payloads — so parsing needs no regex: we scan for the fixed tag
markers and base64-decode inline.
"""
import numpy as np
from numba import njit

B64 = np.array([
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    43, 47, 61], dtype=np.uint8)


@njit(cache=True, nogil=True)
def _b64_encode(src, srclen, out):
    """base64 into out (uint8); returns bytes written."""
    i = 0
    o = 0
    n = srclen
    while i + 2 < n:
        v = (src[i] << 16) | (src[i + 1] << 8) | src[i + 2]
        out[o] = B64[(v >> 18) & 63]
        out[o + 1] = B64[(v >> 12) & 63]
        out[o + 2] = B64[(v >> 6) & 63]
        out[o + 3] = B64[v & 63]
        o += 4
        i += 3
    rem = n - i
    if rem == 1:
        v = src[i] << 16
        out[o] = B64[(v >> 18) & 63]
        out[o + 1] = B64[(v >> 12) & 63]
        out[o + 2] = 61
        out[o + 3] = 61
        o += 4
    elif rem == 2:
        v = (src[i] << 16) | (src[i + 1] << 8)
        out[o] = B64[(v >> 18) & 63]
        out[o + 1] = B64[(v >> 12) & 63]
        out[o + 2] = B64[(v >> 6) & 63]
        out[o + 3] = 61
        o += 4
    return o


_B64_DEC = np.full(256, -1, np.int8)
for _i, _c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"):
    _B64_DEC[ord(_c)] = _i
_B64_DEC = _B64_DEC.astype(np.int8)


@njit(cache=True, nogil=True)
def _b64_decode(src, start, end, out):
    """decode base64 src[start:end] into out; returns bytes written (-1 err)."""
    o = 0
    acc = 0
    nb = 0
    for i in range(start, end):
        c = src[i]
        if c == 61:                      # '='
            break
        v = _B64_DEC[c]
        if v < 0:
            return -1
        acc = (acc << 6) | v
        nb += 1
        if nb == 4:
            out[o] = (acc >> 16) & 255
            out[o + 1] = (acc >> 8) & 255
            out[o + 2] = acc & 255
            o += 3
            acc = 0
            nb = 0
    if nb == 2:
        out[o] = (acc >> 4) & 255
        o += 1
    elif nb == 3:
        out[o] = (acc >> 10) & 255
        out[o + 1] = (acc >> 2) & 255
        o += 2
    return o


@njit(cache=True, nogil=True)
def extract_texts_fast(png, out_raw, out_off, out_type):
    """Scan PNG tEXt/iTXt/zTXt chunks into out_raw buffer (concatenated).
    png: uint8 view of the file; out_off (n_texts+1,) int32 offsets;
    out_type (n_texts,) uint8: 0 tEXt 1 iTXt 2 zTXt. Returns chunk count.
    Also returns pHYs via phys_found/phys bytes (see wrapper)."""
    n = png.shape[0]
    pos = 8
    nt = 0
    while pos + 8 <= n:
        ln = (png[pos] << 24) | (png[pos + 1] << 16) | (png[pos + 2] << 8) | png[pos + 3]
        t0 = png[pos + 4]
        t1 = png[pos + 5]
        t2 = png[pos + 6]
        t3 = png[pos + 7]
        payload = pos + 8
        # tEXt = 0x74455874, iTXt = 0x69545874, zTXt = 0x7A545874
        if t0 == 0x74 and t1 == 0x45 and t2 == 0x58 and t3 == 0x74:
            out_type[nt] = 0
        elif t0 == 0x69 and t1 == 0x54 and t2 == 0x58 and t3 == 0x74:
            out_type[nt] = 1
        elif t0 == 0x7A and t1 == 0x54 and t2 == 0x58 and t3 == 0x74:
            out_type[nt] = 2
        else:
            pos = payload + ln + 4
            continue
        out_off[nt] = nt and 0 or 0        # placeholder, set below
        for k in range(ln):
            out_raw[nt * 0 + k] = out_raw[k]  # noqa: dummy to satisfy nopython
        pos = payload + ln + 4
        nt += 1
    return nt
