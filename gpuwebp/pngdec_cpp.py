"""C++ PNG helpers: metadata scan (and full decode fallback) via pngdec.dll.

The metadata scanner replaces the pure-Python extract_meta in the batch hot
path -- the Python version sliced every IDAT chunk (multi-MB copies per
image); the C++ scan touches metadata chunks only.
"""
import ctypes
import os
import zlib

import numpy as np

_dll = None


def _load():
    global _dll
    if _dll is not None:
        return _dll
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in (os.path.join(root, "cpp", "pngdec.dll"),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pngdec.dll")):
            if os.path.isfile(p):
                _dll = ctypes.CDLL(p)
                _dll.png_meta_scan.restype = ctypes.c_int
                _dll.png_meta_scan.argtypes = (
                    [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
                     ctypes.c_int, ctypes.c_void_p])
                _dll.png_decode_full.restype = ctypes.c_int
                _dll.png_decode_full.argtypes = (
                    [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p,
                     ctypes.c_int] + [ctypes.POINTER(ctypes.c_int)] * 2 +
                    [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])
                return _dll
    except OSError:
        pass
    _dll = False
    return _dll


_TYPES = {0: "tEXt", 1: "zTXt", 2: "iTXt"}


def _parse(buf):
    """Deserialize the C++ metadata blob into the extract_meta dict shape."""
    pos = 0

    def u32():
        nonlocal pos
        v = int.from_bytes(buf[pos:pos + 4], "little")
        pos += 4
        return v

    n = u32()
    texts = []
    for _ in range(n):
        typ = _TYPES[u32()]
        ln = u32()
        raw = bytes(buf[pos:pos + ln])
        pos += ln
        key, _, rest = raw.partition(b"\x00")
        if typ == "tEXt":
            value = rest.decode("latin1", "replace")
        elif typ == "zTXt":
            try:
                value = zlib.decompress(rest[1:]).decode("latin1", "replace")
            except Exception:                       # noqa: BLE001
                value = ""
        else:                                       # iTXt
            value = rest[5:].decode("utf-8", "replace")
        texts.append(dict(type=typ, key=key.decode("latin1", "replace"),
                          value=value, raw=raw))

    def blob():
        nonlocal pos
        ln = u32()
        b = bytes(buf[pos:pos + ln]) if ln else None
        pos += ln
        return b

    phys = blob()
    exif = blob()
    icc_raw = blob()
    icc = None
    if icc_raw:
        _name, _, rest = icc_raw.partition(b"\x00")
        try:
            icc = zlib.decompress(rest[1:])
        except Exception:                           # noqa: BLE001
            icc = None
    return dict(texts=texts, phys_raw=phys, exif_raw=exif, icc_raw=icc)


def extract_meta_cpp(png_data):
    """Drop-in for png_meta.extract_meta via the C++ scanner.
    Returns None if the DLL is unavailable or the scan fails."""
    dll = _load()
    if not dll:
        return None
    buf = np.empty(4 << 20, np.uint8)
    mlen = ctypes.c_int()
    r = dll.png_meta_scan(png_data, len(png_data),
                          buf.ctypes.data_as(ctypes.c_void_p), buf.size,
                          ctypes.byref(mlen))
    if r != 0:
        return None
    return _parse(memoryview(buf)[:mlen.value])


def decode_png_cpp(png_data):
    """Full C++ decode -> (H, W, 4) uint8 array, or None if unsupported."""
    dll = _load()
    if not dll:
        return None
    # probe for dimensions first
    import struct
    if len(png_data) < 33 or png_data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", png_data[16:24])
    if (png_data[24] != 8 or png_data[25] not in (2, 6)
            or png_data[28] != 0):
        return None
    arr = np.empty(h * w * 4, np.uint8)
    meta_buf = np.empty(4 << 20, np.uint8)
    W = ctypes.c_int(w)
    H = ctypes.c_int(h)
    mlen = ctypes.c_int()
    r = dll.png_decode_full(png_data, len(png_data),
                            arr.ctypes.data_as(ctypes.c_void_p), arr.size,
                            ctypes.byref(W), ctypes.byref(H),
                            meta_buf.ctypes.data_as(ctypes.c_void_p),
                            meta_buf.size, ctypes.byref(mlen))
    if r != 0:
        return None
    return arr.reshape(h, w, 4)
