# -*- coding: utf-8 -*-
"""扩展格式输入 → WebP。

解码两层: imagecodecs 原生解码器(快, GIL-free)按魔数分发; 失败落
Pillow(格式覆盖最广)。统一输出 RGBA uint8。

元数据: JPEG 的 EXIF/ICC、WebP 源的 EXIF/XMP/ICC 块原样保留。
支持的输入(常见): PNG/JPEG/WebP/BMP/TIFF/GIF/JP2/JXL/AVIF/HEIC/
QOI/DDS/APNG + Pillow 插件覆盖的其余格式(TGA/ICO/PSD/PNM/SGI/...)。
"""
import io
import os
import struct

import numpy as np

# 采集扩展名(不含 PNG): 常见 + Pillow 生态格式
EXTRA_EXTS = (
    ".jpg", ".jpeg", ".jfif",
    ".webp",
    ".bmp", ".dib",
    ".tif", ".tiff",
    ".gif",
    ".jp2", ".j2k", ".jpc",
    ".jxl",
    ".avif", ".heic", ".heif", ".hif",
    ".qoi",
    ".dds",
    ".apng",
    ".tga", ".icb", ".vda", ".vst",
    ".ico", ".cur",
    ".psd", ".psb",
    ".ppm", ".pgm", ".pbm", ".pnm", ".pam",
    ".sgi", ".rgb", ".rgba", ".bw", ".int", ".inta",
    ".pcx", ".dcx",
    ".im", ".xpm", ".xbm",
    ".pic", ".pixar",
    ".msp", ".wal", ".mpt", ".tex",
    ".fit", ".fits", ".fts",
)


def decode_any(data):
    """任意支持格式 -> RGBA (H, W, 4) uint8; 失败抛异常。"""
    import imagecodecs
    from PIL import Image
    try:
        arr = _decode_native(imagecodecs, data)
        if arr is None:
            raise ValueError("unknown magic")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.concatenate([arr] * 3, -1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        return np.ascontiguousarray(arr[..., :4])
    except Exception:
        img = Image.open(io.BytesIO(data))
        if getattr(img, "is_animated", False):
            img.seek(0)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert(
                "RGBA" if "A" in img.getbands()
                or "transparency" in img.info else "RGB")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
        return np.ascontiguousarray(arr[..., :4])


def _decode_native(ic, d):
    """魔数 -> imagecodecs 解码器; 未知返回 None(交 Pillow)。"""
    if d[:3] == b"\xff\xd8\xff":
        return ic.jpeg_decode(d)
    if d[:4] == b"RIFF" and d[8:12] == b"WEBP":
        return ic.webp_decode(d)
    if d[:2] == b"BM":
        return ic.bmp_decode(d)
    if d[:4] in (b"II*\x00", b"MM\x00*"):
        return ic.tiff_decode(d)
    if d[:6] in (b"GIF87a", b"GIF89a"):
        a = ic.gif_decode(d)
        return a[0] if a.ndim == 4 else a
    if d[:4] == b"\xff\x4f\xff\x51":
        return ic.jpeg2k_decode(d)
    if d[:4] == b"II\xbc\x01":                      # JXL raw codestream
        return ic.jpegxl_decode(d)
    if len(d) > 12 and d[12:16] == b"JXL ":         # JXL container
        return ic.jpegxl_decode(d)
    if d[4:8] == b"ftyp":                            # AVIF/HEIC (ISOBMFF)
        try:
            return ic.avif_decode(d)
        except Exception:
            return ic.heif_decode(d)
    if d[:4] == b"DDS ":
        return ic.dds_decode(d)
    if d[:3] == b"QOI":
        return ic.qoi_decode(d)
    if d[:8] == b"\x89PNG\r\n\x1a\n":
        return ic.png_decode(d)
    return None


def _webp_meta(data):
    """RIFF WebP 容器里的 EXIF / XMP / ICCP 原始块。"""
    meta = {}
    try:
        pos = 12
        end = min(len(data), 8 + struct.unpack("<I", data[4:8])[0])
        while pos + 8 <= end:
            cc = data[pos:pos + 4]
            sz = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            body = data[pos + 8:pos + 8 + sz]
            if cc == b"EXIF" and "exif_raw" not in meta:
                meta["exif_raw"] = body
            elif cc == b"XMP " and "xmp_raw" not in meta:
                meta["xmp_raw"] = body
            elif cc == b"ICCP" and "icc_raw" not in meta:
                meta["icc_raw"] = body
            pos += 8 + sz + (sz & 1)
    except Exception:
        pass
    return meta


def _jpeg_meta(data):
    meta = {}
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        exif = img.info.get("exif")
        icc = img.info.get("icc_profile")
        if exif:
            meta["exif_raw"] = bytes(exif)
        if icc:
            meta["icc_raw"] = bytes(icc)
    except Exception:
        pass
    return meta


def decode_any_meta(data):
    """非 PNG 输入的 (arr, meta)。meta 字段与 png_meta.extract_meta 同名:
    exif_raw / icc_raw / xmp_raw。"""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return decode_any(data), _webp_meta(data)
    if data[:3] == b"\xff\xd8\xff":
        return decode_any(data), _jpeg_meta(data)
    return decode_any(data), {}


def collect_images(src, recursive):
    """所有支持格式的输入文件(含 PNG)。"""
    from .pipeline import collect_pngs
    extra = []
    if recursive:
        for root, _dirs, files in os.walk(src):
            for fn in files:
                if fn.lower().endswith(EXTRA_EXTS):
                    extra.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(src):
            if fn.lower().endswith(EXTRA_EXTS):
                extra.append(os.path.join(src, fn))
    return sorted(collect_pngs(src, recursive) + extra)
