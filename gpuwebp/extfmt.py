# -*- coding: utf-8 -*-
"""扩展格式输入: JPEG/BMP/TIFF/GIF/WebP/JP2 → WebP。

- 解码: imagecodecs 各格式解码器 + Pillow 回退, 统一输出 RGBA uint8
- 元数据: JPEG 的 EXIF/ICC、WebP 源的 EXIF/XMP/ICC 块原样保留
"""
import io
import os
import struct

import numpy as np


def decode_any(data):
    """任意支持格式 → RGBA (H, W, 4) uint8; 失败抛异常。"""
    import imagecodecs
    from PIL import Image
    try:
        if data[:3] == b"\xff\xd8\xff":
            arr = imagecodecs.jpeg_decode(data)
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            arr = imagecodecs.webp_decode(data)
        elif data[:2] == b"BM":
            arr = imagecodecs.bmp_decode(data)
        elif data[:4] in (b"II*\x00", b"MM\x00*"):
            arr = imagecodecs.tiff_decode(data)
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            arr = imagecodecs.gif_decode(data)
            if arr.ndim == 4:            # 动图: 取第一帧
                arr = arr[0]
        elif data[:4] == b"\xff\x4f\xff\x51":
            arr = imagecodecs.jpeg2k_decode(data)
        else:
            raise ValueError("未知格式")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
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
            img = img.convert("RGBA")
        return np.ascontiguousarray(np.asarray(img))


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
    exts = (".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
            ".webp", ".gif", ".jp2")
    if recursive:
        for root, _dirs, files in os.walk(src):
            for fn in files:
                if fn.lower().endswith(exts):
                    extra.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(src):
            if fn.lower().endswith(exts):
                extra.append(os.path.join(src, fn))
    return sorted(collect_pngs(src, recursive) + extra)
