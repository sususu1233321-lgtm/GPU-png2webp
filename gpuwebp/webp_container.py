"""WebP (RIFF) container assembly / parsing with metadata chunk injection."""

import struct


def _chunk(fourcc, payload):
    out = fourcc + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        out += b"\x00"
    return out


def make_simple(vp8_payload):
    """Simple format: RIFF/WEBP + single VP8 chunk."""
    body = _chunk(b"VP8 ", vp8_payload)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


def make_extended(vp8_payload, alpha=None, iccp=None, exif=None, xmp=None):
    """Extended format with VP8X (needed for alpha and/or metadata)."""
    flags = 0
    if iccp:
        flags |= 0x20
    if alpha is not None:
        flags |= 0x10
    if exif:
        flags |= 0x08
    if xmp:
        flags |= 0x04
    # canvas size from vp8 frame header (10th byte offset 6..9)
    w = vp8_payload[6] | (vp8_payload[7] << 8)
    h = vp8_payload[8] | (vp8_payload[9] << 8)
    vp8x = struct.pack("<I", flags & 0xFFFFFF) + \
        struct.pack("<I", (w - 1) & 0xFFFFFF)[:3] + \
        struct.pack("<I", (h - 1) & 0xFFFFFF)[:3]
    body = _chunk(b"VP8X", vp8x)
    if iccp:
        body += _chunk(b"ICCP", iccp)
    if alpha is not None:
        body += _chunk(b"ALPH", alpha)
    body += _chunk(b"VP8 ", vp8_payload)
    if exif:
        body += _chunk(b"EXIF", exif)
    if xmp:
        body += _chunk(b"XMP ", xmp)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


def parse(data):
    """Return dict of chunks: {fourcc: payload} (last occurrence wins)."""
    out = {}
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP file")
    pos = 12
    end = len(data)
    while pos + 8 <= end:
        fourcc = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        payload = data[pos + 8:pos + 8 + size]
        out[fourcc.decode("latin1")] = payload
        pos += 8 + size + (size & 1)
    return out
