import base64, zlib
_KEY = b"GPUYaTu-2024-" + bytes.fromhex("9d012abde0") + b"#webp"

def dec(blob):
    raw = base64.b85decode(blob)
    return zlib.decompress(bytes(
        b ^ _KEY[i % len(_KEY)] for i, b in enumerate(raw))
    ).decode()
