"""PNG metadata extraction -> WebP XMP/EXIF/ICCP packaging with byte-exact
round-trip verification.

NAI-generated PNGs carry their prompt/parameters in tEXt chunks (Title,
Description, Software, Source, Generation_time, Comment) plus a pHYs
resolution chunk.  We store every metadata chunk's RAW payload base64-encoded
inside an XMP packet (so restoration is byte-exact for any encoding), the
original key/value as readable UTF-8 fields, eXIf chunk bytes into the WebP
EXIF chunk and iCCP profiles (decompressed) into the WebP ICCP chunk.
"""
import base64
import struct
import xml.sax.saxutils as sx

META_CHUNK_TYPES = ("tEXt", "iTXt", "zTXt", "pHYs", "eXIf", "iCCP")


def parse_png_chunks(data):
    """Return [(type, payload)] for a PNG byte stream (CRCs unchecked)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    out = []
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8].decode("latin1")
        payload = data[pos + 8:pos + 8 + ln]
        out.append((typ, payload))
        pos += 12 + ln
    return out


def extract_meta(png_data):
    """Collect metadata chunks. Returns dict:
    texts: [{type, key, value, raw}] in file order  (value: best-effort str)
    phys_raw / exif_raw / icc_raw: bytes or None (icc decompressed)
    """
    texts = []
    phys = exif = icc = None
    for typ, payload in parse_png_chunks(png_data):
        if typ in ("tEXt", "iTXt", "zTXt"):
            key, _, rest = payload.partition(b"\x00")
            if typ == "tEXt":
                value = rest.decode("latin1", "replace")
            elif typ == "zTXt":
                import zlib
                try:
                    value = zlib.decompress(rest[1:]).decode("latin1", "replace")
                except Exception:
                    value = ""
            else:  # iTXt: comp_flag, comp_method, lang\0, trans_key\0, text
                value = rest[5:].decode("utf-8", "replace")
            texts.append(dict(type=typ, key=key.decode("latin1", "replace"),
                              value=value, raw=payload))
        elif typ == "pHYs":
            phys = payload
        elif typ == "eXIf":
            exif = payload
        elif typ == "iCCP":
            import zlib
            _name, _, rest = payload.partition(b"\x00")
            try:
                icc = zlib.decompress(rest[1:])
            except Exception:
                icc = None
    return dict(texts=texts, phys_raw=phys, exif_raw=exif, icc_raw=icc)


_XMP_TMPL = (
    '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
    ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    '  <rdf:Description rdf:about=""\n'
    '    xmlns:pngmeta="https://gpuwebp.local/pngmeta/">\n'
    "{fields}"
    "  </rdf:Description>\n"
    " </rdf:RDF>\n"
    "</x:xmpmeta>\n"
    '<?xpacket end="w"?>\n'
)


def build_xmp(meta):
    """Build the XMP packet carrying every raw metadata chunk."""
    fields = []
    for i, t in enumerate(meta["texts"]):
        esc = sx.escape
        fields.append(
            f"   <pngmeta:Text{i}Type>{esc(t['type'])}</pngmeta:Text{i}Type>\n"
            f"   <pngmeta:Text{i}Key>{esc(t['key'])}</pngmeta:Text{i}Key>\n"
            f"   <pngmeta:Text{i}Value>{esc(t['value'])}</pngmeta:Text{i}Value>\n"
            f"   <pngmeta:Text{i}Raw>{base64.b64encode(t['raw']).decode()}"
            f"</pngmeta:Text{i}Raw>\n")
    if meta["phys_raw"] is not None:
        b64 = base64.b64encode(meta["phys_raw"]).decode()
        px, py, unit = struct.unpack(">IIB", meta["phys_raw"])
        fields.append(
            f"   <pngmeta:PhysPixelsX>{px}</pngmeta:PhysPixelsX>\n"
            f"   <pngmeta:PhysPixelsY>{py}</pngmeta:PhysPixelsY>\n"
            f"   <pngmeta:PhysUnit>{unit}</pngmeta:PhysUnit>\n"
            f"   <pngmeta:PhysRaw>{b64}</pngmeta:PhysRaw>\n")
    return _XMP_TMPL.format(fields="".join(fields)).encode("utf-8")


def _parse_xmp(xmp_bytes):
    """Pull our fields back out of the XMP packet (no XML dep, robust)."""
    import re
    s = xmp_bytes.decode("utf-8", "replace")
    out = dict(texts=[], phys_raw=None)
    for i in range(10000):
        m = re.search(rf"<pngmeta:Text{i}Raw>([A-Za-z0-9+/=]*)</pngmeta:Text{i}Raw>", s)
        if not m:
            break
        out["texts"].append(dict(raw=base64.b64decode(m.group(1))))
    m = re.search(r"<pngmeta:PhysRaw>([A-Za-z0-9+/=]*)</pngmeta:PhysRaw>", s)
    if m and m.group(1):
        out["phys_raw"] = base64.b64decode(m.group(1))
    return out


def extract_from_webp(webp_data):
    """Rebuild the meta structure from a WebP file's chunks."""
    from .webp_container import parse as parse_webp
    chunks = parse_webp(webp_data)
    meta = dict(texts=[], phys_raw=None, exif_raw=None, icc_raw=None)
    if "XMP " in chunks:
        back = _parse_xmp(chunks["XMP "])
        meta["texts"] = back["texts"]
        meta["phys_raw"] = back["phys_raw"]
    if "EXIF" in chunks:
        meta["exif_raw"] = chunks["EXIF"]
    if "ICCP" in chunks:
        meta["icc_raw"] = chunks["ICCP"]
    return meta


def verify_pre(meta, webp_data):
    """Byte-level check of an output webp against a PRE-EXTRACTED meta dict
    (same rules as verify(); used by the subprocess verifier)."""
    dst = extract_from_webp(webp_data)
    problems = []
    n_src = len(meta["texts"])
    if len(dst["texts"]) != n_src:
        problems.append(f"text chunk count {len(dst['texts'])} != {n_src}")
    for i in range(min(n_src, len(dst["texts"]))):
        if meta["texts"][i]["raw"] != dst["texts"][i]["raw"]:
            problems.append(f"text chunk {i} payload differs")
    if (meta["phys_raw"] or None) != (dst["phys_raw"] or None):
        problems.append("pHYs differs")
    if (meta["exif_raw"] or None) != (dst["exif_raw"] or None):
        problems.append("eXIf differs")
    if (meta["icc_raw"] or None) != (dst["icc_raw"] or None):
        problems.append("iCCP differs")
    return (not problems), problems


def verify(png_data, webp_data):
    """Byte-level check that every metadata chunk survived. Returns
    (ok, problems[list of str])."""
    src = extract_meta(png_data)
    dst = extract_from_webp(webp_data)
    problems = []
    n_src = len(src["texts"])
    if len(dst["texts"]) != n_src:
        problems.append(f"text chunk count {len(dst['texts'])} != {n_src}")
    for i in range(min(n_src, len(dst["texts"]))):
        if src["texts"][i]["raw"] != dst["texts"][i]["raw"]:
            problems.append(f"text chunk {i} payload differs")
    if (src["phys_raw"] or None) != (dst["phys_raw"] or None):
        problems.append("pHYs differs")
    if (src["exif_raw"] or None) != (dst["exif_raw"] or None):
        problems.append("eXIf differs")
    if (src["icc_raw"] or None) != (dst["icc_raw"] or None):
        problems.append("iCCP differs")
    return (not problems), problems


def restore_png_text_chunks(meta):
    """Original PNG metadata chunks (type, raw payload) for rebuilding."""
    out = [(t["type"], t["raw"]) for t in meta["texts"]]
    if meta.get("phys_raw"):
        out.append(("pHYs", meta["phys_raw"]))
    return out
