"""High-level encoder: RGB(A) PNG in -> WebP bytes out (GPU or CPU engine)."""
import numpy as np

from . import vp8_encode
from . import vp8_tables as T
from .bool_coder import bool_encode
from numba import njit


@njit(cache=True, nogil=True)
def _encode_streams(mb_w, mb_h, bq, fl, use_skip, skip_proba, skip,
                     is_i4, i16_mode, uv_mode, i4_modes, y_dc, y_ac, uv_lv,
                     ops0, buf0, ops, buf):
    """partition0 + token partition + both bool encodes in one nogil call."""
    pos0 = vp8_encode.write_partition0(
        mb_w, mb_h, bq, -2, 0, fl, 0, use_skip, skip_proba,
        skip, is_i4, i16_mode, uv_mode, i4_modes, ops0)
    n0 = bool_encode(ops0[:pos0], buf0)
    pos = vp8_encode.write_token_partition(
        mb_w, mb_h, 0, 1, use_skip, skip, is_i4, y_dc, y_ac, uv_lv, ops)
    nb = bool_encode(ops[:pos], buf)
    return n0, nb, pos0, pos
from .webp_container import make_simple, make_extended
from .alpha_enc import make_alph_chunk


def _encode_from_yuv(y, u, v, quality, alpha, engine, device, meta=None):
    if engine == "gpu":
        from . import gpu_engine as GE
        import cupy as cp
        with cp.cuda.Device(device):
            return _gpu_encode(y, u, v, quality, alpha, meta)
    return _cpu_encode(y, u, v, quality, alpha, meta)


def _cpu_encode(y, u, v, quality, alpha, meta=None):
    H, W = y.shape
    mb_h, mb_w = (H + 15) // 16, (W + 15) // 16
    bq, y1, y2, uv_m, fl = vp8_encode.setup_quant(quality)
    Y = vp8_encode.pad_to_mb(y, mb_h, mb_w)
    U = vp8_encode.pad_to_mb(u, mb_h, mb_w, half=True)
    V = vp8_encode.pad_to_mb(v, mb_h, mb_w, half=True)
    r = vp8_encode.analyze(Y, U, V, y1, y2, uv_m, bq)
    is_i4, i16_mode, uv_mode, i4_modes = r[2], r[3], r[4], r[5]
    y_dc, y_ac, uv_lv, skip = r[6], r[7], r[8], r[9]
    return _assemble(W, H, mb_w, mb_h, bq, fl, is_i4, i16_mode, uv_mode,
                     i4_modes, y_dc, y_ac, uv_lv, skip, alpha, meta)


def _gpu_encode(y, u, v, quality, alpha, meta=None):
    import cupy as cp
    from . import gpu_engine as GE
    from .closed_loop_jit import closed_loop_full

    H, W = y.shape
    mb_h, mb_w = (H + 15) // 16, (W + 15) // 16
    bq, y1, y2, uv_m, fl = vp8_encode.setup_quant(quality)
    Y = GE.pad_to_mb_gpu(cp.asarray(y), mb_h, mb_w)
    U = GE.pad_to_mb_gpu(cp.asarray(u), mb_h, mb_w, half=True)
    V = GE.pad_to_mb_gpu(cp.asarray(v), mb_h, mb_w, half=True)

    # GPU: parallel mode decision; CPU: exact closed-loop levels + recon
    modes = GE.gpu_modes_pass(Y, U, V, y1)

    Yn = cp.asnumpy(Y)
    Un = cp.asnumpy(U)
    Vn = cp.asnumpy(V)
    y2ac = max(8, int(T.AC_TABLE2[bq]))
    y1deq = np.array([T.DC_TABLE[bq]] + [T.AC_TABLE[bq]] * 15, dtype=np.int64)
    y2deq = np.array([T.DC_TABLE[bq] * 2] + [y2ac] * 15, dtype=np.int64)
    uvdeq = np.array([T.DC_TABLE[max(0, min(117, bq - 2))]] + [T.AC_TABLE[bq]] * 15,
                     dtype=np.int64)
    y_dc, y_ac, uv_lv, _rY, _rU, _rV = closed_loop_full(
        Yn, Un, Vn,
        modes["is_i4"], modes["i16_mode"], modes["uv_mode"], modes["i4_modes"],
        y1.q, y1.iq, y1.bias, y1.zthresh, y1.sharpen,
        y2.q, y2.iq, y2.bias, y2.zthresh, y2.sharpen,
        uv_m.q, uv_m.iq, uv_m.bias, uv_m.zthresh, uv_m.sharpen,
        y1deq, y2deq, uvdeq)

    skip = ~(y_dc.any(-1) | y_ac.any(-1).any(-1) | uv_lv.any(-1).any(-1))
    return _assemble(W, H, mb_w, mb_h, bq, fl,
                     modes["is_i4"], modes["i16_mode"], modes["uv_mode"],
                     modes["i4_modes"], y_dc, y_ac, uv_lv, skip, alpha, meta)


def _assemble(W, H, mb_w, mb_h, bq, fl, is_i4, i16_mode, uv_mode,
              i4_modes, y_dc, y_ac, uv_lv, skip, alpha, meta=None):
    n_mb = mb_w * mb_h
    skip_proba = (n_mb - int(skip.sum())) * 255 // n_mb
    use_skip = skip_proba < 250

    ops0 = np.empty(n_mb * 256 + 4096, dtype=np.int32)
    buf0 = np.empty(n_mb * 300 + 4096, dtype=np.uint8)
    ops = np.empty(n_mb * 8200 + 64, dtype=np.int32)
    buf = np.empty(n_mb * 2100 + 4096, dtype=np.uint8)
    n0, nb, pos0, pos = _encode_streams(
        mb_w, mb_h, bq, fl, use_skip, skip_proba, skip,
        is_i4, i16_mode, uv_mode, i4_modes, y_dc, y_ac, uv_lv,
        ops0, buf0, ops, buf)
    p0 = buf0[:n0].tobytes()
    parts = [buf[:nb].tobytes()]

    vp8 = bytearray()
    vp8 += ((1 << 4) | (len(p0) << 5)).to_bytes(3, "little")
    vp8 += (0x9D012A).to_bytes(3, "big")
    vp8 += (W & 0x3FFF).to_bytes(2, "little")
    vp8 += (H & 0x3FFF).to_bytes(2, "little")
    vp8 += p0
    for pb in parts:
        vp8 += pb
    if len(vp8) & 1:
        vp8 += b"\x00"

    xmp = exif = iccp = None
    if meta is not None:
        from .png_meta import build_xmp
        if meta.get("texts") or meta.get("phys_raw") is not None:
            xmp = build_xmp(meta)
        exif = meta.get("exif_raw")
        iccp = meta.get("icc_raw")
    if alpha is not None or xmp or exif or iccp:
        return make_extended(bytes(vp8), alpha=alpha, xmp=xmp,
                             exif=exif, iccp=iccp)
    return make_simple(bytes(vp8))


def encode_rgba(rgba, quality=90, engine="gpu", device=0, meta=None,
                png_data=None):
    """rgba: uint8 (H,W,4). meta: dict from png_meta.extract_meta (or
    png_data: raw PNG bytes to extract it from). Returns webp bytes."""
    H, W = rgba.shape[:2]
    if H % 2 or W % 2:
        raise ValueError("image dimensions must be even")
    from .rgb import rgb_to_yuv420
    y, u, v = rgb_to_yuv420(rgba)
    a = rgba[..., 3]
    has_alpha = not bool((a == 255).all())
    alpha = make_alph_chunk(a) if has_alpha else None
    if meta is None and png_data is not None:
        from .png_meta import extract_meta
        meta = extract_meta(png_data)
    return _encode_from_yuv(y, u, v, quality, alpha, engine, device, meta)


def encode_rgb(rgb, quality=90, engine="gpu", device=0):
    H, W = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], axis=2)
    return encode_rgba(rgba, quality, engine, device)
