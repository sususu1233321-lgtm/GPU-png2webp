"""GPU (CuPy) encode engine.

Replaces the two CPU bottlenecks (mode decision + closed-loop residual) with
fully parallel batch kernels, using a two-pass iterative scheme:

  pass 1:  context = source plane  -> modes + quantized levels + recon1
  pass 2:  context = recon1        -> final levels (context now matches what
           the decoder will actually use -> residuals are near-closed-loop)

All fixed-point math is a vectorized port of the validated scalar reference
(vp8_encode / closed_loop), matching the libwebp decoder semantics.
"""
import numpy as np
import cupy as cp

from . import vp8_tables as T
from .vp8_encode import setup_quant


def _patch_nvrtc_include():
    """NVRTC cannot open cupy's bundled headers when they live on a path with
    non-ASCII characters (e.g. a Chinese install folder).  Mirror the headers
    to a stable ASCII location and add it as an extra -I for every NVRTC
    compile (harmless no-op on ASCII installs)."""
    import os
    import shutil
    try:
        cp.__file__.encode("ascii")
        return                      # cupy package path is ASCII: nothing to do
    except UnicodeEncodeError:
        pass
    base = None
    for cand in (os.environ.get("LOCALAPPDATA"),
                 os.environ.get("SystemDrive", "C:") + os.sep
                 + os.path.join("Users", "Public")):
        if cand:
            try:
                cand.encode("ascii")
            except UnicodeEncodeError:
                continue
            if os.path.isdir(cand):
                base = os.path.join(cand, "gpupic_cuda_shim")
                break
    if not base:
        return
    src_inc = os.path.join(os.path.dirname(cp.__file__), "_core", "include")
    dst_inc = os.path.join(base, "include")
    try:
        stamp = os.path.join(base, ".stamp")
        if not os.path.exists(stamp) and os.path.isdir(src_inc):
            shutil.rmtree(base, ignore_errors=True)
            shutil.copytree(src_inc, dst_inc)
            os.makedirs(os.path.join(base, "bin"), exist_ok=True)
            open(stamp, "w").write("1")
        if not os.path.isdir(dst_inc):
            return
        from cupy.cuda import compiler
        orig = compiler._NVRTCProgram.compile

        def patched(self, options=(), log_stream=None):
            return orig(self, tuple(options) + (f"-I{dst_inc}",), log_stream)

        compiler._NVRTCProgram.compile = patched
    except Exception:
        pass                        # keep going; Pillow fallback covers us


_patch_nvrtc_include()

ZIG = cp.array(T.ZIGZAG)

RD_MULT = 256
LAMBDA_D_I16 = 106
LAMBDA_D_I4 = 11
LAMBDA_D_UV = 120
FIXED_COSTS_I16 = cp.array(T.FIXED_COSTS_I16, dtype=cp.int64)
FIXED_COSTS_UV = cp.array(T.FIXED_UV if False else T.FIXED_COSTS_UV, dtype=cp.int64)


def _mul1(a):
    return ((a * 20091) >> 16) + a


def _mul2(a):
    return (a * 35468) >> 16


def _clip(v):
    return cp.minimum(cp.maximum(v, 0), 255)


def rgb_to_yuv420_gpu(rgba, int16_out=False):
    """rgba (H,W,4) or (B,H,W,4) uint8 -> y/u/v planes (batched leading dim).
    int16_out: emit int16 planes directly (what the encode kernels consume);
    for 16-aligned images this makes padding a zero-copy no-op."""
    r = rgba[..., 0].astype(cp.int32)
    g = rgba[..., 1].astype(cp.int32)
    b = rgba[..., 2].astype(cp.int32)
    YFIX, HALF = 16, 1 << 15
    y = (16839 * r + 33059 * g + 6420 * b + HALF + (16 << YFIX)) >> YFIX
    pre = y.shape[:-2]
    H, W = y.shape[-2:]
    r2 = r.reshape(pre + (H // 2, 2, W // 2, 2)).sum(axis=(-3, -1))
    g2 = g.reshape(pre + (H // 2, 2, W // 2, 2)).sum(axis=(-3, -1))
    b2 = b.reshape(pre + (H // 2, 2, W // 2, 2)).sum(axis=(-3, -1))

    def clip_uv(v):
        x = (v + (HALF << 2) + (128 << YFIX << 2)) >> (YFIX + 2)
        return cp.clip(x, 0, 255)

    u = clip_uv(-9719 * r2 - 19081 * g2 + 28800 * b2)
    v = clip_uv(28800 * r2 - 24116 * g2 - 4684 * b2)
    if int16_out:
        return (cp.clip(y, 0, 255).astype(cp.int16),
                u.astype(cp.int16), v.astype(cp.int16))
    return (cp.clip(y, 0, 255).astype(cp.uint8), u.astype(cp.uint8), v.astype(cp.uint8))


def pad_to_mb_gpu(p, mb_h, mb_w, half=False):
    bh = mb_h * 16 // (2 if half else 1)
    bw = mb_w * 16 // (2 if half else 1)
    p = p.astype(cp.int16)
    if p.shape == (bh, bw):
        return p
    out = cp.zeros((bh, bw), dtype=cp.int16)
    h, w = p.shape
    out[:h, :w] = p
    if w < bw:
        out[:, w:] = out[:, w - 1:w]
    if h < bh:
        out[h:, :] = out[h - 1:h, :]
    return out


def make_borders(P):
    """P (H,W) int16 -> bordered (H+1, W+1) with top row 127, left col 129."""
    H, W = P.shape
    B = cp.zeros((H + 1, W + 1), dtype=cp.int16)
    B[0, :] = 127
    B[:, 0] = 129
    B[1:, 1:] = P
    return B


def gather_block_edges(B, gh, gw, blk):
    H, W = B.shape[0] - 1, B.shape[1] - 1
    n = gh * gw
    by = (cp.arange(n) // gw) * blk
    bx = (cp.arange(n) % gw) * blk
    rows = by[:, None] + cp.zeros((1, blk), dtype=cp.int64)
    cols = bx[:, None] + 1 + cp.arange(blk)[None, :]
    top = B[rows, cols].astype(cp.int32)
    lrows = by[:, None] + 1 + cp.arange(blk)[None, :]
    left = B[lrows, bx[:, None]].astype(cp.int32)
    X = B[by, bx].astype(cp.int32)
    return top, left, X


def block_preds_batch(top, left, X, blk, valid_t, valid_l):
    st = top.sum(axis=1)
    sl = left.sum(axis=1)
    sh = 4 if blk == 8 else 5
    dc = cp.where(valid_t & valid_l, (st + sl + blk) >> sh, 0)
    dc = cp.where(valid_t & ~valid_l, (st + blk // 2) >> (sh - 1), dc)
    dc = cp.where(~valid_t & valid_l, (sl + blk // 2) >> (sh - 1), dc)
    dc = cp.where(~valid_t & ~valid_l, 128, dc)
    preds = cp.empty((4,) + top.shape + (blk,), dtype=cp.int32)
    preds[0] = dc[:, None, None]
    preds[1] = _clip(top[:, None, :] + left[:, :, None] - X[:, None, None])
    preds[2] = cp.broadcast_to(top[:, None, :], top.shape + (blk,))
    preds[3] = cp.broadcast_to(left[:, :, None], top.shape + (blk,))
    return preds


def i4_edges_batch(B, gh, gw):
    H, W = B.shape[0] - 1, B.shape[1] - 1
    n = gh * gw
    gy = cp.arange(n) // gw
    gx = cp.arange(n) % gw
    py = gy * 4
    px = gx * 4
    mbx = gx // 4
    mby = gy // 4
    tt = cp.empty((n, 9), dtype=cp.int32)
    tt[:, 0] = B[py, px]
    for k in range(4):
        tt[:, 1 + k] = B[py, px + 1 + k]
    for k in range(4):
        # TR[k] = row above the subblock, column px+4+k; for x==3 subblocks
        # that row is the one above the MB (px+4+k already equals (mbx+1)*16+k)
        c = cp.minimum(px + 4 + k, W - 1)
        src_row = cp.where(gx % 4 == 3, mby * 16, py)
        tt[:, 5 + k] = B[src_row, c + 1]
    lrows = py[:, None] + 1 + cp.arange(4)[None, :]
    L = B[lrows, px[:, None]]
    return tt.astype(cp.int32), L.astype(cp.int32)


def i4_preds_batch(tt, L):
    """All ten 4x4 predictions, vectorized (port of dsp/dec.c).
    tt (N,9)=[X,T0..3,TR0..3], L (N,4). -> (10,N,4,4) int32."""
    X = tt[:, 0]
    A, B, C, D, E, F, G, H = (tt[:, 1 + i] for i in range(8))
    l0, l1, l2, l3 = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    N = tt.shape[0]
    p = cp.zeros((10, N, 4, 4), dtype=cp.int32)
    a2 = lambda a, b: (a + b + 1) >> 1
    a3 = lambda a, b, c: (a + 2 * b + c + 2) >> 2
    # 0 BDC_PRED
    p[0] = ((4 + A + B + C + D + l0 + l1 + l2 + l3) >> 3)[:, None, None]
    # 1 BTM_PRED
    p[1] = _clip(tt[:, 1:5][:, None, :] + L[:, :, None] - X[:, None, None])
    # 2 BVE_PRED: each row = [a3(X,A,B), a3(A,B,C), a3(B,C,D), a3(C,D,E)]
    p[2, :, :, 0] = a3(X, A, B)[:, None]
    p[2, :, :, 1] = a3(A, B, C)[:, None]
    p[2, :, :, 2] = a3(B, C, D)[:, None]
    p[2, :, :, 3] = a3(C, D, E)[:, None]
    # 3 BHE_PRED: rows a3(X,l0,l1), a3(l0,l1,l2), a3(l1,l2,l3), a3(l2,l3,l3)
    for r, v in enumerate([a3(X, l0, l1), a3(l0, l1, l2), a3(l1, l2, l3), a3(l2, l3, l3)]):
        p[3, :, r, :] = v[:, None]
    I, J, K, Lm = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    # modes 4..9 auto-generated (verified vs _i4_preds)
    p[4, :, 0, 0] = a3(A, X, I)
    p[4, :, 0, 1] = a3(X, A, B)
    p[4, :, 0, 2] = a3(A, B, C)
    p[4, :, 0, 3] = a3(B, C, D)
    p[4, :, 1, 0] = a3(X, I, J)
    p[4, :, 1, 1] = a3(A, X, I)
    p[4, :, 1, 2] = a3(X, A, B)
    p[4, :, 1, 3] = a3(A, B, C)
    p[4, :, 2, 0] = a3(I, J, K)
    p[4, :, 2, 1] = a3(X, I, J)
    p[4, :, 2, 2] = a3(A, X, I)
    p[4, :, 2, 3] = a3(X, A, B)
    p[4, :, 3, 0] = a3(J, K, Lm)
    p[4, :, 3, 1] = a3(I, J, K)
    p[4, :, 3, 2] = a3(X, I, J)
    p[4, :, 3, 3] = a3(A, X, I)
    p[5, :, 0, 0] = a2(X, A)
    p[5, :, 0, 1] = a2(A, B)
    p[5, :, 0, 2] = a2(B, C)
    p[5, :, 0, 3] = a2(C, D)
    p[5, :, 1, 0] = a3(A, X, I)
    p[5, :, 1, 1] = a3(X, A, B)
    p[5, :, 1, 2] = a3(A, B, C)
    p[5, :, 1, 3] = a3(B, C, D)
    p[5, :, 2, 0] = a3(X, I, J)
    p[5, :, 2, 1] = a2(X, A)
    p[5, :, 2, 2] = a2(A, B)
    p[5, :, 2, 3] = a2(B, C)
    p[5, :, 3, 0] = a3(I, J, K)
    p[5, :, 3, 1] = a3(A, X, I)
    p[5, :, 3, 2] = a3(X, A, B)
    p[5, :, 3, 3] = a3(A, B, C)
    p[6, :, 0, 0] = a3(A, B, C)
    p[6, :, 0, 1] = a3(B, C, D)
    p[6, :, 0, 2] = a3(C, D, E)
    p[6, :, 0, 3] = a3(D, E, F)
    p[6, :, 1, 0] = a3(B, C, D)
    p[6, :, 1, 1] = a3(C, D, E)
    p[6, :, 1, 2] = a3(D, E, F)
    p[6, :, 1, 3] = a3(E, F, G)
    p[6, :, 2, 0] = a3(C, D, E)
    p[6, :, 2, 1] = a3(D, E, F)
    p[6, :, 2, 2] = a3(E, F, G)
    p[6, :, 2, 3] = a3(F, G, H)
    p[6, :, 3, 0] = a3(D, E, F)
    p[6, :, 3, 1] = a3(E, F, G)
    p[6, :, 3, 2] = a3(F, G, H)
    p[6, :, 3, 3] = a3(G, H, H)
    p[7, :, 0, 0] = a2(A, B)
    p[7, :, 0, 1] = a2(B, C)
    p[7, :, 0, 2] = a2(C, D)
    p[7, :, 0, 3] = a2(D, E)
    p[7, :, 1, 0] = a3(A, B, C)
    p[7, :, 1, 1] = a3(B, C, D)
    p[7, :, 1, 2] = a3(C, D, E)
    p[7, :, 1, 3] = a3(D, E, F)
    p[7, :, 2, 0] = a2(B, C)
    p[7, :, 2, 1] = a2(C, D)
    p[7, :, 2, 2] = a2(D, E)
    p[7, :, 2, 3] = a3(E, F, G)
    p[7, :, 3, 0] = a3(B, C, D)
    p[7, :, 3, 1] = a3(C, D, E)
    p[7, :, 3, 2] = a3(D, E, F)
    p[7, :, 3, 3] = a3(F, G, H)
    p[8, :, 0, 0] = a2(X, I)
    p[8, :, 0, 1] = a3(A, X, I)
    p[8, :, 0, 2] = a3(X, A, B)
    p[8, :, 0, 3] = a3(A, B, C)
    p[8, :, 1, 0] = a2(I, J)
    p[8, :, 1, 1] = a3(X, I, J)
    p[8, :, 1, 2] = a2(X, I)
    p[8, :, 1, 3] = a3(A, X, I)
    p[8, :, 2, 0] = a2(J, K)
    p[8, :, 2, 1] = a3(I, J, K)
    p[8, :, 2, 2] = a2(I, J)
    p[8, :, 2, 3] = a3(X, I, J)
    p[8, :, 3, 0] = a2(K, Lm)
    p[8, :, 3, 1] = a3(J, K, Lm)
    p[8, :, 3, 2] = a2(J, K)
    p[8, :, 3, 3] = a3(I, J, K)
    p[9, :, 0, 0] = a2(I, J)
    p[9, :, 0, 1] = a3(I, J, K)
    p[9, :, 0, 2] = a2(J, K)
    p[9, :, 0, 3] = a3(J, K, Lm)
    p[9, :, 1, 0] = a2(J, K)
    p[9, :, 1, 1] = a3(J, K, Lm)
    p[9, :, 1, 2] = a2(K, Lm)
    p[9, :, 1, 3] = a3(K, Lm, Lm)
    p[9, :, 2, 0] = a2(K, Lm)
    p[9, :, 2, 1] = a3(K, Lm, Lm)
    p[9, :, 2, 2] = Lm
    p[9, :, 2, 3] = Lm
    p[9, :, 3, 0] = Lm
    p[9, :, 3, 1] = Lm
    p[9, :, 3, 2] = Lm
    p[9, :, 3, 3] = Lm
    return p


# ---------------------------------------------------------------- transforms

def fdct_batch(res):
    """FTransform_C on (...,4,4) int32 residual -> (...,16) slots int32."""
    d0, d1, d2, d3 = res[..., 0], res[..., 1], res[..., 2], res[..., 3]
    a0 = d0 + d3; a1 = d1 + d2; a2 = d1 - d2; a3 = d0 - d3
    t = cp.empty(res.shape[:-2] + (16,), dtype=cp.int64)
    t[..., 0::4] = (a0 + a1) * 8
    t[..., 1::4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9
    t[..., 2::4] = (a0 - a1) * 8
    t[..., 3::4] = (a3 * 2217 - a2 * 5352 + 937) >> 9
    # horizontal pass: for each column i (0..3) combine T rows 0,1,2,3 at stride 4
    t4 = t.reshape(t.shape[:-1] + (4, 4))       # [row r, col i]
    a0 = t4[..., 0, :] + t4[..., 3, :]
    a1 = t4[..., 1, :] + t4[..., 2, :]
    a2 = t4[..., 1, :] - t4[..., 2, :]
    a3 = t4[..., 0, :] - t4[..., 3, :]
    o4 = cp.empty(t4.shape, dtype=cp.int64)
    o4[..., 0, :] = (a0 + a1 + 7) >> 4
    o4[..., 1, :] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0)
    o4[..., 2, :] = (a0 - a1 + 7) >> 4
    o4[..., 3, :] = (a3 * 2217 - a2 * 5352 + 51000) >> 16
    return o4.reshape(t.shape).astype(cp.int32)


def fwht_batch(in256):
    """FTransformWHT_C: (...,256) 16 blocks coeffs -> (...,16)."""
    shp = in256.shape[:-1]
    def blk(i, j):   # block i (0..3), coefficient j
        return in256[..., i * 64 + j * 16].astype(cp.int64)
    t = cp.empty(shp + (16,), dtype=cp.int64)
    for i in range(4):
        a0 = blk(i, 0) + blk(i, 2)
        a1 = blk(i, 1) + blk(i, 3)
        a2 = blk(i, 1) - blk(i, 3)
        a3 = blk(i, 0) - blk(i, 2)
        t[..., 0 + i * 4] = a0 + a1
        t[..., 1 + i * 4] = a3 + a2
        t[..., 2 + i * 4] = a3 - a2
        t[..., 3 + i * 4] = a0 - a1
    out = cp.empty(shp + (16,), dtype=cp.int64)
    for i in range(4):
        a0 = t[..., 0 + i] + t[..., 8 + i]
        a1 = t[..., 4 + i] + t[..., 12 + i]
        a2 = t[..., 4 + i] - t[..., 12 + i]
        a3 = t[..., 0 + i] - t[..., 8 + i]
        out[..., 0 + i] = (a0 + a1) >> 1
        out[..., 4 + i] = (a3 + a2) >> 1
        out[..., 8 + i] = (a3 - a2) >> 1
        out[..., 12 + i] = (a0 - a1) >> 1
    return out


def quant_batch(coeff, mtx, first):
    """QuantizeBlock_C on (...,16) slot coeffs -> (...,15|16) zigzag int16."""
    cz = coeff[..., ZIG]
    sh = cp.asarray(mtx.sharpen)[ZIG]
    iq = cp.asarray(mtx.iq)[ZIG]
    bias = cp.asarray(mtx.bias)[ZIG]
    zt = cp.asarray(mtx.zthresh)[ZIG]
    x = cp.abs(cz) + sh
    lvl = cp.where(x > zt, (x * iq + bias) >> 17, 0)
    lvl = cp.minimum(lvl, 2047)
    lvl = lvl * cp.sign(cz)
    if first:
        lvl = lvl[..., 1:]
    return lvl.astype(cp.int16)


def dequant_batch(levels, deq2, first):
    """levels (...,K) zigzag -> t16 (...,16) slots."""
    if first:
        t = cp.zeros(levels.shape[:-1] + (16,), dtype=cp.int32)
        t[..., ZIG[1:]] = levels * deq2[1]
    else:
        t = cp.zeros(levels.shape[:-1] + (16,), dtype=cp.int32)
        t[..., ZIG] = levels * cp.where(cp.arange(16) == 0, deq2[0], deq2[1])[ZIG]
    return t


def idct_full_batch(t16, ref):
    """ITransformOne_C -> (...,4,4) int32 clipped reconstruction."""
    in_ = t16.reshape(t16.shape[:-1] + (4, 4))
    a = in_[..., 0, :] + in_[..., 2, :]
    b = in_[..., 0, :] - in_[..., 2, :]
    c = _mul2(in_[..., 1, :]) - _mul1(in_[..., 3, :])
    d = _mul1(in_[..., 1, :]) + _mul2(in_[..., 3, :])
    # tmp[i, k] = C[i*4 + k]; horizontal reads rows: flat[0+i]=row0, flat[8+i]=row2
    tmp = cp.stack([a + d, b + c, b - c, a - d], axis=-1)
    dc = tmp[..., 0, :] + 4
    a = dc + tmp[..., 2, :]
    b = dc - tmp[..., 2, :]
    c = _mul2(tmp[..., 1, :]) - _mul1(tmp[..., 3, :])
    d = _mul1(tmp[..., 1, :]) + _mul2(tmp[..., 3, :])
    out = cp.empty(tmp.shape, dtype=cp.int32)
    out[..., 0] = ref[..., 0] + ((a + d) >> 3)
    out[..., 1] = ref[..., 1] + ((b + c) >> 3)
    out[..., 2] = ref[..., 2] + ((b - c) >> 3)
    out[..., 3] = ref[..., 3] + ((a - d) >> 3)
    return cp.clip(out, 0, 255)


def iwht_batch(in16):
    """TransformWHT_C (decoder inverse WHT): (...,16) -> (...,256) block DCs."""
    shp = in16.shape[:-1]
    in16 = in16.astype(cp.int64)
    t = cp.empty(shp + (16,), dtype=cp.int64)
    for i in range(4):
        a0 = in16[..., 0 + i] + in16[..., 12 + i]
        a1 = in16[..., 4 + i] + in16[..., 8 + i]
        a2 = in16[..., 4 + i] - in16[..., 8 + i]
        a3 = in16[..., 0 + i] - in16[..., 12 + i]
        t[..., 0 + i] = a0 + a1
        t[..., 8 + i] = a0 - a1
        t[..., 4 + i] = a3 + a2
        t[..., 12 + i] = a3 - a2
    out = cp.zeros(shp + (256,), dtype=cp.int32)
    for i in range(4):
        dc = t[..., 0 + i * 4] + 3
        a0 = dc + t[..., 3 + i * 4]
        a1 = t[..., 1 + i * 4] + t[..., 2 + i * 4]
        a2 = t[..., 1 + i * 4] - t[..., 2 + i * 4]
        a3 = dc - t[..., 3 + i * 4]
        out[..., i * 64 + 0] = (a0 + a1) >> 3
        out[..., i * 64 + 16] = (a3 + a2) >> 3
        out[..., i * 64 + 32] = (a0 - a1) >> 3
        out[..., i * 64 + 48] = (a3 - a2) >> 3
    return out


# ---------------------------------------------------------------- full pass

def gpu_modes_pass(Y, U, V, y1):
    """Mode decision only (open loop, context = source borders).
    Returns dict(is_i4, i16_mode, uv_mode, i4_modes)."""
    return gpu_modes_pass_batch(Y[None], U[None], V[None], y1, split=False)


def gpu_modes_pass_batch(Yb, Ub, Vb, y1, split=True, select=True):
    """Batched mode decision. Yb (B,H,W) int16 padded luma (all images share
    the same padded dims), Ub/Vb (B,H/2,W/2).  split=True returns a list of
    per-image dicts (is_i4/i16_mode/uv_mode/i4_modes, select_modes applied);
    split=False applies select_modes on image 0 only (legacy single call).
    Mode scores come from the fused single-launch CUDA kernel (bit-identical
    to the former vectorized path)."""
    from .closed_loop_gpu import mode_search_batch_gpu
    if not select:
        return mode_search_batch_gpu(Yb, Ub, Vb, y1)
    _raw = mode_search_batch_gpu(Yb, Ub, Vb, y1)
    return _select_from_raw(_raw, len(Yb), y1, split)


def _select_from_raw(raw, B, y1, split):
    import numpy as _np
    from .vp8_encode import select_modes as _sel
    n_mb = raw["i16_mode"].shape[0] // B
    out = []
    for i in range(B):
        if not split and i > 0:
            break
        i4_modes_np, is_i4_np, _ = _sel(
            _np.ascontiguousarray(raw["sse4"][i]),
            raw["i16_score"][i * n_mb:(i + 1) * n_mb],
            raw["i16_mode"][i * n_mb:(i + 1) * n_mb],
            raw["mb_w"], raw["mb_h"], 1000 * y1.q_avg * y1.q_avg)
        out.append(dict(is_i4=is_i4_np.astype(bool),
                        i16_mode=raw["i16_mode"][i * n_mb:(i + 1) * n_mb],
                        uv_mode=raw["uv_mode"][i * n_mb:(i + 1) * n_mb],
                        i4_modes=i4_modes_np))
    return out if split else out[0]


def _gpu_modes_pass_batch_vec(Yb, Ub, Vb, y1, split=True, select=True):
    """Former vectorized implementation (kept as reference/debug path)."""
    B, H, W = Yb.shape
    mb_h, mb_w = H // 16, W // 16
    n_mb = mb_h * mb_w
    gh, gw = mb_h * 4, mb_w * 4
    N = B * n_mb
    img_stride = H + 1
    N4img = gh * gw

    def borders(Pb):
        b, h, w = Pb.shape
        out = cp.zeros((b, h + 1, w + 1), cp.int16)
        out[:, 0, :] = 127
        out[:, :, 0] = 129
        out[:, 1:, 1:] = Pb
        return out.reshape(b * (h + 1), w + 1)

    ctxY = borders(Yb)
    ctxU = borders(Ub)
    ctxV = borders(Vb)

    idx = cp.arange(N)
    bidx = idx // n_mb
    mbidx = idx % n_mb
    valid_t = (mbidx // mb_w) > 0
    valid_l = (mbidx % mb_w) > 0

    def gather_bat(Bf, blk, stride=None):
        st = img_stride if stride is None else stride
        by = bidx * st + (mbidx // mb_w) * blk
        bx = (mbidx % mb_w) * blk
        cols = bx[:, None] + 1 + cp.arange(blk)[None, :]
        top = Bf[by[:, None] + cp.zeros((1, blk), cp.int64), cols].astype(cp.int32)
        lrows = by[:, None] + 1 + cp.arange(blk)[None, :]
        left = Bf[lrows, bx[:, None]].astype(cp.int32)
        X = Bf[by, bx].astype(cp.int32)
        return top, left, X

    top, left, X = gather_bat(ctxY, 16)
    p16 = block_preds_batch(top, left, X, 16, valid_t, valid_l)
    src16 = (Yb.reshape(B, mb_h, 16, mb_w, 16)
             .transpose(0, 1, 3, 2, 4).reshape(N, 16, 16).astype(cp.int32))
    sse16 = cp.empty((4, N), dtype=cp.int64)
    for m in range(4):
        dd = src16 - p16[m]
        sse16[m] = (dd * dd).sum(axis=2).sum(axis=1)
    cost16 = sse16 * 256 + FIXED_COSTS_I16[:, None] * 106
    i16_mode = cost16.argmin(axis=0).astype(cp.uint8)
    i16_score = cp.asnumpy(cost16.min(axis=0))

    sse_uv = cp.zeros((4, N), dtype=cp.int64)
    half_stride = H // 2 + 1              # chroma bordered-plane row stride
    for P, C in ((Ub, ctxU), (Vb, ctxV)):
        t8, l8, x8 = gather_bat(C, 8, half_stride)
        pr = block_preds_batch(t8, l8, x8, 8, valid_t, valid_l)
        src8 = (P.reshape(B, mb_h, 8, mb_w, 8)
                .transpose(0, 1, 3, 2, 4).reshape(N, 8, 8).astype(cp.int32))
        for m in range(4):
            dd = src8 - pr[m]
            sse_uv[m] += (dd * dd).sum(axis=2).sum(axis=1)
    cost_uv = sse_uv * 256 + FIXED_COSTS_UV[:, None] * 120
    uv_mode = cp.asnumpy(cost_uv.argmin(axis=0).astype(cp.uint8))

    # ---- batched i4 edges + preds ----
    n4 = B * N4img
    j = cp.arange(n4)
    jb = j // N4img
    jloc = j % N4img
    gy = jloc // gw
    gx = jloc % gw
    py = jb * img_stride + gy * 4
    px = gx * 4
    mbx = gx // 4
    mby = gy // 4
    tt = cp.empty((n4, 9), dtype=cp.int32)
    tt[:, 0] = ctxY[py, px]
    for k in range(4):
        tt[:, 1 + k] = ctxY[py, px + 1 + k]
    for k in range(4):
        # TR[k]: row above the subblock, col px+4+k; for x==3 subblocks that
        # row is the one above the MB (px+4+k already equals (mbx+1)*16+k)
        c = cp.minimum(px + 4 + k, W - 1)
        src_row = cp.where(gx % 4 == 3, jb * img_stride + mby * 16, py)
        tt[:, 5 + k] = ctxY[src_row, c + 1]
    lrows = py[:, None] + 1 + cp.arange(4)[None, :]
    L = ctxY[lrows, px[:, None]].astype(cp.int32)
    p4 = i4_preds_batch(tt.astype(cp.int32), L)

    src4 = (Yb.reshape(B, gh, 4, gw, 4)
            .transpose(0, 1, 3, 2, 4).reshape(-1, 4, 4).astype(cp.int32))
    # reorder into per-MB-contiguous slots across the batch
    slot = cp.arange(n4)
    g = slot % (n_mb * 16)
    s_mb_local, s_k = g // 16, g % 16
    jb2 = slot // (n_mb * 16)
    fgy = (s_mb_local // mb_w) * 4 + s_k // 4
    fgx = (s_mb_local % mb_w) * 4 + s_k % 4
    perm = jb2 * N4img + fgy * gw + fgx
    src4 = src4[perm]
    # per-mode SSE with per-mode gather keeps peak VRAM ~10x lower
    sse4 = cp.empty((n4, 10), dtype=cp.int64)
    for m in range(10):
        dd = src4 - p4[m][perm]
        sse4[:, m] = (dd * dd).sum(axis=2).sum(axis=1)
    sse4_np = cp.asnumpy(sse4).reshape(B, n_mb, 16, 10)

    if not select:
        return dict(i16_mode=cp.asnumpy(i16_mode), i16_score=i16_score,
                    uv_mode=uv_mode, sse4=sse4_np, mb_w=mb_w, mb_h=mb_h)

    from .vp8_encode import select_modes
    out = []
    i16m_np = cp.asnumpy(i16_mode)
    for i in range(B):
        if not split and i > 0:
            break
        i4_modes_np, is_i4_np, _ = select_modes(
            np.ascontiguousarray(sse4_np[i]), i16_score[i * n_mb:(i + 1) * n_mb],
            i16m_np[i * n_mb:(i + 1) * n_mb], mb_w, mb_h,
            1000 * y1.q_avg * y1.q_avg)
        out.append(dict(is_i4=is_i4_np.astype(bool),
                        i16_mode=i16m_np[i * n_mb:(i + 1) * n_mb],
                        uv_mode=uv_mode[i * n_mb:(i + 1) * n_mb],
                        i4_modes=i4_modes_np))
    if split:
        return out
    return out[0]


def _subblock_flat_idx(gh, gw, W):
    """Flat plane indices (N4,4,4) for every 4x4 subblock."""
    n = gh * gw
    gy = cp.arange(n) // gw
    gx = cp.arange(n) % gw
    r = cp.arange(4)
    rows = (gy * 4)[:, None, None] + r[None, :, None]          # (N,4,1)
    cols = (gx * 4)[:, None, None] + r[None, None, :]          # (N,1,4)
    return rows * W + cols                                       # (N,4,4)


def _mb_flat_idx(mb_h, mb_w, W, blk):
    n = mb_h * mb_w
    my = cp.arange(n) // mb_w
    mx = cp.arange(n) % mb_w
    r = cp.arange(blk)
    rows = (my * blk)[:, None, None] + r[None, :, None]
    cols = (mx * blk)[:, None, None] + r[None, None, :]
    return rows * W + cols


def gpu_encode_pass(Y, U, V, ctxY, ctxU, ctxV, base_q, y1, y2, uv_m,
                    y1deq2, y2deq2, uvdeq2, prev=None):
    """One parallel encode+reconstruct pass with ctx* planes as prediction
    context. Returns (modes dict, levels dict, recon planes)."""
    mb_h, mb_w = Y.shape[0] // 16, Y.shape[1] // 16
    n_mb = mb_h * mb_w
    gh, gw = mb_h * 4, mb_w * 4
    W, H = Y.shape[1], Y.shape[0]
    HW, HH = W // 2, H // 2
    mbidx = cp.arange(n_mb)
    valid_t = (mbidx // mb_w) > 0
    valid_l = (mbidx % mb_w) > 0

    # ---------- mode decision (context = ctx planes) ----------
    top, left, X = gather_block_edges(ctxY, mb_h, mb_w, 16)
    p16 = block_preds_batch(top, left, X, 16, valid_t, valid_l)
    src16 = Y.reshape(mb_h, 16, mb_w, 16).transpose(0, 2, 1, 3).reshape(n_mb, 16, 16).astype(cp.int32)
    sse16 = cp.empty((4, N), dtype=cp.int64)
    for m in range(4):
        dd = src16 - p16[m]
        sse16[m] = (dd * dd).sum(axis=2).sum(axis=1)
    cost16 = sse16 * 256 + FIXED_COSTS_I16[:, None] * 106
    i16_mode = cost16.argmin(axis=0).astype(cp.uint8)
    i16_score = cp.asnumpy(cost16.min(axis=0))

    sse_uv = cp.zeros((4, n_mb), dtype=cp.int64)
    for P, C in ((U, ctxU), (V, ctxV)):
        t8, l8, x8 = gather_block_edges(C, mb_h, mb_w, 8)
        pr = block_preds_batch(t8, l8, x8, 8, valid_t, valid_l)
        src8 = P.reshape(mb_h, 8, mb_w, 8).transpose(0, 2, 1, 3).reshape(n_mb, 8, 8).astype(cp.int32)
        dd = src8[None] - pr
        sse_uv += (dd.astype(cp.int64) * dd).sum(axis=(2, 3))
    cost_uv = sse_uv * 256 + FIXED_COSTS_UV[:, None] * 120
    uv_mode = cp.asnumpy(cost_uv.argmin(axis=0).astype(cp.uint8))

    tt, L = i4_edges_batch(ctxY, gh, gw)
    p4 = i4_preds_batch(tt, L)
    src4 = Y.reshape(gh, 4, gw, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 4).astype(cp.int32)
    # reorder from frame-subblock order (gy, gx) into per-MB-contiguous order
    # (mb*16 + k, k = subblock row*4+col) so MB grouping downstream is correct.
    # perm maps new slot -> old flat index (gather order).
    N4 = gh * gw
    slot = cp.arange(N4)
    s_mb, s_k = slot // 16, slot % 16
    perm = ((s_mb // mb_w) * 4 + s_k // 4) * gw + (s_mb % mb_w) * 4 + s_k % 4
    src4 = src4[perm]
    p4 = p4[:, perm]
    dd = src4[None] - p4
    sse4 = (dd.astype(cp.int64) * dd).sum(axis=(2, 3)).T
    sse4_np = cp.asnumpy(sse4).reshape(n_mb, 16, 10)
    from .vp8_encode import select_modes
    i4_modes_np, is_i4_np, _ = select_modes(
        np.ascontiguousarray(sse4_np), i16_score, cp.asnumpy(i16_mode),
        mb_w, mb_h, 1000 * y1.q_avg * y1.q_avg)
    is_i4_np = is_i4_np.astype(bool)
    is_i4_c = cp.asarray(is_i4_np)

    # ---------- luma: i4 MBs ----------
    y_ac = np.zeros((n_mb, 16, 16), dtype=np.int16)
    y_dc = np.zeros((n_mb, 16), dtype=np.int16)
    uv_lv = np.zeros((n_mb, 8, 16), dtype=np.int16)
    reconY = cp.zeros((H, W), dtype=cp.uint8)
    sb_idx = _subblock_flat_idx(gh, gw, W)[perm]

    mb_of_sb = cp.asarray(np.repeat(np.arange(n_mb), 16))
    sel = cp.flatnonzero(is_i4_c[mb_of_sb])
    if sel.size:
        modes_flat = cp.asarray(i4_modes_np.reshape(-1))
        pr4 = p4[modes_flat[sel], sel]
        sr4 = src4[sel]
        coef4 = fdct_batch(sr4 - pr4)
        lv4 = quant_batch(coef4, y1, 0)
        y_ac[is_i4_np] = cp.asnumpy(lv4).reshape(-1, 16, 16)
        t4 = dequant_batch(lv4, y1deq2, 0)
        rec4 = idct_full_batch(t4, pr4)
        reconY.reshape(-1)[sb_idx[sel].reshape(-1)] = rec4.reshape(-1).astype(cp.uint8)

    # ---------- luma: i16 MBs ----------
    idx16 = cp.flatnonzero(~is_i4_c)
    if idx16.size:
        pm = i16_mode[idx16]
        pr16 = p16[pm, idx16]
        sr16 = src16[idx16]
        blocks16 = (sr16 - pr16).reshape(idx16.size, 4, 4, 4, 4).reshape(idx16.size * 16, 4, 4)
        coef16 = fdct_batch(blocks16)                       # (n*16,16)
        coef256 = coef16.reshape(idx16.size, 256)
        wht = fwht_batch(coef256)
        lv_dc = quant_batch(wht, y2, 0)
        y_dc_np = cp.asnumpy(lv_dc)
        y_dc[idx16.get()] = y_dc_np
        dc_deq = dequant_batch(lv_dc, y2deq2, 0)
        dcs = iwht_batch(dc_deq)                           # (n,256) block DCs
        # decoder shortcut: y2 with only the DC level nonzero (nz<=1) puts
        # (dc0+3)>>3 into EVERY block's DC slot instead of the full inverse WHT
        nz_last = cp.where(lv_dc != 0, cp.arange(1, 17, dtype=lv_dc.dtype), 0).max(axis=-1)
        smask = (nz_last <= 1)[:, None]
        dc0 = ((dc_deq[:, 0] + 3) >> 3)[:, None]
        dcs[..., 0::16] = cp.where(smask, dc0, dcs[..., 0::16])
        coef256 = coef256.copy()
        coef256[..., 0::16] = dcs[..., 0::16]
        coef256[..., 1::16] = 0
        lv_ac = quant_batch(coef256.reshape(-1, 16), y1, 1)      # (n*16, 15)
        lv_ac_full = cp.zeros((lv_ac.shape[0], 16), dtype=cp.int16)
        lv_ac_full[:, 1:] = lv_ac
        y_ac[idx16.get()] = cp.asnumpy(lv_ac_full).reshape(idx16.size, 16, 16)
        t16 = coef256.copy() * 0
        t16[..., 0::16] = dcs[..., 0::16]
        t16b = t16.reshape(-1, 16)
        dq = dequant_batch(lv_ac, y1deq2, 1)     # (n*16, 16) with slot0 = 0
        t16b[:, 1:] = dq[:, 1:]
        rec16 = idct_full_batch(t16.reshape(-1, 16), pr16.reshape(-1, 4, 4)).reshape(-1, 16, 4, 4)
        mb16_idx = _mb_flat_idx(mb_h, mb_w, W, 16)
        reconY.reshape(-1)[mb16_idx[idx16].reshape(-1)] = rec16.reshape(-1).astype(cp.uint8)

    # ---------- chroma ----------
    reconU = cp.zeros((HH, HW), dtype=cp.uint8)
    reconV = cp.zeros((HH, HW), dtype=cp.uint8)
    uv_mb_idx8 = _mb_flat_idx(mb_h, mb_w, HW, 8)
    for ci, (P, C, RC) in enumerate(((U, ctxU, reconU), (V, ctxV, reconV))):
        t8, l8, x8 = gather_block_edges(C, mb_h, mb_w, 8)
        pr8all = block_preds_batch(t8, l8, x8, 8, valid_t, valid_l)
        pm = uv_mode
        pr8 = pr8all[cp.asarray(pm), mbidx]
        src8 = P.reshape(mb_h, 8, mb_w, 8).transpose(0, 2, 1, 3).reshape(n_mb, 8, 8).astype(cp.int32)
        res = (src8 - pr8).reshape(n_mb, 2, 2, 4, 4).reshape(n_mb * 4, 4, 4)
        coef8 = fdct_batch(res)
        lv8 = quant_batch(coef8, uv_m, 0)
        uv_lv[:, ci * 4:(ci + 1) * 4] = cp.asnumpy(lv8).reshape(n_mb, 4, 16)
        t8b = dequant_batch(lv8, uvdeq2, 0)
        rec8 = idct_full_batch(t8b, pr8.reshape(-1, 4, 4)).reshape(n_mb, 4, 4, 4)
        RC.reshape(-1)[uv_mb_idx8.reshape(-1)] = rec8.reshape(-1).astype(cp.uint8)

    modes = dict(is_i4=is_i4_np, i16_mode=cp.asnumpy(i16_mode),
                 uv_mode=uv_mode, i4_modes=i4_modes_np)
    return modes, y_dc, y_ac, uv_lv, reconY, reconU, reconV
