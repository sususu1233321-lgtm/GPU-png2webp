"""VP8 (WebP lossy) intra-frame encoder.

From-scratch encoder producing a spec-conformant VP8 keyframe bitstream.
Constant tables and bitstream syntax are ported 1:1 from libwebp v1.5.0 (BSD);
the pipeline is an open-loop, GPU-friendly design of our own:
  - all intra predictions are made from *source* pixels (fully parallel),
  - mode decision uses libwebp's fast-path distortion+header-cost formula,
  - residual -> FTransform -> quantization produce zigzag levels,
  - bool-coding of headers/tokens runs in numba at machine-code speed.

The heavy per-pixel stages (predict/SSE/transform/quantize) live in
analyze_cpu(); a CuPy implementation with the same interface is in gpu_kernels.
"""
import numpy as np
from numba import njit

from . import vp8_tables as T
from .bool_coder import bool_encode

VP8_SIGNATURE = 0x9D012A
QFIX = 17
ZIG = np.array(T.ZIGZAG, dtype=np.intp)
BANDS = np.array(T.ENC_BANDS, dtype=np.int64)

COEFFS_PROBA0 = np.array(T.COEFFS_PROBA0, dtype=np.int64).reshape(-1)
COEFFS_UPDATE_PROBA = np.array(T.COEFFS_UPDATE_PROBA, dtype=np.int64).reshape(-1)
KF_BMODE_PROBA = np.array(T.KF_BMODE_PROBA, dtype=np.int64).reshape(-1)
FIXED_COSTS_I4 = np.array(T.FIXED_COSTS_I4, dtype=np.int64).reshape(-1)
FIXED_COSTS_I16 = np.array(T.FIXED_COSTS_I16, dtype=np.int64)
FIXED_COSTS_UV = np.array(T.FIXED_COSTS_UV, dtype=np.int64)

RD_MULT = 256          # libwebp RefineUsingDistortion constants
LAMBDA_D_I16 = 106
LAMBDA_D_I4 = 11
LAMBDA_D_UV = 120


# ----------------------------------------------------------------------------
# quantization setup


def quality_to_q(quality):
    c = max(1.0, min(100.0, float(quality))) / 100.0
    linear_c = c * (2.0 / 3.0) if c < 0.75 else 2.0 * c - 1.0
    c_base = linear_c ** (1.0 / 3.0)
    q = int(127.0 * (1.0 - c_base))
    return max(0, min(127, q))


class QuantMatrix:
    __slots__ = ("q", "iq", "bias", "zthresh", "sharpen", "q_avg")

    def __init__(self, q0, q1, bias_dc, bias_ac, is_luma_ac):
        q = np.empty(16, dtype=np.int64)
        iq = np.empty(16, dtype=np.int64)
        bias = np.empty(16, dtype=np.int64)
        zt = np.empty(16, dtype=np.int64)
        q[0], q[1] = q0, q1
        iq[0] = (1 << QFIX) // q0
        iq[1] = (1 << QFIX) // q1
        bias[0] = bias_dc << (QFIX - 8)
        bias[1] = bias_ac << (QFIX - 8)
        zt[0] = ((1 << QFIX) - 1 - bias[0]) // iq[0]
        zt[1] = ((1 << QFIX) - 1 - bias[1]) // iq[1]
        q[2:] = q[1]; iq[2:] = iq[1]; bias[2:] = bias[1]; zt[2:] = zt[1]
        if is_luma_ac:
            sharpen = (np.array(T.FREQ_SHARPENING, dtype=np.int64) * q) >> 11
        else:
            sharpen = np.zeros(16, dtype=np.int64)
        self.q, self.iq, self.bias, self.zthresh, self.sharpen = q, iq, bias, zt, sharpen
        self.q_avg = (int(q.sum()) + 8) >> 4


def setup_quant(quality):
    q = quality_to_q(quality)
    y1 = QuantMatrix(T.DC_TABLE[q], T.AC_TABLE[q], 96, 110, True)
    y2 = QuantMatrix(T.DC_TABLE[q] * 2, T.AC_TABLE2[q], 96, 108, False)
    uv = QuantMatrix(T.DC_TABLE[max(0, min(117, q - 2))], T.AC_TABLE[q], 110, 115, False)
    qstep = min(T.AC_TABLE[q] >> 2, 63)
    level = T.FILTER_LEVEL_FROM_DELTA_SHARP0[qstep] * 300 // 356
    level = 0 if level < 2 else min(63, level)
    return q, y1, y2, uv, level


# ----------------------------------------------------------------------------
# numba op primitives   ops[i] = (prob << 1) | bit


@njit(cache=True, nogil=True, inline="always")
def _op(ops, pos, bit, prob):
    ops[pos] = (prob << 1) | (bit & 1)
    return pos + 1


@njit(cache=True, nogil=True, inline="always")
def _put_bits(ops, pos, value, n_bits):
    mask = 1 << (n_bits - 1)
    while mask != 0:
        pos = _op(ops, pos, 1 if (value & mask) != 0 else 0, 128)
        mask >>= 1
    return pos


@njit(cache=True, nogil=True, inline="always")
def _put_signed(ops, pos, value, n_bits):
    if value == 0:
        return _op(ops, pos, 0, 128)
    pos = _op(ops, pos, 1, 128)
    if value < 0:
        return _put_bits(ops, pos, ((-value) << 1) | 1, n_bits + 1)
    return _put_bits(ops, pos, value << 1, n_bits + 1)


# ----------------------------------------------------------------------------
# partition 0 writer


@njit(cache=True, nogil=True)
def write_partition0(mb_w, mb_h, base_quant, dq_uv_dc, dq_uv_ac,
                     filter_level, num_parts_log2, use_skip, skip_proba,
                     skip, is_i4, i16_mode, uv_mode, i4_modes, ops):
    pos = 0
    pos = _op(ops, pos, 0, 128)              # colorspace
    pos = _op(ops, pos, 0, 128)              # clamp type
    pos = _op(ops, pos, 0, 128)              # segmentation disabled
    pos = _op(ops, pos, 1, 128)              # simple loop filter
    pos = _put_bits(ops, pos, filter_level, 6)
    pos = _put_bits(ops, pos, 0, 3)          # sharpness
    pos = _op(ops, pos, 0, 128)              # no lf deltas
    pos = _put_bits(ops, pos, num_parts_log2, 2)
    pos = _put_bits(ops, pos, base_quant, 7)
    pos = _put_signed(ops, pos, 0, 4)        # dq_y1_dc
    pos = _put_signed(ops, pos, 0, 4)        # dq_y2_dc
    pos = _put_signed(ops, pos, 0, 4)        # dq_y2_ac
    pos = _put_signed(ops, pos, dq_uv_dc, 4)
    pos = _put_signed(ops, pos, dq_uv_ac, 4)
    pos = _op(ops, pos, 0, 128)              # refresh_entropy_probs = 0
    for i in range(4 * 8 * 3 * 11):
        pos = _op(ops, pos, 0, COEFFS_UPDATE_PROBA[i])
    if use_skip:
        pos = _op(ops, pos, 1, 128)
        pos = _put_bits(ops, pos, skip_proba, 8)
    else:
        pos = _op(ops, pos, 0, 128)

    for mby in range(mb_h):
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            if use_skip:
                pos = _op(ops, pos, 1 if skip[mb] else 0, skip_proba)
            if is_i4[mb]:
                pos = _op(ops, pos, 0, 145)
                for y in range(4):
                    left = 0 if mbx == 0 else i4_modes[mb - 1, y * 4 + 3]
                    for x in range(4):
                        if y == 0:
                            top = 0 if mby == 0 else i4_modes[mb - mb_w, 12 + x]
                        else:
                            top = i4_modes[mb, (y - 1) * 4 + x]
                        p0 = KF_BMODE_PROBA[(top * 10 + left) * 9 + 0]
                        mode = i4_modes[mb, y * 4 + x]
                        pos = _put_i4(ops, pos, mode, top, left)
                        left = mode
            else:
                pos = _op(ops, pos, 1, 145)
                mode = i16_mode[mb]     # {DC=0, TM=1, V=2, H=3}
                b1 = 1 if (mode == 1 or mode == 3) else 0   # TM or H
                pos = _op(ops, pos, b1, 156)
                if b1 == 1:
                    pos = _op(ops, pos, 1 if mode == 1 else 0, 128)  # TM or H
                else:
                    pos = _op(ops, pos, 1 if mode == 2 else 0, 163)  # V or DC
            umode = uv_mode[mb]         # {DC=0, TM=1, V=2, H=3}
            pos = _op(ops, pos, 1 if umode != 0 else 0, 142)
            if umode != 0:
                pos = _op(ops, pos, 1 if umode != 2 else 0, 114)      # V or {TM,H}
                if umode != 2:
                    pos = _op(ops, pos, 1 if umode == 1 else 0, 183)  # TM or H
    return pos


@njit(cache=True, nogil=True, inline="always")
def _put_i4(ops, pos, mode, top, left):
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 0]
    pos = _op(ops, pos, 1 if mode != 0 else 0, b)
    if mode == 0:
        return pos
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 1]
    pos = _op(ops, pos, 1 if mode != 1 else 0, b)
    if mode == 1:
        return pos
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 2]
    pos = _op(ops, pos, 1 if mode != 2 else 0, b)
    if mode == 2:
        return pos
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 3]
    ge6 = 1 if mode >= 6 else 0
    pos = _op(ops, pos, ge6, b)
    if ge6 == 0:
        b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 4]
        pos = _op(ops, pos, 1 if mode != 3 else 0, b)
        if mode == 3:
            return pos
        return _op(ops, pos, 1 if mode != 4 else 0,
                   KF_BMODE_PROBA[(top * 10 + left) * 9 + 5])
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 6]
    pos = _op(ops, pos, 1 if mode != 6 else 0, b)
    if mode == 6:
        return pos
    b = KF_BMODE_PROBA[(top * 10 + left) * 9 + 7]
    pos = _op(ops, pos, 1 if mode != 7 else 0, b)
    if mode == 7:
        return pos
    return _op(ops, pos, 1 if mode != 8 else 0,
               KF_BMODE_PROBA[(top * 10 + left) * 9 + 8])


# ----------------------------------------------------------------------------
# token partition writer


CAT3 = np.array(T.CAT3, dtype=np.int64)
CAT4 = np.array(T.CAT4, dtype=np.int64)
CAT5 = np.array(T.CAT5, dtype=np.int64)
CAT6 = np.array(T.CAT6, dtype=np.int64)


@njit(cache=True, nogil=True, inline="always")
def _emit_block(levels, first, ctype, ctx, ops, pos):
    last = -1
    for n in range(first, 16):
        if levels[n] != 0:
            last = n
    band = BANDS[first]
    base = ((ctype * 8 + band) * 3 + ctx) * 11
    pos = _op(ops, pos, 1 if last >= 0 else 0, COEFFS_PROBA0[base])
    if last < 0:
        return pos
    n = first
    while n < 16:
        c = levels[n]
        n += 1
        v = c if c >= 0 else -c
        sign = 1 if c < 0 else 0
        band = BANDS[n] if n < 16 else 0
        pos = _op(ops, pos, 1 if v != 0 else 0, COEFFS_PROBA0[base + 1])
        if v == 0:
            base = ((ctype * 8 + band) * 3 + 0) * 11
            continue
        pos = _op(ops, pos, 1 if v > 1 else 0, COEFFS_PROBA0[base + 2])
        if v <= 1:
            base = ((ctype * 8 + band) * 3 + 1) * 11
        else:
            pos = _op(ops, pos, 1 if v > 4 else 0, COEFFS_PROBA0[base + 3])
            if v <= 4:
                pos = _op(ops, pos, 1 if v != 2 else 0, COEFFS_PROBA0[base + 4])
                if v != 2:
                    pos = _op(ops, pos, 1 if v == 4 else 0, COEFFS_PROBA0[base + 5])
            elif v <= 10:
                pos = _op(ops, pos, 0, COEFFS_PROBA0[base + 6])   # not >10
                pos = _op(ops, pos, 1 if v > 6 else 0, COEFFS_PROBA0[base + 7])
                if v <= 6:                     # category 1
                    pos = _op(ops, pos, 1 if v == 6 else 0, 159)
                else:                          # category 2
                    pos = _op(ops, pos, 1 if v >= 9 else 0, 165)
                    pos = _op(ops, pos, 0 if (v & 1) != 0 else 1, 145)
            else:
                pos = _op(ops, pos, 1, COEFFS_PROBA0[base + 6])   # >10
                residue = v - 3
                if residue < 16:               # category 3
                    pos = _op(ops, pos, 0, COEFFS_PROBA0[base + 8])
                    pos = _op(ops, pos, 0, COEFFS_PROBA0[base + 9])
                    residue -= 8
                    mask = 4
                    for k in range(3):
                        pos = _op(ops, pos, 1 if (residue & mask) != 0 else 0, CAT3[k])
                        mask >>= 1
                elif residue < 32:             # category 4
                    pos = _op(ops, pos, 0, COEFFS_PROBA0[base + 8])
                    pos = _op(ops, pos, 1, COEFFS_PROBA0[base + 9])
                    residue -= 16
                    mask = 8
                    for k in range(4):
                        pos = _op(ops, pos, 1 if (residue & mask) != 0 else 0, CAT4[k])
                        mask >>= 1
                elif residue < 64:             # category 5
                    pos = _op(ops, pos, 1, COEFFS_PROBA0[base + 8])
                    pos = _op(ops, pos, 0, COEFFS_PROBA0[base + 10])
                    residue -= 32
                    mask = 16
                    for k in range(5):
                        pos = _op(ops, pos, 1 if (residue & mask) != 0 else 0, CAT5[k])
                        mask >>= 1
                else:                          # category 6
                    pos = _op(ops, pos, 1, COEFFS_PROBA0[base + 8])
                    pos = _op(ops, pos, 1, COEFFS_PROBA0[base + 10])
                    residue -= 64
                    mask = 1 << 10
                    for k in range(11):
                        pos = _op(ops, pos, 1 if (residue & mask) != 0 else 0, CAT6[k])
                        mask >>= 1
            base = ((ctype * 8 + band) * 3 + 2) * 11
        pos = _op(ops, pos, sign, 128)
        if n == 16:
            return pos
        pos = _op(ops, pos, 1 if n <= last else 0, COEFFS_PROBA0[base])
        if n > last:
            return pos
    return pos


@njit(cache=True, nogil=True)
def write_token_partition(mb_w, mb_h, part, num_parts, use_skip, skip, is_i4,
                          y_dc, y_ac, uv, ops):
    pos = 0
    top_nz = np.zeros((mb_w, 9), dtype=np.int64)
    left_nz = np.zeros(9, dtype=np.int64)
    for mby in range(mb_h):
        for k in range(9):
            left_nz[k] = 0
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            emit = ((mb % num_parts) == part) and (use_skip == 0 or not skip[mb])
            # y2 (i16 only)
            if not is_i4[mb]:
                nzdc = 0
                for n in range(16):
                    if y_dc[mb, n] != 0:
                        nzdc = 1
                        break
                if emit:
                    ctx = top_nz[mbx, 8] + left_nz[8]
                    pos = _emit_block(y_dc[mb], 0, 1, ctx, ops, pos)
                top_nz[mbx, 8] = nzdc
                left_nz[8] = nzdc
            for y in range(4):
                for x in range(4):
                    blk = x + y * 4
                    nzb = 0
                    for n in range(16):
                        if y_ac[mb, blk, n] != 0:
                            nzb = 1
                            break
                    if emit:
                        ctx = top_nz[mbx, x] + left_nz[y]
                        if is_i4[mb]:
                            pos = _emit_block(y_ac[mb, blk], 0, 3, ctx, ops, pos)
                        else:
                            pos = _emit_block(y_ac[mb, blk], 1, 0, ctx, ops, pos)
                    top_nz[mbx, x] = nzb
                    left_nz[y] = nzb
            for n in range(8):
                ch = n >> 2
                x = n & 1
                y = (n >> 1) & 1
                slot = 4 + ch * 2 + x
                lslot = 4 + ch * 2 + y
                nzb = 0
                for k in range(16):
                    if uv[mb, n, k] != 0:
                        nzb = 1
                        break
                if emit:
                    ctx = top_nz[mbx, slot] + left_nz[lslot]
                    pos = _emit_block(uv[mb, n], 0, 2, ctx, ops, pos)
                top_nz[mbx, slot] = nzb
                left_nz[lslot] = nzb
    return pos


# ----------------------------------------------------------------------------
# i4 mode selection (sequential context within each MB)


@njit(cache=True, nogil=True)
def select_modes(sse4, i16_score, i16_mode_arr, mb_w, mb_h, i4_penalty):
    """Sequential per-MB mode decision with exact context propagation.
    i16 MBs fill their 16 context slots with the i16 mode value (libwebp/VP8
    semantics). Returns (out_modes, is_i4, i4_score)."""
    n_mb = mb_w * mb_h
    out_modes = np.zeros((n_mb, 16), dtype=np.uint8)
    is_i4 = np.zeros(n_mb, dtype=np.uint8)
    i4_score = np.zeros(n_mb, dtype=np.int64)
    for mby in range(mb_h):
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            total = np.int64(i4_penalty)
            for y in range(4):
                left = 0 if mbx == 0 else out_modes[mb - 1, y * 4 + 3]
                for x in range(4):
                    if y == 0:
                        top = 0 if mby == 0 else out_modes[mb - mb_w, 12 + x]
                    else:
                        top = out_modes[mb, (y - 1) * 4 + x]
                    best = np.int64(1) << 60
                    best_m = 0
                    for m in range(10):
                        cost = FIXED_COSTS_I4[(top * 10 + left) * 10 + m]
                        s = sse4[mb, y * 4 + x, m] * RD_MULT + cost * LAMBDA_D_I4
                        if s < best:
                            best = s
                            best_m = m
                    out_modes[mb, y * 4 + x] = best_m
                    total += best
                    left = best_m
            i4_score[mb] = total
            if total < i16_score[mb]:
                is_i4[mb] = 1
                # keep i4 modes
            else:
                m16 = i16_mode_arr[mb]
                for k in range(16):
                    out_modes[mb, k] = m16
    return out_modes, is_i4, i4_score


# ----------------------------------------------------------------------------
# numpy pixel-domain analysis (CPU reference; GPU kernels mirror this)


def _avg2(a, b):
    return (a + b + 1) >> 1


def _avg3(a, b, c):
    return (a + 2 * b + c + 2) >> 2


def pad_to_mb(p, mb_h, mb_w, half=False):
    bh, bw = mb_h * 16, mb_w * 16
    if half:
        bh //= 2; bw //= 2
    p = p.astype(np.int16)
    if p.shape == (bh, bw):
        return p
    out = np.zeros((bh, bw), dtype=np.int16)
    h, w = p.shape
    out[:h, :w] = p
    if w < bw:
        out[:, w:] = p[:, w - 1:w]
    if h < bh:
        out[h:, :] = out[h - 1:h, :]
    return out


def _grid_edges(P, blk):
    """top/left/above-left for blk-sized blocks (16 or 8), decoder semantics."""
    H, W = P.shape
    gh, gw = H // blk, W // blk
    top = np.full((gh, gw, blk), 127, dtype=np.int16)
    if gh > 1:
        rows = (np.arange(1, gh) * blk - 1)[:, None]
        cols = np.arange(gw * blk)[None, :]
        top[1:] = P[rows, cols].reshape(gh - 1, gw, blk)
    left = np.full((gh, gw, blk), 129, dtype=np.int16)
    if gw > 1:
        rows = (np.arange(gh) * blk)[:, None, None] + np.arange(blk)[None, None, :]
        cols = (np.arange(1, gw) * blk - 1)[None, :, None] + np.zeros((1, 1, 1), dtype=np.intp)
        left[:, 1:] = P[np.broadcast_to(rows, (gh, gw - 1, blk)),
                        np.broadcast_to(cols, (gh, gw - 1, blk))]
    X = np.empty((gh, gw), dtype=np.int16)
    X[0, :] = 127
    X[1:, 0] = 129
    if gh > 1 and gw > 1:
        X[1:, 1:] = P[(np.arange(1, gh) * blk - 1)[:, None],
                      (np.arange(1, gw) * blk - 1)[None, :]]
    return top, left, X


def _block_preds(top, left, X, valid=None):
    """4 predictions in VP8 mode-enum order {DC=0, TM=1, V=2, H=3}.
    valid: optional (has_top, has_left) override; default inferred from shape
    (row/col 0 = unavailable), which is only correct for whole-plane grids."""
    gh, gw, blk = top.shape
    if valid is not None:
        has_top = np.array([[bool(valid[0])]])
        has_left = np.array([[bool(valid[1])]])
    preds = np.empty((4, gh, gw, blk, blk), dtype=np.int16)
    st = top.astype(np.int64).sum(-1)
    sl = left.astype(np.int64).sum(-1)
    if valid is None:
        has_top = np.zeros((gh, 1), dtype=bool); has_top[1:] = True
        has_left = np.zeros((1, gw), dtype=bool); has_left[:, 1:] = True
    vt = has_top & np.ones((1, gw), dtype=bool)
    vl = np.ones((gh, 1), dtype=bool) & has_left
    sh = 4 if blk == 8 else 5
    # single-sided DC doubles the available sum: (2*s + blk) >> sh == (s + blk//2) >> (sh-1)
    dc = np.where(vt & vl, (st + sl + blk) >> sh, 0)
    dc = np.where(vt & ~vl, (st + blk // 2) >> (sh - 1), dc)
    dc = np.where(~vt & vl, (sl + blk // 2) >> (sh - 1), dc)
    dc = np.where(~vt & ~vl, 128, dc)
    preds[0] = dc[:, :, None, None]                                       # DC
    preds[1] = np.clip(top[:, :, None, :].astype(np.int32)                # TM
                       + left[:, :, :, None].astype(np.int32)
                       - X[:, :, None, None].astype(np.int32), 0, 255)
    preds[2] = np.broadcast_to(top[:, :, None, :], (gh, gw, blk, blk))    # V
    preds[3] = np.broadcast_to(left[:, :, :, None], (gh, gw, blk, blk))   # H
    return preds


def _i4_edges(Y):
    """Per-4x4-subblock edges: tt[N,9] = [X,T0..T3,TR0..TR3], L[N,4], src[N,4,4].
    TR follows decoder semantics: for the rightmost subblock of a MB (x==3),
    TR is replicated from the row above the MB (or the MB's last pixel at the
    frame edge)."""
    H, W = Y.shape
    gh, gw = H // 4, W // 4
    mb_w = W // 16
    Yp = np.empty((H + 1, W + 13), dtype=np.int16)
    Yp[0, :] = 127
    Yp[1:, :5] = 129
    Yp[1:, 5:5 + W] = Y
    Yp[1:, 5 + W:] = Y[:, W - 1:W]
    # top/X: row (py) hits the 127 pad row when py == 0; left: rows (1+py) are pixel rows
    tt = Yp[(4 * np.arange(gh))[:, None, None] + np.zeros((1, 1, 9), dtype=np.intp),
            (4 + 4 * np.arange(gw))[None, :, None] + np.arange(9)]
    # fix TR for rightmost subblocks (x == 3): rows -> MB-above row, cols clamped
    mbh = gh // 4
    for mby in range(mbh):
        for mbx in range(mb_w):
            tr_row = mby * 16                    # Yp row of MB-above bottom row
            for sy in range(4):
                for k in range(4):
                    c = mbx * 16 + 16 + k
                    if c > W - 1:
                        c = mbx * 16 + 15
                    tt[mby * 4 + sy, mbx * 4 + 3, 5 + k] = Yp[tr_row, 5 + c]
    L = Yp[(1 + 4 * np.arange(gh))[:, None, None] + np.arange(4)[None, None, :],
           (4 + 4 * np.arange(gw))[None, :, None] + np.zeros((1, 1, 4), dtype=np.intp)]
    src = np.ascontiguousarray(Y.reshape(gh, 4, gw, 4).transpose(0, 2, 1, 3).reshape(gh, gw, 4, 4))
    return tt, L, src


def _i4_preds(tt, L):
    """All ten 4x4 predictions. tt[N,9]=[X,T0..3,TR0..3], L[N,4]. -> (10,N,4,4)."""
    N = tt.shape[0]
    X = tt[:, 0].astype(np.int32)
    A, B, C, D, E, F, G, H = (tt[:, 1 + i].astype(np.int32) for i in range(8))
    l0, l1, l2, l3 = (L[:, i].astype(np.int32) for i in range(4))
    p = np.empty((10, N, 4, 4), dtype=np.int16)
    # 0 B_DC_PRED
    p[0] = ((4 + A + B + C + D + l0 + l1 + l2 + l3) >> 3)[:, None, None]
    # 1 B_TM_PRED
    p[1] = np.clip(tt[:, 1:5].astype(np.int32)[:, None, :] + L.astype(np.int32)[:, :, None]
                   - X[:, None, None], 0, 255)
    # 2 B_VE_PRED
    ve = np.stack([_avg3(X, A, B), _avg3(A, B, C), _avg3(B, C, D), _avg3(C, D, E)], axis=1)
    p[2] = ve[:, None, :]
    # 3 B_HE_PRED
    he = np.stack([_avg3(X, l0, l1), _avg3(l0, l1, l2), _avg3(l1, l2, l3), _avg3(l2, l3, l3)], axis=1)
    p[3] = he[:, :, None]
    # 4 B_RD_PRED   (I=l0 J=l1 K=l2 L=l3)  [y, x] layout
    I, J, K, L_ = l0, l1, l2, l3
    pr = p[4]
    pr[:, 3, 0] = _avg3(J, K, L_)                       # DST(0,3)
    pr[:, 2, 0] = pr[:, 3, 1] = _avg3(I, J, K)          # DST(0,2) DST(1,3)
    pr[:, 1, 0] = pr[:, 2, 1] = pr[:, 3, 2] = _avg3(X, I, J)
    pr[:, 0, 0] = pr[:, 1, 1] = pr[:, 2, 2] = pr[:, 3, 3] = _avg3(A, X, I)
    pr[:, 0, 1] = pr[:, 1, 2] = pr[:, 2, 3] = _avg3(B, A, X)
    pr[:, 0, 2] = pr[:, 1, 3] = _avg3(C, B, A)
    pr[:, 0, 3] = _avg3(D, C, B)                        # DST(3,0)
    # 5 B_VR_PRED
    pr = p[5]
    pr[:, 0, 0] = pr[:, 2, 1] = _avg2(X, A)             # DST(0,0) DST(1,2)
    pr[:, 0, 1] = pr[:, 2, 2] = _avg2(A, B)
    pr[:, 0, 2] = pr[:, 2, 3] = _avg2(B, C)
    pr[:, 0, 3] = _avg2(C, D)                           # DST(3,0)
    pr[:, 3, 0] = _avg3(K, J, I)                        # DST(0,3)
    pr[:, 2, 0] = _avg3(J, I, X)                        # DST(0,2)
    pr[:, 1, 0] = pr[:, 3, 1] = _avg3(I, X, A)
    pr[:, 1, 1] = pr[:, 3, 2] = _avg3(X, A, B)
    pr[:, 1, 2] = pr[:, 3, 3] = _avg3(A, B, C)
    pr[:, 1, 3] = _avg3(B, C, D)
    # 6 B_LD_PRED
    pr = p[6]
    pr[:, 0, 0] = _avg3(A, B, C)
    pr[:, 0, 1] = pr[:, 1, 0] = _avg3(B, C, D)
    pr[:, 0, 2] = pr[:, 1, 1] = pr[:, 2, 0] = _avg3(C, D, E)
    pr[:, 0, 3] = pr[:, 1, 2] = pr[:, 2, 1] = pr[:, 3, 0] = _avg3(D, E, F)
    pr[:, 1, 3] = pr[:, 2, 2] = pr[:, 3, 1] = _avg3(E, F, G)
    pr[:, 2, 3] = pr[:, 3, 2] = _avg3(F, G, H)
    pr[:, 3, 3] = _avg3(G, H, H)
    # 7 B_VL_PRED
    pr = p[7]
    pr[:, 0, 0] = _avg2(A, B)
    pr[:, 0, 1] = pr[:, 2, 0] = _avg2(B, C)             # DST(1,0) DST(0,2)
    pr[:, 0, 2] = pr[:, 2, 1] = _avg2(C, D)
    pr[:, 0, 3] = pr[:, 2, 2] = _avg2(D, E)
    pr[:, 1, 0] = _avg3(A, B, C)
    pr[:, 1, 1] = pr[:, 3, 0] = _avg3(B, C, D)          # DST(1,1) DST(0,3)
    pr[:, 1, 2] = pr[:, 3, 1] = _avg3(C, D, E)
    pr[:, 1, 3] = pr[:, 3, 2] = _avg3(D, E, F)
    pr[:, 2, 3] = _avg3(E, F, G)                        # DST(3,2)
    pr[:, 3, 3] = _avg3(F, G, H)
    # 8 B_HD_PRED
    pr = p[8]
    pr[:, 0, 0] = pr[:, 1, 2] = _avg2(I, X)             # DST(0,0) DST(2,1)
    pr[:, 1, 0] = pr[:, 2, 2] = _avg2(J, I)
    pr[:, 2, 0] = pr[:, 3, 2] = _avg2(K, J)
    pr[:, 3, 0] = _avg2(L_, K)
    pr[:, 0, 3] = _avg3(A, B, C)                        # DST(3,0)
    pr[:, 0, 2] = _avg3(X, A, B)                        # DST(2,0)
    pr[:, 0, 1] = pr[:, 1, 3] = _avg3(I, X, A)
    pr[:, 1, 1] = pr[:, 2, 3] = _avg3(J, I, X)
    pr[:, 2, 1] = pr[:, 3, 3] = _avg3(K, J, I)
    pr[:, 3, 1] = _avg3(L_, K, J)
    # 9 B_HU_PRED
    pr = p[9]
    pr[:, 0, 0] = _avg2(I, J)
    pr[:, 0, 2] = pr[:, 1, 0] = _avg2(J, K)             # DST(2,0) DST(0,1)
    pr[:, 1, 2] = pr[:, 2, 0] = _avg2(K, L_)
    pr[:, 0, 1] = _avg3(I, J, K)
    pr[:, 0, 3] = pr[:, 1, 1] = _avg3(J, K, L_)
    pr[:, 1, 3] = pr[:, 2, 1] = _avg3(K, L_, L_)
    pr[:, 2, 2] = pr[:, 2, 3] = pr[:, 3, 0] = pr[:, 3, 1] = pr[:, 3, 2] = pr[:, 3, 3] = L_
    return p


def fdct(src, ref):
    """libwebp FTransform_C. src/ref (...,4,4) int [y,x]. -> (...,16) slot=k*4+i."""
    d = src.astype(np.int32) - ref.astype(np.int32)
    d0, d1, d2, d3 = d[..., 0], d[..., 1], d[..., 2], d[..., 3]   # columns; y axis remains
    a0 = d0 + d3; a1 = d1 + d2; a2 = d1 - d2; a3 = d0 - d3
    # vertical pass: T[..., y, j]
    T = np.stack([
        (a0 + a1) * 8,
        (a2 * 2217 + a3 * 5352 + 1812) >> 9,
        (a0 - a1) * 8,
        (a3 * 2217 - a2 * 5352 + 937) >> 9,
    ], axis=-1)
    # horizontal pass over rows of T: A[.., i] = T[.., y, i]
    A0 = T[..., 0, :] + T[..., 3, :]
    A1 = T[..., 1, :] + T[..., 2, :]
    A2 = T[..., 1, :] - T[..., 2, :]
    A3 = T[..., 0, :] - T[..., 3, :]
    o0 = (A0 + A1 + 7) >> 4
    o1 = ((A2 * 2217 + A3 * 5352 + 12000) >> 16) + (A3 != 0)
    o2 = (A0 - A1 + 7) >> 4
    o3 = (A3 * 2217 - A2 * 5352 + 51000) >> 16
    out = np.stack([o0, o1, o2, o3], axis=-2)   # (..., k, i)
    return out.reshape(out.shape[:-2] + (16,)).astype(np.int64)


def fwht(dcs):
    """libwebp FTransformWHT_C. dcs (n,4,4) block DC grid -> (n,16) k*4+i."""
    b = dcs.astype(np.int64)
    a0 = b[:, :, 0] + b[:, :, 2]; a1 = b[:, :, 1] + b[:, :, 3]
    a2 = b[:, :, 1] - b[:, :, 3]; a3 = b[:, :, 0] - b[:, :, 2]
    t = np.stack([a0 + a1, a3 + a2, a3 - a2, a0 - a1], axis=-1)   # (n,4,4) [row][j]
    a0 = t[:, 0] + t[:, 2]; a1 = t[:, 1] + t[:, 3]
    a2 = t[:, 1] - t[:, 3]; a3 = t[:, 0] - t[:, 2]
    b0 = (a0 + a1) >> 1; b1 = (a3 + a2) >> 1; b2 = (a3 - a2) >> 1; b3 = (a0 - a1) >> 1
    out = np.stack([b0, b1, b2, b3], axis=-2)   # (n, k, i)
    return out.reshape(-1, 16)


def quantize(coeff, mtx, zero_dc=False):
    """coeff (...,16) slot order -> levels (...,16) zigzag order (int16), nz (bool)."""
    cz = coeff[..., ZIG]
    if zero_dc:
        cz[..., 0] = 0
    sh = mtx.sharpen[ZIG]; iq = mtx.iq[ZIG]
    bias = mtx.bias[ZIG]; zt = mtx.zthresh[ZIG]
    x = np.abs(cz) + sh
    lvl = np.where(x > zt, (x * iq + bias) >> QFIX, 0)
    lvl = np.minimum(lvl, 2047)
    lvl = (lvl * np.sign(cz)).astype(np.int16)
    nz = lvl.any(axis=-1)
    return lvl, nz


# ----------------------------------------------------------------------------
# main analysis


def analyze(Y, U, V, y1, y2, uv_m, base_q=9):
    """Open-loop analysis on padded planes. Returns mode/level arrays."""
    mb_h, mb_w = Y.shape[0] // 16, Y.shape[1] // 16
    n_mb = mb_h * mb_w

    # ---- i16 ----
    top, left, X = _grid_edges(Y, 16)
    preds16 = _block_preds(top, left, X)
    src16 = np.ascontiguousarray(Y.reshape(mb_h, 16, mb_w, 16).transpose(0, 2, 1, 3))
    sse16 = np.empty((4, n_mb), dtype=np.int64)
    for m in range(4):
        d = src16.astype(np.int32) - preds16[m].astype(np.int32)
        sse16[m] = (d * d).sum((-1, -2)).reshape(n_mb)
    cost16 = sse16 * RD_MULT + FIXED_COSTS_I16[:, None] * LAMBDA_D_I16
    i16_mode = cost16.argmin(0).astype(np.uint8)
    i16_score = cost16.min(0)

    # ---- chroma ----
    sse_uv = np.zeros((4, n_mb), dtype=np.int64)
    uv_preds = []
    for P in (U, V):
        t8, l8, x8 = _grid_edges(P, 8)
        pr = _block_preds(t8, l8, x8)
        uv_preds.append(pr)
        src8 = np.ascontiguousarray(P.reshape(mb_h, 8, mb_w, 8).transpose(0, 2, 1, 3))
        for m in range(4):
            d = src8.astype(np.int32) - pr[m].astype(np.int32)
            sse_uv[m] += (d * d).sum((-1, -2)).reshape(n_mb)
    cost_uv = sse_uv * RD_MULT + FIXED_COSTS_UV[:, None] * LAMBDA_D_UV
    uv_mode = cost_uv.argmin(0).astype(np.uint8)

    # ---- i4 ----
    tt, L4, src4 = _i4_edges(Y)
    N4 = tt.shape[0] * tt.shape[1]
    tt = tt.reshape(-1, 9)
    L4 = L4.reshape(-1, 4)
    preds4 = _i4_preds(tt, L4)                       # (10, N4, 4, 4)
    sse4_flat = np.empty((10, N4), dtype=np.int64)
    src4_flat = src4.reshape(-1, 4, 4)
    for m in range(10):
        d = src4_flat.astype(np.int32) - preds4[m]
        sse4_flat[m] = (d * d).sum((-1, -2))
    sse4 = np.ascontiguousarray(sse4_flat.T.reshape(mb_h, mb_w, 16, 10).reshape(n_mb, 16, 10))
    i4_penalty = 1000 * y1.q_avg * y1.q_avg
    i4_modes_u8, is_i4_u8, _ = select_modes(sse4, i16_score, i16_mode,
                                            mb_w, mb_h, i4_penalty)
    i4_modes = i4_modes_u8          # i16 rows filled with i16 mode for context
    is_i4 = is_i4_u8.astype(bool)

    # ---- closed-loop residual + transform + quantize ----
    from .closed_loop import closed_loop_residuals
    y2ac = max(8, int(T.AC_TABLE2[base_q]))   # decoder dequant = kAcTable2 raw
    y1deq = np.array([T.DC_TABLE[base_q]] + [T.AC_TABLE[base_q]] * 15, dtype=np.int64)
    y2deq = np.array([T.DC_TABLE[base_q] * 2] + [y2ac] * 15, dtype=np.int64)
    uvdeq = np.array([T.DC_TABLE[max(0, min(117, base_q - 2))]] + [T.AC_TABLE[base_q]] * 15, dtype=np.int64)
    y_dc_levels, y_ac_levels, uv_levels, _rY, _rU, _rV = closed_loop_residuals(
        Y, U, V, is_i4, i16_mode, uv_mode, i4_modes,
        y1, y2, uv_m, y2deq, y1deq, uvdeq)

    nz_any = (y_dc_levels.any(-1) | y_ac_levels.any(-1).any(-1) | uv_levels.any(-1).any(-1))
    skip = ~nz_any
    return (mb_w, mb_h, is_i4, i16_mode, uv_mode, i4_modes,
            y_dc_levels, y_ac_levels, uv_levels, skip)


# ----------------------------------------------------------------------------
# frame assembly


def encode(y, u, v, quality, num_parts=1):
    """y: HxW uint8, u/v: (H/2)x(W/2) uint8. Returns VP8 chunk payload bytes."""
    H, W = y.shape
    mb_w, mb_h = (W + 15) // 16, (H + 15) // 16
    base_quant, y1, y2, uv_m, filter_level = setup_quant(quality)
    Y = pad_to_mb(y, mb_h, mb_w)
    U = pad_to_mb(u, mb_h, mb_w, half=True)
    V = pad_to_mb(v, mb_h, mb_w, half=True)

    (mb_w, mb_h, is_i4, i16_mode, uv_mode, i4_modes,
     y_dc_levels, y_ac_levels, uv_levels, skip) = analyze(Y, U, V, y1, y2, uv_m, base_quant)

    n_mb = mb_w * mb_h
    nb_skip = int(skip.sum())
    skip_proba = (n_mb - nb_skip) * 255 // n_mb if n_mb else 255
    use_skip = skip_proba < 250

    num_parts_log2 = {1: 0, 2: 1, 4: 2, 8: 3}[num_parts]

    ops0 = np.empty(n_mb * 256 + 4096, dtype=np.int32)
    pos0 = write_partition0(mb_w, mb_h, base_quant, -2, 0, filter_level,
                            num_parts_log2, use_skip, skip_proba,
                            skip, is_i4, i16_mode, uv_mode, i4_modes, ops0)
    buf0 = np.empty(pos0 + 16, dtype=np.uint8)
    n0 = bool_encode(ops0[:pos0], buf0)
    p0 = buf0[:n0].tobytes()

    parts = []
    for p in range(num_parts):
        ops = np.empty(n_mb * 8200 + 64, dtype=np.int32)
        pos = write_token_partition(mb_w, mb_h, p, num_parts, use_skip, skip,
                                    is_i4, y_dc_levels, y_ac_levels, uv_levels, ops)
        buf = np.empty(pos + 16, dtype=np.uint8)
        nb = bool_encode(ops[:pos], buf)
        parts.append(buf[:nb].tobytes())

    sizes = b"".join(len(pb).to_bytes(3, "little") for pb in parts[:-1])
    vp8 = bytearray()
    bits = (0) | (0 << 1) | (1 << 4) | (len(p0) << 5)
    vp8 += bits.to_bytes(3, "little")
    vp8 += VP8_SIGNATURE.to_bytes(3, "big")
    vp8 += (W & 0x3FFF).to_bytes(2, "little")
    vp8 += (H & 0x3FFF).to_bytes(2, "little")
    vp8 += p0 + sizes
    for pb in parts:
        vp8 += pb
    if len(vp8) & 1:
        vp8 += b"\x00"
    return bytes(vp8)
