"""Single-@njit closed-loop VP8 encoder core (exact decoder semantics).

Same algorithm as closed_loop.closed_loop_residuals but as one jitted function
with scalar inlined predictors: ~20x faster (no per-MB numpy allocations, no
Python driver).  Kernels are imported from closed_loop (already @njit).

Prediction formulas are the decoder-verified ports:
  i16/uv preds {DC,TM,V,H}: dsp/dec.c VP8PredModes (block_preds_batch)
  i4 preds 0..9:           dsp/dec.c B_*_PRED (i4_preds_batch)
"""
import numpy as np
from numba import njit

from .closed_loop import (_fdct, _fwht, _iwht, _idct_full, _idct_ac3,
                          _idct_dc, _quantize, _dequant_into)


@njit(cache=True, nogil=True, inline="always")
def _pred_blk(mode, rC, y0, x0, blk, out):
    """{DC,TM,V,H} pred of a blk x blk region from bordered recon rC.
    Border convention: rC[r+1, c+1] = pixel(r, c); row 0 / col 0 = border."""
    has_t = y0 > 0
    has_l = x0 > 0
    if mode == 2:                      # V: replicate the row above
        for c in range(blk):
            v = rC[y0, x0 + 1 + c]
            for r in range(blk):
                out[r, c] = v
        return
    if mode == 3:                      # H: replicate the column left
        for r in range(blk):
            v = rC[y0 + 1 + r, x0]
            for c in range(blk):
                out[r, c] = v
        return
    if mode == 1:                      # TM
        X = rC[y0, x0]
        for r in range(blk):
            l = rC[y0 + 1 + r, x0]
            for c in range(blk):
                v = rC[y0, x0 + 1 + c] + l - X
                out[r, c] = min(255, max(0, v))
        return
    # DC
    sh = 4 if blk == 8 else 5
    if has_t and has_l:
        st = 0
        sl = 0
        for c in range(blk):
            st += rC[y0, x0 + 1 + c]
        for r in range(blk):
            sl += rC[y0 + 1 + r, x0]
        dc = (st + sl + blk) >> sh
    elif has_t:
        st = 0
        for c in range(blk):
            st += rC[y0, x0 + 1 + c]
        dc = (st + (blk >> 1)) >> (sh - 1)
    elif has_l:
        sl = 0
        for r in range(blk):
            sl += rC[y0 + 1 + r, x0]
        dc = (sl + (blk >> 1)) >> (sh - 1)
    else:
        dc = 128
    for r in range(blk):
        for c in range(blk):
            out[r, c] = dc


@njit(cache=True, nogil=True, inline="always")
def _i4_pred(mode, rY, py, px, py0, W, out):
    """4x4 intra pred (B_*_PRED, enum order) from bordered recon rY.
    tt = [X, A,B,C,D, TR0..3]; A..D = row above, TR = above-right."""
    X = rY[py, px]
    A = rY[py, px + 1]
    B = rY[py, px + 2]
    C = rY[py, px + 3]
    D = rY[py, px + 4]
    l0 = rY[py + 1, px]
    l1 = rY[py + 2, px]
    l2 = rY[py + 3, px]
    l3 = rY[py + 4, px]
    # top-right: same row for x<3; for x==3 the row above the MB (decoder
    # semantics), clamped to the MB's own top row at the frame right edge
    sx = (px >> 2) & 3        # subblock column within MB (px is source coord)
    tr_row = py0 if sx == 3 else py
    E = rY[tr_row, min(px + 5, W)]
    F = rY[tr_row, min(px + 6, W)]
    G = rY[tr_row, min(px + 7, W)]
    H = rY[tr_row, min(px + 8, W)]
    I, J, K, Lm = l0, l1, l2, l3
    if mode == 0:              # B_DC_PRED
        dc = (4 + A + B + C + D + l0 + l1 + l2 + l3) >> 3
        for r in range(4):
            for c in range(4):
                out[r, c] = dc
    elif mode == 1:            # B_TM_PRED
        for r in range(4):
            l = rY[py + 1 + r, px]
            for c in range(4):
                v = rY[py, px + 1 + c] + l - X
                out[r, c] = min(255, max(0, v))
    elif mode == 2:            # B_VE_PRED
        for r in range(4):
            out[r, 0] = (X + 2 * A + B + 2) >> 2
            out[r, 1] = (A + 2 * B + C + 2) >> 2
            out[r, 2] = (B + 2 * C + D + 2) >> 2
            out[r, 3] = (C + 2 * D + E + 2) >> 2
    elif mode == 3:            # B_HE_PRED
        v0 = (X + 2 * l0 + l1 + 2) >> 2
        v1 = (l0 + 2 * l1 + l2 + 2) >> 2
        v2 = (l1 + 2 * l2 + l3 + 2) >> 2
        v3 = (l2 + 3 * l3 + 2) >> 2
        for c in range(4):
            out[0, c] = v0
            out[1, c] = v1
            out[2, c] = v2
            out[3, c] = v3
    elif mode == 4:            # B_LD_PRED
        out[0, 0] = (A + 2 * X + I + 2) >> 2
        out[0, 1] = (X + 2 * A + B + 2) >> 2
        out[0, 2] = (A + 2 * B + C + 2) >> 2
        out[0, 3] = (B + 2 * C + D + 2) >> 2
        out[1, 0] = (X + 2 * I + J + 2) >> 2
        out[1, 1] = (A + 2 * X + I + 2) >> 2
        out[1, 2] = (X + 2 * A + B + 2) >> 2
        out[1, 3] = (A + 2 * B + C + 2) >> 2
        out[2, 0] = (I + 2 * J + K + 2) >> 2
        out[2, 1] = (X + 2 * I + J + 2) >> 2
        out[2, 2] = (A + 2 * X + I + 2) >> 2
        out[2, 3] = (X + 2 * A + B + 2) >> 2
        out[3, 0] = (J + 2 * K + Lm + 2) >> 2
        out[3, 1] = (I + 2 * J + K + 2) >> 2
        out[3, 2] = (X + 2 * I + J + 2) >> 2
        out[3, 3] = (A + 2 * X + I + 2) >> 2
    elif mode == 5:            # B_RD_PRED
        out[0, 0] = (X + A + 1) >> 1
        out[0, 1] = (A + B + 1) >> 1
        out[0, 2] = (B + C + 1) >> 1
        out[0, 3] = (C + D + 1) >> 1
        out[1, 0] = (A + 2 * X + I + 2) >> 2
        out[1, 1] = (X + 2 * A + B + 2) >> 2
        out[1, 2] = (A + 2 * B + C + 2) >> 2
        out[1, 3] = (B + 2 * C + D + 2) >> 2
        out[2, 0] = (X + 2 * I + J + 2) >> 2
        out[2, 1] = (X + A + 1) >> 1
        out[2, 2] = (A + B + 1) >> 1
        out[2, 3] = (B + C + 1) >> 1
        out[3, 0] = (I + 2 * J + K + 2) >> 2
        out[3, 1] = (A + 2 * X + I + 2) >> 2
        out[3, 2] = (X + 2 * A + B + 2) >> 2
        out[3, 3] = (A + 2 * B + C + 2) >> 2
    elif mode == 6:            # B_VR_PRED
        out[0, 0] = (A + 2 * B + C + 2) >> 2
        out[0, 1] = (B + 2 * C + D + 2) >> 2
        out[0, 2] = (C + 2 * D + E + 2) >> 2
        out[0, 3] = (D + 2 * E + F + 2) >> 2
        out[1, 0] = (B + 2 * C + D + 2) >> 2
        out[1, 1] = (C + 2 * D + E + 2) >> 2
        out[1, 2] = (D + 2 * E + F + 2) >> 2
        out[1, 3] = (E + 2 * F + G + 2) >> 2
        out[2, 0] = (C + 2 * D + E + 2) >> 2
        out[2, 1] = (D + 2 * E + F + 2) >> 2
        out[2, 2] = (E + 2 * F + G + 2) >> 2
        out[2, 3] = (F + 2 * G + H + 2) >> 2
        out[3, 0] = (D + 2 * E + F + 2) >> 2
        out[3, 1] = (E + 2 * F + G + 2) >> 2
        out[3, 2] = (F + 2 * G + H + 2) >> 2
        out[3, 3] = (G + 3 * H + 2) >> 2
    elif mode == 7:            # B_VL_PRED
        out[0, 0] = (A + B + 1) >> 1
        out[0, 1] = (B + C + 1) >> 1
        out[0, 2] = (C + D + 1) >> 1
        out[0, 3] = (D + E + 1) >> 1
        out[1, 0] = (A + 2 * B + C + 2) >> 2
        out[1, 1] = (B + 2 * C + D + 2) >> 2
        out[1, 2] = (C + 2 * D + E + 2) >> 2
        out[1, 3] = (D + 2 * E + F + 2) >> 2
        out[2, 0] = (B + C + 1) >> 1
        out[2, 1] = (C + D + 1) >> 1
        out[2, 2] = (D + E + 1) >> 1
        out[2, 3] = (E + 2 * F + G + 2) >> 2
        out[3, 0] = (B + 2 * C + D + 2) >> 2
        out[3, 1] = (C + 2 * D + E + 2) >> 2
        out[3, 2] = (D + 2 * E + F + 2) >> 2
        out[3, 3] = (F + 2 * G + H + 2) >> 2
    elif mode == 8:            # B_HD_PRED
        out[0, 0] = (X + I + 1) >> 1
        out[0, 1] = (A + 2 * X + I + 2) >> 2
        out[0, 2] = (X + 2 * A + B + 2) >> 2
        out[0, 3] = (A + 2 * B + C + 2) >> 2
        out[1, 0] = (I + J + 1) >> 1
        out[1, 1] = (X + 2 * I + J + 2) >> 2
        out[1, 2] = (X + I + 1) >> 1
        out[1, 3] = (A + 2 * X + I + 2) >> 2
        out[2, 0] = (J + K + 1) >> 1
        out[2, 1] = (I + 2 * J + K + 2) >> 2
        out[2, 2] = (I + J + 1) >> 1
        out[2, 3] = (X + 2 * I + J + 2) >> 2
        out[3, 0] = (K + Lm + 1) >> 1
        out[3, 1] = (J + 2 * K + Lm + 2) >> 2
        out[3, 2] = (J + K + 1) >> 1
        out[3, 3] = (I + 2 * J + K + 2) >> 2
    else:                      # B_HU_PRED (9)
        out[0, 0] = (I + J + 1) >> 1
        out[0, 1] = (I + 2 * J + K + 2) >> 2
        out[0, 2] = (J + K + 1) >> 1
        out[0, 3] = (J + 2 * K + Lm + 2) >> 2
        out[1, 0] = (J + K + 1) >> 1
        out[1, 1] = (J + 2 * K + Lm + 2) >> 2
        out[1, 2] = (K + Lm + 1) >> 1
        out[1, 3] = (K + 3 * Lm + 2) >> 2
        out[2, 0] = (K + Lm + 1) >> 1
        out[2, 1] = (K + 3 * Lm + 2) >> 2
        out[2, 2] = Lm
        out[2, 3] = Lm
        out[3, 0] = Lm
        out[3, 1] = Lm
        out[3, 2] = Lm
        out[3, 3] = Lm


@njit(cache=True, nogil=True)
def closed_loop_full(Y, U, V, is_i4, i16_mode, uv_mode, i4_modes,
                     y1q, y1iq, y1b, y1z, y1s,
                     y2q, y2iq, y2b, y2z, y2s,
                     uvq, uviq, uvb, uvz, uvs,
                     y1deq, y2deq, uvdeq):
    """Sequential closed-loop encode with exact decoder reconstruction.
    Returns (y_dc (n,16), y_ac (n,16,16), uv_lv (n,8,16)) int16, all zigzag."""
    mb_h = Y.shape[0] // 16
    mb_w = Y.shape[1] // 16
    n_mb = mb_h * mb_w
    H = Y.shape[0]
    W = Y.shape[1]
    HH = H // 2
    HW = W // 2

    rY = np.empty((H + 1, W + 1), np.int16)
    rY[0, :] = 127
    rY[:, 0] = 129
    rU = np.empty((HH + 1, HW + 1), np.int16)
    rU[0, :] = 127
    rU[:, 0] = 129
    rV = np.empty((HH + 1, HW + 1), np.int16)
    rV[0, :] = 127
    rV[:, 0] = 129

    y_dc = np.zeros((n_mb, 16), np.int16)
    y_ac = np.zeros((n_mb, 16, 16), np.int16)
    uv_lv = np.zeros((n_mb, 8, 16), np.int16)

    pred16 = np.empty((16, 16), np.int16)
    pred8 = np.empty((8, 8), np.int16)
    pred4 = np.empty((4, 4), np.int16)
    res = np.empty((4, 4), np.int64)
    tmp = np.empty(16, np.int64)
    t16 = np.empty(16, np.int64)
    lv = np.zeros(16, np.int64)
    dc16 = np.empty(16, np.int64)
    dc_deq = np.empty(16, np.int64)
    in256 = np.empty(256, np.int64)
    coeff = np.empty((16, 16), np.int64)
    rb = np.empty((4, 4), np.int16)
    group_t = np.empty((4, 16), np.int64)
    group_nz = np.empty(4, np.int64)

    for mby in range(mb_h):
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            py0 = mby * 16
            px0 = mbx * 16

            if not is_i4[mb]:
                # ---------------- I16 ----------------
                _pred_blk(i16_mode[mb], rY, py0, px0, 16, pred16)
                for b in range(16):
                    by = b >> 2
                    bx = b & 3
                    for r in range(4):
                        for c in range(4):
                            res[r, c] = (Y[py0 + by * 4 + r, px0 + bx * 4 + c]
                                         - pred16[by * 4 + r, bx * 4 + c])
                    _fdct(res, tmp)
                    for k in range(16):
                        in256[b * 16 + k] = tmp[k]
                _fwht(in256, dc16)
                for k in range(16):
                    lv[k] = 0
                nz2 = _quantize(dc16, y2q, y2iq, y2b, y2z, y2s, lv, 0)
                for k in range(16):
                    y_dc[mb, k] = lv[k]
                for k in range(16):
                    dc_deq[k] = 0
                _dequant_into(lv, y2deq, dc_deq, 0)
                if nz2 > 1:
                    _iwht(dc_deq, in256)
                    for b in range(16):
                        for k in range(16):
                            coeff[b, k] = in256[b * 16 + k]
                else:
                    # decoder shortcut puts dc0 in every block's DC slot; the
                    # AC coefficients are still the ORIGINAL fdct outputs
                    dc0 = (dc_deq[0] + 3) >> 3
                    for b in range(16):
                        coeff[b, 0] = dc0
                        for k in range(1, 16):
                            coeff[b, k] = in256[b * 16 + k]
                for b in range(16):
                    by = b >> 2
                    bx = b & 3
                    for k in range(16):
                        lv[k] = 0
                    nz1 = _quantize(coeff[b], y1q, y1iq, y1b, y1z, y1s, lv, 1)
                    for k in range(16):
                        y_ac[mb, b, k] = lv[k]
                    for k in range(16):
                        t16[k] = 0
                    t16[0] = coeff[b, 0]
                    _dequant_into(lv, y1deq, t16, 1)
                    dz = nz1 if nz1 > 0 else 1
                    for r in range(4):
                        for c in range(4):
                            rb[r, c] = pred16[by * 4 + r, bx * 4 + c]
                    if dz > 3:
                        _idct_full(t16, rb, rb)
                    elif dz > 1:
                        _idct_ac3(t16, rb, rb)
                    elif t16[0] != 0:
                        _idct_dc(t16, rb, rb)
                    for r in range(4):
                        for c in range(4):
                            rY[py0 + 1 + by * 4 + r, px0 + 1 + bx * 4 + c] = rb[r, c]
            else:
                # ---------------- I4 ----------------
                for sb in range(16):
                    sy = sb >> 2
                    sx = sb & 3
                    py = py0 + sy * 4
                    px = px0 + sx * 4
                    _i4_pred(i4_modes[mb, sb], rY, py, px, py0, W, pred4)
                    for r in range(4):
                        for c in range(4):
                            res[r, c] = Y[py + r, px + c] - pred4[r, c]
                    _fdct(res, tmp)
                    for k in range(16):
                        lv[k] = 0
                    nz1 = _quantize(tmp, y1q, y1iq, y1b, y1z, y1s, lv, 0)
                    for k in range(16):
                        y_ac[mb, sb, k] = lv[k]
                    for k in range(16):
                        t16[k] = 0
                    _dequant_into(lv, y1deq, t16, 0)
                    for r in range(4):
                        for c in range(4):
                            rb[r, c] = pred4[r, c]
                    if nz1 > 3:
                        _idct_full(t16, rb, rb)
                    elif nz1 > 1:
                        _idct_ac3(t16, rb, rb)
                    elif t16[0] != 0:
                        _idct_dc(t16, rb, rb)
                    for r in range(4):
                        for c in range(4):
                            rY[py + 1 + r, px + 1 + c] = rb[r, c]

            # ---------------- chroma (always) ----------------
            for ci in range(2):
                rC = rU if ci == 0 else rV
                srcC = U if ci == 0 else V
                cy0 = mby * 8
                cx0 = mbx * 8
                _pred_blk(uv_mode[mb], rC, cy0, cx0, 8, pred8)
                for b in range(4):
                    by = b >> 1
                    bx = b & 1
                    for r in range(4):
                        for c in range(4):
                            res[r, c] = (srcC[cy0 + by * 4 + r, cx0 + bx * 4 + c]
                                         - pred8[by * 4 + r, bx * 4 + c])
                    _fdct(res, tmp)
                    for k in range(16):
                        lv[k] = 0
                    nzb = _quantize(tmp, uvq, uviq, uvb, uvz, uvs, lv, 0)
                    for k in range(16):
                        uv_lv[mb, ci * 4 + b, k] = lv[k]
                    group_nz[b] = nzb
                    for k in range(16):
                        t16[k] = 0
                    _dequant_into(lv, uvdeq, t16, 0)
                    for k in range(16):
                        group_t[b, k] = t16[k]
                anybig = False
                for b in range(4):
                    if group_nz[b] > 1:
                        anybig = True
                for b in range(4):
                    by = b >> 1
                    bx = b & 1
                    for r in range(4):
                        for c in range(4):
                            rb[r, c] = pred8[by * 4 + r, bx * 4 + c]
                    if anybig:
                        for k in range(16):
                            t16[k] = group_t[b, k]
                        _idct_full(t16, rb, rb)
                    elif group_nz[b] == 1 and group_t[b, 0] != 0:
                        for k in range(16):
                            t16[k] = group_t[b, k]
                        _idct_dc(t16, rb, rb)
                    for r in range(4):
                        for c in range(4):
                            rC[cy0 + 1 + by * 4 + r, cx0 + 1 + bx * 4 + c] = rb[r, c]

    return y_dc, y_ac, uv_lv, rY, rU, rV
