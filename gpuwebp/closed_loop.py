"""Closed-loop VP8 encoder core.

Sequentially reconstructs the image exactly like libwebp's decoder would
(prediction from RECONSTRUCTED neighbours, exact transform variants), and
computes residuals against that true prediction.  Predictions reuse the
verified numpy implementations in vp8_encode (edge samples are gathered from
the reconstruction plane instead of the source).

Transforms / quantization are numba ports 1:1 from libwebp v1.5.0 (BSD):
  fdct/fwht: src/dsp/enc.c FTransform_C / FTransformWHT_C
  quantize:  src/dsp/enc.c QuantizeBlock_C
  idct/iwht: src/dsp/dec.c ITransformOne_C / TransformDC_C / TransformAC3_C /
             TransformWHT_C
  dispatch:  src/dec/frame_dec.c DoTransform / DoUVTransform / ParseResiduals
"""
import numpy as np
from numba import njit

QFIX = 17
ZIGZAG = np.array((0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15),
                  dtype=np.int64)


# ---------------------------------------------------------------- transforms

@njit(cache=True, nogil=True, inline="always")
def _mul1(a):
    return ((a * 20091) >> 16) + a


@njit(cache=True, nogil=True, inline="always")
def _mul2(a):
    return (a * 35468) >> 16


@njit(cache=True, nogil=True, inline="always")
def _fdct(res, out):
    """FTransform_C on a residual 4x4 (int). out: 16 coeffs, slot=k*4+i."""
    tmp = np.empty(16, np.int64)
    for i in range(4):    # rows
        d0 = res[i, 0]
        d1 = res[i, 1]
        d2 = res[i, 2]
        d3 = res[i, 3]
        a0 = d0 + d3
        a1 = d1 + d2
        a2 = d1 - d2
        a3 = d0 - d3
        tmp[0 + i * 4] = (a0 + a1) * 8
        tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9
        tmp[2 + i * 4] = (a0 - a1) * 8
        tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9
    for i in range(4):
        a0 = tmp[0 + i] + tmp[12 + i]
        a1 = tmp[4 + i] + tmp[8 + i]
        a2 = tmp[4 + i] - tmp[8 + i]
        a3 = tmp[0 + i] - tmp[12 + i]
        out[0 + i] = (a0 + a1 + 7) >> 4
        out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (1 if a3 != 0 else 0)
        out[8 + i] = (a0 - a1 + 7) >> 4
        out[12 + i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16


@njit(cache=True, nogil=True, inline="always")
def _fwht(in256, out):
    """FTransformWHT_C. in256: 16 blocks x 16 coeffs (raster), out: 16."""
    tmp = np.empty(16, np.int64)
    for i in range(4):
        b = i * 64
        a0 = in256[b + 0] + in256[b + 32]
        a1 = in256[b + 16] + in256[b + 48]
        a2 = in256[b + 16] - in256[b + 48]
        a3 = in256[b + 0] - in256[b + 32]
        tmp[0 + i * 4] = a0 + a1
        tmp[1 + i * 4] = a3 + a2
        tmp[2 + i * 4] = a3 - a2
        tmp[3 + i * 4] = a0 - a1
    for i in range(4):
        a0 = tmp[0 + i] + tmp[8 + i]
        a1 = tmp[4 + i] + tmp[12 + i]
        a2 = tmp[4 + i] - tmp[12 + i]
        a3 = tmp[0 + i] - tmp[8 + i]
        out[0 + i] = (a0 + a1) >> 1
        out[4 + i] = (a3 + a2) >> 1
        out[8 + i] = (a3 - a2) >> 1
        out[12 + i] = (a0 - a1) >> 1


@njit(cache=True, nogil=True, inline="always")
def _iwht(in16, out256):
    """TransformWHT_C (decoder inverse WHT). in16: 16, out256: block DC slots."""
    tmp = np.empty(16, np.int64)
    for i in range(4):
        a0 = in16[0 + i] + in16[12 + i]
        a1 = in16[4 + i] + in16[8 + i]
        a2 = in16[4 + i] - in16[8 + i]
        a3 = in16[0 + i] - in16[12 + i]
        tmp[0 + i] = a0 + a1
        tmp[8 + i] = a0 - a1
        tmp[4 + i] = a3 + a2
        tmp[12 + i] = a3 - a2
    p = 0
    for i in range(4):
        dc = tmp[0 + i * 4] + 3
        a0 = dc + tmp[3 + i * 4]
        a1 = tmp[1 + i * 4] + tmp[2 + i * 4]
        a2 = tmp[1 + i * 4] - tmp[2 + i * 4]
        a3 = dc - tmp[3 + i * 4]
        out256[p + 0] = (a0 + a1) >> 3
        out256[p + 16] = (a3 + a2) >> 3
        out256[p + 32] = (a0 - a1) >> 3
        out256[p + 48] = (a3 - a2) >> 3
        p += 64


@njit(cache=True, nogil=True, inline="always")
def _idct_full(in16, ref, dst):
    """ITransformOne_C: adds to ref, clips, writes dst (4x4)."""
    tmp = np.empty(16, np.int64)
    for i in range(4):
        a = in16[0 + i] + in16[8 + i]
        b = in16[0 + i] - in16[8 + i]
        c = _mul2(in16[4 + i]) - _mul1(in16[12 + i])
        d = _mul1(in16[4 + i]) + _mul2(in16[12 + i])
        tmp[0 + i * 4] = a + d
        tmp[1 + i * 4] = b + c
        tmp[2 + i * 4] = b - c
        tmp[3 + i * 4] = a - d
    for i in range(4):
        dc = tmp[i] + 4
        a = dc + tmp[8 + i]
        b = dc - tmp[8 + i]
        c = _mul2(tmp[4 + i]) - _mul1(tmp[12 + i])
        d = _mul1(tmp[4 + i]) + _mul2(tmp[12 + i])
        v = (a + d) >> 3
        dst[i, 0] = min(255, max(0, ref[i, 0] + v))
        v = (b + c) >> 3
        dst[i, 1] = min(255, max(0, ref[i, 1] + v))
        v = (b - c) >> 3
        dst[i, 2] = min(255, max(0, ref[i, 2] + v))
        v = (a - d) >> 3
        dst[i, 3] = min(255, max(0, ref[i, 3] + v))


@njit(cache=True, nogil=True, inline="always")
def _idct_dc(in16, ref, dst):
    dc = (in16[0] + 4) >> 3
    for j in range(4):
        for i in range(4):
            dst[j, i] = min(255, max(0, ref[j, i] + dc))


@njit(cache=True, nogil=True, inline="always")
def _idct_ac3(in16, ref, dst):
    """TransformAC3_C (valid when only zigzag positions 0..2 can be nonzero)."""
    a = in16[0] + 4
    c4 = _mul2(in16[4])
    d4 = _mul1(in16[4])
    c1 = _mul2(in16[1])
    d1 = _mul1(in16[1])
    for y in range(4):
        if y == 0:
            dc = a + d4
        elif y == 1:
            dc = a + c4
        elif y == 2:
            dc = a - c4
        else:
            dc = a - d4
        for x in range(4):
            if x == 0:
                v = dc + d1
            elif x == 1:
                v = dc + c1
            elif x == 2:
                v = dc - c1
            else:
                v = dc - d1
            dst[y, x] = min(255, max(0, ref[y, x] + (v >> 3)))


@njit(cache=True, nogil=True, inline="always")
def _quantize(coeff, q, iq, bias, zt, sh, out, first):
    """QuantizeBlock_C with zthresh+sharpen; out in zigzag order.
    Returns nz = last nonzero zigzag position + 1 (GetCoeffs return value)."""
    last = -1
    for n in range(first, 16):
        j = ZIGZAG[n]
        c = coeff[j]
        sign = 1 if c < 0 else 0
        if sign:
            c = -c
        c += sh[j]
        if c > zt[j]:
            level = (c * iq[j] + bias[j]) >> QFIX
            if level > 2047:
                level = 2047
            if sign:
                level = -level
            out[n] = level
            if level != 0:
                last = n
        else:
            out[n] = 0
    return last + 1


@njit(cache=True, nogil=True, inline="always")
def _dequant_into(levels, deq, tmp16, first):
    """tmp16[slot] = levels[n] * deq[n>0] for n >= first."""
    for n in range(first, 16):
        tmp16[ZIGZAG[n]] = levels[n] * (deq[0] if n == 0 else deq[1])


# ---------------------------------------------------------------- driver


def closed_loop_residuals(Y, U, V, is_i4, i16_mode, uv_mode, i4_modes,
                          y1, y2, uv_m, y2deq, y1deq, uvdeq):
    """Sequentially reconstruct (exact decoder semantics) and emit levels.
    Y/U/V: padded planes (int16, mb grid).  Returns (y_dc, y_ac, uv_levels)."""
    from .vp8_encode import _i4_preds, _block_preds   # verified numpy preds

    mb_h, mb_w = Y.shape[0] // 16, Y.shape[1] // 16
    n_mb = mb_h * mb_w
    H, W = Y.shape
    HH, HW = H // 2, W // 2

    rY = np.zeros((H + 1, W + 1), np.int16)
    rY[0, :] = 127
    rY[:, 0] = 129
    rU = np.zeros((HH + 1, HW + 1), np.int16)
    rU[0, :] = 127
    rU[:, 0] = 129
    rV = np.zeros((HH + 1, HW + 1), np.int16)
    rV[0, :] = 127
    rV[:, 0] = 129

    y_dc = np.zeros((n_mb, 16), np.int16)
    y_ac = np.zeros((n_mb, 16, 16), np.int16)
    uv_lv = np.zeros((n_mb, 8, 16), np.int16)

    tmp = np.zeros(16, np.int64)
    t16 = np.zeros(16, np.int64)
    lv = np.zeros(16, np.int64)
    dc16 = np.zeros(16, np.int64)
    in256 = np.zeros(256, np.int64)

    y1a = (y1.q, y1.iq, y1.bias, y1.zthresh, y1.sharpen)
    y2a = (y2.q, y2.iq, y2.bias, y2.zthresh, y2.sharpen)
    uva = (uv_m.q, uv_m.iq, uv_m.bias, uv_m.zthresh, uv_m.sharpen)

    for mby in range(mb_h):
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            py0 = mby * 16
            px0 = mbx * 16

            if not is_i4[mb]:
                # ---- I16 ----
                top = rY[py0, px0 + 1:px0 + 17].reshape(1, 1, 16)
                left = rY[py0 + 1:py0 + 17, px0].reshape(1, 1, 16)
                X = rY[py0, px0].reshape(1, 1)
                pred = _block_preds(top, left, X, valid=(mby > 0, mbx > 0))[i16_mode[mb], 0, 0]
                src16 = Y[py0:py0 + 16, px0:px0 + 16]
                coeff = np.zeros((16, 16), np.int64)
                for b in range(16):
                    by, bx = b // 4, b % 4
                    res = src16[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] - pred[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
                    _fdct(res, coeff[b])
                np.copyto(in256, coeff.reshape(-1))
                _fwht(in256, dc16)
                lv[:] = 0
                nz2 = _quantize(dc16, y2a[0], y2a[1], y2a[2], y2a[3], y2a[4], lv, 0)
                y_dc[mb] = lv
                dc_deq = np.zeros(16, np.int64)
                _dequant_into(lv, y2deq, dc_deq, 0)
                coeff = in256.reshape(16, 16)
                if nz2 > 1:
                    _iwht(dc_deq, in256)
                    coeff = in256.reshape(16, 16).copy()
                else:
                    coeff[:, 0] = (dc_deq[0] + 3) >> 3
                recon16 = np.zeros((16, 16), np.int16)
                for b in range(16):
                    by, bx = b // 4, b % 4
                    lv[:] = 0
                    nz1 = _quantize(coeff[b], y1a[0], y1a[1], y1a[2], y1a[3], y1a[4], lv, 1)
                    y_ac[mb, b] = lv
                    np.copyto(t16, np.zeros(16, np.int64))
                    t16[0] = coeff[b, 0]
                    _dequant_into(lv, y1deq, t16, 1)
                    dz = nz1 if nz1 > 0 else 1
                    pb = pred[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
                    rb = recon16[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
                    if dz > 3:
                        _idct_full(t16, pb, rb)
                    elif dz > 1:
                        _idct_ac3(t16, pb, rb)
                    elif t16[0] != 0:
                        _idct_dc(t16, pb, rb)
                    else:
                        rb[:] = pb
                rY[py0 + 1:py0 + 17, px0 + 1:px0 + 17] = recon16
            else:
                # ---- I4 ----
                for sy in range(4):
                    for sx in range(4):
                        py = py0 + sy * 4
                        px = px0 + sx * 4
                        mode = i4_modes[mb, sy * 4 + sx]
                        tt = np.empty(9, np.int16)
                        tt[0] = rY[py, px]
                        tt[1:5] = rY[py, px + 1:px + 5]
                        if sx == 3:
                            # rightmost subblock: TR comes from the row above the
                            # MB (vertically replicated by the decoder), clamped
                            # to the MB's own last pixel at the frame edge
                            for k in range(4):
                                c = px0 + 16 + k
                                if c > W - 1:
                                    c = px0 + 15
                                tt[5 + k] = rY[py0, c + 1]
                        else:
                            for k in range(4):
                                c = min(px + 4 + k, W - 1)
                                tt[5 + k] = rY[py, c + 1]
                        L = rY[py + 1:py + 5, px]
                        pred = _i4_preds(tt.reshape(1, 9), L.reshape(1, 4))[mode, 0]
                        srcb = Y[py:py + 4, px:px + 4]
                        _fdct(srcb - pred, tmp)
                        lv[:] = 0
                        nzb = _quantize(tmp, y1a[0], y1a[1], y1a[2], y1a[3], y1a[4], lv, 0)
                        y_ac[mb, sy * 4 + sx] = lv
                        np.copyto(t16, np.zeros(16, np.int64))
                        _dequant_into(lv, y1deq, t16, 0)
                        rb = np.zeros((4, 4), np.int16)
                        if nzb > 3:
                            _idct_full(t16, pred, rb)
                        elif nzb > 1:
                            _idct_ac3(t16, pred, rb)
                        elif nzb == 1 and t16[0] != 0:
                            _idct_dc(t16, pred, rb)
                        else:
                            rb[:] = pred
                        rY[py + 1:py + 5, px + 1:px + 5] = rb

            # ---- chroma (always) ----
            for ci, (rC, srcC) in enumerate(((rU, U), (rV, V))):
                cy0 = mby * 8
                cx0 = mbx * 8
                top = rC[cy0, cx0 + 1:cx0 + 9].reshape(1, 1, 8)
                left = rC[cy0 + 1:cy0 + 9, cx0].reshape(1, 1, 8)
                X = rC[cy0, cx0].reshape(1, 1)
                pred = _block_preds(top, left, X, valid=(mby > 0, mbx > 0))[uv_mode[mb], 0, 0]
                src8 = srcC[cy0:cy0 + 8, cx0:cx0 + 8]
                group_nz = np.zeros(4, np.int64)
                group_lv = np.zeros((4, 16), np.int64)
                group_t = np.zeros((4, 16), np.int64)
                for b in range(4):
                    by, bx = b // 2, b % 2
                    res = src8[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] - pred[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
                    _fdct(res, tmp)
                    lv[:] = 0
                    nzb = _quantize(tmp, uva[0], uva[1], uva[2], uva[3], uva[4], lv, 0)
                    uv_lv[mb, ci * 4 + b] = lv
                    group_nz[b] = nzb
                    _dequant_into(lv, uvdeq, t16, 0)
                    group_t[b] = t16
                for b in range(4):
                    by, bx = b // 2, b % 2
                    pb = pred[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4]
                    rb = np.zeros((4, 4), np.int16)
                    anybig = group_nz.max() > 1
                    if anybig:
                        _idct_full(group_t[b], pb, rb)
                    elif group_nz[b] == 1 and group_t[b][0] != 0:
                        _idct_dc(group_t[b], pb, rb)
                    else:
                        rb[:] = pb
                    rC[cy0 + by * 4 + 1:cy0 + by * 4 + 5, cx0 + bx * 4 + 1:cx0 + bx * 4 + 5] = rb

    return y_dc, y_ac, uv_lv, rY, rU, rV
