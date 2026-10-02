
// __nanosleep needs sm_70+; older archs (Pascal sm_61 etc.) spin on a
// threadfence loop instead so the wavefront kernels stay multi-arch
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
__device__ __forceinline__ void __DEVICE_SPIN(unsigned ns) {
    volatile int _s = 0;
    for (unsigned _i = 0; _i < (ns >> 3) + 1; ++_i) _s += 1;
}
#else
#define __DEVICE_SPIN __nanosleep
#endif


#define QFIX 17
typedef long long i64;
typedef unsigned char u8;
typedef short i16;

__device__ const int ZIG[16] = {0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15};

__device__ i64 mini64(i64 a, i64 b) { return a < b ? a : b; }
__device__ i64 maxi64(i64 a, i64 b) { return a > b ? a : b; }
__device__ int clip255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// ---------------- transforms (ports of closed_loop.py, int64) ---------------

__device__ void fdct(const int* res, i64* out) {
    i64 tmp[16];
    for (int i = 0; i < 4; ++i) {
        i64 d0 = res[i*4+0], d1 = res[i*4+1], d2 = res[i*4+2], d3 = res[i*4+3];
        i64 a0 = d0 + d3, a1 = d1 + d2, a2 = d1 - d2, a3 = d0 - d3;
        tmp[0+i*4] = (a0 + a1) * 8;
        tmp[1+i*4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[2+i*4] = (a0 - a1) * 8;
        tmp[3+i*4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }
    for (int i = 0; i < 4; ++i) {
        i64 a0 = tmp[0+i] + tmp[12+i];
        i64 a1 = tmp[4+i] + tmp[8+i];
        i64 a2 = tmp[4+i] - tmp[8+i];
        i64 a3 = tmp[0+i] - tmp[12+i];
        out[0+i]  = (a0 + a1 + 7) >> 4;
        out[4+i]  = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0 ? 1 : 0);
        out[8+i]  = (a0 - a1 + 7) >> 4;
        out[12+i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16;
    }
}

__device__ void fwht(const i64* in256, i64* out) {
    i64 tmp[16];
    for (int i = 0; i < 4; ++i) {
        i64 b = i * 64;
        i64 a0 = in256[b+0] + in256[b+32];
        i64 a1 = in256[b+16] + in256[b+48];
        i64 a2 = in256[b+16] - in256[b+48];
        i64 a3 = in256[b+0] - in256[b+32];
        tmp[0+i*4] = a0 + a1;
        tmp[1+i*4] = a3 + a2;
        tmp[2+i*4] = a3 - a2;
        tmp[3+i*4] = a0 - a1;
    }
    for (int i = 0; i < 4; ++i) {
        i64 a0 = tmp[0+i] + tmp[8+i];
        i64 a1 = tmp[4+i] + tmp[12+i];
        i64 a2 = tmp[4+i] - tmp[12+i];
        i64 a3 = tmp[0+i] - tmp[8+i];
        out[0+i] = (a0 + a1) >> 1;
        out[4+i] = (a3 + a2) >> 1;
        out[8+i] = (a3 - a2) >> 1;
        out[12+i] = (a0 - a1) >> 1;
    }
}

__device__ void iwht(const i64* in16, i64* out256) {
    i64 tmp[16];
    for (int i = 0; i < 4; ++i) {
        i64 a0 = in16[0+i] + in16[12+i];
        i64 a1 = in16[4+i] + in16[8+i];
        i64 a2 = in16[4+i] - in16[8+i];
        i64 a3 = in16[0+i] - in16[12+i];
        tmp[0+i] = a0 + a1;
        tmp[8+i] = a0 - a1;
        tmp[4+i] = a3 + a2;
        tmp[12+i] = a3 - a2;
    }
    int p = 0;
    for (int i = 0; i < 4; ++i) {
        i64 dc = tmp[0+i*4] + 3;
        i64 a0 = dc + tmp[3+i*4];
        i64 a1 = tmp[1+i*4] + tmp[2+i*4];
        i64 a2 = tmp[1+i*4] - tmp[2+i*4];
        i64 a3 = dc - tmp[3+i*4];
        out256[p+0] = (a0 + a1) >> 3;
        out256[p+16] = (a3 + a2) >> 3;
        out256[p+32] = (a0 - a1) >> 3;
        out256[p+48] = (a3 - a2) >> 3;
        p += 64;
    }
}

__device__ i64 mul1(i64 a) { return ((a * 20091) >> 16) + a; }
__device__ i64 mul2(i64 a) { return (a * 35468) >> 16; }

__device__ void idct_full(const i64* in16, i16* ref, i16* dst) {
    i64 tmp[16];
    for (int i = 0; i < 4; ++i) {
        i64 a = in16[0+i] + in16[8+i];
        i64 b = in16[0+i] - in16[8+i];
        i64 c = mul2(in16[4+i]) - mul1(in16[12+i]);
        i64 d = mul1(in16[4+i]) + mul2(in16[12+i]);
        tmp[0+i*4] = a + d;
        tmp[1+i*4] = b + c;
        tmp[2+i*4] = b - c;
        tmp[3+i*4] = a - d;
    }
    for (int i = 0; i < 4; ++i) {
        i64 dc = tmp[i] + 4;
        i64 a = dc + tmp[8+i];
        i64 b = dc - tmp[8+i];
        i64 c = mul2(tmp[4+i]) - mul1(tmp[12+i]);
        i64 d = mul1(tmp[4+i]) + mul2(tmp[12+i]);
        dst[i*4+0] = (i16)clip255((int)(ref[i*4+0] + ((a + d) >> 3)));
        dst[i*4+1] = (i16)clip255((int)(ref[i*4+1] + ((b + c) >> 3)));
        dst[i*4+2] = (i16)clip255((int)(ref[i*4+2] + ((b - c) >> 3)));
        dst[i*4+3] = (i16)clip255((int)(ref[i*4+3] + ((a - d) >> 3)));
    }
}

__device__ void idct_dc(const i64* in16, i16* ref, i16* dst) {
    int dc = (int)((in16[0] + 4) >> 3);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            dst[j*4+i] = (i16)clip255((int)ref[j*4+i] + dc);
}

__device__ void idct_ac3(const i64* in16, i16* ref, i16* dst) {
    i64 a = in16[0] + 4;
    i64 c4 = mul2(in16[4]);
    i64 d4 = mul1(in16[4]);
    i64 c1 = mul2(in16[1]);
    i64 d1 = mul1(in16[1]);
    for (int y = 0; y < 4; ++y) {
        i64 dc;
        if (y == 0) dc = a + d4;
        else if (y == 1) dc = a + c4;
        else if (y == 2) dc = a - c4;
        else dc = a - d4;
        for (int x = 0; x < 4; ++x) {
            i64 v;
            if (x == 0) v = dc + d1;
            else if (x == 1) v = dc + c1;
            else if (x == 2) v = dc - c1;
            else v = dc - d1;
            dst[y*4+x] = (i16)clip255((int)ref[y*4+x] + (int)(v >> 3));
        }
    }
}

__device__ int quantize(const i64* coeff, const i64* q, const i64* iq,
                        const i64* bias, const i64* zt, const i64* sh,
                        i64* out, int first) {
    int last = -1;
    for (int n = first; n < 16; ++n) {
        int j = ZIG[n];
        i64 c = coeff[j];
        int sign = c < 0 ? 1 : 0;
        if (sign) c = -c;
        c += sh[j];
        if (c > zt[j]) {
            i64 level = (c * iq[j] + bias[j]) >> QFIX;
            if (level > 2047) level = 2047;
            if (sign) level = -level;
            out[n] = level;
            if (level != 0) last = n;
        } else {
            out[n] = 0;
        }
    }
    return last + 1;
}

__device__ void dequant_into(const i64* levels, const i64* deq, i64* tmp16,
                             int first) {
    for (int n = first; n < 16; ++n)
        tmp16[ZIG[n]] = levels[n] * (n == 0 ? deq[0] : deq[1]);
}

// ---------------- intra prediction (ports of closed_loop_jit) --------------

// The generic pred reads via a stride argument; defined as a macro-free
// function taking the plane stride.
__device__ void pred_blk_s(int mode, const i16* rC, int stride, int y0,
                           int x0, int blk, i16* out) {
    int has_t = y0 > 0;
    int has_l = x0 > 0;
    if (mode == 2) {            // V: replicate the row above
        for (int c = 0; c < blk; ++c) {
            int v = rC[y0 * stride + (x0 + 1 + c)];
            for (int r = 0; r < blk; ++r) out[r * blk + c] = (i16)v;
        }
        return;
    }
    if (mode == 3) {            // H: replicate the column left
        for (int r = 0; r < blk; ++r) {
            int v = rC[(y0 + 1 + r) * stride + x0];
            for (int c = 0; c < blk; ++c) out[r * blk + c] = (i16)v;
        }
        return;
    }
    if (mode == 1) {            // TM
        int X = rC[y0 * stride + x0];
        for (int r = 0; r < blk; ++r) {
            int l = rC[(y0 + 1 + r) * stride + x0];
            for (int c = 0; c < blk; ++c)
                out[r * blk + c] = (i16)clip255(
                    rC[y0 * stride + (x0 + 1 + c)] + l - X);
        }
        return;
    }
    int sh = (blk == 8) ? 4 : 5;    // DC
    int dc;
    if (has_t && has_l) {
        int st = 0, sl = 0;
        for (int c = 0; c < blk; ++c) st += rC[y0 * stride + (x0 + 1 + c)];
        for (int r = 0; r < blk; ++r) sl += rC[(y0 + 1 + r) * stride + x0];
        dc = (st + sl + blk) >> sh;
    } else if (has_t) {
        int st = 0;
        for (int c = 0; c < blk; ++c) st += rC[y0 * stride + (x0 + 1 + c)];
        dc = (st + (blk >> 1)) >> (sh - 1);
    } else if (has_l) {
        int sl = 0;
        for (int r = 0; r < blk; ++r) sl += rC[(y0 + 1 + r) * stride + x0];
        dc = (sl + (blk >> 1)) >> (sh - 1);
    } else {
        dc = 128;
    }
    for (int r = 0; r < blk; ++r)
        for (int c = 0; c < blk; ++c) out[r * blk + c] = (i16)dc;
}

__device__ i64 a2_(i64 a, i64 b) { return (a + b + 1) >> 1; }
__device__ i64 a3_(i64 a, i64 b, i64 c) { return (a + 2 * b + c + 2) >> 2; }

__device__ void i4_pred(int mode, const i16* rY, int stride, int py, int px,
                        int py0, int W, i16* out) {
    i64 X = rY[py * stride + px];
    i64 A = rY[py * stride + px + 1];
    i64 B = rY[py * stride + px + 2];
    i64 C = rY[py * stride + px + 3];
    i64 D = rY[py * stride + px + 4];
    i64 l0 = rY[(py + 1) * stride + px];
    i64 l1 = rY[(py + 2) * stride + px];
    i64 l2 = rY[(py + 3) * stride + px];
    i64 l3 = rY[(py + 4) * stride + px];
    int sx = (px >> 2) & 3;          // subblock column within MB
    int tr_row = (sx == 3) ? py0 : py;
    i64 E = rY[tr_row * stride + mini64(px + 5, W)];
    i64 F = rY[tr_row * stride + mini64(px + 6, W)];
    i64 G = rY[tr_row * stride + mini64(px + 7, W)];
    i64 Hh = rY[tr_row * stride + mini64(px + 8, W)];
    i64 I = l0, J = l1, K = l2, Lm = l3;
    i64 o[16];
    if (mode == 0) {              // B_DC_PRED
        i64 dc = (4 + A + B + C + D + l0 + l1 + l2 + l3) >> 3;
        for (int i = 0; i < 16; ++i) o[i] = dc;
    } else if (mode == 1) {       // B_TM_PRED
        for (int r = 0; r < 4; ++r) {
            i64 l = rY[(py + 1 + r) * stride + px];
            for (int c = 0; c < 4; ++c)
                o[r * 4 + c] = clip255((int)(rY[py * stride + (px + 1 + c)] + l - X));
        }
    } else if (mode == 2) {       // B_VE_PRED
        i64 v0 = a3_(X, A, B), v1 = a3_(A, B, C), v2 = a3_(B, C, D), v3 = a3_(C, D, E);
        for (int r = 0; r < 4; ++r) {
            o[r*4+0] = v0; o[r*4+1] = v1; o[r*4+2] = v2; o[r*4+3] = v3;
        }
    } else if (mode == 3) {       // B_HE_PRED
        i64 v0 = a3_(X, l0, l1), v1 = a3_(l0, l1, l2);
        i64 v2 = a3_(l1, l2, l3), v3 = a3_(l2, l3, l3);
        for (int r = 0; r < 4; ++r) {
            i64 v = (r == 0) ? v0 : (r == 1) ? v1 : (r == 2) ? v2 : v3;
            for (int c = 0; c < 4; ++c) o[r * 4 + c] = v;
        }
    } else if (mode == 4) {       // B_LD_PRED
        o[0*4+0]=a3_(A,X,I); o[0*4+1]=a3_(X,A,B); o[0*4+2]=a3_(A,B,C); o[0*4+3]=a3_(B,C,D);
        o[1*4+0]=a3_(X,I,J); o[1*4+1]=a3_(A,X,I); o[1*4+2]=a3_(X,A,B); o[1*4+3]=a3_(A,B,C);
        o[2*4+0]=a3_(I,J,K); o[2*4+1]=a3_(X,I,J); o[2*4+2]=a3_(A,X,I); o[2*4+3]=a3_(X,A,B);
        o[3*4+0]=a3_(J,K,Lm); o[3*4+1]=a3_(I,J,K); o[3*4+2]=a3_(X,I,J); o[3*4+3]=a3_(A,X,I);
    } else if (mode == 5) {       // B_RD_PRED
        o[0*4+0]=a2_(X,A); o[0*4+1]=a2_(A,B); o[0*4+2]=a2_(B,C); o[0*4+3]=a2_(C,D);
        o[1*4+0]=a3_(A,X,I); o[1*4+1]=a3_(X,A,B); o[1*4+2]=a3_(A,B,C); o[1*4+3]=a3_(B,C,D);
        o[2*4+0]=a3_(X,I,J); o[2*4+1]=a2_(X,A); o[2*4+2]=a2_(A,B); o[2*4+3]=a2_(B,C);
        o[3*4+0]=a3_(I,J,K); o[3*4+1]=a3_(A,X,I); o[3*4+2]=a3_(X,A,B); o[3*4+3]=a3_(A,B,C);
    } else if (mode == 6) {       // B_VR_PRED
        o[0*4+0]=a3_(A,B,C); o[0*4+1]=a3_(B,C,D); o[0*4+2]=a3_(C,D,E); o[0*4+3]=a3_(D,E,F);
        o[1*4+0]=a3_(B,C,D); o[1*4+1]=a3_(C,D,E); o[1*4+2]=a3_(D,E,F); o[1*4+3]=a3_(E,F,G);
        o[2*4+0]=a3_(C,D,E); o[2*4+1]=a3_(D,E,F); o[2*4+2]=a3_(E,F,G); o[2*4+3]=a3_(F,G,Hh);
        o[3*4+0]=a3_(D,E,F); o[3*4+1]=a3_(E,F,G); o[3*4+2]=a3_(F,G,Hh); o[3*4+3]=a3_(G,Hh,Hh);
    } else if (mode == 7) {       // B_VL_PRED
        o[0*4+0]=a2_(A,B); o[0*4+1]=a2_(B,C); o[0*4+2]=a2_(C,D); o[0*4+3]=a2_(D,E);
        o[1*4+0]=a3_(A,B,C); o[1*4+1]=a3_(B,C,D); o[1*4+2]=a3_(C,D,E); o[1*4+3]=a3_(D,E,F);
        o[2*4+0]=a2_(B,C); o[2*4+1]=a2_(C,D); o[2*4+2]=a2_(D,E); o[2*4+3]=a3_(E,F,G);
        o[3*4+0]=a3_(B,C,D); o[3*4+1]=a3_(C,D,E); o[3*4+2]=a3_(D,E,F); o[3*4+3]=a3_(F,G,Hh);
    } else if (mode == 8) {       // B_HD_PRED
        o[0*4+0]=a2_(X,I); o[0*4+1]=a3_(A,X,I); o[0*4+2]=a3_(X,A,B); o[0*4+3]=a3_(A,B,C);
        o[1*4+0]=a2_(I,J); o[1*4+1]=a3_(X,I,J); o[1*4+2]=a2_(X,I); o[1*4+3]=a3_(A,X,I);
        o[2*4+0]=a2_(J,K); o[2*4+1]=a3_(I,J,K); o[2*4+2]=a2_(I,J); o[2*4+3]=a3_(X,I,J);
        o[3*4+0]=a2_(K,Lm); o[3*4+1]=a3_(J,K,Lm); o[3*4+2]=a2_(J,K); o[3*4+3]=a3_(I,J,K);
    } else {                      // B_HU_PRED (9)
        o[0*4+0]=a2_(I,J); o[0*4+1]=a3_(I,J,K); o[0*4+2]=a2_(J,K); o[0*4+3]=a3_(J,K,Lm);
        o[1*4+0]=a2_(J,K); o[1*4+1]=a3_(J,K,Lm); o[1*4+2]=a2_(K,Lm); o[1*4+3]=a3_(K,Lm,Lm);
        o[2*4+0]=a2_(K,Lm); o[2*4+1]=a3_(K,Lm,Lm); o[2*4+2]=Lm; o[2*4+3]=Lm;
        o[3*4+0]=Lm; o[3*4+1]=Lm; o[3*4+2]=Lm; o[3*4+3]=Lm;
    }
    for (int i = 0; i < 16; ++i) out[i] = (i16)o[i];
}


// ------------------ trellis (RD-optimized) quantization ------------------
// Faithful port of libwebp's TrellisQuantizeBlock (quant_enc.c, BSD):
// per-block Viterbi over level deltas {level0, level0+1} minimising
//   score = rate*lambda + 256*weighted_distortion
// with the rate model from the DEFAULT probability tables (frame
// independent -> bit-deterministic on any GPU). Rates are in 1/256-bit
// fixed point; distortion weights follow kWeightTrellis(USE_TDISTO=1).
#include "trellis_tables.inc"

// coefficient bands with the position-16 sentinel (BANDS[n+1] reads)
__device__ const unsigned char TRE_BANDS[17] = {
    0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0
};

#define TRE_MAX_COST 0x7fffffffffffff00LL

// VP8LevelCost on the flattened [4][8][3][68] level-cost table
__device__ __forceinline__ int tre_levelcost(int tbl_base, int level) {
    int v = level > 67 ? 67 : level;
    return TRE_LEVEL_FIXED[level] + TRE_LEVEL_COST[tbl_base + v];
}
__device__ __forceinline__ int tre_tbl(int ctype, int band, int ctx) {
    return ((ctype * 8 + band) * 3 + ctx) * 68;
}

// returns last+1 (index past the highest nonzero level), like quantize();
// out[n] = signed level in ZIGZAG order. Scratch arrays are per-thread.
__device__ int trellis_quantize(const i64* coeff, const i64* q,
                                const i64* iq, const i64* sh,
                                i64* out, int first, int ctype,
                                int ctx0, i64 lambda,
                                short* n_lvl, short* n_sign,
                                int* n_prev, int* best_path) {
    const int band0 = TRE_BANDS[first];
    i64 thresh = q[1] * q[1] / 4;
    int last = first - 1;
    for (int n = 15; n >= first; --n) {
        int j = ZIG[n];
        if (coeff[j] * coeff[j] > thresh) { last = n; break; }
    }
    if (last < 15) ++last;

    int p0 = (int)COEFFS_PROBA0_T[((ctype * 8 + band0) * 3 + ctx0) * 11];
    i64 skip_rate = TRE_ENTROPY_COST[p0];              // BitCost(0, p0)
    i64 on_rate = (ctx0 == 0) ? TRE_ENTROPY_COST[255 - p0] : 0;

    i64 score[2] = { on_rate * lambda, on_rate * lambda };
    int tbl[2] = { tre_tbl(ctype, band0, ctx0),
                   tre_tbl(ctype, band0, ctx0) };
    int have[2] = { 1, 1 };
    i64 best_score = skip_rate * lambda;
    best_path[0] = -1;

    for (int n = first; n <= last; ++n) {
        const int j = ZIG[n];
        const i64 Q = q[j], iQ = iq[j];
        const int sign = coeff[j] < 0;
        const i64 c0 = (sign ? -coeff[j] : coeff[j]) + sh[j];
        int level0 = (int)((c0 * iQ) >> QFIX);                       // B = 0
        int thresh_level = (int)((c0 * iQ + (128 << (QFIX - 8))) >> QFIX);
        if (level0 > 2047) level0 = 2047;
        if (thresh_level > 2047) thresh_level = 2047;
        if (level0 < 0) level0 = 0;
        if (thresh_level < 0) thresh_level = 0;
        const i64 w = TRE_WEIGHT[j];
        const int band_next = TRE_BANDS[n + 1];

        i64 nscore[2];
        int ntbl[2], nhave[2];
        for (int m = 0; m < 2; ++m) {
            const int level = level0 + m;
            const int ctx = level > 2 ? 2 : level;
            if (level > thresh_level) {
                nhave[m] = 0;
                nscore[m] = TRE_MAX_COST;
                ntbl[m] = tre_tbl(ctype, band_next, 1);
                continue;
            }
            const i64 new_err = c0 - (i64)level * Q;
            const i64 delta_err = w * (new_err * new_err - c0 * c0);
            i64 best = TRE_MAX_COST;
            int bp = 0;
            for (int p = 0; p < 2; ++p) {
                if (!have[p]) continue;
                i64 rate = (i64)tre_levelcost(tbl[p], level) * lambda;
                i64 s = score[p] + rate;
                if (s < best) { best = s; bp = p; }
            }
            best += delta_err * 256;
            nscore[m] = best;
            nhave[m] = 1;
            ntbl[m] = tre_tbl(ctype, band_next, ctx);
            n_lvl[n * 2 + m] = (short)level;
            n_sign[n] = (short)sign;
            n_prev[n * 2 + m] = bp;
            if (level != 0 && best < best_score) {
                int p0b = (int)COEFFS_PROBA0_T[
                    ((ctype * 8 + band_next) * 3 + ctx) * 11];
                i64 lp = (n < 15)
                         ? (i64)TRE_ENTROPY_COST[p0b] * lambda : 0;
                if (best + lp < best_score) {
                    best_score = best + lp;
                    best_path[0] = n;
                    best_path[1] = m;
                    best_path[2] = bp;
                }
            }
        }
        score[0] = nscore[0]; score[1] = nscore[1];
        tbl[0] = ntbl[0]; tbl[1] = ntbl[1];
        have[0] = nhave[0]; have[1] = nhave[1];
    }

    for (int n = first; n < 16; ++n) out[n] = 0;
    if (best_path[0] == -1) return 0;
    int node = best_path[1];
    n_prev[best_path[0] * 2 + node] = best_path[2];
    int nz = 0;
    for (int n = best_path[0]; n >= first; --n) {
        short lv = n_lvl[n * 2 + node];
        out[n] = n_sign[n] ? -(i64)lv : (i64)lv;
        if (lv && n + 1 > nz) nz = n + 1;   // highest nonzero position
        node = n_prev[n * 2 + node];
    }
    return nz;
}

__device__ int g_trellis_on = 0;

// ------------------------- the per-image closed loop ------------------------

extern "C" __global__ void closed_loop_kernel(
    const i16* __restrict__ Y, const i16* __restrict__ U,
    const i16* __restrict__ V,
    const u8* __restrict__ is_i4, const u8* __restrict__ i16_mode,
    const u8* __restrict__ uv_mode, const u8* __restrict__ i4_modes,
    const i64* __restrict__ y1q, const i64* __restrict__ y1iq,
    const i64* __restrict__ y1b, const i64* __restrict__ y1z,
    const i64* __restrict__ y1s,
    const i64* __restrict__ y2q, const i64* __restrict__ y2iq,
    const i64* __restrict__ y2b, const i64* __restrict__ y2z,
    const i64* __restrict__ y2s,
    const i64* __restrict__ uvq, const i64* __restrict__ uviq,
    const i64* __restrict__ uvb, const i64* __restrict__ uvz,
    const i64* __restrict__ uvs,
    const i64* __restrict__ y1deq, const i64* __restrict__ y2deq,
    const i64* __restrict__ uvdeq,
    const i64* __restrict__ tre_lam,         // [2]: lambda i4, i16
    i16* __restrict__ y_dc, i16* __restrict__ y_ac, i16* __restrict__ uv_lv,
    i16* __restrict__ rY, i16* __restrict__ rU, i16* __restrict__ rV,
    unsigned int* __restrict__ flags,
    int B, int mb_h, int mb_w, int H, int W, const int* w_clamp)
{
    // one thread drives one MB row of one image; rows pipeline behind each
    // other via per-MB completion flags.  The launcher caps B*mb_h so that
    // every block is resident (spinning blocks must not wait on unscheduled
    // producers) — no deadlock is possible under that invariant.
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= B * mb_h) return;
    int b = blk / mb_h;
    int mby = blk % mb_h;
    int HH = H / 2, HW = W / 2;
    int n_mb = mb_h * mb_w;
    int ystride = W + 1, cstride = HW + 1;
    const i16* Yb_ = Y + (size_t)b * H * W;
    const i16* Ub_ = U + (size_t)b * HH * HW;
    const i16* Vb_ = V + (size_t)b * HH * HW;
    i16* rYb = rY + (size_t)b * (H + 1) * (W + 1);
    i16* rUb = rU + (size_t)b * (HH + 1) * (HW + 1);
    i16* rVb = rV + (size_t)b * (HH + 1) * (HW + 1);

    if (mby == 0) {
        // row 0 of each image also initialises the bordered recon planes
        // (single-thread launch: plain writes are enough). The corner [0,0]
        // belongs to the TOP border (127), matching libwebp's first-row
        // memset(y_dst - BPS - 1, 127, 21); writing 129 there desynced the
        // closed-loop reconstruction from the decoder and the error
        // cascaded down the prediction chains.
        for (int i = 0; i < W + 1; ++i) rYb[i] = 127;
        for (int r = 1; r < H + 1; ++r) rYb[r * ystride] = 129;
        for (int i = 0; i < HW + 1; ++i) { rUb[i] = 127; rVb[i] = 127; }
        for (int r = 1; r < HH + 1; ++r) {
            rUb[r * cstride] = 129;
            rVb[r * cstride] = 129;
        }
        __threadfence();
        // rows > 0 wait on our flags, which we only set after this fence,
        // so the borders are visible to them before any recon read.
    }

    i16 pred16[256], pred8[64], pred4[16], rb[16];
    int res[16];
    i64 tmp[16], t16[16], lv[16], dc16[16], dc_deq[16], in256[256],
        coeff[256], group_t[4][16];
    i64 group_nz[4];
    i16* ydc = y_dc + (size_t)b * n_mb * 16;
    i16* yac = y_ac + (size_t)b * n_mb * 256;
    i16* uvl = uv_lv + (size_t)b * n_mb * 128;
    const u8* isf = is_i4 + (size_t)b * n_mb;
    const u8* i16m = i16_mode + (size_t)b * n_mb;
    const u8* uvm = uv_mode + (size_t)b * n_mb;
    const u8* i4m = i4_modes + (size_t)b * n_mb * 16;
    volatile unsigned int* myflags = flags + (size_t)b * n_mb;

    // trellis RD-quantization state (per thread): scratch + token nz ctx
    short tr_lvl[32]; short tr_sign[16]; int tr_prev[32]; int tr_bp[4];
    int col_nz[4];     // running Y block-column nz (top ctx)
    int left_nz[4];    // running Y block-row nz (left ctx)

    for (int mbx = 0; mbx < mb_w; ++mbx) {
        int mb = mby * mb_w + mbx;
        if (mby > 0) {
            // wait for the MB above AND its right neighbour: the i4
            // top-right pixels of our x==3 subblocks read the row above
            // the MB at columns px0+16..19 (i.e. MB(r-1, c+1)'s recon)
            while (myflags[(size_t)(mby - 1) * mb_w + mbx] == 0) {
                __DEVICE_SPIN(64);
            }
            if (mbx + 1 < mb_w) {
                while (myflags[(size_t)(mby - 1) * mb_w + mbx + 1] == 0) {
                    __DEVICE_SPIN(64);
                }
            }
            __threadfence();
        }
        // token-context init at row start: top nz from the above MB's
        // bottom-row blocks (complete: we waited on its flags)
        if (mbx == 0) {
            for (int x = 0; x < 4; ++x) {
                col_nz[x] = 0;
                if (mby > 0) {
                    const i16* ab = yac + (size_t)(mb - mb_w) * 256
                                    + (size_t)(3 * 4 + x) * 16;
                    for (int t = 0; t < 16; ++t)
                        if (ab[t]) { col_nz[x] = 1; break; }
                }
            }
            for (int y = 0; y < 4; ++y) left_nz[y] = 0;
        }
        int py0 = mby * 16, px0 = mbx * 16;
        if (!isf[mb]) {
            pred_blk_s(i16m[mb], rYb, ystride, py0, px0, 16, pred16);
            for (int blk = 0; blk < 16; ++blk) {
                int by = blk >> 2, bx = blk & 3;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        res[r*4+c] = (int)Yb_[(py0+by*4+r)*W + (px0+bx*4+c)]
                                   - (int)pred16[(by*4+r)*16 + bx*4+c];
                fdct(res, tmp);
                for (int k = 0; k < 16; ++k) in256[blk*16+k] = tmp[k];
            }
            fwht(in256, dc16);
            for (int k = 0; k < 16; ++k) lv[k] = 0;
            int nz2 = quantize(dc16, y2q, y2iq, y2b, y2z, y2s, lv, 0);
            for (int k = 0; k < 16; ++k) ydc[mb*16+k] = (i16)lv[k];
            for (int k = 0; k < 16; ++k) dc_deq[k] = 0;
            dequant_into(lv, y2deq, dc_deq, 0);
            if (nz2 > 1) {
                iwht(dc_deq, in256);
                for (int k = 0; k < 256; ++k) coeff[k] = in256[k];
            } else {
                i64 dc0 = (dc_deq[0] + 3) >> 3;
                for (int blk2 = 0; blk2 < 16; ++blk2) {
                    coeff[blk2*16+0] = dc0;
                    for (int k = 1; k < 16; ++k)
                        coeff[blk2*16+k] = in256[blk2*16+k];
                }
            }
            for (int blk = 0; blk < 16; ++blk) {
                int by = blk >> 2, bx = blk & 3;
                for (int k = 0; k < 16; ++k) lv[k] = 0;
                int ctx1 = col_nz[bx] + left_nz[by];
                int nz1 = g_trellis_on
                    ? trellis_quantize(coeff + blk*16, y1q, y1iq, y1s, lv,
                                       1, 0, ctx1, tre_lam[1],
                                       tr_lvl, tr_sign, tr_prev, tr_bp)
                    : quantize(coeff + blk*16, y1q, y1iq, y1b, y1z,
                               y1s, lv, 1);
                col_nz[bx] = left_nz[by] = nz1 ? 1 : 0;
                for (int k = 0; k < 16; ++k)
                    yac[mb*256+blk*16+k] = (i16)lv[k];
                for (int k = 0; k < 16; ++k) t16[k] = 0;
                t16[0] = coeff[blk*16+0];
                dequant_into(lv, y1deq, t16, 1);
                int dz = nz1 > 0 ? nz1 : 1;
                i16* pb = pred16 + (by*4)*16 + bx*4;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) rb[r*4+c] = pb[r*16+c];
                if (dz > 3) idct_full(t16, rb, rb);
                else if (dz > 1) idct_ac3(t16, rb, rb);
                else if (t16[0] != 0) idct_dc(t16, rb, rb);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        rYb[(py0+1+by*4+r)*ystride + (px0+1+bx*4+c)]
                            = rb[r*4+c];
            }
        } else {
            for (int sb = 0; sb < 16; ++sb) {
                int sy = sb >> 2, sx = sb & 3;
                int py = py0 + sy * 4, px = px0 + sx * 4;
                i4_pred(i4m[mb*16+sb], rYb, ystride, py, px, py0, w_clamp[b], pred4);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        res[r*4+c] = (int)Yb_[(py + r)*W + px + c]
                                   - (int)pred4[r*4+c];
                fdct(res, tmp);
                for (int k = 0; k < 16; ++k) lv[k] = 0;
                int ctx1 = col_nz[sx] + left_nz[sy];
                int nz1 = g_trellis_on
                    ? trellis_quantize(tmp, y1q, y1iq, y1s, lv, 0, 3, ctx1,
                                       tre_lam[0], tr_lvl, tr_sign, tr_prev,
                                       tr_bp)
                    : quantize(tmp, y1q, y1iq, y1b, y1z, y1s, lv, 0);
                col_nz[sx] = left_nz[sy] = nz1 ? 1 : 0;
                for (int k = 0; k < 16; ++k)
                    yac[mb*256+sb*16+k] = (i16)lv[k];
                for (int k = 0; k < 16; ++k) t16[k] = 0;
                dequant_into(lv, y1deq, t16, 0);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) rb[r*4+c] = pred4[r*4+c];
                if (nz1 > 3) idct_full(t16, rb, rb);
                else if (nz1 > 1) idct_ac3(t16, rb, rb);
                else if (t16[0] != 0) idct_dc(t16, rb, rb);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        rYb[(py+1+r)*ystride + (px+1+c)] = rb[r*4+c];
            }
        }
        for (int ci = 0; ci < 2; ++ci) {
            i16* rC = ci == 0 ? rUb : rVb;
            const i16* srcC = ci == 0 ? Ub_ : Vb_;
            int cy0 = mby * 8, cx0 = mbx * 8;
            pred_blk_s(uvm[mb], rC, cstride, cy0, cx0, 8, pred8);
            for (int blk = 0; blk < 4; ++blk) {
                int by = blk >> 1, bx = blk & 1;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        res[r*4+c] = (int)srcC[(cy0+by*4+r)*HW + (cx0+bx*4+c)]
                                   - (int)pred8[(by*4+r)*8 + bx*4+c];
                fdct(res, tmp);
                for (int k = 0; k < 16; ++k) lv[k] = 0;
                int nzb = quantize(tmp, uvq, uviq, uvb, uvz, uvs, lv, 0);
                for (int k = 0; k < 16; ++k)
                    uvl[mb*128 + ci*64 + blk*16 + k] = (i16)lv[k];
                group_nz[blk] = nzb;
                for (int k = 0; k < 16; ++k) t16[k] = 0;
                dequant_into(lv, uvdeq, t16, 0);
                for (int k = 0; k < 16; ++k) group_t[blk][k] = t16[k];
            }
            int anybig = 0;
            for (int blk = 0; blk < 4; ++blk)
                if (group_nz[blk] > 1) anybig = 1;
            for (int blk = 0; blk < 4; ++blk) {
                int by = blk >> 1, bx = blk & 1;
                i16* pb = pred8 + (by*4)*8 + bx*4;
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) rb[r*4+c] = pb[r*8+c];
                if (anybig) idct_full(group_t[blk], rb, rb);
                else if (group_nz[blk] == 1 && group_t[blk][0] != 0)
                    idct_dc(group_t[blk], rb, rb);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        rC[(cy0+1+by*4+r)*cstride + (cx0+1+bx*4+c)]
                            = rb[r*4+c];
            }
        }
        __threadfence();
        myflags[(size_t)mby * mb_w + mbx] = 1;
    }
}

// Fused intra-mode search: one CUDA thread per macroblock computes the
// i16/uv best modes plus the full 16x10 i4 SSE table in a single launch
// (replaces ~50 separate cupy dispatches per batch).
extern "C" __global__ void mode_search_kernel(
    const i16* __restrict__ bY, const i16* __restrict__ bU,
    const i16* __restrict__ bV,
    const i64* __restrict__ fc_i16, const i64* __restrict__ fc_uv,
    u8* __restrict__ i16_mode, i64* __restrict__ i16_score,
    u8* __restrict__ uv_mode, int* __restrict__ sse4,
    int B, int mb_h, int mb_w, int H, int W, const int* w_clamp)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_mb = mb_h * mb_w;
    if (t >= B * n_mb) return;
    int b = t / n_mb;
    int mb = t % n_mb;
    int mby = mb / mb_w, mbx = mb % mb_w;
    int HH = H / 2, HW = W / 2;
    int ystride = W + 1, cstride = HW + 1;
    const i16* Y = bY + (size_t)b * (H + 1) * (W + 1);
    const i16* U = bU + (size_t)b * (HH + 1) * (HW + 1);
    const i16* V = bV + (size_t)b * (HH + 1) * (HW + 1);
    int py0 = mby * 16, px0 = mbx * 16;   // content coords, == bordered idx

    // ---- i16 modes (DC/TM/V/H on the 16x16 block) ----
    i16 pred16[256];
    i64 best16c = 0; int best16 = 0;
    for (int m = 0; m < 4; ++m) {
        pred_blk_s(m, Y, ystride, py0, px0, 16, pred16);
        i64 sse = 0;
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                int src = Y[(py0 + 1 + r) * ystride + (px0 + 1 + c)];
                i64 d = src - pred16[r * 16 + c];
                sse += d * d;
            }
        i64 cost = sse * 256 + fc_i16[m] * 106;
        if (m == 0 || cost < best16c) { best16c = cost; best16 = m; }
    }
    i16_mode[t] = (u8)best16;
    i16_score[t] = best16c;

    // ---- chroma modes (U+V summed, 8x8) ----
    i16 pred8[64];
    int bestuv = 0; i64 bestuvc = 0;
    for (int m = 0; m < 4; ++m) {
        i64 sse = 0;
        for (int ci = 0; ci < 2; ++ci) {
            const i16* C = ci == 0 ? U : V;
            pred_blk_s(m, C, cstride, mby * 8, mbx * 8, 8, pred8);
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c) {
                    int src = C[(mby * 8 + 1 + r) * cstride + (mbx * 8 + 1 + c)];
                    i64 d = src - pred8[r * 8 + c];
                    sse += d * d;
                }
        }
        i64 cost = sse * 256 + fc_uv[m] * 120;
        if (m == 0 || cost < bestuvc) { bestuvc = cost; bestuv = m; }
    }
    uv_mode[t] = (u8)bestuv;

    // ---- i4 subblock SSE table (16 subblocks x 10 modes) ----
    i16 pred4[16];
    for (int sb = 0; sb < 16; ++sb) {
        int sy = sb >> 2, sx = sb & 3;
        int py = py0 + sy * 4, px = px0 + sx * 4;
        for (int m = 0; m < 10; ++m) {
            i4_pred(m, Y, ystride, py, px, py0, w_clamp[b], pred4);
            i64 sse = 0;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    int src = Y[(py + 1 + r) * ystride + (px + 1 + c)];
                    i64 d = src - pred4[r * 4 + c];
                    sse += d * d;
                }
            sse4[((size_t)t * 16 + sb) * 10 + m] = sse;
        }
    }
}

// Batched GPU PNG defilter: one thread per image walks the (already
// zlib-inflated) filtered byte stream sequentially and writes RGBA.
// Rows depend on the previous reconstructed row, so a single thread per
// image is the natural mapping; the batch of ~96 images runs in parallel.
__device__ __forceinline__ int paethd(int a, int b, int c) {
    int p = a + b - c;
    int pa = p - a; if (pa < 0) pa = -pa;
    int pb = p - b; if (pb < 0) pb = -pb;
    int pc = p - c; if (pc < 0) pc = -pc;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

extern "C" __global__ void png_defilter_kernel(
    const unsigned char* __restrict__ raw,   // (B, rows, rstride)
    unsigned char* __restrict__ out,         // (B, H, W, 4)
    int B, int rows, int rstride, int W, int bpp)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const unsigned char* r = raw + (size_t)b * rows * rstride;
    unsigned char* o = out + (size_t)b * rows * W * 4;
    int stride = W * bpp;
    for (int y = 0; y < rows; ++y) {
        const unsigned char* rp = r + (size_t)y * rstride;
        int f = rp[0];
        const unsigned char* rv = rp + 1;
        unsigned char* orow = o + (size_t)y * stride;
        unsigned char* prev = y > 0 ? o + (size_t)(y - 1) * stride : 0;
        if (f == 0) {
            if (bpp == 4) {
                for (int x = 0; x < stride; ++x) orow[x] = rv[x];
            } else {
                for (int p = 0; p < W; ++p) {
                    orow[p*3+0] = rv[p*3+0];
                    orow[p*3+1] = rv[p*3+1];
                    orow[p*3+2] = rv[p*3+2];
                }
            }
        } else if (f == 1) {
            // per-pixel structure: the byte-level x-bpp loop-carried
            // dependency gets mis-vectorised under __restrict__ on some
            // nvcc versions (observed: wrong left-neighbour for x%bpp!=0)
            for (int c = 0; c < bpp && c < stride; ++c) orow[c] = rv[c];
            for (int p = 1; p < W; ++p)
                for (int c = 0; c < bpp; ++c) {
                    int x = p * bpp + c;
                    if (x >= stride) break;
                    orow[x] = (unsigned char)(rv[x] + orow[x - bpp]);
                }
        } else if (f == 2) {
            if (y == 0) {
                for (int x = 0; x < stride; ++x) orow[x] = rv[x];
            } else {
                for (int x = 0; x < stride; ++x)
                    orow[x] = (unsigned char)(rv[x] + prev[x]);
            }
        } else if (f == 3) {
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < bpp; ++c) {
                    int x = p * bpp + c;
                    if (x >= stride) break;
                    int left = p > 0 ? orow[x - bpp] : 0;
                    int up = y > 0 ? prev[x] : 0;
                    orow[x] = (unsigned char)(rv[x] + ((left + up) >> 1));
                }
        } else {
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < bpp; ++c) {
                    int x = p * bpp + c;
                    if (x >= stride) break;
                    int left = p > 0 ? orow[x - bpp] : 0;
                    int up = y > 0 ? prev[x] : 0;
                    int ul = (y > 0 && p > 0) ? prev[x - bpp] : 0;
                    orow[x] = (unsigned char)(rv[x] + paethd(left, up, ul));
                }
        }
    }
    if (bpp == 3) {   // expand RGB -> RGBA in place (back to front)
        for (int y = rows - 1; y >= 0; --y) {
            unsigned char* row = o + (size_t)y * W * 4;
            const unsigned char* src = o + (size_t)y * stride;
            for (int p = W - 1; p >= 0; --p) {
                row[p*4+0] = src[p*3+0];
                row[p*4+1] = src[p*3+1];
                row[p*4+2] = src[p*3+2];
                row[p*4+3] = 255;
            }
        }
    }
}

// one thread per image: flag[b]=1 if any alpha byte != 255
extern "C" __global__ void alpha_scan_kernel(
    const unsigned char* __restrict__ rgba,
    unsigned char* __restrict__ flags, int B, long npix)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const unsigned char* a = rgba + (size_t)b * npix * 4 + 3;
    unsigned int f = 0;
    for (long i = 0; i < npix; ++i)
        if (a[i * 4] != 255) { f = 1; break; }
    flags[b] = f;
}

// padded-batch staging scatter: one block per image; copies the real
// content rows (top-left anchored) and zeroes the pad region.
// src rows are Wr*4 bytes (packed back to back), dst rows stride W*4.
// All row lengths are multiples of 64 bytes (Wr, W are 16-multiples), so
// int4 traffic is fully aligned.
extern "C" __global__ void pad_scatter_kernel(
    const unsigned char* __restrict__ src,   // packed images at off[i]
    unsigned char* __restrict__ dst,         // n * W*H*4 zero-padded grid
    const long long* __restrict__ off,       // per-image src byte offsets
    const int* __restrict__ w_reals,
    const int* __restrict__ h_reals,
    int W, int H)
{
    int i = blockIdx.x;
    const unsigned char* s = src + off[i];
    unsigned char* d = dst + (size_t)i * W * H * 4;
    int Wr = w_reals[i], Hr = h_reals[i];
    int chunks_real = Wr * 4 / 16;
    int chunks_row  = W * 4 / 16;
    for (int r = threadIdx.x; r < H; r += blockDim.x) {
        int4* dr = (int4*)(d + (size_t)r * W * 4);
        if (r < Hr) {
            const int4* sr = (const int4*)(s + (size_t)r * Wr * 4);
            for (int k = 0; k < chunks_real; k++) dr[k] = sr[k];
            const int4 z = make_int4(0, 0, 0, 0);
            for (int k = chunks_real; k < chunks_row; k++) dr[k] = z;
        } else {
            const int4 z = make_int4(0, 0, 0, 0);
            for (int k = 0; k < chunks_row; k++) dr[k] = z;
        }
    }
}

// GPU mode selection: exact port of the CPU select_modes_one recurrence.
// One thread drives one MB row of one image; each MB's 16 i4 decisions read
// the LEFT MB's final modes (same thread, sequential) and the TOP MB's
// final modes (previous row's thread, gated by per-MB flags — same
// pipelining contract as closed_loop_kernel, so the launcher must keep
// every block resident).
// rate-aware variant: refines the top-SSE candidate modes with an
// estimated token rate (FDCT + quantize + level costs on the bordered
// SOURCE plane, same open-loop approximation the SSE tables use).
// score = sse*256 + (header_cost + token_rate) * 11   (1/256-bit units)
extern "C" __global__ void select_modes_rdo_kernel(
    const int* __restrict__ sse4,
    const long long* __restrict__ i16_score,
    const unsigned char* __restrict__ i16_mode,
    const unsigned char* __restrict__ uv_mode,
    const long long* __restrict__ fixed_costs,
    long long penalty,
    const i16* __restrict__ bY,             // bordered source luma
    const i64* __restrict__ y1q, const i64* __restrict__ y1iq,
    const i64* __restrict__ y1b, const i64* __restrict__ y1z,
    const i64* __restrict__ y1s,
    unsigned char* __restrict__ is_i4_out,
    unsigned char* __restrict__ i4_modes_out,
    unsigned int* __restrict__ flags,
    int B, int mb_h, int mb_w, int H, int W, const int* w_clamp)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= B * mb_h) return;
    int b = blk / mb_h;
    int mby = blk % mb_h;
    int n_mb = mb_h * mb_w;
    int ystride = W + 1;
    const int* sse_b = sse4 + (size_t)b * n_mb * 160;
    const long long* score_b = i16_score + (size_t)b * n_mb;
    unsigned char* isf = is_i4_out + (size_t)b * n_mb;
    unsigned char* om = i4_modes_out + (size_t)b * n_mb * 16;
    const i16* Yb = bY + (size_t)b * (H + 1) * (W + 1);
    volatile unsigned int* fl = flags + (size_t)b * n_mb;

    i16 pred4[16];
    int res[16];
    i64 ftmp[16];
    i64 lv[16];
    int cand[3];

    for (int mbx = 0; mbx < mb_w; ++mbx) {
        int mb = mby * mb_w + mbx;
        if (mby > 0) {
            while (fl[(size_t)(mby - 1) * mb_w + mbx] == 0) __DEVICE_SPIN(64);
            __threadfence();
        }
        int py0 = mby * 16, px0 = mbx * 16;
        unsigned char* omm = om + (size_t)mb * 16;
        long long total = penalty;
        for (int y = 0; y < 4; y++) {
            int left = (mbx == 0) ? 0
                : om[(size_t)(mb - 1) * 16 + y * 4 + 3];
            for (int x = 0; x < 4; x++) {
                int top = (y == 0)
                    ? ((mby == 0) ? 0 : om[(size_t)(mb - mb_w) * 16 + 12 + x])
                    : omm[(y - 1) * 4 + x];
                const int* s = sse_b + (size_t)mb * 160 + (y * 4 + x) * 10;
                const long long* fc = fixed_costs + (top * 10 + left) * 10;
                // top-3 modes by SSE
                int c0 = -1, c1 = -1, c2 = -1;
                long long b0 = 1LL << 60, b1 = b0, b2 = b0;
                for (int m = 0; m < 10; m++) {
                    long long v = s[m];
                    if (v < b0) { b2 = b1; c2 = c1; b1 = b0; c1 = c0;
                                  b0 = v; c0 = m; }
                    else if (v < b1) { b2 = b1; c2 = c1; b1 = v; c1 = m; }
                    else if (v < b2) { b2 = v; c2 = m; }
                }
                cand[0] = c0; cand[1] = c1; cand[2] = c2;
                int py = py0 + y * 4, px = px0 + x * 4;
                long long bestr[3] = {1LL << 60, 1LL << 60, 1LL << 60};
                #pragma unroll
                for (int ci = 0; ci < 3; ci++) {
                    int m = cand[ci];
                    if (m < 0) continue;
                    i4_pred(m, Yb, ystride, py, px, py0, w_clamp[b], pred4);
                    i64 sse = 0;
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++) {
                            int v = (int)Yb[(py + 1 + r) * ystride
                                            + (px + 1 + c)]
                                  - (int)pred4[r * 4 + c];
                            res[r * 4 + c] = v;
                            sse += (i64)v * v;
                        }
                    for (int t2 = 0; t2 < 16; t2++) ftmp[t2] = res[t2];
                    fdct(res, ftmp);
                    for (int t2 = 0; t2 < 16; t2++) lv[t2] = 0;
                    quantize(ftmp, y1q, y1iq, y1b, y1z, y1s, lv, 0);
                    // token rate: level costs + sign bits, ctx=1 approx
                    i64 rate = 0;
                    for (int n2 = 0; n2 < 16; n2++) {
                        i64 l2 = lv[n2];
                        if (l2 == 0) continue;
                        int al = (int)(l2 < 0 ? -l2 : l2);
                        rate += tre_levelcost(
                            tre_tbl(3, TRE_BANDS[n2], 1), al) + 256;
                    }
                    bestr[ci] = sse * 256 + (fc[m] + rate) * 11;
                }
                long long best = s[0] * 256 + fc[0] * 11;  // fallback c0
                int best_m = 0;
                long long s0 = s[c0 < 0 ? 0 : c0] * 256
                             + fc[c0 < 0 ? 0 : c0] * 11;
                best = s0; best_m = c0 < 0 ? 0 : c0;
                for (int ci = 0; ci < 3; ci++)
                    if (bestr[ci] < best) { best = bestr[ci];
                                            best_m = cand[ci]; }
                omm[y * 4 + x] = (unsigned char)best_m;
                total += best;
                left = best_m;
            }
        }
        if (total < score_b[mb]) {
            isf[mb] = 1;
        } else {
            isf[mb] = 0;
            unsigned char m16 = i16_mode[mb];
            for (int k2 = 0; k2 < 16; k2++) omm[k2] = m16;
        }
        __threadfence();
        fl[mb] = 1;
    }
}

extern "C" __global__ void select_modes_kernel(
    const int* __restrict__ sse4,            // [n*n_mb * 160]
    const long long* __restrict__ i16_score, // [n*n_mb]
    const unsigned char* __restrict__ i16_mode,
    const unsigned char* __restrict__ uv_mode,   // unused here, kept for parity
    const long long* __restrict__ fixed_costs,   // [10*10*10]
    long long penalty,
    unsigned char* __restrict__ is_i4_out,   // [n*n_mb]
    unsigned char* __restrict__ i4_modes_out,// [n*n_mb*16] (context slots)
    unsigned int* __restrict__ flags,        // [n*n_mb], pre-zeroed
    int B, int mb_h, int mb_w)
{
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= B * mb_h) return;
    int b = blk / mb_h;
    int mby = blk % mb_h;
    int n_mb = mb_h * mb_w;
    const int* sse_b = sse4 + (size_t)b * n_mb * 160;
    const long long* score_b = i16_score + (size_t)b * n_mb;
    const unsigned char* i16m_b = i16_mode + (size_t)b * n_mb;
    unsigned char* isf = is_i4_out + (size_t)b * n_mb;
    unsigned char* om = i4_modes_out + (size_t)b * n_mb * 16;
    volatile unsigned int* fl = flags + (size_t)b * n_mb;

    for (int mbx = 0; mbx < mb_w; ++mbx) {
        int mb = mby * mb_w + mbx;
        if (mby > 0) {
            while (fl[(size_t)(mby - 1) * mb_w + mbx] == 0) __DEVICE_SPIN(64);
            __threadfence();
        }
        unsigned char* omm = om + (size_t)mb * 16;
        long long total = penalty;
        for (int y = 0; y < 4; y++) {
            int left = (mbx == 0) ? 0
                : om[(size_t)(mb - 1) * 16 + y * 4 + 3];
            for (int x = 0; x < 4; x++) {
                int top = (y == 0)
                    ? ((mby == 0) ? 0 : om[(size_t)(mb - mb_w) * 16 + 12 + x])
                    : omm[(y - 1) * 4 + x];
                const int* s = sse_b + (size_t)mb * 160 + (y * 4 + x) * 10;
                long long best = 1LL << 60;
                int best_m = 0;
                const long long* fc = fixed_costs + (top * 10 + left) * 10;
                for (int m = 0; m < 10; m++) {
                    long long sc = (long long)s[m] * 256
                        + fc[m] * 11;
                    if (sc < best) { best = sc; best_m = m; }
                }
                omm[y * 4 + x] = (unsigned char)best_m;
                total += best;
                left = best_m;
            }
        }
        if (total < score_b[mb]) {
            isf[mb] = 1;               // keep the i4 modes just written
        } else {
            isf[mb] = 0;
            unsigned char m16 = i16m_b[mb];
            #pragma unroll
            for (int k = 0; k < 16; k++) omm[k] = m16;
        }
        __threadfence();
        fl[mb] = 1;
    }
}

// per-image SSE between the source YUV planes and the closed-loop
// reconstruction (= exactly what the decoder will rebuild from the
// bitstream). One block per image; bounds follow the REAL image dims so
// padded batches only count real pixels. Output: sse_out[3b+{0,1,2}] =
// {Y, U, V} squared-error sums (u64).
extern "C" __global__ void recon_sse_kernel(
    const i16* __restrict__ Y, const i16* __restrict__ U,
    const i16* __restrict__ V,
    const i16* __restrict__ rY, const i16* __restrict__ rU,
    const i16* __restrict__ rV,
    unsigned long long* __restrict__ sse_out,
    const int* __restrict__ w_reals, const int* __restrict__ h_reals,
    int B, int H, int W)
{
    int b = blockIdx.x;
    if (b >= B) return;
    int Wr = w_reals[b], Hr = h_reals[b];
    int HW = W / 2, HH = H / 2;
    int HWr = Wr / 2, HHr = Hr / 2;
    const i16* Yb = Y + (size_t)b * H * W;
    const i16* Ub = U + (size_t)b * HH * HW;
    const i16* Vb_ = V + (size_t)b * HH * HW;
    const i16* rYb = rY + (size_t)b * (H + 1) * (W + 1);
    const i16* rUb = rU + (size_t)b * (HH + 1) * (HW + 1);
    const i16* rVb = rV + (size_t)b * (HH + 1) * (HW + 1);
    unsigned long long sy = 0, su = 0, sv = 0;
    // source and recon planes are laid out at PAD strides (W / HW); only
    // the real top-left region [0..Hr) x [0..Wr) is compared
    size_t ny = (size_t)Hr * Wr;
    for (size_t k = threadIdx.x; k < ny; k += blockDim.x) {
        size_t yy = k / Wr, xx = k % Wr;
        int d = (int)Yb[yy * W + xx]
              - (int)rYb[(yy + 1) * (W + 1) + (xx + 1)];
        sy += (unsigned long long)((long long)d * d);
    }
    size_t nuv = (size_t)HHr * HWr;
    for (size_t k = threadIdx.x; k < nuv; k += blockDim.x) {
        size_t yy = k / HWr, xx = k % HWr;
        int du = (int)Ub[yy * HW + xx]
               - (int)rUb[(yy + 1) * (HW + 1) + (xx + 1)];
        int dv = (int)Vb_[yy * HW + xx]
               - (int)rVb[(yy + 1) * (HW + 1) + (xx + 1)];
        su += (unsigned long long)((long long)du * du);
        sv += (unsigned long long)((long long)dv * dv);
    }
    __shared__ unsigned long long red[3 * 256];
    red[threadIdx.x] = sy;
    red[256 + threadIdx.x] = su;
    red[512 + threadIdx.x] = sv;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
            red[256 + threadIdx.x] += red[256 + threadIdx.x + s];
            red[512 + threadIdx.x] += red[512 + threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sse_out[(size_t)b * 3 + 0] = red[0];
        sse_out[(size_t)b * 3 + 1] = red[256];
        sse_out[(size_t)b * 3 + 2] = red[512];
    }
}
