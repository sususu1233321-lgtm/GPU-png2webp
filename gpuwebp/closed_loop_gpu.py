"""GPU closed-loop encoder kernel (one CUDA thread per image).

The closed loop is sequential per image by algorithm (each macroblock's
prediction uses the reconstructed pixels of its left/above neighbours), so
the wavefront cannot be parallelised inside one image.  Instead we exploit
batch-level parallelism: one thread runs the ENTIRE sequential loop for one
image; a batch of 32-64 images runs as 32-64 threads in a single kernel
launch.  All arithmetic is a 1:1 port of closed_loop_jit.py (int64
fixed-point) so the emitted levels are bit-identical to the CPU version.
"""
import numpy as np
import cupy as cp

_CUDA_SRC = r"""
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
    i16* __restrict__ y_dc, i16* __restrict__ y_ac, i16* __restrict__ uv_lv,
    i16* __restrict__ rY, i16* __restrict__ rU, i16* __restrict__ rV,
    unsigned int* __restrict__ flags,
    int B, int mb_h, int mb_w, int H, int W)
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
        // corner [0,0] = 127 (top border owns it, like libwebp's decoder)
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

    for (int mbx = 0; mbx < mb_w; ++mbx) {
        int mb = mby * mb_w + mbx;
        if (mby > 0) {
            // wait for the MB above AND its right neighbour: the i4
            // top-right pixels of our x==3 subblocks read the row above
            // the MB at columns px0+16..19 (i.e. MB(r-1, c+1)'s recon)
            while (myflags[(size_t)(mby - 1) * mb_w + mbx] == 0) {
                __nanosleep(64);
            }
            if (mbx + 1 < mb_w) {
                while (myflags[(size_t)(mby - 1) * mb_w + mbx + 1] == 0) {
                    __nanosleep(64);
                }
            }
            __threadfence();
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
                int nz1 = quantize(coeff + blk*16, y1q, y1iq, y1b, y1z,
                                   y1s, lv, 1);
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
                i4_pred(i4m[mb*16+sb], rYb, ystride, py, px, py0, W, pred4);
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        res[r*4+c] = (int)Yb_[(py + r)*W + px + c]
                                   - (int)pred4[r*4+c];
                fdct(res, tmp);
                for (int k = 0; k < 16; ++k) lv[k] = 0;
                int nz1 = quantize(tmp, y1q, y1iq, y1b, y1z, y1s, lv, 0);
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
    int B, int mb_h, int mb_w, int H, int W)
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
            i4_pred(m, Y, ystride, py, px, py0, W, pred4);
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
            for (int x = 0; x < bpp && x < stride; ++x) orow[x] = rv[x];
            for (int x = bpp; x < stride; ++x)
                orow[x] = (unsigned char)(rv[x] + orow[x - bpp]);
        } else if (f == 2) {
            if (y == 0) {
                for (int x = 0; x < stride; ++x) orow[x] = rv[x];
            } else {
                for (int x = 0; x < stride; ++x)
                    orow[x] = (unsigned char)(rv[x] + prev[x]);
            }
        } else if (f == 3) {
            for (int x = 0; x < stride; ++x) {
                int left = x >= bpp ? orow[x - bpp] : 0;
                int up = y > 0 ? prev[x] : 0;
                orow[x] = (unsigned char)(rv[x] + ((left + up) >> 1));
            }
        } else {
            for (int x = 0; x < stride; ++x) {
                int left = x >= bpp ? orow[x - bpp] : 0;
                int up = y > 0 ? prev[x] : 0;
                int ul = (y > 0 && x >= bpp) ? prev[x - bpp] : 0;
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

"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("closed_loop_kernel",)
        ).get_function("closed_loop_kernel")
    return _kernel


def closed_loop_batch_gpu(Yb, Ub, Vb, modes_list,
                          y1, y2, uv_m, y1deq, y2deq, uvdeq):
    """Batched GPU closed loop.  Yb (B,H,W) / Ub,Vb (B,H/2,W/2) int16 padded;
    modes_list: list of per-image mode dicts (numpy).  Returns
    (y_dc, y_ac, uv_lv) numpy int16 arrays with leading dim B."""
    B, H, W = Yb.shape
    HH, HW = H // 2, W // 2
    mb_h, mb_w = H // 16, W // 16
    n_mb = mb_h * mb_w
    dev = cp.cuda.Device()
    is_i4 = cp.asarray(np.concatenate(
        [m["is_i4"].astype(np.uint8) for m in modes_list]))
    i16m = cp.asarray(np.concatenate([m["i16_mode"] for m in modes_list]))
    uvm = cp.asarray(np.concatenate([m["uv_mode"] for m in modes_list]))
    i4m = cp.asarray(np.concatenate([m["i4_modes"].reshape(-1)
                                     for m in modes_list]))
    q = [cp.asarray(np.ascontiguousarray(a, np.int64))
         for a in (y1.q, y1.iq, y1.bias, y1.zthresh, y1.sharpen,
                   y2.q, y2.iq, y2.bias, y2.zthresh, y2.sharpen,
                   uv_m.q, uv_m.iq, uv_m.bias, uv_m.zthresh, uv_m.sharpen,
                   y1deq, y2deq, uvdeq)]
    y_dc = cp.zeros(B * n_mb * 16, cp.int16)
    y_ac = cp.empty(B * n_mb * 256, cp.int16)
    uv_lv = cp.empty(B * n_mb * 128, cp.int16)
    # recon planes MUST be zeroed: the kernel reads a few locations it never
    # writes (edge top-right clamps); with cp.empty those were stale pool
    # bytes — nondeterministic output depending on allocation history
    rY = cp.zeros(B * (H + 1) * (W + 1), cp.int16)
    rU = cp.zeros(B * (HH + 1) * (HW + 1), cp.int16)
    rV = cp.zeros(B * (HH + 1) * (HW + 1), cp.int16)
    flags = cp.zeros(B * n_mb, cp.uint32)
    kern = _get_kernel()
    kern((B * mb_h,), (1,),
         (Yb, Ub, Vb, is_i4, i16m, uvm, i4m,
          q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9],
          q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17],
          y_dc, y_ac, uv_lv, rY, rU, rV, flags,
          np.int32(B), np.int32(mb_h), np.int32(mb_w),
          np.int32(H), np.int32(W)))
    out = cp.concatenate([y_dc.reshape(B, n_mb, 16),
                          y_ac.reshape(B, n_mb, 256),
                          uv_lv.reshape(B, n_mb, 128)], axis=2)
    o = cp.asnumpy(out)                 # single D2H transfer
    return (o[:, :, :16], o[:, :, 16:272].reshape(B, n_mb, 16, 16),
            o[:, :, 272:].reshape(B, n_mb, 8, 16))


_mode_kernel = None


def _get_mode_kernel():
    global _mode_kernel
    if _mode_kernel is None:
        _mode_kernel = cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("mode_search_kernel",)
        ).get_function("mode_search_kernel")
    return _mode_kernel


def mode_search_batch_gpu(Yb, Ub, Vb, y1):
    """Fused mode search on padded planes Yb (B,H,W) etc. int16.
    Returns the raw bundle (i16_mode, i16_score, uv_mode, sse4, mb_w, mb_h)
    bit-identical to the vectorized gpu_modes_pass_batch(select=False)."""
    B, H, W = Yb.shape
    HH, HW = H // 2, W // 2
    mb_h, mb_w = H // 16, W // 16
    n_mb = mb_h * mb_w

    def borders(P):
        b, h, w = P.shape
        out = cp.zeros((b, h + 1, w + 1), cp.int16)
        out[:, :, 0] = 129
        out[:, 0, :] = 127     # top row owns the corner (libwebp semantics)
        out[:, 1:, 1:] = P
        return out

    bY, bU, bV = borders(Yb), borders(Ub), borders(Vb)
    from . import vp8_tables as _T
    fc_i16 = cp.asarray(np.array(_T.FIXED_COSTS_I16, np.int64))
    fc_uv = cp.asarray(np.array(_T.FIXED_COSTS_UV, np.int64))
    i16_mode = cp.empty(B * n_mb, cp.uint8)
    i16_score = cp.empty(B * n_mb, cp.int64)
    uv_mode = cp.empty(B * n_mb, cp.uint8)
    sse4 = cp.empty(B * n_mb * 160, cp.int32)   # 16*255^2 fits int32 easily
    n = B * n_mb
    kern = _get_mode_kernel()
    kern(((n + 127) // 128,), (128,),
         (bY, bU, bV, fc_i16, fc_uv, i16_mode, i16_score, uv_mode, sse4,
          np.int32(B), np.int32(mb_h), np.int32(mb_w),
          np.int32(H), np.int32(W)))
    return dict(i16_mode=cp.asnumpy(i16_mode), i16_score=cp.asnumpy(i16_score),
                uv_mode=cp.asnumpy(uv_mode),
                sse4=cp.asnumpy(sse4).reshape(B, n_mb, 16, 10),
                mb_w=mb_w, mb_h=mb_h)  # sse4 int32: half the D2H bytes


_defilter_kernel = None


def png_defilter_batch(raw, rows, rstride, W, bpp):
    """raw: (B, rows, rstride) uint8 filter-prefixed inflated rows.
    Returns (B, rows, W, 4) uint8 RGBA (RGB expanded with alpha=255)."""
    global _defilter_kernel
    import cupy as _cp
    if _defilter_kernel is None:
        _defilter_kernel = _cp.RawModule(
            code=_CUDA_SRC, options=("-std=c++17",),
            name_expressions=("png_defilter_kernel",)
        ).get_function("png_defilter_kernel")
    B = raw.shape[0]
    d_raw = _cp.asarray(raw)
    out = _cp.empty((B, rows, W, 4), _cp.uint8)
    _defilter_kernel(((B + 31) // 32 * 32,), (32,),
                     (d_raw, out, np.int32(B), np.int32(rows),
                      np.int32(rstride), np.int32(W), np.int32(bpp)))
    return out
