// GPU PNG inflate: one thread per image stream, ported from pngdec.cpp
// inflate_raw. Two-level Huffman tables (9-bit root + sub tables, zlib
// style) keep the per-thread working set L1-resident; a single-level
// 1<<15 table (the CPU version) thrashes local memory on random probes.
//
// Output is the PNG "raw" scanline blob (filter byte + packed row bytes
// per row) laid out exactly as png_defilter_kernel consumes. adler32 of
// the zlib stream is folded into the byte loops (deferred mod-65521 in
// batches of 4096) and verified against the stream trailer: any inflate
// divergence from the encoder's data is reported per-image.
#pragma once

namespace gpuinfl {

constexpr int HT_ROOT_MAX = 9;          // root table bits
constexpr int HT_SUB_MAX = 1 << 14;     // sub-table entry budget per tree
constexpr int ADLER_BATCH = 4096;       // additions between %65521 (fits u32)

struct HTree {
    unsigned root[1 << HT_ROOT_MAX];    // entry: (len<<16)|sym, or
                                        // 0x80000000|(subbits<<24)|offset
    unsigned short sub[HT_SUB_MAX];     // entry: (len<<12)|sym
    int rootbits;
};

// Huffman tables live in a per-image slice of a GLOBAL workspace, not in
// local memory: random table probes from co-resident threads serialize on
// the local-memory banks (measured 60x slowdown at 64 streams), while
// global probes go through L2 with no conflict serialization.
constexpr size_t HT_WS_STRIDE = (3 * sizeof(HTree) + 255) & ~(size_t)255;

struct Adler {
    unsigned a, b, cnt;
};

__device__ __forceinline__ void ad_reset(Adler* s) {
    s->a = 1; s->b = 0; s->cnt = 0;
}

__device__ __forceinline__ void ad_add(Adler* s, unsigned char v) {
#ifdef GPUINFL_NO_ADLER
    (void)s; (void)v;
#else
    s->a += v;
    s->b += s->a;
    if (++s->cnt == ADLER_BATCH) {
        s->a %= 65521u; s->b %= 65521u; s->cnt = 0;
    }
#endif
}

__device__ __forceinline__ unsigned ad_final(const Adler* s) {
    return ((s->b % 65521u) << 16) | (s->a % 65521u);
}

__device__ __forceinline__ unsigned rev_bits(unsigned c, int l) {
    unsigned r = 0;
    #pragma unroll
    for (int b = 0; b < l; b++) { r = (r << 1) | (c & 1); c >>= 1; }
    return r;
}

// Build a two-level canonical Huffman table from code lengths.
// Semantics mirror pngdec.cpp build_tree: over-subscribed -> error,
// incomplete tolerated (unused slots decode as error when hit).
__device__ int ht_build(HTree* t, const unsigned char* lengths, int n) {
    int counts[16];
    #pragma unroll
    for (int l = 0; l <= 15; l++) counts[l] = 0;
    for (int i = 0; i < n; i++) counts[lengths[i]]++;
    if (counts[0] == n) return -1;
    int left = 1;
    #pragma unroll
    for (int l = 1; l <= 15; l++) {
        left <<= 1;
        left -= counts[l];
        if (left < 0) return -2;              // over-subscribed
    }
    int maxlen = 15;
    while (maxlen > 1 && counts[maxlen] == 0) maxlen--;
    int R = maxlen < HT_ROOT_MAX ? maxlen : HT_ROOT_MAX;
    t->rootbits = R;
    const unsigned rmask = (1u << R) - 1;
    for (int i = 0; i < (1 << R); i++) t->root[i] = 0;

    int nextcode[16];
    int code = 0;
    for (int l = 1; l <= 15; l++) {
        code = (code + counts[l - 1]) << 1;
        nextcode[l] = code;
    }
    unsigned codes[320];                      // max alphabet 288+32
    for (int i = 0; i < n; i++)
        codes[i] = lengths[i] ? (unsigned)nextcode[lengths[i]]++ : 0u;

    // direct codes (len <= R)
    for (int i = 0; i < n; i++) {
        int l = lengths[i];
        if (!l || l > R) continue;
        unsigned rc = rev_bits(codes[i], l);
        unsigned e = ((unsigned)l << 16) | (unsigned)i;
        for (unsigned k = 0; k < (1u << (R - l)); k++)
            t->root[rc + (k << l)] = e;
    }
    // long codes: group by root prefix
    int pmax[1 << HT_ROOT_MAX];
    int poff[1 << HT_ROOT_MAX];
    for (int i = 0; i < (1 << R); i++) pmax[i] = 0;
    for (int i = 0; i < n; i++) {
        int l = lengths[i];
        if (l <= R) continue;
        unsigned rc = rev_bits(codes[i], l);
        int p = (int)(rc & rmask);
        if (l > pmax[p]) pmax[p] = l;
    }
    int sub_used = 0;
    for (int p = 0; p < (1 << R); p++) {
        if (!pmax[p]) continue;
        int sb = pmax[p] - R;
        int S = 1 << sb;
        if (sub_used + S > HT_SUB_MAX) return -3;
        int off = sub_used;
        sub_used += S;
        poff[p] = off;
        for (int k = 0; k < S; k++) t->sub[off + k] = 0;
        t->root[p] = 0x80000000u | ((unsigned)sb << 24) | (unsigned)off;
    }
    for (int i = 0; i < n; i++) {
        int l = lengths[i];
        if (l <= R) continue;
        unsigned rc = rev_bits(codes[i], l);
        int p = (int)(rc & rmask);
        int sb = pmax[p] - R;
        int idx = (int)((rc >> R) & ((1u << sb) - 1));
        for (unsigned k = 0; k < (1u << (pmax[p] - l)); k++)
            t->sub[poff[p] + idx + (k << (l - R))]
                = (unsigned short)((l << 12) | i);
    }
    return 0;
}

struct InflG {
    const unsigned char* in;
    int in_len, ipos;
    unsigned long long buf;      // u64: enables 4-byte bulk input loads
    int bitcnt;
};

__device__ __forceinline__ int fill_bits(InflG* s, int need) {
    while (s->bitcnt < need) {
        if (s->ipos + 4 <= s->in_len) {
            // 4 independent byte loads (any alignment); their latencies
            // overlap, so the serial chain pays ~1 load instead of 4
            const unsigned char* p = s->in + s->ipos;
            unsigned v = (unsigned)p[0] | ((unsigned)p[1] << 8)
                       | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
            s->buf |= (unsigned long long)v << s->bitcnt;
            s->ipos += 4;
            s->bitcnt += 32;
        } else {
            if (s->ipos >= s->in_len) return 0;
            s->buf |= (unsigned long long)s->in[s->ipos++] << s->bitcnt;
            s->bitcnt += 8;
        }
    }
    return 1;
}

__device__ int ht_decode(InflG* s, const HTree* t) {
    fill_bits(s, t->rootbits);
    unsigned e = t->root[(unsigned)(s->buf & ((1ull << t->rootbits) - 1))];
    int len;
    unsigned sym;
    if (e & 0x80000000u) {
        int sb = (int)((e >> 24) & 15);
        int off = (int)(e & 0x3fffu);
        fill_bits(s, t->rootbits + sb);
        unsigned e2 = t->sub[off + (unsigned)((s->buf >> t->rootbits)
                                    & ((1ull << sb) - 1))];
        len = e2 >> 12;
        sym = e2 & 0xfff;
    } else {
        len = (int)(e >> 16);
        sym = e & 0xffffu;
    }
    if (len == 0 || len > s->bitcnt) return -1;
    s->buf >>= len;
    s->bitcnt -= len;
    return (int)sym;
}

__device__ signed char CLORDER[19] =
    {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
__device__ short LBASE[29] =
    {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,
     99,115,131,163,195,227,258};
__device__ unsigned char LEXT[29] =
    {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
__device__ short DBASE[30] =
    {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
     1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
__device__ unsigned char DEXT[30] =
    {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,
     12,12,13,13};

// in: zlib stream (2-byte header + deflate + BE adler32 trailer).
// out: raw bytes. ws: per-image table workspace (>= HT_WS_STRIDE).
// Returns 0 on success (adler verified), negative on error.
__device__ int gpu_inflate_one(const unsigned char* in, int in_len,
                               unsigned char* out, int ocap, int* olen,
                               unsigned char* ws) {
    if (in_len < 6) return -40;
    unsigned cmf = in[0], flg = in[1];
    if ((cmf & 0x0fu) != 8 || (cmf * 256u + flg) % 31u != 0 || (flg & 0x20u))
        return -41;                            // bad wrapper / FDICT
    unsigned want = ((unsigned)in[in_len - 4] << 24)
                  | ((unsigned)in[in_len - 3] << 16)
                  | ((unsigned)in[in_len - 2] << 8)
                  |  (unsigned)in[in_len - 1];

    InflG s;
    s.in = in + 2; s.in_len = in_len - 6; s.ipos = 0;
    s.buf = 0; s.bitcnt = 0;
    int opos = 0;
    Adler ad;
    ad_reset(&ad);

    HTree* lit = (HTree*)(ws);
    HTree* dist = (HTree*)(ws + sizeof(HTree));
    HTree* clt = (HTree*)(ws + 2 * sizeof(HTree));

    for (;;) {
        if (!fill_bits(&s, 3)) return -1;
        int final_blk = (int)(s.buf & 1); s.buf >>= 1; s.bitcnt -= 1;
        int type = (int)(s.buf & 3); s.buf >>= 2; s.bitcnt -= 2;

        if (type == 0) {                       // stored
            while (s.bitcnt >= 8) { s.ipos--; s.bitcnt -= 8; }
            s.buf = 0; s.bitcnt = 0;
            if (s.ipos + 4 > s.in_len) return -2;
            int len = s.in[s.ipos] | (s.in[s.ipos + 1] << 8);
            int nlen = s.in[s.ipos + 2] | (s.in[s.ipos + 3] << 8);
            s.ipos += 4;
            if ((len ^ 0xffff) != nlen) return -3;
            if (opos + len > ocap) return -4;
            if (s.ipos + len > s.in_len) return -5;
            for (int i = 0; i < len; i++) {
                unsigned char v = s.in[s.ipos + i];
                out[opos++] = v;
                ad_add(&ad, v);
            }
            s.ipos += len;
        } else if (type == 1 || type == 2) {
            if (type == 1) {                   // fixed tables
                unsigned char lengths[288];
                for (int i = 0; i < 144; i++) lengths[i] = 8;
                for (int i = 144; i < 256; i++) lengths[i] = 9;
                for (int i = 256; i < 280; i++) lengths[i] = 7;
                for (int i = 280; i < 288; i++) lengths[i] = 8;
                if (ht_build(lit, lengths, 288) < 0) return -6;
                unsigned char dl[30];
                for (int i = 0; i < 30; i++) dl[i] = 5;
                if (ht_build(dist, dl, 30) < 0) return -7;
            } else {                           // dynamic tables
                if (!fill_bits(&s, 14)) return -8;
                int hlit = (int)(s.buf & 0x1f) + 257; s.buf >>= 5; s.bitcnt -= 5;
                int hdist = (int)(s.buf & 0x1f) + 1;  s.buf >>= 5; s.bitcnt -= 5;
                int hclen = (int)(s.buf & 0xf) + 4;   s.buf >>= 4; s.bitcnt -= 4;
                unsigned char clens[19];
                for (int i = 0; i < 19; i++) clens[i] = 0;
                for (int i = 0; i < hclen; i++) {
                    if (!fill_bits(&s, 3)) return -9;
                    clens[CLORDER[i]] = (unsigned char)(s.buf & 7);
                    s.buf >>= 3; s.bitcnt -= 3;
                }
                if (ht_build(clt, clens, 19) < 0) return -10;
                int nlen = hlit + hdist;
                if (nlen > 320) return -11;
                unsigned char lens[320];
                int i = 0;
                while (i < nlen) {
                    int sym = ht_decode(&s, clt);
                    if (sym < 0) return -12;
                    if (sym < 16) {
                        lens[i++] = (unsigned char)sym;
                    } else if (sym == 16) {
                        if (i == 0) return -13;
                        if (!fill_bits(&s, 2)) return -14;
                        int rep = 3 + (int)(s.buf & 3); s.buf >>= 2; s.bitcnt -= 2;
                        unsigned char prev = lens[i - 1];
                        while (rep-- && i < nlen) lens[i++] = prev;
                    } else if (sym == 17) {
                        if (!fill_bits(&s, 3)) return -15;
                        int rep = 3 + (int)(s.buf & 7); s.buf >>= 3; s.bitcnt -= 3;
                        while (rep-- && i < nlen) lens[i++] = 0;
                    } else {
                        if (!fill_bits(&s, 7)) return -16;
                        int rep = 11 + (int)(s.buf & 0x7f); s.buf >>= 7; s.bitcnt -= 7;
                        while (rep-- && i < nlen) lens[i++] = 0;
                    }
                }
                if (lens[256] == 0) return -17;
                if (ht_build(lit, lens, hlit) < 0) return -18;
                if (ht_build(dist, lens + hlit, hdist) < 0) return -19;
            }
            for (;;) {
                int sym = ht_decode(&s, lit);
                if (sym < 0) return -20;
                if (sym < 256) {
                    if (opos >= ocap) return -21;
                    unsigned char v = (unsigned char)sym;
                    out[opos++] = v;
                    ad_add(&ad, v);
                } else if (sym == 256) {
                    break;
                } else {
                    sym -= 257;
                    if (sym >= 29) return -22;
                    int le = LEXT[sym];
                    if (!fill_bits(&s, le)) return -23;
                    int len = LBASE[sym] + (int)(s.buf & ((1ull << le) - 1));
                    s.buf >>= le; s.bitcnt -= le;
                    int dsym = ht_decode(&s, dist);
                    if (dsym < 0) return -24;
                    if (dsym >= 30) return -25;
                    int de = DEXT[dsym];
                    if (!fill_bits(&s, de)) return -26;
                    int d = DBASE[dsym] + (int)(s.buf & ((1ull << de) - 1));
                    s.buf >>= de; s.bitcnt -= de;
                    if (d > opos) return -27;
                    if (opos + len > ocap) return -28;
                    int rem = len;
                    while (rem > 0) {
                        int chunk = rem < d ? rem : d;
                        for (int k = 0; k < chunk; k++) {
                            unsigned char v = out[opos - d + k];
                            out[opos + k] = v;
                            ad_add(&ad, v);
                        }
                        opos += chunk;
                        rem -= chunk;
                    }
                }
            }
        } else {
            return -29;
        }
        if (final_blk) break;
    }
#ifndef GPUINFL_NO_ADLER
    if (ad_final(&ad) != want) return -30;     // adler mismatch
#endif
    *olen = opos;
    return 0;
}

// n streams in flat blobs; per-image flat offsets and expected raw length.
__global__ void inflate_png_kernel(
    const unsigned char* __restrict__ idat,
    const int* __restrict__ ioff,
    const int* __restrict__ ilen,
    unsigned char* __restrict__ fraw,
    const int* __restrict__ ooff,
    const int* __restrict__ oexp,
    int* __restrict__ err,
    unsigned char* __restrict__ ws,
    int n)
{
    // one stream per WARP, lane 0 active: 32 independent streams per
    // block would diverge at every branch and serialize the whole warp
    // (measured 50x); a solo lane runs its serial chain at full speed
    int idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if ((threadIdx.x & 31) != 0 || idx >= n) return;
    int got = 0;
    int e = gpu_inflate_one(idat + ioff[idx], ilen[idx],
                            fraw + ooff[idx], oexp[idx], &got,
                            ws + (size_t)idx * HT_WS_STRIDE);
    if (e == 0 && got != oexp[idx]) e = -50;   // raw length mismatch
    err[idx] = e;
}

static int run_inflate_batch(cudaStream_t stream,
                             const unsigned char* d_idat,
                             const int* d_ioff, const int* d_ilen,
                             unsigned char* d_fraw,
                             const int* d_ooff, const int* d_oexp,
                             int* d_err, unsigned char* d_ws, int n) {
    int threads = 256;                        // 8 warps = 8 streams/block
    int blocks = (n + 7) / 8;                 // 1 active lane per warp
    inflate_png_kernel<<<blocks, threads, 0, stream>>>(
        d_idat, d_ioff, d_ilen, d_fraw, d_ooff, d_oexp, d_err, d_ws, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}  // namespace gpuinfl
