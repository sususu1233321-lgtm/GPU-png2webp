// entropy.cpp -- VP8 entropy coding (partition0 + token partition + boolean
// range coder) as a standalone CPU DLL. Fused single pass: the (prob,bit)
// ops feed the range coder directly, no intermediate ops buffer.
//
// Bit-exact port of gpuwebp/vp8_encode.py write_partition0 /
// write_token_partition and gpuwebp/bool_coder.py bool_encode (which are
// 1:1 ports of libwebp, BSD).
//
// Thread-safe: no mutable global state. Called concurrently from the
// finish worker threads (ctypes releases the GIL).
//
// Build (see build_entropy.bat): cl /O2 /LD /MD entropy.cpp /Fe:entropy.dll

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>

#include "entropy_tables.inc"

// ------------------------------------------------------------ bool coder

struct BC {
    int range_;
    long long value;
    int run;
    int nb_bits;
    int pos;
    unsigned char* out;
};

static void bc_init(BC* bc, unsigned char* out) {
    bc->range_ = 254;
    bc->value = 0;
    bc->run = 0;
    bc->nb_bits = -8;
    bc->pos = 0;
    bc->out = out;
}

static void bc_flush(BC* bc) {
    int s = 8 + bc->nb_bits;
    long long bits = bc->value >> s;
    bc->value = bc->value - (bits << s);
    bc->nb_bits -= 8;
    if ((bits & 0xff) != 0xff) {
        if ((bits & 0x100) != 0) {                 // carry
            if (bc->pos > 0) bc->out[bc->pos - 1] += 1;
        }
        if (bc->run > 0) {
            int fill = ((bits & 0x100) == 0) ? 0xff : 0;
            for (int i = 0; i < bc->run; i++) bc->out[bc->pos++] = (unsigned char)fill;
            bc->run = 0;
        }
        bc->out[bc->pos++] = (unsigned char)(bits & 0xff);
    } else {
        bc->run += 1;
    }
}

static inline void bc_put(BC* bc, int bit, int prob) {
    int split = (bc->range_ * prob) >> 8;
    if (bit) {
        bc->value += split + 1;
        bc->range_ -= split + 1;
    } else {
        bc->range_ = split;
    }
    if (bc->range_ < 127) {
        int shift = K_NORM[bc->range_];
        bc->range_ = K_NEW_RANGE[bc->range_];
        bc->value <<= shift;
        bc->nb_bits += shift;
        if (bc->nb_bits > 0) bc_flush(bc);
    }
}

static void bc_finish(BC* bc) {
    int n_zero = 9 - bc->nb_bits;
    for (int i = 0; i < n_zero; i++) {
        int split = bc->range_ >> 1;
        bc->range_ = split;
        if (bc->range_ < 127) {
            bc->range_ = K_NEW_RANGE[bc->range_];
            bc->value <<= 1;
            bc->nb_bits += 1;
            if (bc->nb_bits > 0) bc_flush(bc);
        }
    }
    bc->nb_bits = 0;
    bc_flush(bc);
}

// uniform bits, MSB first (port of _put_bits with prob 128)
static inline void bc_put_bits(BC* bc, int value, int n_bits) {
    int mask = 1 << (n_bits - 1);
    while (mask != 0) {
        bc_put(bc, (value & mask) != 0, 128);
        mask >>= 1;
    }
}

static inline void bc_put_signed(BC* bc, int value, int n_bits) {
    if (value == 0) {
        bc_put(bc, 0, 128);
        return;
    }
    bc_put(bc, 1, 128);
    if (value < 0)
        bc_put_bits(bc, ((-value) << 1) | 1, n_bits + 1);
    else
        bc_put_bits(bc, value << 1, n_bits + 1);
}

// ------------------------------------------------------------ partition 0

static inline void put_i4(BC* bc, int mode, int top, int left) {
    const unsigned char* p = KF_BMODE_PROBA + (top * 10 + left) * 9;
    bc_put(bc, mode != 0, p[0]);
    if (mode == 0) return;
    bc_put(bc, mode != 1, p[1]);
    if (mode == 1) return;
    bc_put(bc, mode != 2, p[2]);
    if (mode == 2) return;
    int ge6 = mode >= 6;
    bc_put(bc, ge6, p[3]);
    if (ge6 == 0) {
        bc_put(bc, mode != 3, p[4]);
        if (mode == 3) return;
        bc_put(bc, mode != 4, p[5]);
        return;
    }
    bc_put(bc, mode != 6, p[6]);
    if (mode == 6) return;
    bc_put(bc, mode != 7, p[7]);
    if (mode == 7) return;
    bc_put(bc, mode != 8, p[8]);
}

static int write_partition0(BC* bc, int mb_w, int mb_h, int base_quant,
                            int dq_uv_dc, int dq_uv_ac, int filter_level,
                            int num_parts_log2, int use_skip, int skip_proba,
                            const unsigned char* skip,
                            const unsigned char* is_i4,
                            const unsigned char* i16_mode,
                            const unsigned char* uv_mode,
                            const unsigned char* i4_modes,
                            const unsigned char* upd_tab = nullptr) {
    bc_put(bc, 0, 128);                    // colorspace
    bc_put(bc, 0, 128);                    // clamp type
    bc_put(bc, 0, 128);                    // segmentation disabled
    bc_put(bc, 1, 128);                    // simple loop filter
    bc_put_bits(bc, filter_level, 6);
    bc_put_bits(bc, 0, 3);                 // sharpness
    bc_put(bc, 0, 128);                    // no lf deltas
    bc_put_bits(bc, num_parts_log2, 2);
    bc_put_bits(bc, base_quant, 7);
    bc_put_signed(bc, 0, 4);               // dq_y1_dc
    bc_put_signed(bc, 0, 4);               // dq_y2_dc
    bc_put_signed(bc, 0, 4);               // dq_y2_ac
    bc_put_signed(bc, dq_uv_dc, 4);
    bc_put_signed(bc, dq_uv_ac, 4);
    bc_put(bc, 0, 128);                    // refresh_entropy_probs = 0
    for (int i = 0; i < 4 * 8 * 3 * 11; i++) {
        int upd = upd_tab ? upd_tab[i] : 0;
        bc_put(bc, upd != 0, COEFFS_UPDATE_PROBA[i]);
        if (upd)
            bc_put_bits(bc, upd, 8);
    }
    if (use_skip) {
        bc_put(bc, 1, 128);
        bc_put_bits(bc, skip_proba, 8);
    } else {
        bc_put(bc, 0, 128);
    }

    for (int mby = 0; mby < mb_h; mby++) {
        for (int mbx = 0; mbx < mb_w; mbx++) {
            int mb = mby * mb_w + mbx;
            if (use_skip)
                bc_put(bc, skip[mb] != 0, skip_proba);
            if (is_i4[mb]) {
                bc_put(bc, 0, 145);
                for (int y = 0; y < 4; y++) {
                    int left = (mbx == 0) ? 0 : i4_modes[(mb - 1) * 16 + y * 4 + 3];
                    for (int x = 0; x < 4; x++) {
                        int top = (y == 0)
                            ? ((mby == 0) ? 0 : i4_modes[(mb - mb_w) * 16 + 12 + x])
                            : i4_modes[mb * 16 + (y - 1) * 4 + x];
                        int mode = i4_modes[mb * 16 + y * 4 + x];
                        put_i4(bc, mode, top, left);
                        left = mode;
                    }
                }
            } else {
                bc_put(bc, 1, 145);
                int mode = i16_mode[mb];       // {DC=0, TM=1, V=2, H=3}
                int b1 = (mode == 1 || mode == 3);   // TM or H
                bc_put(bc, b1, 156);
                if (b1)
                    bc_put(bc, mode == 1, 128);       // TM or H
                else
                    bc_put(bc, mode == 2, 163);       // V or DC
            }
            int umode = uv_mode[mb];
            bc_put(bc, umode != 0, 142);
            if (umode != 0) {
                bc_put(bc, umode != 2, 114);          // V or {TM,H}
                if (umode != 2)
                    bc_put(bc, umode == 1, 183);      // TM or H
            }
        }
    }
    bc_finish(bc);
    return bc->pos;
}

// ------------------------------------------------------------ tokens

// per-frame adaptation: stats[slot*11 + node][bit] counts tree decisions so
// the token probabilities can be fitted to this frame (same tokens, same
// decoded pixels -- only the entropy model changes)
static __declspec(thread) unsigned (*g_stats)[2] = nullptr;

static void emit_block(BC* bc, const short* levels, int first, int ctype,
                       int ctx, const unsigned char* probs) {
    int last = -1;
    for (int n = first; n < 16; n++)
        if (levels[n] != 0) last = n;
    int band = BANDS[first];
    int base = ((ctype * 8 + band) * 3 + ctx) * 11;
    if (g_stats) g_stats[base][(last >= 0) ? 1 : 0]++;
        bc_put(bc, last >= 0, probs[base]);
    if (last < 0) return;
    int n = first;
    while (n < 16) {
        int c = levels[n];
        n += 1;
        int v = c >= 0 ? c : -c;
        int sign = c < 0;
        band = (n < 16) ? BANDS[n] : 0;
        if (g_stats) g_stats[base + 1][(v != 0) ? 1 : 0]++;
        bc_put(bc, v != 0, probs[base + 1]);
        if (v == 0) {
            base = ((ctype * 8 + band) * 3 + 0) * 11;
            continue;
        }
        if (g_stats) g_stats[base + 2][(v > 1) ? 1 : 0]++;
        bc_put(bc, v > 1, probs[base + 2]);
        if (v <= 1) {
            base = ((ctype * 8 + band) * 3 + 1) * 11;
        } else {
            if (g_stats) g_stats[base + 3][(v > 4) ? 1 : 0]++;
        bc_put(bc, v > 4, probs[base + 3]);
            if (v <= 4) {
                if (g_stats) g_stats[base + 4][(v != 2) ? 1 : 0]++;
        bc_put(bc, v != 2, probs[base + 4]);
                if (v != 2) {
                    if (g_stats) g_stats[base + 5][(v == 4) ? 1 : 0]++;
                    bc_put(bc, v == 4, probs[base + 5]);
                }
            } else if (v <= 10) {
                if (g_stats) g_stats[base + 6][(0) ? 1 : 0]++;
        bc_put(bc, 0, probs[base + 6]);   // not >10
                if (g_stats) g_stats[base + 7][(v > 6) ? 1 : 0]++;
        bc_put(bc, v > 6, probs[base + 7]);
                if (v <= 6) {                             // category 1
                    bc_put(bc, v == 6, 159);
                } else {                                  // category 2
                    bc_put(bc, v >= 9, 165);
                    bc_put(bc, (v & 1) != 0 ? 0 : 1, 145);
                }
            } else {
                if (g_stats) g_stats[base + 6][(1) ? 1 : 0]++;
        bc_put(bc, 1, probs[base + 6]);   // >10
                int residue = v - 3;
                if (residue < 16) {                       // category 3
                    if (g_stats) g_stats[base + 8][(0) ? 1 : 0]++;
        bc_put(bc, 0, probs[base + 8]);
                    if (g_stats) g_stats[base + 9][(0) ? 1 : 0]++;
        bc_put(bc, 0, probs[base + 9]);
                    residue -= 8;
                    int mask = 4;
                    for (int k = 0; k < 3; k++) {
                        bc_put(bc, (residue & mask) != 0, CAT3[k]);
                        mask >>= 1;
                    }
                } else if (residue < 32) {                // category 4
                    if (g_stats) g_stats[base + 8][(0) ? 1 : 0]++;
        bc_put(bc, 0, probs[base + 8]);
                    if (g_stats) g_stats[base + 9][(1) ? 1 : 0]++;
        bc_put(bc, 1, probs[base + 9]);
                    residue -= 16;
                    int mask = 8;
                    for (int k = 0; k < 4; k++) {
                        bc_put(bc, (residue & mask) != 0, CAT4[k]);
                        mask >>= 1;
                    }
                } else if (residue < 64) {                // category 5
                    if (g_stats) g_stats[base + 8][(1) ? 1 : 0]++;
        bc_put(bc, 1, probs[base + 8]);
                    if (g_stats) g_stats[base + 10][(0) ? 1 : 0]++;
        bc_put(bc, 0, probs[base + 10]);
                    residue -= 32;
                    int mask = 16;
                    for (int k = 0; k < 5; k++) {
                        bc_put(bc, (residue & mask) != 0, CAT5[k]);
                        mask >>= 1;
                    }
                } else {                                  // category 6
                    if (g_stats) g_stats[base + 8][(1) ? 1 : 0]++;
        bc_put(bc, 1, probs[base + 8]);
                    if (g_stats) g_stats[base + 10][(1) ? 1 : 0]++;
        bc_put(bc, 1, probs[base + 10]);
                    residue -= 64;
                    int mask = 1 << 10;
                    for (int k = 0; k < 11; k++) {
                        bc_put(bc, (residue & mask) != 0, CAT6[k]);
                        mask >>= 1;
                    }
                }
            }
            base = ((ctype * 8 + band) * 3 + 2) * 11;
        }
        bc_put(bc, sign, 128);
        if (n == 16) return;
        if (g_stats) g_stats[base][(n <= last) ? 1 : 0]++;
        bc_put(bc, n <= last, probs[base]);
        if (n > last) return;
    }
}

static int write_token_partition(BC* bc, int mb_w, int mb_h,
                                 int use_skip, const unsigned char* skip,
                                 const unsigned char* is_i4,
                                 const short* y_dc, const short* y_ac,
                                 const short* uv,
                                 const unsigned char* probs) {
    std::vector<int> top_nz((size_t)mb_w * 9, 0);
    int left_nz[9];
    for (int mby = 0; mby < mb_h; mby++) {
        for (int k = 0; k < 9; k++) left_nz[k] = 0;
        for (int mbx = 0; mbx < mb_w; mbx++) {
            int mb = mby * mb_w + mbx;
            int emit = (use_skip == 0) || (skip[mb] == 0);
            if (!is_i4[mb]) {                       // y2 (i16 only)
                int nzdc = 0;
                for (int n = 0; n < 16; n++)
                    if (y_dc[(size_t)mb * 16 + n] != 0) { nzdc = 1; break; }
                if (emit) {
                    int ctx = top_nz[(size_t)mbx * 9 + 8] + left_nz[8];
                    emit_block(bc, y_dc + (size_t)mb * 16, 0, 1, ctx, probs);
                }
                top_nz[(size_t)mbx * 9 + 8] = nzdc;
                left_nz[8] = nzdc;
            }
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    int blk = x + y * 4;
                    int nzb = 0;
                    for (int n = 0; n < 16; n++)
                        if (y_ac[(size_t)mb * 256 + blk * 16 + n] != 0) { nzb = 1; break; }
                    if (emit) {
                        int ctx = top_nz[(size_t)mbx * 9 + x] + left_nz[y];
                        if (is_i4[mb])
                            emit_block(bc, y_ac + (size_t)mb * 256 + blk * 16, 0, 3, ctx, probs);
                        else
                            // i16: AC blocks start at position 1 (slot 0 is
                            // the DC, coded in the WHT block); ctype 0
                            emit_block(bc, y_ac + (size_t)mb * 256 + blk * 16, 1, 0, ctx, probs);
                    }
                    top_nz[(size_t)mbx * 9 + x] = nzb;
                    left_nz[y] = nzb;
                }
            }
            for (int n = 0; n < 8; n++) {
                int ch = n >> 2;
                int x = n & 1;
                int y = (n >> 1) & 1;
                int slot = 4 + ch * 2 + x;
                int lslot = 4 + ch * 2 + y;
                int nzb = 0;
                for (int k = 0; k < 16; k++)
                    if (uv[(size_t)mb * 128 + n * 16 + k] != 0) { nzb = 1; break; }
                if (emit) {
                    int ctx = top_nz[(size_t)mbx * 9 + slot] + left_nz[lslot];
                    emit_block(bc, uv + (size_t)mb * 128 + n * 16, 0, 2, ctx, probs);
                }
                top_nz[(size_t)mbx * 9 + slot] = nzb;
                left_nz[lslot] = nzb;
            }
        }
    }
    bc_finish(bc);
    return bc->pos;
}

// ------------------------------------------------------------ single entry

extern "C" __declspec(dllexport)
int encode_vp8_streams(
    int mb_w, int mb_h, int bq, int filter_level,
    const unsigned char* skip,          // n_mb, 0/1
    const unsigned char* is_i4,         // n_mb
    const unsigned char* i16_mode,      // n_mb
    const unsigned char* uv_mode,       // n_mb
    const unsigned char* i4_modes,      // n_mb*16
    const short* y_dc,                  // n_mb*16
    const short* y_ac,                  // n_mb*256
    const short* uv_lv,                 // n_mb*128
    unsigned char* out0, int cap0, int* n0,
    unsigned char* out1, int cap1, int* n1)
{
    if (mb_w <= 0 || mb_h <= 0) return -1;
    int n_mb = mb_w * mb_h;

    // skip probability (matches _assemble)
    int n_nonskip = 0;
    for (int i = 0; i < n_mb; i++)
        if (skip[i] == 0) n_nonskip++;
    int skip_proba = n_nonskip * 255 / n_mb;
    int use_skip = skip_proba < 250;

    BC bc;
    bc_init(&bc, out0);
    int p0 = write_partition0(&bc, mb_w, mb_h, bq, -2, 0, filter_level, 0,
                              use_skip, skip_proba, skip, is_i4, i16_mode,
                              uv_mode, i4_modes);
    if (p0 > cap0) return -10;
    *n0 = p0;

    bc_init(&bc, out1);
    int p1 = write_token_partition(&bc, mb_w, mb_h, use_skip, skip, is_i4,
                                   y_dc, y_ac, uv_lv, COEFFS_PROBA0);
    if (p1 > cap1) return -11;
    *n1 = p1;
    return 0;
}

// ------------------------------------------------------------ batch entry
// One call per GPU batch: encodes n images concurrently on an internal
// thread pool (no GIL, no per-image Python glue). Each image's partition0 +
// token partition is written into its own out buffer (p0 first, then p1).

struct BatchArgs {
    int mb_w, mb_h;
    int bq, fl;
    // per-image input pointers
    const unsigned char* const* skip;
    const unsigned char* const* is_i4;
    const unsigned char* const* i16_mode;
    const unsigned char* const* uv_mode;
    const unsigned char* const* i4_modes;
    const short* const* y_dc;
    const short* const* y_ac;
    const short* const* uv_lv;
    unsigned char* const* out;         // n buffers, each cap bytes
    int cap;
    int* out_len;                      // 2n: [i]=p0 len, [n+i]=total len
    int n;
};

// decide per-slot prob updates from the frame's own decision statistics.
// A slot is updated when the fitted probability saves more bits than the
// ~10-bit header cost of carrying it.
// The coefficient table has 4*8*3*11 = 1056 entries; each entry is one
// (type, band, ctx, tree-node) probability and is updated independently
// (flag + 8-bit literal in the frame header).
static int build_upd_tab(const unsigned (*stats)[2],
                         unsigned char* upd_tab) {
    int n_upd = 0;
    for (int e = 0; e < 4 * 8 * 3 * 11; e++) {
        upd_tab[e] = 0;
        unsigned n0 = stats[e][0], n1 = stats[e][1];
        unsigned t = n0 + n1;
        if (t < 24) continue;
        int oldp = COEFFS_PROBA0[e];
        // table prob = P(bit==0): the bool coder's bit=0 interval is
        // [0, split] with split = range*prob>>8
        int newp = (int)((n0 * 256 + t / 2) / t);
        if (newp < 1) newp = 1;
        if (newp > 255) newp = 255;
        if (newp == oldp) continue;
        double ho = 0, hn = 0;
        for (int p = 0; p < 2; p++) {
            double po = (p ? oldp : 256 - oldp) / 256.0;
            double pn = (p ? newp : 256 - newp) / 256.0;
            unsigned cnt = p ? n1 : n0;
            if (cnt) {
                ho -= cnt * log2(po);
                hn -= cnt * log2(pn);
            }
        }
        if (ho - hn > 14.0) {          // > flag + 8-bit literal + slack
            upd_tab[e] = (unsigned char)newp;
            n_upd++;
        }
    }
    return n_upd;
}

static int encode_one(const BatchArgs* A, int i) {
    int n_mb = A->mb_w * A->mb_h;
    int n_nonskip = 0;
    for (int k = 0; k < n_mb; k++)
        if (A->skip[i][k] == 0) n_nonskip++;
    int skip_proba = n_nonskip * 255 / n_mb;
    int use_skip = skip_proba < 250;

    BC bc;
    unsigned char* o = A->out[i];
    // pass A: default tables, collect this frame's decision statistics
    static __declspec(thread) unsigned (*stats)[2] = nullptr;
    if (!stats)
        stats = (unsigned (*)[2])malloc(sizeof(unsigned) * 2 * 1056);
    memset(stats, 0, sizeof(unsigned) * 2 * 1056);
    if (!getenv("ENTROPY_NOADAPT"))
        g_stats = stats;   // only collect when adaptation may run
    bc_init(&bc, o);
    int p0 = write_partition0(&bc, A->mb_w, A->mb_h, A->bq, -2, 0, A->fl, 0,
                              use_skip, skip_proba, A->skip[i], A->is_i4[i],
                              A->i16_mode[i], A->uv_mode[i], A->i4_modes[i]);
    if (p0 > A->cap - 4096) {
        A->out_len[i] = -1; A->out_len[A->n + i] = -1;
        return -10;
    }
    bc_init(&bc, o + p0);      // token partition = independent bool stream
    int p1 = write_token_partition(&bc, A->mb_w, A->mb_h, use_skip, A->skip[i],
                                   A->is_i4[i], A->y_dc[i], A->y_ac[i],
                                   A->uv_lv[i], COEFFS_PROBA0);
    g_stats = nullptr;
    if (p0 + p1 > A->cap) {
        A->out_len[i] = -1; A->out_len[A->n + i] = -1;
        return -11;
    }
    int total_def = p0 + p1;

    // pass B: fitted probabilities (identical tokens -> identical decoded
    // pixels; only the entropy model and the header update table change)
    unsigned char* upd_tab = (unsigned char*)malloc(1056);
    unsigned char* probs2 = (unsigned char*)malloc(1056);
    unsigned char* alt = (unsigned char*)malloc(A->cap);
    int rc = -1;
    if (upd_tab && probs2 && alt) {
        int n_upd = getenv("ENTROPY_NOADAPT") ? 0
                    : build_upd_tab(stats, upd_tab);
        if (n_upd > 0) {
            memcpy(probs2, COEFFS_PROBA0, 1056);
            for (int j = 0; j < 1056; j++)
                if (upd_tab[j])
                    probs2[j] = upd_tab[j];
            BC bc2;
            bc_init(&bc2, alt);
            int q0 = write_partition0(&bc2, A->mb_w, A->mb_h, A->bq, -2, 0,
                                      A->fl, 0, use_skip, skip_proba,
                                      A->skip[i], A->is_i4[i],
                                      A->i16_mode[i], A->uv_mode[i],
                                      A->i4_modes[i], upd_tab);
            int q1 = 0;
            if (q0 <= A->cap - 64) {
                bc_init(&bc2, alt + q0);
                q1 = write_token_partition(&bc2, A->mb_w, A->mb_h, use_skip,
                                           A->skip[i], A->is_i4[i],
                                           A->y_dc[i], A->y_ac[i],
                                           A->uv_lv[i], probs2);
            }
            if (q0 + q1 < total_def && q0 + q1 <= A->cap) {
                memcpy(o, alt, q0 + q1);
                p0 = q0; p1 = q1;
            }
        }
    }
    free(upd_tab); free(probs2); free(alt);
    rc = 0;
    A->out_len[i] = p0;
    A->out_len[A->n + i] = p0 + p1;
    return rc;
}

extern "C" __declspec(dllexport)
int encode_vp8_batch(
    int n, int mb_w, int mb_h, int bq, int fl,
    void* const* skip, void* const* is_i4, void* const* i16_mode,
    void* const* uv_mode, void* const* i4_modes,
    void* const* y_dc, void* const* y_ac, void* const* uv_lv,
    void* const* out, int cap, int* out_len)
{
    if (n <= 0 || mb_w <= 0 || mb_h <= 0) return -1;
    BatchArgs A;
    A.mb_w = mb_w; A.mb_h = mb_h; A.bq = bq; A.fl = fl;
    A.skip = (const unsigned char* const*)skip;
    A.is_i4 = (const unsigned char* const*)is_i4;
    A.i16_mode = (const unsigned char* const*)i16_mode;
    A.uv_mode = (const unsigned char* const*)uv_mode;
    A.i4_modes = (const unsigned char* const*)i4_modes;
    A.y_dc = (const short* const*)y_dc;
    A.y_ac = (const short* const*)y_ac;
    A.uv_lv = (const short* const*)uv_lv;
    A.out = (unsigned char* const*)out;
    A.cap = cap; A.out_len = out_len; A.n = n;

    if (n == 1) return encode_one(&A, 0);

    // capped: the host pipeline also runs decode/finish/verify pools; an
    // unbounded pool here once stacked 96 threads and froze the machine
    int nthreads = 4;
    nthreads = n < nthreads ? n : nthreads;
    if (nthreads <= 1) {
        for (int i = 0; i < n; i++)
            if (encode_one(&A, i) != 0) return -20;
        return 0;
    }
    std::vector<std::thread> pool;
    std::atomic<int> next(0);
    std::atomic<int> failed(0);
    auto worker = [&]() {
        while (true) {
            int i = next.fetch_add(1);
            if (i >= A.n) break;
            if (encode_one(&A, i) != 0) failed.fetch_add(1);
        }
    };
    for (int t = 1; t < nthreads; t++) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    return failed.load() ? -20 : 0;
}

// ------------------------------------------------------------ benchmark

extern "C" __declspec(dllexport)
double bench_entropy(int runs,
                     int mb_w, int mb_h, int bq, int filter_level,
                     const unsigned char* skip, const unsigned char* is_i4,
                     const unsigned char* i16_mode, const unsigned char* uv_mode,
                     const unsigned char* i4_modes,
                     const short* y_dc, const short* y_ac, const short* uv_lv,
                     unsigned char* out0, int cap0, int* n0,
                     unsigned char* out1, int cap1, int* n1)
{
    encode_vp8_streams(mb_w, mb_h, bq, filter_level, skip, is_i4, i16_mode,
                       uv_mode, i4_modes, y_dc, y_ac, uv_lv,
                       out0, cap0, n0, out1, cap1, n1);   // warmup
    clock_t t0 = clock();
    for (int i = 0; i < runs; i++)
        encode_vp8_streams(mb_w, mb_h, bq, filter_level, skip, is_i4, i16_mode,
                           uv_mode, i4_modes, y_dc, y_ac, uv_lv,
                           out0, cap0, n0, out1, cap1, n1);
    return (double)(clock() - t0) / CLOCKS_PER_SEC / runs * 1000.0;
}
