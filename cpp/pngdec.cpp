// pngdec.cpp -- self-contained PNG decoder + metadata scanner as a CPU DLL.
// Replaces Pillow decode + Python chunk parsing in the batch pipeline's hot
// path. No external dependencies (own table-driven DEFLATE/inflate).
//
// Fast path supports: 8-bit RGB / RGBA, non-interlaced. Everything else
// returns PNG_UNSUPPORTED (=1) so the caller falls back to Pillow.
//
// Thread-safe: no global mutable state.
//
// Build (see build_pngdec.bat): cl /O2 /LD /MD pngdec.cpp /Fe:pngdec.dll

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <windows.h>
#include <vector>

#define MAXBITS 15
#define PNG_UNSUPPORTED 1

// ------------------------------------------------------------ zlib (dll)

// Real zlib via dynamically loaded zlib1.dll -- several candidate paths are
// probed (python DLLs dir first). Falls back to the built-in scalar inflate.
typedef struct z_stream_s z_stream;            // opaque here
static void* (*pz_inflateInit_)(void*, const char*, int) = nullptr;
static int   (*pz_inflate)(void*, int) = nullptr;
static int   (*pz_inflateEnd)(void*) = nullptr;
static void* (*pz_inflateInit2_)(void*, int, const char*, int) = nullptr;
static bool g_zlib_tried = false;

static bool load_zlib() {
    if (g_zlib_tried) return pz_inflate != nullptr;
    g_zlib_tried = true;
    const char* candidates[] = {
        "zlib1.dll",                        // PATH / app dir
        "zlib.dll",
        nullptr
    };
    // extra candidates from env (python DLLs dir)
    const char* env = getenv("ZLIB1_DLL");
    HMODULE h = nullptr;
    if (env && env[0]) h = LoadLibraryA(env);
    for (int i = 0; !h && candidates[i]; i++)
        h = LoadLibraryA(candidates[i]);
    if (!h) return false;
    pz_inflateInit_ = (void* (*)(void*, const char*, int))
        GetProcAddress(h, "inflateInit_");
    pz_inflateInit2_ = (void* (*)(void*, int, const char*, int))
        GetProcAddress(h, "inflateInit2_");
    pz_inflate = (int (*)(void*, int))GetProcAddress(h, "inflate");
    pz_inflateEnd = (int (*)(void*))GetProcAddress(h, "inflateEnd");
    return pz_inflate && pz_inflateInit_ && pz_inflateInit2_;
}

// inflate a zlib-wrapped stream with the real zlib. Returns 0 ok.
struct ZStr {                 // matches zlib's z_stream layout (64-bit)
    const unsigned char* next_in;
    unsigned avail_in;
    unsigned long total_in;
    unsigned char* next_out;
    unsigned avail_out;
    unsigned long total_out;
    const char* msg;
    void* state;
    int (*zalloc)(void*, void*, unsigned, unsigned);
    void (*zfree)(void*, void*, void*);
    void* opaque;
    int data_type;
    unsigned long adler;
    unsigned long reserved;
};

static int zlib_inflate_to(const unsigned char* in, size_t in_len,
                           unsigned char* out, size_t out_cap,
                           size_t* out_len)
{
    if (!load_zlib()) return -100;
    ZStr s;
    memset(&s, 0, sizeof(s));
    s.next_in = in; s.avail_in = (unsigned)in_len;
    s.next_out = out; s.avail_out = (unsigned)out_cap;
    if (pz_inflateInit_(&s, "1.2.13", (int)sizeof(s)) != 0) return -101;
    int rc = pz_inflate(&s, 4 /*FINISH*/);
    pz_inflateEnd(&s);
    if (rc != 1 /*Z_STREAM_END*/) return -102;
    *out_len = s.total_out;
    return 0;
}

// ------------------------------------------------------------ inflate

struct Tree {
    std::vector<unsigned> tbl;    // 1<<bits entries: (len << 16) | symbol
    int bits;
};

struct Infl {
    const unsigned char* in;
    size_t in_len, ipos;
    unsigned buf;                 // bit buffer, LSB-first
    int bitcnt;
    unsigned char* out;
    size_t opos, ocap;
};

static int fill_bits(Infl* s, int need) {
    while (s->bitcnt < need) {
        if (s->ipos >= s->in_len) return 0;
        s->buf |= (unsigned)s->in[s->ipos++] << s->bitcnt;
        s->bitcnt += 8;
    }
    return 1;
}

// canonical Huffman. The bit buffer is LSB-first while deflate Huffman
// codes are MSB-first, so table indices use the bit-REVERSED code; a code
// of length l occupies window bits 0..l-1 -> matching indices are
// rc + k*2^l (scattered with stride 2^l), NOT a contiguous range.
static int build_tree(Tree* t, const unsigned char* lengths, int n) {
    int counts[MAXBITS + 1] = {0};
    for (int i = 0; i < n; i++) counts[lengths[i]]++;
    if (counts[0] == n) return -1;
    int left = 1;
    for (int l = 1; l <= MAXBITS; l++) {
        left <<= 1;
        left -= counts[l];
        if (left < 0) return -2;              // over-subscribed
    }
    int maxlen = MAXBITS;
    while (maxlen > 1 && counts[maxlen] == 0) maxlen--;
    t->bits = maxlen;
    t->tbl.assign((size_t)1 << maxlen, 0u);
    int nextcode[MAXBITS + 1] = {0};
    int code = 0;
    for (int l = 1; l <= MAXBITS; l++) {
        code = (code + counts[l - 1]) << 1;
        nextcode[l] = code;
    }
    for (int i = 0; i < n; i++) {
        int l = lengths[i];
        if (!l) continue;
        unsigned c = (unsigned)nextcode[l]++;
        unsigned rc = 0, cc = c;
        for (int b = 0; b < l; b++) { rc = (rc << 1) | (cc & 1); cc >>= 1; }
        int shift = maxlen - l;
        unsigned entry = ((unsigned)l << 16) | (unsigned)i;
        size_t cnt = (size_t)1 << shift;
        for (size_t k = 0; k < cnt; k++) t->tbl[(size_t)rc + (k << l)] = entry;
    }
    return maxlen;
}

static int decode_sym(Infl* s, const Tree* t) {
    if (s->bitcnt < t->bits) fill_bits(s, t->bits);   // best effort at EOF
    unsigned e = t->tbl[s->buf & ((1u << t->bits) - 1)];
    int len = (int)(e >> 16);
    if (len == 0 || len > s->bitcnt) return -1;
    s->buf >>= len;
    s->bitcnt -= len;
    return (int)(e & 0xffff);
}

static const unsigned char CLORDER[19] =
    {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

static int inflate_raw(const unsigned char* in, size_t in_len,
                       unsigned char* out, size_t ocap, size_t* out_len) {
    Infl s;
    s.in = in; s.in_len = in_len; s.ipos = 0;
    s.buf = 0; s.bitcnt = 0;
    s.out = out; s.opos = 0; s.ocap = ocap;

    Tree lit, dist;

    for (;;) {
        if (!fill_bits(&s, 3)) return -1;
        int final_blk = (int)(s.buf & 1); s.buf >>= 1; s.bitcnt -= 1;
        int type = (int)(s.buf & 3); s.buf >>= 2; s.bitcnt -= 2;

        if (type == 0) {                       // stored
            // rewind whole lookahead bytes held in the bit buffer, then
            // drop to the byte boundary
            while (s.bitcnt >= 8) { s.ipos--; s.bitcnt -= 8; }
            s.buf = 0; s.bitcnt = 0;
            if (s.ipos + 4 > s.in_len) return -2;
            size_t len = in[s.ipos] | ((size_t)in[s.ipos + 1] << 8);
            size_t nlen = in[s.ipos + 2] | ((size_t)in[s.ipos + 3] << 8);
            s.ipos += 4;
            if ((len ^ 0xffff) != nlen) return -3;
            if (s.opos + len > ocap) return -4;
            if (s.ipos + len > s.in_len) return -5;
            memcpy(out + s.opos, in + s.ipos, len);
            s.ipos += len; s.opos += len;
        } else if (type == 1 || type == 2) {
            if (type == 1) {                   // fixed tables
                unsigned char lengths[288];
                for (int i = 0; i < 144; i++) lengths[i] = 8;
                for (int i = 144; i < 256; i++) lengths[i] = 9;
                for (int i = 256; i < 280; i++) lengths[i] = 7;
                for (int i = 280; i < 288; i++) lengths[i] = 8;
                if (build_tree(&lit, lengths, 288) < 0) return -6;
                unsigned char dl[30];
                for (int i = 0; i < 30; i++) dl[i] = 5;
                if (build_tree(&dist, dl, 30) < 0) return -7;
            } else {                           // dynamic tables
                if (!fill_bits(&s, 14)) return -8;
                int hlit = (int)(s.buf & 0x1f) + 257; s.buf >>= 5; s.bitcnt -= 5;
                int hdist = (int)(s.buf & 0x1f) + 1;  s.buf >>= 5; s.bitcnt -= 5;
                int hclen = (int)(s.buf & 0xf) + 4;   s.buf >>= 4; s.bitcnt -= 4;
                unsigned char clens[19] = {0};
                for (int i = 0; i < hclen; i++) {
                    if (!fill_bits(&s, 3)) return -9;
                    clens[CLORDER[i]] = (unsigned char)(s.buf & 7);
                    s.buf >>= 3; s.bitcnt -= 3;
                }
                Tree clt;
                if (build_tree(&clt, clens, 19) < 0) return -10;
                int nlen = hlit + hdist;
                std::vector<unsigned char> lens(nlen);
                int i = 0;
                while (i < nlen) {
                    int sym = decode_sym(&s, &clt);
                    if (sym < 0) return -11;
                    if (sym < 16) {
                        lens[i++] = (unsigned char)sym;
                    } else if (sym == 16) {
                        if (i == 0) return -12;
                        if (!fill_bits(&s, 2)) return -13;
                        int rep = 3 + (int)(s.buf & 3); s.buf >>= 2; s.bitcnt -= 2;
                        unsigned char prev = lens[i - 1];
                        while (rep-- && i < nlen) lens[i++] = prev;
                    } else if (sym == 17) {
                        if (!fill_bits(&s, 3)) return -14;
                        int rep = 3 + (int)(s.buf & 7); s.buf >>= 3; s.bitcnt -= 3;
                        while (rep-- && i < nlen) lens[i++] = 0;
                    } else {
                        if (!fill_bits(&s, 7)) return -15;
                        int rep = 11 + (int)(s.buf & 0x7f); s.buf >>= 7; s.bitcnt -= 7;
                        while (rep-- && i < nlen) lens[i++] = 0;
                    }
                }
                if (lens[256] == 0) return -16;      // no end-of-block code
                if (build_tree(&lit, lens.data(), hlit) < 0) return -17;
                if (build_tree(&dist, lens.data() + hlit, hdist) < 0) return -18;
            }
            static const short LBASE[29] =
                {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,
                 99,115,131,163,195,227,258};
            static const unsigned char LEXT[29] =
                {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
            static const short DBASE[30] =
                {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
                 1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
            static const unsigned char DEXT[30] =
                {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,
                 12,12,13,13};
            for (;;) {
                int sym = decode_sym(&s, &lit);
                if (sym < 0) return -19;
                if (sym < 256) {
                    if (s.opos >= ocap) return -20;
                    out[s.opos++] = (unsigned char)sym;
                } else if (sym == 256) {
                    break;
                } else {
                    sym -= 257;
                    if (sym >= 29) return -21;
                    int le = LEXT[sym];
                    if (!fill_bits(&s, le)) return -22;
                    int len = LBASE[sym] + (int)(s.buf & ((1u << le) - 1));
                    s.buf >>= le; s.bitcnt -= le;
                    int dsym = decode_sym(&s, &dist);
                    if (dsym < 0) return -23;
                    if (dsym >= 30) return -24;
                    int de = DEXT[dsym];
                    if (!fill_bits(&s, de)) return -25;
                    int d = DBASE[dsym] + (int)(s.buf & ((1u << de) - 1));
                    s.buf >>= de; s.bitcnt -= de;
                    if ((size_t)d > s.opos) return -26;
                    if (s.opos + (size_t)len > ocap) return -27;
                    // chunked copy: chunks of size <= d handle overlap and
                    // let long matches go through memcpy
                    int rem = len;
                    while (rem > 0) {
                        size_t chunk = (size_t)rem < (size_t)d
                                     ? (size_t)rem : (size_t)d;
                        memcpy(out + s.opos, out + s.opos - (size_t)d, chunk);
                        s.opos += chunk;
                        rem -= (int)chunk;
                    }
                }
            }
        } else {
            return -28;
        }
        if (final_blk) break;
    }
    *out_len = s.opos;
    return 0;
}

// ------------------------------------------------------------ defilter

static inline int paeth(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p;
    int pb = p > b ? p - b : b - p;
    int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

static void defilter(const unsigned char* raw,
                     unsigned char* recon, int h, int stride, int bpp) {
    const unsigned char* src = raw;
    for (int y = 0; y < h; y++) {
        int f = *src++;
        unsigned char* r = recon + (size_t)y * stride;
        const unsigned char* up = (y > 0) ? r - stride : NULL;
        switch (f) {
        case 0:
            memcpy(r, src, (size_t)stride);
            break;
        case 1:
            memcpy(r, src, (size_t)bpp);
            for (int x = bpp; x < stride; x++)
                r[x] = (unsigned char)(src[x] + r[x - bpp]);
            break;
        case 2:
            if (up)
                for (int x = 0; x < stride; x++)
                    r[x] = (unsigned char)(src[x] + up[x]);
            else
                memcpy(r, src, (size_t)stride);
            break;
        case 3:
            for (int x = 0; x < bpp; x++)
                r[x] = (unsigned char)(src[x] + ((up ? up[x] : 0) >> 1));
            for (int x = bpp; x < stride; x++)
                r[x] = (unsigned char)(src[x] + ((r[x - bpp] + (up ? up[x] : 0)) >> 1));
            break;
        default:  // 4 Paeth
            for (int x = 0; x < bpp; x++)
                r[x] = (unsigned char)(src[x] + paeth(0, up ? up[x] : 0, 0));
            for (int x = bpp; x < stride; x++)
                r[x] = (unsigned char)(src[x] +
                    paeth(r[x - bpp], up ? up[x] : 0,
                          (up && x >= bpp) ? up[x - bpp] : 0));
            break;
        }
        src += stride;
    }
}

// ------------------------------------------------------------ chunk scan

static unsigned be32(const unsigned char* p) {
    return ((unsigned)p[0] << 24) | ((unsigned)p[1] << 16)
         | ((unsigned)p[2] << 8) | p[3];
}

// metadata serialization (little-endian):
//   u32 n_texts; per text { u32 type(0 tEXt/1 zTXt/2 iTXt); u32 raw_len; raw }
//   u32 phys_len; bytes (0 = none)
//   u32 exif_len; bytes
//   u32 iccp_raw_len; bytes

struct MetaOut {
    unsigned char* meta;
    int meta_cap;
    size_t mpos;
    unsigned n_texts;
    size_t n_texts_pos;
    const unsigned char* phys;  size_t phys_len;
    const unsigned char* exif;  size_t exif_len;
    const unsigned char* iccp;  size_t iccp_len;

    bool wmeta(const void* p, size_t n) {
        if (mpos + n > (size_t)meta_cap) return false;
        memcpy(meta + mpos, p, n);
        mpos += n;
        return true;
    }
    bool wu32(unsigned v) {
        unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                              (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
        return wmeta(b, 4);
    }
    bool scan(const unsigned char* data, int len) {
        mpos = 0; n_texts = 0;
        phys = exif = iccp = NULL;
        phys_len = exif_len = iccp_len = 0;
        if (!wu32(0)) return false;
        n_texts_pos = mpos - 4;
        if (len < 8) return false;
        size_t pos = 8;
        while (pos + 12 <= (size_t)len) {
            unsigned cl = be32(data + pos);
            const unsigned char* typ = data + pos + 4;
            const unsigned char* body = data + pos + 8;
            if (pos + 12 + (size_t)cl > (size_t)len) break;
            if (!memcmp(typ, "tEXt", 4) || !memcmp(typ, "zTXt", 4)
                || !memcmp(typ, "iTXt", 4)) {
                unsigned t = !memcmp(typ, "tEXt", 4) ? 0
                           : !memcmp(typ, "zTXt", 4) ? 1 : 2;
                if (!wu32(t) || !wu32(cl)) return false;
                if (cl && !wmeta(body, cl)) return false;
                n_texts++;
            } else if (!memcmp(typ, "pHYs", 4)) {
                phys = body; phys_len = cl;
            } else if (!memcmp(typ, "eXIf", 4)) {
                exif = body; exif_len = cl;
            } else if (!memcmp(typ, "iCCP", 4)) {
                iccp = body; iccp_len = cl;
            } else if (!memcmp(typ, "IEND", 4)) {
                break;
            }
            pos += 12 + (size_t)cl;
        }
        return true;
    }
    bool finish() {
        if (!wu32((unsigned)phys_len) || (phys_len && !wmeta(phys, phys_len)))
            return false;
        if (!wu32((unsigned)exif_len) || (exif_len && !wmeta(exif, exif_len)))
            return false;
        if (!wu32((unsigned)iccp_len) || (iccp_len && !wmeta(iccp, iccp_len)))
            return false;
        meta[n_texts_pos]     = (unsigned char)n_texts;
        meta[n_texts_pos + 1] = (unsigned char)(n_texts >> 8);
        meta[n_texts_pos + 2] = (unsigned char)(n_texts >> 16);
        meta[n_texts_pos + 3] = (unsigned char)(n_texts >> 24);
        return true;
    }
};

extern "C" __declspec(dllexport)
int png_meta_scan(const unsigned char* data, int len,
                  unsigned char* meta, int meta_cap, int* meta_len) {
    MetaOut mo;
    mo.meta = meta; mo.meta_cap = meta_cap;
    if (!mo.scan(data, len)) return -11;
    if (!mo.finish()) return -14;
    *meta_len = (int)mo.mpos;
    return 0;
}

extern "C" __declspec(dllexport)
int png_probe(const unsigned char* data, int len, int* W, int* H) {
    if (len < 33) return -1;
    static const unsigned char magic[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (memcmp(data, magic, 8) != 0) return -2;
    if (memcmp(data + 12, "IHDR", 4) != 0) return -3;
    unsigned w = be32(data + 16), h = be32(data + 20);
    unsigned bd = data[24], ct = data[25], interlace = data[28];
    if (w == 0 || h == 0 || w > 30000 || h > 30000) return -4;
    *W = (int)w; *H = (int)h;
    if (bd != 8 || (ct != 2 && ct != 6) || interlace != 0) return PNG_UNSUPPORTED;
    return 0;
}

extern "C" __declspec(dllexport)
int png_decode_full(const unsigned char* data, int len,
                    unsigned char* rgba, int rgba_cap,
                    int* W, int* H,
                    unsigned char* meta, int meta_cap, int* meta_len) {
    int w, h;
    int pr = png_probe(data, len, &w, &h);
    if (pr != 0) return pr;
    *W = w; *H = h;
    int bpp = data[25] == 6 ? 4 : 3;
    size_t stride = (size_t)w * bpp;
    size_t raw_need = (size_t)h * (stride + 1);
    if ((size_t)rgba_cap < (size_t)w * h * 4) return -10;

    MetaOut mo;
    mo.meta = meta; mo.meta_cap = meta_cap;

    // one scan: IDAT positions + metadata chunks (no IDAT copies)
    std::vector<size_t> idat_off, idat_len;
    mo.mpos = 0; mo.n_texts = 0;
    mo.phys = mo.exif = mo.iccp = NULL;
    mo.phys_len = mo.exif_len = mo.iccp_len = 0;
    if (!mo.wu32(0)) return -11;
    mo.n_texts_pos = mo.mpos - 4;
    size_t pos = 8;
    while (pos + 12 <= (size_t)len) {
        unsigned cl = be32(data + pos);
        const unsigned char* typ = data + pos + 4;
        const unsigned char* body = data + pos + 8;
        if (pos + 12 + (size_t)cl > (size_t)len) break;
        if (!memcmp(typ, "IDAT", 4)) {
            idat_off.push_back(pos + 8);
            idat_len.push_back(cl);
        } else if (!memcmp(typ, "tEXt", 4) || !memcmp(typ, "zTXt", 4)
                   || !memcmp(typ, "iTXt", 4)) {
            unsigned t = !memcmp(typ, "tEXt", 4) ? 0
                       : !memcmp(typ, "zTXt", 4) ? 1 : 2;
            if (!mo.wu32(t) || !mo.wu32(cl)) return -12;
            if (cl && !mo.wmeta(body, cl)) return -12;
            mo.n_texts++;
        } else if (!memcmp(typ, "pHYs", 4)) {
            mo.phys = body; mo.phys_len = cl;
        } else if (!memcmp(typ, "eXIf", 4)) {
            mo.exif = body; mo.exif_len = cl;
        } else if (!memcmp(typ, "iCCP", 4)) {
            mo.iccp = body; mo.iccp_len = cl;
        } else if (!memcmp(typ, "IEND", 4)) {
            break;
        }
        pos += 12 + (size_t)cl;
    }
    if (idat_off.empty()) return -13;

    // contiguous IDAT fast path (normal case); else gather into scratch
    const unsigned char* zsrc;
    std::vector<unsigned char> zbuf;
    size_t zlen = 0;
    for (size_t i = 0; i < idat_len.size(); i++) zlen += idat_len[i];
    bool contiguous = true;
    for (size_t i = 0; i + 1 < idat_off.size(); i++)
        if (idat_off[i] + idat_len[i] != idat_off[i + 1]) { contiguous = false; break; }
    if (contiguous) {
        zsrc = data + idat_off[0];
    } else {
        zbuf.resize(zlen);
        size_t o = 0;
        for (size_t i = 0; i < idat_off.size(); i++) {
            memcpy(zbuf.data() + o, data + idat_off[i], idat_len[i]);
            o += idat_len[i];
        }
        zsrc = zbuf.data();
    }

    // skip the 2-byte zlib header (CMF/FLG); PNG IDAT is a zlib stream
    if (zlen < 6) return -29;
    if ((zsrc[0] & 0x0f) != 8 || (zsrc[0] >> 4) > 7) return PNG_UNSUPPORTED;
    zsrc += 2; zlen -= 2;

    std::vector<unsigned char> raw(raw_need);
    size_t got = 0;
    int ir = inflate_raw(zsrc, zlen, raw.data(), raw_need, &got);
    if (ir != 0) return ir - 100;             // -101..-128: exact inflate code
    if (got < raw_need) return -50;

    if (bpp == 4) {
        defilter(raw.data(), rgba, h, (int)stride, 4);
    } else {
        std::vector<unsigned char> recon(stride * (size_t)h);
        defilter(raw.data(), recon.data(), h, (int)stride, 3);
        for (int y = 0; y < h; y++) {
            const unsigned char* r = recon.data() + (size_t)y * stride;
            unsigned char* o = rgba + (size_t)y * w * 4;
            for (int x = 0; x < w; x++) {
                o[x * 4]     = r[x * 3];
                o[x * 4 + 1] = r[x * 3 + 1];
                o[x * 4 + 2] = r[x * 3 + 2];
                o[x * 4 + 3] = 255;
            }
        }
    }

    if (!mo.finish()) return -14;
    *meta_len = (int)mo.mpos;
    return 0;
}

// inflate-only fast path: chunk scan + DEFLATE, no defilter. Output = the
// filter-prefixed rows exactly as stored (h rows of [filter_byte | stride]),
// ready for the GPU defilter kernel. Returns 0 ok, PNG_UNSUPPORTED for
// formats outside the fast path, negative on error.
extern "C" __declspec(dllexport)
int png_inflate_raw(const unsigned char* data, int len,
                    unsigned char* raw, int raw_cap, int* out_len,
                    int* W, int* H, int* bpp)
{
    int w, h;
    int pr = png_probe(data, len, &w, &h);
    if (pr != 0) return pr;
    *W = w; *H = h;
    int b = data[25] == 6 ? 4 : 3;
    *bpp = b;
    size_t stride = (size_t)w * b;
    size_t raw_need = (size_t)h * (stride + 1);
    if ((size_t)raw_cap < raw_need) return -60;

    // gather IDAT (contiguous fast path, scratch fallback)
    std::vector<size_t> off, ln;
    size_t pos = 8;
    while (pos + 12 <= (size_t)len) {
        unsigned cl = be32(data + pos);
        const unsigned char* typ = data + pos + 4;
        if (pos + 12 + (size_t)cl > (size_t)len) break;
        if (!memcmp(typ, "IDAT", 4)) {
            off.push_back(pos + 8);
            ln.push_back(cl);
        } else if (!memcmp(typ, "IEND", 4)) {
            break;
        }
        pos += 12 + (size_t)cl;
    }
    if (off.empty()) return -61;
    const unsigned char* zsrc;
    std::vector<unsigned char> zbuf;
    size_t zlen = 0;
    for (size_t i = 0; i < ln.size(); i++) zlen += ln[i];
    bool contiguous = true;
    for (size_t i = 0; i + 1 < off.size(); i++)
        if (off[i] + ln[i] != off[i + 1]) { contiguous = false; break; }
    if (contiguous) {
        zsrc = data + off[0];
    } else {
        zbuf.resize(zlen);
        size_t o = 0;
        for (size_t i = 0; i < off.size(); i++) {
            memcpy(zbuf.data() + o, data + off[i], ln[i]);
            o += ln[i];
        }
        zsrc = zbuf.data();
    }
    if (zlen < 6) return -62;
    if ((zsrc[0] & 0x0f) != 8 || (zsrc[0] >> 4) > 7) return PNG_UNSUPPORTED;
    zsrc += 2; zlen -= 2;

    size_t got = 0;
    int ir = zlib_inflate_to(zsrc, zlen, raw, raw_need, &got);
    if (ir != 0) {                      // dll unavailable: scalar fallback
        ir = inflate_raw(zsrc, zlen, raw, raw_need, &got);
        if (ir != 0) return ir - 100;
    }
    if (got < raw_need) return -63;
    *out_len = (int)raw_need;
    return 0;
}
