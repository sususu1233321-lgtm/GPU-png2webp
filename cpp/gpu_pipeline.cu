#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
// gpu_pipeline.cu — C++ GPU pipeline for PNG→WebP batch encoding.
// Replaces the Python/cupy orchestration layer: direct CUDA API calls,
// pinned-memory async transfers, zero Python overhead in the hot path.
//
// Build: nvcc -O2 -shared -Xcompiler "/LD" -o gpu_pipeline.dll gpu_pipeline.cpp
//   (or use build.bat)
//
// Python interface (ctypes):
//   process_batch(rgba_ptr, n, w, h, quality,
//                 y_dc, y_ac, uv_lv, is_i4, i16_mode, uv_mode, i4_modes)
//   → returns 0 on success, negative on error

#include <cuda_runtime.h>
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


// CUDA kernels (extracted from gpuwebp/closed_loop_gpu.py)
#include "kernels.cuh"

// ---------------------------------------------------------------- helpers

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), \
                __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

// Quant setup (mirrors vp8_encode.setup_quant)
struct QuantTables {
    long long y1q[16], y1iq[16], y1b[16], y1z[16], y1s[16];
    long long y2q[16], y2iq[16], y2b[16], y2z[16], y2s[16];
    long long uvq[16], uviq[16], uvb[16], uvz[16], uvs[16];
    long long y1deq[16], y2deq[16], uvdeq[16];
    int filter_level;
    long long y1_qavg;
    long long lam_i4, lam_i16;      // trellis RD lambdas
};

// DC quant table (from vp8_tables.py, extracted from libwebp)








#include "quant_tables.inc"


// ------------------------------------------------------------------ YUV

// Simple BT.601 RGB→YUV420 conversion as a CUDA kernel
__global__ void rgb_to_yuv420_kernel(
    const unsigned char* __restrict__ rgba, int img_stride,
    short* __restrict__ Y, short* __restrict__ U, short* __restrict__ V,
    int W, int H, int y_stride, int c_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int img = blockIdx.z;
    if (x >= W || y >= H) return;
    const unsigned char* p = rgba + (size_t)img * img_stride + (y * W + x) * 4;
    int r = p[0], g = p[1], b = p[2];
    long long yl = 16839LL * r + 33059LL * g + 6420LL * b + 32768LL + (16LL << 16);
    int yy = (int)(yl >> 16);
    Y[(size_t)img * y_stride + y * W + x] = (short)(yy > 255 ? 255 : yy);
    if ((x & 1) == 0 && (y & 1) == 0) {
        int x2 = x / 2, y2 = y / 2;
        int W2 = W / 2, H2 = H / 2;
        const unsigned char* p00 = p;
        const unsigned char* p01 = p + 4;  if (x + 1 < W) p01 = rgba + (size_t)img * img_stride + (y * W + x + 1) * 4;
        const unsigned char* p10 = p + W * 4;  if (y + 1 < H) p10 = rgba + (size_t)img * img_stride + ((y + 1) * W + x) * 4;
        const unsigned char* p11 = p10 + 4;  if (y + 1 < H && x + 1 < W) p11 = rgba + (size_t)img * img_stride + ((y + 1) * W + x + 1) * 4;
        // Python: r2 = r.reshape(...).sum(axis=(-3,-1)) — SUM not average
        int r2 = p00[0] + p01[0] + p10[0] + p11[0];
        int g2 = p00[1] + p01[1] + p10[1] + p11[1];
        int b2 = p00[2] + p01[2] + p10[2] + p11[2];
        long long uv_in_u = -9719LL * r2 - 19081LL * g2 + 28800LL * b2;
        long long uv_in_v = 28800LL * r2 - 24116LL * g2 - 4684LL * b2;
        int uu = (int)((uv_in_u + 131072LL + 33554432LL) >> 18);
        uu = uu < 0 ? 0 : (uu > 255 ? 255 : uu);
        int vv = (int)((uv_in_v + 131072LL + 33554432LL) >> 18);
        vv = vv < 0 ? 0 : (vv > 255 ? 255 : vv);
        U[(size_t)img * c_stride + y2 * W2 + x2] = (short)uu;
        V[(size_t)img * c_stride + y2 * W2 + x2] = (short)vv;
    }
}

// ------------------------------------------------------------ select_modes (CPU, C++)

// Create bordered planes for mode_search_kernel: row0=127, col0=129
__global__ void make_borders_kernel(
    const short* __restrict__ src, short* __restrict__ dst,
    int H, int W, int val127, int val129)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 0..W (inclusive)
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 0..H (inclusive)
    int img = blockIdx.z;
    int sS = H * W;          // src stride per image
    int dS = (H + 1) * (W + 1);  // dst stride
    if (x > W || y > H) return;
    short* d = dst + (size_t)img * dS;
    if (y == 0) {
        // row 0 owns the corner: 127, like libwebp's first-row top memset
        d[y * (W + 1) + x] = (short)val127;
    } else if (x == 0) {
        d[y * (W + 1) + x] = (short)val129;
    } else if (x <= W && y <= H) {
        const short* s = src + (size_t)img * sS;
        d[y * (W + 1) + x] = s[(y - 1) * W + (x - 1)];
    }
}
// Sequential per-image mode selection with context propagation.
// Port of vp8_encode.select_modes (numba)

struct ModeResult {
    unsigned char is_i4;
    unsigned char i16_mode;
    unsigned char uv_mode;
    unsigned char i4_modes[16];
};

static const int FIXED_COSTS_I16[4] = {0, 0, 0, 0};

static const long long RD_MULT = 256;      // libwebp RefineUsingDistortion
static const long long LAMBDA_D_I4 = 11;

#include "fixed_costs_i4.inc"

static void select_modes_one(
    const int* sse4,     // [n_mb * 16 * 10]
    const long long* i16_score, // [n_mb]
    const unsigned char* i16_mode, // [n_mb]
    const unsigned char* uv_mode,  // [n_mb] (from mode search)
    int mb_w, int mb_h,
    long long penalty,
    ModeResult* out)           // [n_mb]
{
    // Faithful port of select_modes from vp8_encode.py: sequential scan with
    // exact context propagation (left/top 4x4 mode indices feed FIXED_COSTS_I4)
    for (int mby = 0; mby < mb_h; mby++) {
        for (int mbx = 0; mbx < mb_w; mbx++) {
            int mb = mby * mb_w + mbx;
            unsigned char* om = out[mb].i4_modes;   // doubles as context slots
            long long total = penalty;
            for (int y = 0; y < 4; y++) {
                int left = (mbx == 0) ? 0 : out[mb - 1].i4_modes[y * 4 + 3];
                for (int x = 0; x < 4; x++) {
                    int top = (y == 0)
                        ? ((mby == 0) ? 0 : out[mb - mb_w].i4_modes[12 + x])
                        : om[(y - 1) * 4 + x];
                    const int* s = sse4 + mb * 160 + (y * 4 + x) * 10;
                    long long best = 1LL << 60;
                    int best_m = 0;
                    const long long* fc = FIXED_COSTS_I4 + (top * 10 + left) * 10;
                    for (int m = 0; m < 10; m++) {
                        long long sc = (long long)s[m] * RD_MULT + fc[m] * LAMBDA_D_I4;
                        if (sc < best) { best = sc; best_m = m; }
                    }
                    om[y * 4 + x] = (unsigned char)best_m;
                    total += best;
                    left = best_m;
                }
            }
            if (total < i16_score[mb]) {
                out[mb].is_i4 = 1;          // keep i4 modes just written
            } else {
                out[mb].is_i4 = 0;
                unsigned char m16 = i16_mode[mb];   // fill context with i16 mode
                for (int k = 0; k < 16; k++) om[k] = m16;
            }
            out[mb].i16_mode = i16_mode[mb];
            out[mb].uv_mode = uv_mode[mb];
        }
    }
}

// ------------------------------------------------------------ main entry

#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <deque>
#include <condition_variable>
#include <memory>

// Persistent device/host buffer cache: allocations are the dominant
// per-call overhead when batches are small. Grow on demand, never shrink.
// Every buffer has its OWN capacity — sharing caps between buffers makes
// the grow check skip a sibling that still needs reallocation.
struct Buf { void* p = nullptr; size_t cap = 0; bool host = false; };
struct DevCache {
    cudaStream_t stream = nullptr;
    Buf fraw;                // filtered rows (GPU-defilter path)
    Buf wclamp;              // per-image W_clamp ints
    Buf tlam;                // trellis lambdas [2]
    Buf rawpad;              // packed per-image RGBA (padded-batch staging)
    Buf doff;                // per-image src byte offsets (padded batch)
    Buf dhre;                // per-image real heights (padded batch)
    Buf dfc;                 // FIXED_COSTS_I4 table (GPU mode select)
    Buf rsse;                // per-image recon SSE (n*3 u64)
    Buf dhre23;              // per-image real heights at stage23 (SSE kernel)
    Buf dalpha;              // per-image alpha flags (device)
    Buf rgba, Y, U, V, rY, rU, rV, bY, bU, bV, flags;
    Buf is_i4, i16m, uvm, i4m, i16score, sse4, out;
    long long* q[18] = {nullptr};
    long long* fc_i16 = nullptr;     long long* fc_uv = nullptr;
    int q_quality = -1;
    Buf pin_in, pin_sse4, pin_score, pin_mb, pin_out, pin_is4, pin_i4m;
    bool init = false;
};
static DevCache g;          // kept for the sync wrappers
static DevCache S2[2];       // slot buffer sets for the pipeline

static int g_device = 0;
static bool g_dev_fixed = false;   // set before first CUDA call -> sticky

// MUST be called before the first submit/process call. Returns 0 on
// success; -1 if workers already started on another device.
extern "C" __declspec(dllexport)
int gpu_set_device(int d)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || d < 0 || d >= n) return -2;
    if (g_dev_fixed && g_device != d) return -1;
    if (cudaSetDevice(d) != cudaSuccess) return -3;
    g_device = d;
    g_dev_fixed = true;
    return 0;
}

static void init_slot_streams() {
    static std::once_flag once;
    std::call_once(once, [] {
        cudaSetDevice(g_device);      // first CUDA call pins the context
        for (int i = 0; i < 2; i++) cudaStreamCreate(&S2[i].stream);
    });
}

static int grow_d(Buf& b, size_t need) {
    if (b.p && b.cap >= need) return 0;
    if (b.p) cudaFree(b.p);
    cudaError_t e = cudaMalloc(&b.p, need);
    if (e != cudaSuccess) { b.p = nullptr; b.cap = 0; return -1; }
    b.cap = need;
    b.host = false;
    return 0;
}
static int grow_pin(Buf& b, size_t need) {
    if (b.p && b.cap >= need) return 0;
    if (b.p) cudaFreeHost(b.p);
    cudaError_t e = cudaHostAlloc(&b.p, need, cudaHostAllocDefault);
    if (e != cudaSuccess) { b.p = nullptr; b.cap = 0; return -1; }
    b.cap = need;
    b.host = true;
    return 0;
}
static void free_buf(Buf& b) {
    if (!b.p) { b.cap = 0; return; }
    if (b.host) cudaFreeHost(b.p); else cudaFree(b.p);
    b.p = nullptr; b.cap = 0;
}
static void purge_cache(DevCache& c) {
    Buf* bs[] = {&c.fraw, &c.dalpha, &c.rgba, &c.Y, &c.U, &c.V,
                 &c.rY, &c.rU, &c.rV, &c.bY, &c.bU, &c.bV, &c.flags,
                 &c.is_i4, &c.i16m, &c.uvm, &c.i4m, &c.i16score,
                 &c.sse4, &c.out, &c.pin_in, &c.pin_sse4, &c.pin_score,
                 &c.pin_mb, &c.pin_out, &c.pin_is4, &c.pin_i4m};
    for (Buf* b : bs) free_buf(*b);
    for (int i = 0; i < 18; i++) {
        if (c.q[i]) { cudaFree(c.q[i]); c.q[i] = nullptr; }
    }
    if (c.fc_i16) { cudaFree(c.fc_i16); c.fc_i16 = nullptr; }
    if (c.fc_uv) { cudaFree(c.fc_uv); c.fc_uv = nullptr; }
    c.q_quality = -1;
}

static int old_sync_body(
    const unsigned char* h_rgba,   // host (pageable ok): n * H * W * 4 bytes
    int n, int W, int H, int quality,
    short* h_y_dc, short* h_y_ac, short* h_uv_lv,
    unsigned char* h_is_i4, unsigned char* h_i16_mode,
    unsigned char* h_uv_mode, unsigned char* h_i4_modes);

// pointer-array variant: Python passes per-image numpy data pointers, the
// staging memcpy happens inside the DLL (no GIL held, no double copy)
static int old_sync_ptrs(
    void* const* img_ptrs,         // n pointers, each H*W*4 bytes
    int n, int W, int H, int quality,
    short* h_y_dc, short* h_y_ac, short* h_uv_lv,
    unsigned char* h_is_i4, unsigned char* h_i16_mode,
    unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    size_t one = (size_t)H * W * 4;
    if (g.pin_in.cap < (size_t)n * one) {
        // grow the pinned staging first (process_batch would grow it too,
        // but we stage before calling it)
        if (grow_pin(g.pin_in, (size_t)n * one)) return -3;
    }
    unsigned char* dst = (unsigned char*)g.pin_in.p;
    for (int i = 0; i < n; i++)
        memcpy(dst + (size_t)i * one, img_ptrs[i], one);
    return old_sync_body(dst, n, W, H, quality,
                         h_y_dc, h_y_ac, h_uv_lv,
                         h_is_i4, h_i16_mode, h_uv_mode, h_i4_modes);
}

static int old_sync_body(
    const unsigned char* h_rgba,
    int n, int W, int H, int quality,
    short* h_y_dc, short* h_y_ac, short* h_uv_lv,
    unsigned char* h_is_i4, unsigned char* h_i16_mode,
    unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    if (n <= 0 || !h_rgba) return -1;
    if (H % 16 || W % 16) return -2;          // 16-alignment required
    int mb_h = H / 16, mb_w = W / 16;
    int n_mb = mb_h * mb_w;
    int HH = H / 2, HW = W / 2;
    size_t nmb_all = (size_t)n * n_mb;

    QuantTables qt;
    setup_quant_exact(quality, &qt);

    size_t rgb_bytes = (size_t)n * H * W * 4;
    size_t y_bytes = (size_t)n * H * W * sizeof(short);
    size_t c_bytes = (size_t)n * HH * HW * sizeof(short);
    size_t ry_bytes = (size_t)n * (H + 1) * (W + 1) * sizeof(short);
    size_t rc_bytes = (size_t)n * (HH + 1) * (HW + 1) * sizeof(short);
    size_t sse_bytes = nmb_all * 160 * sizeof(int);
    size_t out_bytes = nmb_all * (16 + 256 + 128) * sizeof(short);
    size_t modes_bytes = nmb_all * 17;        // is_i4 + i4_modes(16)

    // ---- grow caches ----
    if (grow_d(g.rgba, rgb_bytes)) return -3;
    if (grow_d(g.Y, y_bytes)) return -3;
    if (grow_d(g.U, c_bytes)) return -3;
    if (grow_d(g.V, c_bytes)) return -3;
    if (grow_d(g.rY, ry_bytes)) return -3;
    if (grow_d(g.rU, rc_bytes)) return -3;
    if (grow_d(g.rV, rc_bytes)) return -3;
    if (grow_d(g.bY, ry_bytes)) return -3;
    if (grow_d(g.bU, rc_bytes)) return -3;
    if (grow_d(g.bV, rc_bytes)) return -3;
    if (grow_d(g.flags, nmb_all * 4)) return -3;
    if (grow_d(g.is_i4, nmb_all)) return -3;
    if (grow_d(g.i16m, nmb_all)) return -3;
    if (grow_d(g.uvm, nmb_all)) return -3;
    if (grow_d(g.i4m, nmb_all * 16)) return -3;
    if (grow_d(g.i16score, nmb_all * 8)) return -3;
    if (grow_d(g.sse4, sse_bytes)) return -3;
    if (grow_d(g.out, out_bytes)) return -3;
    for (int i = 0; i < 18; i++)
        if (!g.q[i] && cudaMalloc(&g.q[i], 16 * sizeof(long long)) != cudaSuccess)
            return -3;
    if (!g.tlam.p && cudaMalloc(&g.tlam.p, 2 * sizeof(long long)) != cudaSuccess)
        return -3;
    if (!g.fc_i16 && cudaMalloc(&g.fc_i16, 4 * sizeof(long long))) return -3;
    if (!g.fc_uv && cudaMalloc(&g.fc_uv, 4 * sizeof(long long))) return -3;
    if (grow_pin(g.pin_in, rgb_bytes)) return -3;
    if (grow_pin(g.pin_sse4, sse_bytes)) return -3;
    if (grow_pin(g.pin_score, nmb_all * 8)) return -3;
    if (grow_pin(g.pin_mb, nmb_all * 3)) return -3;
    if (grow_pin(g.pin_out, out_bytes)) return -3;
    if (grow_pin(g.pin_is4, nmb_all)) return -3;
    if (grow_pin(g.pin_i4m, nmb_all * 16)) return -3;

    short* d_ydc = (short*)g.out.p;
    short* d_yac = (short*)g.out.p + nmb_all * 16;
    short* d_uvlv = (short*)g.out.p + nmb_all * (16 + 256);

    // ---- quant tables (only when quality changes) ----
    if (g.q_quality != quality) {
        long long tmp[16];
        #define UPLOAD_Q(field, idx) { \
            for (int i = 0; i < 16; i++) tmp[i] = qt.field[i]; \
            cudaMemcpy(g.q[idx], tmp, 16 * sizeof(long long), cudaMemcpyHostToDevice); }
        UPLOAD_Q(y1q, 0); UPLOAD_Q(y1iq, 1); UPLOAD_Q(y1b, 2); UPLOAD_Q(y1z, 3); UPLOAD_Q(y1s, 4);
        UPLOAD_Q(y2q, 5); UPLOAD_Q(y2iq, 6); UPLOAD_Q(y2b, 7); UPLOAD_Q(y2z, 8); UPLOAD_Q(y2s, 9);
        UPLOAD_Q(uvq, 10); UPLOAD_Q(uviq, 11); UPLOAD_Q(uvb, 12); UPLOAD_Q(uvz, 13); UPLOAD_Q(uvs, 14);
        UPLOAD_Q(y1deq, 15); UPLOAD_Q(y2deq, 16); UPLOAD_Q(uvdeq, 17);
        #undef UPLOAD_Q
        {
            long long lam[2] = { qt.lam_i4, qt.lam_i16 };
            cudaMemcpy(g.tlam.p, lam, 2 * sizeof(long long),
                       cudaMemcpyHostToDevice);
        }
        long long fc_i16[4] = {663, 919, 872, 919};
        long long fc_uv[4] = {302, 984, 439, 642};
        cudaMemcpy(g.fc_i16, fc_i16, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(g.fc_uv, fc_uv, 32, cudaMemcpyHostToDevice);
        g.q_quality = quality;
    }

    static bool timing = getenv("GPUPIPE_TIMING") != nullptr;
    double t_h2d = 0, t_ms = 0, t_d2h1 = 0, t_sel = 0, t_up = 0, t_cl = 0, t_d2h2 = 0;
    auto clk = []() { return (double)clock() / CLOCKS_PER_SEC; };
    double t0 = clk();

    // ---- upload input via pinned staging ----
    memcpy(g.pin_in.p, h_rgba, rgb_bytes);
    CHECK_CUDA(cudaMemcpy(g.rgba.p, g.pin_in.p, rgb_bytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize(); t_h2d = clk() - t0;

    // ---- YUV + borders + mode search ----
    {
        dim3 block(16, 16, 1);
        dim3 grid(W / 16, H / 16, n);
        rgb_to_yuv420_kernel<<<grid, block>>>(
            (const unsigned char*)g.rgba.p, (int)(H * W * 4),
            (short*)g.Y.p, (short*)g.U.p, (short*)g.V.p, W, H, H * W, HH * HW);
    }
    {
        dim3 blk(32, 8, 1);
        dim3 grd((W + 32) / 32, (H + 8) / 8, n);      // cover 0..W inclusive
        make_borders_kernel<<<grd, blk>>>((const short*)g.Y.p, (short*)g.bY.p, H, W, 127, 129);
        dim3 grd2((HW + 32) / 32, (HH + 8) / 8, n);
        make_borders_kernel<<<grd2, blk>>>((const short*)g.U.p, (short*)g.bU.p, HH, HW, 127, 129);
        make_borders_kernel<<<grd2, blk>>>((const short*)g.V.p, (short*)g.bV.p, HH, HW, 127, 129);

        {
            static int* tmp = nullptr; static int tmpcap = 0;
            if (tmpcap < n) { free(tmp); tmp = (int*)malloc(n * sizeof(int)); tmpcap = n; }
            for (int i = 0; i < n; i++) tmp[i] = W;
            if (grow_d(g.wclamp, (size_t)n * sizeof(int))) return -3;
            cudaMemcpy(g.wclamp.p, tmp, n * sizeof(int), cudaMemcpyHostToDevice);
        }
        int total = (int)nmb_all;
        int threads = 128;
        int blocks = (total + threads - 1) / threads;
        mode_search_kernel<<<blocks, threads>>>(
            (const short*)g.bY.p, (const short*)g.bU.p, (const short*)g.bV.p,
            g.fc_i16, g.fc_uv,
            (unsigned char*)g.i16m.p, (long long*)g.i16score.p,
            (unsigned char*)g.uvm.p, (int*)g.sse4.p,
            n, mb_h, mb_w, H, W, (const int*)g.wclamp.p);
        CHECK_CUDA(cudaGetLastError());
    }
    cudaDeviceSynchronize(); t_ms = clk() - t0 - t_h2d;

    // ---- D2H mode-search outputs into pinned staging ----
    CHECK_CUDA(cudaMemcpy(g.pin_sse4.p, g.sse4.p, sse_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(g.pin_score.p, g.i16score.p, nmb_all * 8, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(g.pin_mb.p, g.i16m.p, nmb_all, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy((char*)g.pin_mb.p + nmb_all, g.uvm.p, nmb_all, cudaMemcpyDeviceToHost));

    t_d2h1 = clk() - t0 - t_h2d - t_ms;
    // ---- GPU mode selection (same kernel as the async pipeline; the old
    // CPU select ran 4 host threads per batch) ----
    if (g.dfc.cap < 1000 * sizeof(long long)) {
        if (grow_d(g.dfc, 1000 * sizeof(long long))) return -3;
        CHECK_CUDA(cudaMemcpy(g.dfc.p, FIXED_COSTS_I4,
                              1000 * sizeof(long long),
                              cudaMemcpyHostToDevice));
    }
    long long penalty = 1000LL * qt.y1_qavg * qt.y1_qavg;
    CHECK_CUDA(cudaMemsetAsync(g.flags.p, 0, nmb_all * sizeof(unsigned int), g.stream));
    select_modes_kernel<<<(n * mb_h + 32 - 1) / 32, 32, 0, g.stream>>>(
        (const int*)g.sse4.p, (const long long*)g.i16score.p,
        (const unsigned char*)g.i16m.p, (const unsigned char*)g.uvm.p,
        (const long long*)g.dfc.p, penalty,
        (unsigned char*)g.is_i4.p, (unsigned char*)g.i4m.p,
        (unsigned int*)g.flags.p, n, mb_h, mb_w);
    CHECK_CUDA(cudaGetLastError());

    // ---- memsets + closed loop ----
    CHECK_CUDA(cudaMemsetAsync(g.rY.p, 0, ry_bytes, g.stream));
    CHECK_CUDA(cudaMemsetAsync(g.rU.p, 0, rc_bytes, g.stream));
    CHECK_CUDA(cudaMemsetAsync(g.rV.p, 0, rc_bytes, g.stream));
    CHECK_CUDA(cudaMemsetAsync(g.flags.p, 0, nmb_all * sizeof(unsigned int), g.stream));
    // kernel skips writing all-zero DC slots — zero the dc region or stale
    // values leak across calls
    CHECK_CUDA(cudaMemsetAsync(g.out.p, 0, nmb_all * 16 * sizeof(short), g.stream));
    closed_loop_kernel<<<(n * mb_h + 32 - 1) / 32, 32>>>(
        (const short*)g.Y.p, (const short*)g.U.p, (const short*)g.V.p,
        (const unsigned char*)g.is_i4.p, (const unsigned char*)g.i16m.p,
        (const unsigned char*)g.uvm.p, (const unsigned char*)g.i4m.p,
        g.q[0], g.q[1], g.q[2], g.q[3], g.q[4],
        g.q[5], g.q[6], g.q[7], g.q[8], g.q[9],
        g.q[10], g.q[11], g.q[12], g.q[13], g.q[14],
        g.q[15], g.q[16], g.q[17],
        (const long long*)g.tlam.p,
        d_ydc, d_yac, d_uvlv,
        (short*)g.rY.p, (short*)g.rU.p, (short*)g.rV.p,
        (unsigned int*)g.flags.p,
        n, mb_h, mb_w, H, W, (const int*)g.wclamp.p);
    CHECK_CUDA(cudaGetLastError());

    // ---- final D2H: results + modes ----
    CHECK_CUDA(cudaMemcpyAsync(g.pin_out.p, g.out.p, out_bytes, cudaMemcpyDeviceToHost, g.stream));
    cudaStreamSynchronize(g.stream);
    memcpy(h_y_dc, g.pin_out.p, nmb_all * 16 * sizeof(short));
    memcpy(h_y_ac, (short*)g.pin_out.p + nmb_all * 16, nmb_all * 256 * sizeof(short));
    memcpy(h_uv_lv, (short*)g.pin_out.p + nmb_all * (16 + 256), nmb_all * 128 * sizeof(short));
    CHECK_CUDA(cudaMemcpy(h_is_i4, g.is_i4.p, nmb_all, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_i16_mode, g.i16m.p, nmb_all, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_uv_mode, g.uvm.p, nmb_all, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_i4_modes, g.i4m.p, nmb_all * 16, cudaMemcpyDeviceToHost));
    return 0;
}

// ---------------------------------------------------------------------
// async pipeline request
// ---------------------------------------------------------------------
struct Req3 {
    int id;
    int slot;
    int filtered = 0;        // 1 = imgs point to filter-prefixed rows
    int bpp = 0;
    unsigned char* alpha_flags = nullptr;   // caller out: n bytes
    std::vector<void*> imgs;            // per-image pointers (filtered path)
    std::vector<void*> src_rows;        // padded/zc path: per-image RGBA rows
    std::vector<unsigned char> input;   // staged copy (sync rgba path)
    int zc = 0;                         // 1 = rgba async path stages from
                                        // src_rows pointers (caller keeps the
                                        // arrays alive until completion)
    int n, W, H, quality;
    int padded = 0;                     // 1 = padded batch: W,H are PAD dims
    int W_real = 0, H_real = 0;         // uniform real dims (legacy single-dim)
    std::vector<int> vWr, vHr;          // per-image real dims (padded path)
    // packed outputs (padded path): caller buffers sized n*n_mb_real
    short* p_ydc = nullptr; short* p_yac = nullptr; short* p_uvlv = nullptr;
    unsigned char* p_is4 = nullptr; unsigned char* p_i16 = nullptr;
    unsigned char* p_uvm = nullptr; unsigned char* p_i4m = nullptr;
    int own_out = 0;                    // padded path mallocs y_dc/... itself
    long long* h_sse = nullptr;         // caller out: n*3 recon SSE (u64)
    short* y_dc; short* y_ac; short* uv_lv;
    unsigned char* is_i4; unsigned char* i16m; unsigned char* uvm;
    unsigned char* i4m;
    std::atomic<int> err;
};
static std::mutex g3_mu;
static std::condition_variable g3_cv_work, g3_cv_done;
static std::deque<Req3*> g3_q, g3_complete;
static std::vector<std::unique_ptr<Req3>> g3_reqs;
static int g3_next_id = 0;
static bool g3_run = false;

static int stage1_v2(Req3* r, struct DevCache* bs) {
    const unsigned char* h_rgba = (r->filtered || r->zc || r->padded)
        ? nullptr : r->input.data();
    int n = r->n, W = r->W, H = r->H, quality = r->quality;
    if (n <= 0 || (!r->filtered && !r->padded && !r->zc && !h_rgba))
        return -1;
    if (H % 16 || W % 16) return -2;          // 16-alignment required
    int mb_h = H / 16, mb_w = W / 16;
    int n_mb = mb_h * mb_w;
    int HH = H / 2, HW = W / 2;
    size_t nmb_all = (size_t)n * n_mb;

    QuantTables qt;
    setup_quant_exact(quality, &qt);

    size_t rgb_bytes = (size_t)n * H * W * 4;
    size_t y_bytes = (size_t)n * H * W * sizeof(short);
    size_t c_bytes = (size_t)n * HH * HW * sizeof(short);
    size_t ry_bytes = (size_t)n * (H + 1) * (W + 1) * sizeof(short);
    size_t rc_bytes = (size_t)n * (HH + 1) * (HW + 1) * sizeof(short);
    size_t sse_bytes = nmb_all * 160 * sizeof(int);
    size_t out_bytes = nmb_all * (16 + 256 + 128) * sizeof(short);
    size_t modes_bytes = nmb_all * 17;        // is_i4 + i4_modes(16)
    (void)modes_bytes;

    // ---- grow caches ----
    if (grow_d(bs->rgba, rgb_bytes)) return -3;
    if (grow_d(bs->Y, y_bytes)) return -3;
    if (grow_d(bs->U, c_bytes)) return -3;
    if (grow_d(bs->V, c_bytes)) return -3;
    if (grow_d(bs->rY, ry_bytes)) return -3;
    if (grow_d(bs->rU, rc_bytes)) return -3;
    if (grow_d(bs->rV, rc_bytes)) return -3;
    if (grow_d(bs->bY, ry_bytes)) return -3;
    if (grow_d(bs->bU, rc_bytes)) return -3;
    if (grow_d(bs->bV, rc_bytes)) return -3;
    if (grow_d(bs->flags, nmb_all * 4)) return -3;
    if (grow_d(bs->is_i4, nmb_all)) return -3;
    if (grow_d(bs->i16m, nmb_all)) return -3;
    if (grow_d(bs->uvm, nmb_all)) return -3;
    if (grow_d(bs->i4m, nmb_all * 16)) return -3;
    if (grow_d(bs->i16score, nmb_all * 8)) return -3;
    if (grow_d(bs->sse4, sse_bytes)) return -3;
    if (grow_d(bs->out, out_bytes)) return -3;
    for (int i = 0; i < 18; i++)
        if (!bs->q[i] && cudaMalloc(&bs->q[i], 16 * sizeof(long long)) != cudaSuccess)
            return -3;
    if (!bs->tlam.p && cudaMalloc(&bs->tlam.p, 2 * sizeof(long long)) != cudaSuccess)
        return -3;
    if (!bs->fc_i16 && cudaMalloc(&bs->fc_i16, 4 * sizeof(long long))) return -3;
    if (!bs->fc_uv && cudaMalloc(&bs->fc_uv, 4 * sizeof(long long))) return -3;
    if (grow_pin(bs->pin_in, rgb_bytes)) return -3;
    if (grow_pin(bs->pin_sse4, sse_bytes)) return -3;
    if (grow_pin(bs->pin_score, nmb_all * 8)) return -3;
    if (grow_pin(bs->pin_mb, nmb_all * 3)) return -3;
    if (grow_pin(bs->pin_out, out_bytes)) return -3;
    if (grow_pin(bs->pin_is4, nmb_all)) return -3;
    if (grow_pin(bs->pin_i4m, nmb_all * 16)) return -3;

    // ---- quant tables (only when quality changes) ----
    if (bs->q_quality != quality) {
        cudaStreamSynchronize(bs->stream);
        long long tmp[16];
        #define UPLOAD_Q(field, idx) { \
            for (int i = 0; i < 16; i++) tmp[i] = qt.field[i]; \
            cudaMemcpy(bs->q[idx], tmp, 16 * sizeof(long long), cudaMemcpyHostToDevice); }
        UPLOAD_Q(y1q, 0); UPLOAD_Q(y1iq, 1); UPLOAD_Q(y1b, 2); UPLOAD_Q(y1z, 3); UPLOAD_Q(y1s, 4);
        UPLOAD_Q(y2q, 5); UPLOAD_Q(y2iq, 6); UPLOAD_Q(y2b, 7); UPLOAD_Q(y2z, 8); UPLOAD_Q(y2s, 9);
        UPLOAD_Q(uvq, 10); UPLOAD_Q(uviq, 11); UPLOAD_Q(uvb, 12); UPLOAD_Q(uvz, 13); UPLOAD_Q(uvs, 14);
        UPLOAD_Q(y1deq, 15); UPLOAD_Q(y2deq, 16); UPLOAD_Q(uvdeq, 17);
        #undef UPLOAD_Q
        {
            long long lam[2] = { qt.lam_i4, qt.lam_i16 };
            cudaMemcpy(bs->tlam.p, lam, 2 * sizeof(long long),
                       cudaMemcpyHostToDevice);
        }
        long long fc_i16[4] = {663, 919, 872, 919};
        long long fc_uv[4] = {302, 984, 439, 642};
        cudaMemcpy(bs->fc_i16, fc_i16, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(bs->fc_uv, fc_uv, 32, cudaMemcpyHostToDevice);
        bs->q_quality = quality;
    }

    // ---- upload input via pinned staging ----
    if (r->padded) {
        // per-image RGBA (Wr*Hr*4, packed) -> H2D as one contiguous block ->
        // GPU scatter into the zero-padded W*H*4 grid. The old CPU loop did
        // ~40K small per-row memcpys per batch on the W1 thread and stalled
        // the interleaved main-size batches by ~1s each.
        if (grow_d(bs->wclamp, (size_t)n * sizeof(int))) return -3;
        static long long* tmpoff = nullptr; static int tmpoffcap = 0;
        static int* tmphr = nullptr; static int* tmpc = nullptr;
        static int tmpccap = 0;
        if (tmpoffcap < n + 1) {
            free(tmpoff); tmpoff = (long long*)malloc((n + 1) * sizeof(long long));
            free(tmphr); tmphr = (int*)malloc((n + 1) * sizeof(int));
            free(tmpc); tmpc = (int*)malloc((n + 1) * sizeof(int));
            tmpoffcap = n + 1; tmpccap = n + 1;
        }
        size_t raw_need = 0;
        for (int i = 0; i < n; i++) {
            tmpoff[i] = (long long)raw_need;
            tmphr[i] = r->vHr[i];
            tmpc[i] = r->vWr[i];           // = wclamp values
            raw_need += (size_t)r->vHr[i] * r->vWr[i] * 4;
            raw_need = (raw_need + 255) & ~(size_t)255;   // int4-safe
        }
        tmpoff[n] = (long long)raw_need;
        if (bs->pin_in.cap < raw_need && grow_pin(bs->pin_in, raw_need))
            return -3;
        unsigned char* pin = (unsigned char*)bs->pin_in.p;
        for (int i = 0; i < n; i++)
            memcpy(pin + tmpoff[i], r->src_rows[i],
                   (size_t)r->vHr[i] * r->vWr[i] * 4);
        r->src_rows.clear();
        r->src_rows.shrink_to_fit();
        if (grow_d(bs->rawpad, raw_need)) return -3;
        if (grow_d(bs->doff, (size_t)(n + 1) * sizeof(long long))) return -3;
        if (grow_d(bs->dhre, (size_t)n * sizeof(int))) return -3;
        CHECK_CUDA(cudaMemcpyAsync(bs->rawpad.p, pin, raw_need,
                                   cudaMemcpyHostToDevice, bs->stream));
        CHECK_CUDA(cudaMemcpyAsync(bs->doff.p, tmpoff,
                                   (n + 1) * sizeof(long long),
                                   cudaMemcpyHostToDevice, bs->stream));
        CHECK_CUDA(cudaMemcpyAsync(bs->dhre.p, tmphr, n * sizeof(int),
                                   cudaMemcpyHostToDevice, bs->stream));
        CHECK_CUDA(cudaMemcpyAsync(bs->wclamp.p, tmpc, n * sizeof(int),
                                   cudaMemcpyHostToDevice, bs->stream));
        pad_scatter_kernel<<<n, 256, 0, bs->stream>>>(
            (const unsigned char*)bs->rawpad.p,
            (unsigned char*)bs->rgba.p,
            (const long long*)bs->doff.p,
            (const int*)bs->wclamp.p,
            (const int*)bs->dhre.p, r->W, r->H);
        CHECK_CUDA(cudaGetLastError());
        if (getenv("PAD_DBG")) { fprintf(stderr, "s1 staged n=%d padW=%d raw=%zu" "\n", n, r->W, raw_need); fflush(stderr); }
    } else if (r->filtered) {
        // filtered-rows input: stage rows, defilter on the GPU into rgba
        size_t rstride = (size_t)W * r->bpp + 1;
        size_t one = (size_t)H * rstride;
        size_t fraw_need = (size_t)n * one;
        if (grow_d(bs->fraw, fraw_need)) return -3;
        unsigned char* pin = (unsigned char*)bs->pin_in.p;
        if (bs->pin_in.cap < fraw_need && grow_pin(bs->pin_in, fraw_need))
            return -3;
        pin = (unsigned char*)bs->pin_in.p;
        for (int i = 0; i < n; i++)
            memcpy(pin + (size_t)i * one, r->imgs[i], one);
        CHECK_CUDA(cudaMemcpyAsync(bs->fraw.p, pin, fraw_need,
                                   cudaMemcpyHostToDevice, bs->stream));
        int tdf = 128;
        int bdf = (n + tdf - 1) / tdf;
        png_defilter_kernel<<<bdf, tdf, 0, bs->stream>>>(
            (const unsigned char*)bs->fraw.p, (unsigned char*)bs->rgba.p,
            n, H, (int)rstride, W, r->bpp);
        if (r->alpha_flags) {
            if (grow_d(bs->dalpha, n)) return -3;
            int th = 128;
            int bl = (n + th - 1) / th;
            alpha_scan_kernel<<<bl, th, 0, bs->stream>>>(
                (const unsigned char*)bs->rgba.p,
                (unsigned char*)bs->dalpha.p, n, (long)H * W);
        }
    } else if (r->zc) {
        // zero-copy async path: stage straight from the caller's per-image
        // RGBA arrays (the pipeline keeps them alive until completion) —
        // the submit-thread vector copy of ~259MB/batch serialized the GPU
        // feed and cost ~15% throughput
        size_t one = (size_t)H * W * 4;
        if (bs->pin_in.cap < rgb_bytes && grow_pin(bs->pin_in, rgb_bytes))
            return -3;
        unsigned char* pin = (unsigned char*)bs->pin_in.p;
        for (int i = 0; i < n; i++)
            memcpy(pin + (size_t)i * one, r->src_rows[i], one);
        r->src_rows.clear();
        r->src_rows.shrink_to_fit();
        CHECK_CUDA(cudaMemcpyAsync(bs->rgba.p, pin, rgb_bytes,
                                   cudaMemcpyHostToDevice, bs->stream));
    } else {
    memcpy(bs->pin_in.p, h_rgba, rgb_bytes);
    // V3-step6: input H2D via the slot stream (pinned source)
    CHECK_CUDA(cudaMemcpyAsync(bs->rgba.p, bs->pin_in.p, rgb_bytes, cudaMemcpyHostToDevice, bs->stream));
    }
    if (!r->padded) {
        // uniform paths (raw RGBA + filtered) need the clamp array too —
        // the kernels always read w_clamp[b]
        if (grow_d(bs->wclamp, (size_t)n * sizeof(int))) return -3;
        static int* tmpf = nullptr; static int tmpfcap = 0;
        if (tmpfcap < n) { free(tmpf); tmpf = (int*)malloc(n * sizeof(int)); tmpfcap = n; }
        for (int i = 0; i < n; i++) tmpf[i] = r->W;
        CHECK_CUDA(cudaMemcpyAsync(bs->wclamp.p, tmpf, n * sizeof(int),
                                  cudaMemcpyHostToDevice, bs->stream));
    }

    // ---- YUV + borders + mode search ----
    {
        dim3 block(16, 16, 1);
        dim3 grid(W / 16, H / 16, n);
        rgb_to_yuv420_kernel<<<grid, block, 0, bs->stream>>>(
            (const unsigned char*)bs->rgba.p, (int)(H * W * 4),
            (short*)bs->Y.p, (short*)bs->U.p, (short*)bs->V.p, W, H, H * W, HH * HW);
    }
    {
        dim3 blk(32, 8, 1);
        dim3 grd((W + 32) / 32, (H + 8) / 8, n);      // cover 0..W inclusive
        make_borders_kernel<<<grd, blk, 0, bs->stream>>>((const short*)bs->Y.p, (short*)bs->bY.p, H, W, 127, 129);
        dim3 grd2((HW + 32) / 32, (HH + 8) / 8, n);
        make_borders_kernel<<<grd2, blk, 0, bs->stream>>>((const short*)bs->U.p, (short*)bs->bU.p, HH, HW, 127, 129);
        make_borders_kernel<<<grd2, blk, 0, bs->stream>>>((const short*)bs->V.p, (short*)bs->bV.p, HH, HW, 127, 129);

        int total = (int)nmb_all;
        int threads = 128;
        int blocks = (total + threads - 1) / threads;
        mode_search_kernel<<<blocks, threads>>>(
            (const short*)bs->bY.p, (const short*)bs->bU.p, (const short*)bs->bV.p,
            bs->fc_i16, bs->fc_uv,
            (unsigned char*)bs->i16m.p, (long long*)bs->i16score.p,
            (unsigned char*)bs->uvm.p, (int*)bs->sse4.p,
            n, mb_h, mb_w, H, W, (const int*)bs->wclamp.p);
        CHECK_CUDA(cudaGetLastError());
    }

    cudaError_t ke = cudaGetLastError();
    if (ke != cudaSuccess) return -50;
    // staging is complete: release the heap input copy now (the Req3 objects
    // live in g3_reqs forever; un-dropped 256MB inputs once accumulated
    // ~27GB across a full corpus run and froze the machine)
    r->input.clear();
    r->input.shrink_to_fit();
    r->imgs.clear();
    r->imgs.shrink_to_fit();
    return 0;
}

static int stage23_v2(Req3* r, struct DevCache* bs) {
    int n = r->n, W = r->W, H = r->H, quality = r->quality;
    short* h_y_dc = r->y_dc; short* h_y_ac = r->y_ac; short* h_uv_lv = r->uv_lv;
    unsigned char* h_is_i4 = r->is_i4; unsigned char* h_i16_mode = r->i16m;
    unsigned char* h_uv_mode = r->uvm; unsigned char* h_i4_modes = r->i4m;
    int mb_h = H / 16, mb_w = W / 16;
    int n_mb = mb_h * mb_w;
    int HH = H / 2, HW = W / 2;
    size_t nmb_all = (size_t)n * n_mb;
    size_t sse_bytes = nmb_all * 160 * sizeof(int);
    size_t out_bytes = nmb_all * (16 + 256 + 128) * sizeof(short);
    size_t ry_bytes = (size_t)n * (H + 1) * (W + 1) * sizeof(short);
    size_t rc_bytes = (size_t)n * (HH + 1) * (HW + 1) * sizeof(short);
    QuantTables qt;
    setup_quant_exact(quality, &qt);
    short* d_ydc = (short*)bs->out.p;
    short* d_yac = (short*)bs->out.p + nmb_all * 16;
    short* d_uvlv = (short*)bs->out.p + nmb_all * (16 + 256);
    cudaStreamSynchronize(bs->stream);   // stage 1 kernels complete
    // stage 2 begins: mode selection runs ON DEVICE (exact port of the CPU
    // recurrence) — the old path pulled 162MB of sse4 per batch down to the
    // host, ran a 4-thread CPU select and re-uploaded the modes, all on
    // this thread's critical path
    if (bs->dfc.cap < 1000 * sizeof(long long)) {
        if (grow_d(bs->dfc, 1000 * sizeof(long long))) return -3;
        CHECK_CUDA(cudaMemcpy(bs->dfc.p, FIXED_COSTS_I4,
                              1000 * sizeof(long long),
                              cudaMemcpyHostToDevice));
    }
    long long penalty = 1000LL * qt.y1_qavg * qt.y1_qavg;
    CHECK_CUDA(cudaMemsetAsync(bs->flags.p, 0, nmb_all * sizeof(unsigned int), bs->stream));
    select_modes_kernel<<<(n * mb_h + 32 - 1) / 32, 32, 0, bs->stream>>>(
        (const int*)bs->sse4.p, (const long long*)bs->i16score.p,
        (const unsigned char*)bs->i16m.p, (const unsigned char*)bs->uvm.p,
        (const long long*)bs->dfc.p, penalty,
        (unsigned char*)bs->is_i4.p, (unsigned char*)bs->i4m.p,
        (unsigned int*)bs->flags.p, n, mb_h, mb_w);
    CHECK_CUDA(cudaGetLastError());
    // V3-step4: recon/flag/output memsets via the slot stream
    CHECK_CUDA(cudaMemsetAsync(bs->rY.p, 0, ry_bytes, bs->stream));
    CHECK_CUDA(cudaMemsetAsync(bs->rU.p, 0, rc_bytes, bs->stream));
    CHECK_CUDA(cudaMemsetAsync(bs->rV.p, 0, rc_bytes, bs->stream));
    CHECK_CUDA(cudaMemsetAsync(bs->flags.p, 0, nmb_all * sizeof(unsigned int), bs->stream));
    // kernel skips writing all-zero DC slots (Python side uses cp.zeros for
    // y_dc) — zero the dc region or stale values leak across calls
    CHECK_CUDA(cudaMemsetAsync(bs->out.p, 0, nmb_all * 16 * sizeof(short), bs->stream));

    // ---- closed loop ----
    {
        closed_loop_kernel<<<(n * mb_h + 32 - 1) / 32, 32>>>(
            (const short*)bs->Y.p, (const short*)bs->U.p, (const short*)bs->V.p,
            (const unsigned char*)bs->is_i4.p, (const unsigned char*)bs->i16m.p,
            (const unsigned char*)bs->uvm.p, (const unsigned char*)bs->i4m.p,
            bs->q[0], bs->q[1], bs->q[2], bs->q[3], bs->q[4],
            bs->q[5], bs->q[6], bs->q[7], bs->q[8], bs->q[9],
            bs->q[10], bs->q[11], bs->q[12], bs->q[13], bs->q[14],
            bs->q[15], bs->q[16], bs->q[17],
            (const long long*)bs->tlam.p,
            d_ydc, d_yac, d_uvlv,
            (short*)bs->rY.p, (short*)bs->rU.p, (short*)bs->rV.p,
            (unsigned int*)bs->flags.p,
            n, mb_h, mb_w, H, W, (const int*)bs->wclamp.p);
        CHECK_CUDA(cudaGetLastError());
    }

    // ---- per-image reconstruction SSE (GPU quality gate) ----
    if (r->h_sse) {
        static int* tmph = nullptr; static int tmphcap = 0;
        if (tmphcap < n) { free(tmph); tmph = (int*)malloc(n * sizeof(int)); tmphcap = n; }
        for (int i = 0; i < n; i++)
            tmph[i] = r->padded ? r->vHr[i] : H;
        if (grow_d(bs->dhre23, (size_t)n * sizeof(int))) return -3;
        if (grow_d(bs->rsse, (size_t)n * 3 * sizeof(long long))) return -3;
        CHECK_CUDA(cudaMemcpyAsync(bs->dhre23.p, tmph, n * sizeof(int),
                                   cudaMemcpyHostToDevice, bs->stream));
        recon_sse_kernel<<<n, 256, 0, bs->stream>>>(
            (const short*)bs->Y.p, (const short*)bs->U.p, (const short*)bs->V.p,
            (const short*)bs->rY.p, (const short*)bs->rU.p, (const short*)bs->rV.p,
            (unsigned long long*)bs->rsse.p,
            (const int*)bs->wclamp.p, (const int*)bs->dhre23.p,
            n, H, W);
        CHECK_CUDA(cudaGetLastError());
    }

    // ---- single D2H for all levels + modes, then scatter ----
    // V3-step1: result D2H via the slot stream (async + explicit sync)
    CHECK_CUDA(cudaMemcpyAsync(bs->pin_out.p, bs->out.p, out_bytes, cudaMemcpyDeviceToHost, bs->stream));
    cudaStreamSynchronize(bs->stream);
    memcpy(h_y_dc, bs->pin_out.p, nmb_all * 16 * sizeof(short));
    memcpy(h_y_ac, (short*)bs->pin_out.p + nmb_all * 16, nmb_all * 256 * sizeof(short));
    memcpy(h_uv_lv, (short*)bs->pin_out.p + nmb_all * (16 + 256), nmb_all * 128 * sizeof(short));
    // V3-step2: direct D2H to caller buffers via the slot stream
    CHECK_CUDA(cudaMemcpyAsync(h_is_i4, bs->is_i4.p, nmb_all, cudaMemcpyDeviceToHost, bs->stream));
    CHECK_CUDA(cudaMemcpyAsync(h_i16_mode, bs->i16m.p, nmb_all, cudaMemcpyDeviceToHost, bs->stream));
    CHECK_CUDA(cudaMemcpyAsync(h_uv_mode, bs->uvm.p, nmb_all, cudaMemcpyDeviceToHost, bs->stream));
    CHECK_CUDA(cudaMemcpyAsync(h_i4_modes, bs->i4m.p, nmb_all * 16, cudaMemcpyDeviceToHost, bs->stream));
    if (r->filtered && r->alpha_flags)
        CHECK_CUDA(cudaMemcpyAsync(r->alpha_flags, bs->dalpha.p, n,
                                   cudaMemcpyDeviceToHost, bs->stream));
    if (r->h_sse)
        CHECK_CUDA(cudaMemcpyAsync(r->h_sse, bs->rsse.p,
                                   (size_t)n * 3 * sizeof(long long),
                                   cudaMemcpyDeviceToHost, bs->stream));
    cudaStreamSynchronize(bs->stream);

    // ---- padded path: gather real MB rows into the packed caller buffers ----
    if (r->padded) {
        if (getenv("PAD_DBG")) { fprintf(stderr, "s23 gather n=%d" "\n", r->n); fflush(stderr); }
        int mb_w_p = r->W / 16;
        size_t off = 0;   // packed dst base = sum of preceding images' real MBs
        for (int i = 0; i < r->n; i++) {
            int mb_h_r = r->vHr[i] / 16, mb_w_r = r->vWr[i] / 16;
            size_t nmb_r = (size_t)mb_h_r * mb_w_r;
            if (getenv("PAD_DBG")) { fprintf(stderr, "gather i=%d Wr=%d Hr=%d" "\n", i, r->vWr[i], r->vHr[i]); fflush(stderr); }
            const unsigned char* srcI4 = r->i4m + (size_t)i * ((size_t)(r->H/16) * mb_w_p) * 16;
            const unsigned char* srcIs4 = r->is_i4 + (size_t)i * ((size_t)(r->H/16) * mb_w_p);
            const unsigned char* srcI16 = r->i16m + (size_t)i * ((size_t)(r->H/16) * mb_w_p);
            const unsigned char* srcUvm = r->uvm + (size_t)i * ((size_t)(r->H/16) * mb_w_p);
            const short* srcDc = r->y_dc + (size_t)i * ((size_t)(r->H/16) * mb_w_p) * 16;
            const short* srcAc = r->y_ac + (size_t)i * ((size_t)(r->H/16) * mb_w_p) * 256;
            const short* srcUv = r->uv_lv + (size_t)i * ((size_t)(r->H/16) * mb_w_p) * 128;
            unsigned char* dstI4 = r->p_i4m + off * 16;
            unsigned char* dstIs4 = r->p_is4 + off;
            unsigned char* dstI16 = r->p_i16 + off;
            unsigned char* dstUvm = r->p_uvm + off;
            short* dstDc = r->p_ydc + off * 16;
            short* dstAc = r->p_yac + off * 256;
            short* dstUv = r->p_uvlv + off * 128;
            for (int row = 0; row < mb_h_r; row++) {
                size_t sp = (size_t)row * mb_w_p;      // padded-grid row start
                size_t dp = (size_t)row * mb_w_r;      // packed row start
                memcpy(dstDc + dp * 16, srcDc + sp * 16, (size_t)mb_w_r * 16 * 2);
                memcpy(dstAc + dp * 256, srcAc + sp * 256, (size_t)mb_w_r * 256 * 2);
                memcpy(dstUv + dp * 128, srcUv + sp * 128, (size_t)mb_w_r * 128 * 2);
                memcpy(dstIs4 + dp, srcIs4 + sp, mb_w_r);
                memcpy(dstI16 + dp, srcI16 + sp, mb_w_r);
                memcpy(dstUvm + dp, srcUvm + sp, mb_w_r);
                memcpy(dstI4 + dp * 16, srcI4 + sp * 16, (size_t)mb_w_r * 16);
            }
            off += nmb_r;
        }
        free(r->y_dc); free(r->y_ac); free(r->uv_lv);
        free(r->is_i4); free(r->i16m); free(r->uvm); free(r->i4m);
        r->own_out = 0;
    }

    return 0;
}
static std::deque<Req3*> g3_handoff;          // W1 -> W2
static std::condition_variable g3_cv_w2;
static bool g3_slot_free[2] = {true, true};
static std::condition_variable g3_cv_slot;

typedef void (__stdcall *feeder_cb_t)(void*, int);
static feeder_cb_t g_ext_cb = nullptr;
static void* g_ext_user = nullptr;
static int g_zc_submit = 0;   // submit_batch stages from caller arrays

// Pre-grow a slot cache for one batch shape. The first real batch of each
// new (n, W, H) used to pay its full cudaMalloc bill (up to ~1s of GB-size
// allocations) mid-run and stalled the interleaved main-size batches.
static int grow_shape(struct DevCache* bs, int n, int W, int H)
{
    if (H % 16 || W % 16 || n <= 0) return -1;
    int HH = H / 2, HW = W / 2;
    size_t nmb_all = (size_t)n * (H / 16) * (W / 16);
    size_t rgb_bytes = (size_t)n * H * W * 4;
    size_t c_bytes = (size_t)n * HH * HW * sizeof(short);
    size_t ry_bytes = (size_t)n * (H + 1) * (W + 1) * sizeof(short);
    size_t rc_bytes = (size_t)n * (HH + 1) * (HW + 1) * sizeof(short);
    size_t sse_bytes = nmb_all * 160 * sizeof(int);
    size_t out_bytes = nmb_all * (16 + 256 + 128) * sizeof(short);
    if (grow_d(bs->rgba, rgb_bytes)) return -3;
    if (grow_d(bs->Y, rgb_bytes / 2)) return -3;
    if (grow_d(bs->U, c_bytes)) return -3;
    if (grow_d(bs->V, c_bytes)) return -3;
    if (grow_d(bs->rY, ry_bytes)) return -3;
    if (grow_d(bs->rU, rc_bytes)) return -3;
    if (grow_d(bs->rV, rc_bytes)) return -3;
    if (grow_d(bs->bY, ry_bytes)) return -3;
    if (grow_d(bs->bU, rc_bytes)) return -3;
    if (grow_d(bs->bV, rc_bytes)) return -3;
    if (grow_d(bs->flags, nmb_all * 4)) return -3;
    if (grow_d(bs->is_i4, nmb_all)) return -3;
    if (grow_d(bs->i16m, nmb_all)) return -3;
    if (grow_d(bs->uvm, nmb_all)) return -3;
    if (grow_d(bs->i4m, nmb_all * 16)) return -3;
    if (grow_d(bs->i16score, nmb_all * 8)) return -3;
    if (grow_d(bs->sse4, sse_bytes)) return -3;
    if (grow_d(bs->out, out_bytes)) return -3;
    if (grow_d(bs->rawpad, rgb_bytes)) return -3;
    if (grow_d(bs->doff, (size_t)(n + 1) * sizeof(long long))) return -3;
    if (grow_d(bs->dhre, (size_t)n * sizeof(int))) return -3;
    if (grow_d(bs->dhre23, (size_t)n * sizeof(int))) return -3;
    if (grow_d(bs->rsse, (size_t)n * 3 * sizeof(long long))) return -3;
    if (grow_d(bs->wclamp, (size_t)n * sizeof(int))) return -3;
    if (grow_pin(bs->pin_in, rgb_bytes)) return -3;
    if (grow_pin(bs->pin_sse4, sse_bytes)) return -3;
    if (grow_pin(bs->pin_score, nmb_all * 8)) return -3;
    if (grow_pin(bs->pin_mb, nmb_all * 3)) return -3;
    if (grow_pin(bs->pin_out, out_bytes)) return -3;
    if (grow_pin(bs->pin_is4, nmb_all)) return -3;
    if (grow_pin(bs->pin_i4m, nmb_all * 16)) return -3;
    return 0;
}

extern "C" __declspec(dllexport)
int gpu_warm(int n, int W, int H)
{
    cudaSetDevice(g_device);
    init_slot_streams();
    int rc = grow_shape(&S2[0], n, W, H);
    if (rc) return rc;
    return grow_shape(&S2[1], n, W, H);
}

extern "C" __declspec(dllexport)
int set_trellis(int on)
{
    cudaSetDevice(g_device);
    int v = on ? 1 : 0;
    cudaMemcpyToSymbol(g_trellis_on, &v, sizeof(int));
    return 0;
}

extern "C" __declspec(dllexport)
int set_zc_submit(int on)
{
    g_zc_submit = on ? 1 : 0;
    return 0;
}

extern "C" __declspec(dllexport)
int set_completion_cb(feeder_cb_t cb, void* user)
{
    g_ext_cb = cb;
    g_ext_user = user;
    return 0;
}

// Free every cached buffer. The grow-only caches are the right design for a
// steady stream of same-size batches, but a many-sizes corpus walks every
// buffer up to its global max and fragments the device to a standstill
// (observed on a 16GB V100: 0 bytes free, 34s for a single-image batch).
// The pipeline calls this when free device memory runs low. Slots mid-flight
// in the async workers are skipped: slot_free gates them, and the sync cache
// `g` is only ever used by the non-pipeline entry points.
extern "C" __declspec(dllexport)
int gpu_purge()
{
    cudaSetDevice(g_device);
    std::lock_guard<std::mutex> lk(g3_mu);
    purge_cache(g);
    for (int i = 0; i < 2; i++)
        if (g3_slot_free[i]) purge_cache(S2[i]);
    return 0;
}

static void w1_loop() {
    init_slot_streams();
    cudaSetDevice(g_device);   // per-thread current device
    // the pipeline process runs ~20 more threads (decode/entropy/GPU feed)
    // on a 16-core host; without a bump these two device-feeding threads
    // get preempted mid-batch and stall batches by ~1s
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    while (g3_run) {
        Req3* r = nullptr;
        {
            std::unique_lock<std::mutex> lk(g3_mu);
            g3_cv_work.wait(lk, [] { return !g3_q.empty() || !g3_run; });
            if (g3_q.empty()) continue;
            // wait for a free slot
            g3_cv_slot.wait(lk, [] { return g3_slot_free[0] || g3_slot_free[1]; });
            r = g3_q.front(); g3_q.pop_front();
            if (g3_slot_free[0]) { g3_slot_free[0] = false; r->slot = 0; }
            else { g3_slot_free[1] = false; r->slot = 1; }
        }
        int rc = stage1_v2(r, &S2[r->slot]);
        if (rc != 0) {
            if (r->own_out) { free(r->y_dc); free(r->y_ac); free(r->uv_lv);
                free(r->is_i4); free(r->i16m); free(r->uvm); free(r->i4m);
                r->own_out = 0; }
            r->err.store(rc);
            std::lock_guard<std::mutex> lk(g3_mu);
            g3_complete.push_back(r);
            g3_slot_free[r->slot] = true;
            g3_cv_done.notify_all();
            g3_cv_slot.notify_one();
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(g3_mu);
            g3_handoff.push_back(r);
        }
        g3_cv_w2.notify_one();
    }
}

static int g_dbg_slot = -1, g_dbg_n = 0, g_dbg_H = 0, g_dbg_W = 0;

static void w2_loop() {
    cudaSetDevice(g_device);   // per-thread current device
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    while (g3_run) {
        Req3* r = nullptr;
        {
            std::unique_lock<std::mutex> lk(g3_mu);
            g3_cv_w2.wait(lk, [] { return !g3_handoff.empty() || !g3_run; });
            if (g3_handoff.empty()) continue;
            r = g3_handoff.front(); g3_handoff.pop_front();
        }
        int rc = stage23_v2(r, &S2[r->slot]);
        r->err.store(rc);
        if (getenv("RECON_DBG")) {
            g_dbg_slot = r->slot; g_dbg_n = r->n;
            g_dbg_H = r->H; g_dbg_W = r->W;
        }
        {
            std::lock_guard<std::mutex> lk(g3_mu);
            g3_complete.push_back(r);
            g3_slot_free[r->slot] = true;
        }
        if (g_ext_cb) g_ext_cb(g_ext_user, r->id);
        g3_cv_done.notify_all();
        g3_cv_slot.notify_one();
    }
}

static int submit_impl(void* const* img_ptrs, int n, int W, int H,
                       int quality, int filtered, int bpp,
                       unsigned char* alpha_flags,
                       short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                       unsigned char* h_is_i4, unsigned char* h_i16_mode,
                       unsigned char* h_uv_mode, unsigned char* h_i4_modes,
                       int padded = 0, int W_real = 0, int H_real = 0,
                       const int* w_reals = nullptr, const int* h_reals = nullptr,
                       long long* h_sse = nullptr)
{
    if (n <= 0 || !img_ptrs) return -1;
    if (H % 16 || W % 16) return -2;
    if (padded && (W_real % 16 || H_real % 16 || W_real > W || H_real > H))
        return -4;
    init_slot_streams();
    {
        static std::once_flag once;
        std::call_once(once, [] {
            g3_run = true;
            std::thread(w1_loop).detach();
            std::thread(w2_loop).detach();
        });
    }
    std::unique_ptr<Req3> up(new Req3());
    Req3* r = up.get();
    r->id = ++g3_next_id;
    if (getenv("PAD_DBG")) { fprintf(stderr, "submit n=%d pad=%d vWr=%zu\n",
        n, padded, r->vWr.size()); fflush(stderr); }
    if (padded) {
        r->src_rows.assign(img_ptrs, img_ptrs + n);
    } else if (filtered) {
        r->imgs.assign(img_ptrs, img_ptrs + n);   // W1 stages from these
    } else if (g_zc_submit) {
        // async zero-copy: W1 stages from the caller's arrays directly;
        // the caller (pipeline) holds them until poll reports completion
        r->src_rows.assign(img_ptrs, img_ptrs + n);
        r->zc = 1;
    } else {
        size_t one = (size_t)H * W * 4;
        r->input.resize((size_t)n * one);
        for (int i = 0; i < n; i++)
            memcpy(r->input.data() + (size_t)i * one, img_ptrs[i], one);
    }
    r->n = n; r->W = W; r->H = H; r->quality = quality;
    r->padded = padded; r->W_real = W_real; r->H_real = H_real;
    if (padded && w_reals && h_reals) {
        r->vWr.assign(w_reals, w_reals + n);
        r->vHr.assign(h_reals, h_reals + n);
    }
    r->filtered = filtered; r->bpp = bpp; r->alpha_flags = alpha_flags;
    if (filtered) {
        size_t one = (size_t)H * ((size_t)W * bpp + 1);
        (void)one;
    }
    if (padded) {
        // raw padded-grid outputs land in malloc scratch; stage23 gathers the
        // real MB rows into the caller's packed buffers via the p_* pointers
        size_t nmb_pad = (size_t)n * (H / 16) * (W / 16);
        r->y_dc = (short*)malloc(nmb_pad * 16 * 2);
        r->y_ac = (short*)malloc(nmb_pad * 256 * 2);
        r->uv_lv = (short*)malloc(nmb_pad * 128 * 2);
        r->is_i4 = (unsigned char*)malloc(nmb_pad);
        r->i16m = (unsigned char*)malloc(nmb_pad);
        r->uvm = (unsigned char*)malloc(nmb_pad);
        r->i4m = (unsigned char*)malloc(nmb_pad * 16);
        r->own_out = 1;
        r->p_ydc = h_y_dc; r->p_yac = h_y_ac; r->p_uvlv = h_uv_lv;
        r->p_is4 = h_is_i4; r->p_i16 = h_i16_mode;
        r->p_uvm = h_uv_mode; r->p_i4m = h_i4_modes;
    } else {
        r->y_dc = h_y_dc; r->y_ac = h_y_ac; r->uv_lv = h_uv_lv;
        r->is_i4 = h_is_i4; r->i16m = h_i16_mode; r->uvm = h_uv_mode;
        r->i4m = h_i4_modes;
    }
    r->h_sse = h_sse;
    r->err.store(0);
    {
        std::lock_guard<std::mutex> lk(g3_mu);
        g3_reqs.push_back(std::move(up));
        g3_q.push_back(r);
    }
    g3_cv_work.notify_one();
    return r->id;
}

extern "C" __declspec(dllexport)
int submit_batch(void* const* img_ptrs, int n, int W, int H, int quality,
                 short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                 unsigned char* h_is_i4, unsigned char* h_i16_mode,
                 unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    return submit_impl(img_ptrs, n, W, H, quality, 0, 0, nullptr,
                       h_y_dc, h_y_ac, h_uv_lv, h_is_i4, h_i16_mode,
                       h_uv_mode, h_i4_modes);
}

extern "C" __declspec(dllexport)
int submit_batch_filtered(void* const* row_ptrs, int n, int W, int H,
                          int bpp, int quality,
                          short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                          unsigned char* h_is_i4, unsigned char* h_i16_mode,
                          unsigned char* h_uv_mode, unsigned char* h_i4_modes,
                          unsigned char* h_alpha_flags)
{
    return submit_impl(row_ptrs, n, W, H, quality, 1, bpp, h_alpha_flags,
                       h_y_dc, h_y_ac, h_uv_lv, h_is_i4, h_i16_mode,
                       h_uv_mode, h_i4_modes);
}

// padded batch: per-image RGBA rows are W_real*H_real*4; processed on a
// zero-padded W x H grid, real-MB outputs gathered to packed buffers
extern "C" __declspec(dllexport)
int submit_batch_padded(void* const* row_ptrs, int n,
                        const int* w_reals, const int* h_reals,
                        int W_pad, int H_pad,
                        int quality,
                        short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                        unsigned char* h_is_i4, unsigned char* h_i16_mode,
                        unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    return submit_impl(row_ptrs, n, W_pad, H_pad, quality, 0, 0, nullptr,
                       h_y_dc, h_y_ac, h_uv_lv, h_is_i4, h_i16_mode,
                       h_uv_mode, h_i4_modes, 1, 0, 0, w_reals, h_reals);
}

// 8-output variants: same as above plus per-image recon SSE (n*3 u64,
// YUV-domain quantisation error of what the decoder will rebuild).
// Separate names keep the 7-arg exports ABI-stable for existing callers.
extern "C" __declspec(dllexport)
int submit_batch2(void* const* img_ptrs, int n, int W, int H, int quality,
                  short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                  unsigned char* h_is_i4, unsigned char* h_i16_mode,
                  unsigned char* h_uv_mode, unsigned char* h_i4_modes,
                  long long* h_sse)
{
    return submit_impl(img_ptrs, n, W, H, quality, 0, 0, nullptr,
                       h_y_dc, h_y_ac, h_uv_lv, h_is_i4, h_i16_mode,
                       h_uv_mode, h_i4_modes, 0, 0, 0, nullptr, nullptr,
                       h_sse);
}

extern "C" __declspec(dllexport)
int submit_batch_padded2(void* const* row_ptrs, int n,
                         const int* w_reals, const int* h_reals,
                         int W_pad, int H_pad,
                         int quality,
                         short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                         unsigned char* h_is_i4, unsigned char* h_i16_mode,
                         unsigned char* h_uv_mode, unsigned char* h_i4_modes,
                         long long* h_sse)
{
    return submit_impl(row_ptrs, n, W_pad, H_pad, quality, 0, 0, nullptr,
                       h_y_dc, h_y_ac, h_uv_lv, h_is_i4, h_i16_mode,
                       h_uv_mode, h_i4_modes, 1, 0, 0, w_reals, h_reals,
                       h_sse);
}

// debug: copy the most recent completed batch's reconstruction planes
extern "C" __declspec(dllexport)
int dump_recon(short* y_out, short* u_out, short* v_out)
{
    cudaSetDevice(g_device);
    if (g_dbg_slot < 0) return -1;
    DevCache* bs = &S2[g_dbg_slot];
    int n = g_dbg_n, H = g_dbg_H, W = g_dbg_W;
    int HH = H / 2, HW = W / 2;
    cudaMemcpy(y_out, bs->rY.p, (size_t)n * (H + 1) * (W + 1) * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(u_out, bs->rU.p, (size_t)n * (HH + 1) * (HW + 1) * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(v_out, bs->rV.p, (size_t)n * (HH + 1) * (HW + 1) * 2,
               cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" __declspec(dllexport)
int poll_batch(int block_ms, int* err_out)
{
    if (err_out) *err_out = 0;
    std::unique_lock<std::mutex> lk(g3_mu);
    if (g3_complete.empty()) {
        if (block_ms == 0) return 0;
        if (block_ms < 0) g3_cv_done.wait(lk, [] { return !g3_complete.empty(); });
        else g3_cv_done.wait_for(lk, std::chrono::milliseconds(block_ms),
                                 [] { return !g3_complete.empty(); });
        if (g3_complete.empty()) return 0;
    }
    Req3* r = g3_complete.front();
    g3_complete.pop_front();
    if (err_out) *err_out = r->err.load();
    return r->id;
}

extern "C" __declspec(dllexport)
int process_batch(const unsigned char* h_rgba, int n, int W, int H, int quality,
                  short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                  unsigned char* h_is_i4, unsigned char* h_i16_mode,
                  unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    return old_sync_body(h_rgba, n, W, H, quality, h_y_dc, h_y_ac, h_uv_lv,
                         h_is_i4, h_i16_mode, h_uv_mode, h_i4_modes);
}

extern "C" __declspec(dllexport)
int process_batch_ptrs(void* const* img_ptrs, int n, int W, int H, int quality,
                       short* h_y_dc, short* h_y_ac, short* h_uv_lv,
                       unsigned char* h_is_i4, unsigned char* h_i16_mode,
                       unsigned char* h_uv_mode, unsigned char* h_i4_modes)
{
    return old_sync_ptrs(img_ptrs, n, W, H, quality, h_y_dc, h_y_ac, h_uv_lv,
                         h_is_i4, h_i16_mode, h_uv_mode, h_i4_modes);
}
