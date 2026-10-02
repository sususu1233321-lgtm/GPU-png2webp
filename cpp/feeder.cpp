// feeder.cpp -- C++ feed pipeline: file IO + PNG inflate + size grouping +
// direct submission to the GPU DLL. Replaces the 16 Python decode threads
// (their GIL-bound glue was the end-to-end cap at ~135/s).
//
// Flow per group (H, W, bpp): K worker threads read+scan+inflate files
// (in file order), a group buffer collects the filtered-row images, and a
// dedicated submit thread assembles batches and calls gpu_pipeline_v2.dll
// submit_batch_filtered. Completion callback frees the row buffers.
//
// Python polls feeder_poll() once per COMPLETED BATCH and gets an opaque
// handle; all arrays/metadata are exposed via pointer getters (zero copy).
//
// Build: cl /O2 /LD /MD feeder.cpp /Fe:feeder.dll   (links pngdec + gpu dlls at runtime)

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define NOMINMAX
#include <windows.h>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <algorithm>

// ---- pngdec.dll entry (loaded once) ----
typedef int (*png_inflate_raw_t)(const unsigned char*, int, unsigned char*,
                                 int, int*, int*, int*, int*);
typedef int (*png_meta_scan_t)(const unsigned char*, int, unsigned char*,
                               int, int*);
static png_inflate_raw_t p_inflate = nullptr;
static png_meta_scan_t p_metascan = nullptr;
static HMODULE h_pngdec = nullptr;

// ---- gpu_pipeline_v2.dll entries ----
typedef int (*submit_rgba_t)(void* const*, int, int, int, int,
                             short*, short*, short*,
                             unsigned char*, unsigned char*,
                             unsigned char*, unsigned char*);
static submit_rgba_t p_submit = nullptr;
typedef int (*submit_filtered_t)(void* const*, int, int, int, int, int,
                                 short*, short*, short*,
                                 unsigned char*, unsigned char*,
                                 unsigned char*, unsigned char*,
                                 unsigned char*);
static submit_filtered_t p_submit_f = nullptr;

static HMODULE h_gpu = nullptr;

static HMODULE load_w(const char* utf8) {
    int wl = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    wchar_t* buf = new wchar_t[wl];
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, buf, wl);
    HMODULE h = LoadLibraryW(buf);
    delete[] buf;
    return h;
}

static bool load_deps(const char* pngdec_path, const char* gpu_path) {
    static bool tried = false;
    if (tried) return p_inflate && p_submit;
    tried = true;
    h_pngdec = load_w(pngdec_path);
    if (h_pngdec) {
        p_inflate = (png_inflate_raw_t)GetProcAddress(h_pngdec, "png_inflate_raw");
        p_metascan = (png_meta_scan_t)GetProcAddress(h_pngdec, "png_meta_scan");
    }
    h_gpu = load_w(gpu_path);
    if (h_gpu) {
        p_submit = (submit_rgba_t)GetProcAddress(h_gpu, "submit_batch");
        p_submit_f = (submit_filtered_t)GetProcAddress(
            h_gpu, "submit_batch_filtered");
    }
    if (h_gpu)
        
    return p_inflate && p_submit;
}

// ---- structures ----
struct ImgJob {
    std::string path;           // utf-8
    int W = 0, H = 0, bpp = 0;
    unsigned char* rows = nullptr;   // defiltered RGBA (malloc)
    int rows_len = 0;
    int has_alpha = 0;
    unsigned char* aplane = nullptr; // alpha plane H*W (malloc, alpha imgs)
    unsigned char* meta = nullptr;   // serialized metadata blob
    int meta_len = 0;
    int ok = 0;                 // 1 = inflated fine
};

struct BatchRec {
    int req_id = 0;
    int n = 0, W = 0, H = 0, bpp = 0;
    std::vector<ImgJob*> jobs;      // owned (freed on release)
    // caller-visible output buffers (malloc'd, owned until release)
    short* y_dc = nullptr; short* y_ac = nullptr; short* uv_lv = nullptr;
    unsigned char* is_i4 = nullptr; unsigned char* i16m = nullptr;
    unsigned char* uvm = nullptr; unsigned char* i4m = nullptr;
    unsigned char* alpha = nullptr; // n bytes (flags)
    unsigned char* aplains = nullptr; // concatenated alpha planes (H*W each)
};

static std::vector<std::string> g_files;
static std::vector<int> g_order;          // indices sorted by (H,W,bpp,path)
static std::vector<ImgJob> g_jobs;        // parallel to g_files
static std::atomic<size_t> g_next{0};
static std::vector<BatchRec*> g_done;
static std::mutex g_mu;
static std::condition_variable g_cv_done;
static std::atomic<bool> g_run{false};
static std::atomic<bool> g_all_done{false};
static std::atomic<int> g_err{0};
static std::vector<int> g_unsupported;
static std::atomic<int> g_active{0};
static int g_quality = 90, g_batch_max = 96;
static size_t g_groups_left = 0;

// group bookkeeping
struct GroupKey { int H, W, bpp; };
struct GroupState {
    std::vector<ImgJob*> pending;
    size_t remaining;          // files not yet inflated
};
static std::vector<GroupState> g_groups;
static std::vector<GroupKey> g_keys;
static std::mutex g_gmu;       // protects g_groups pending vectors

static void on_batch_done(int req_id) {
    // called (from the GPU dll's worker) when a submitted batch completed
    std::lock_guard<std::mutex> lk(g_gmu);
    for (auto& gs : g_groups) {
        for (size_t i = 0; i < gs.pending.size(); i++) {
            // find the batch by req id is not tracked here; handled by feeder
        }
    }
    (void)req_id;
}

// simple registry of outstanding batches by req id
struct OutRec { int req_id; BatchRec* rec; };
static std::vector<OutRec> g_out;
static std::mutex g_omu;

static void __stdcall completion_cb(void* user, int req_id) {
    (void)user;
    BatchRec* rec = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_omu);
        for (size_t i = 0; i < g_out.size(); i++) {
            if (g_out[i].req_id == req_id) {
                rec = g_out[i].rec;
                g_out.erase(g_out.begin() + i);
                break;
            }
        }
    }
    if (!rec) return;
    // free row buffers now (W1 has staged them long ago)
    for (auto* j : rec->jobs) {
        free(j->rows); j->rows = nullptr;
        free(j->aplane); j->aplane = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_done.push_back(rec);
    }
    g_cv_done.notify_one();
}

typedef void (__stdcall *cb_t)(void*, int);

// submit any group whose pending set is full or whose files are exhausted
static void try_submit_groups(bool final) {
    std::lock_guard<std::mutex> lk(g_gmu);
    for (size_t gi = 0; gi < g_groups.size(); gi++) {
        auto& gs = g_groups[gi];
        while (gs.pending.size() >= (size_t)g_batch_max
               || (final && gs.remaining == 0 && !gs.pending.empty())) {
            int n = (int)(std::min)(gs.pending.size(), (size_t)g_batch_max);
            BatchRec* rec = new BatchRec();
            rec->n = n; rec->W = g_keys[gi].W; rec->H = g_keys[gi].H;
            rec->bpp = g_keys[gi].bpp;
            rec->jobs.assign(gs.pending.begin(), gs.pending.begin() + n);
            gs.pending.erase(gs.pending.begin(), gs.pending.begin() + n);
            size_t nmb = (size_t)n * (rec->H / 16) * (rec->W / 16);
            rec->y_dc = (short*)malloc(nmb * 16 * 2);
            rec->y_ac = (short*)malloc(nmb * 256 * 2);
            rec->uv_lv = (short*)malloc(nmb * 128 * 2);
            rec->is_i4 = (unsigned char*)malloc(nmb);
            rec->i16m = (unsigned char*)malloc(nmb);
            rec->uvm = (unsigned char*)malloc(nmb);
            rec->i4m = (unsigned char*)malloc(nmb * 16);
            rec->alpha = (unsigned char*)malloc(n);
            {
                size_t cap2 = 0;
                for (int i = 0; i < n; i++)
                    if (rec->jobs[i]->aplane) cap2 += (size_t)rec->H * rec->W;
                if (cap2) {
                    rec->aplains = (unsigned char*)malloc(cap2);
                    size_t o = 0;
                    for (int i = 0; i < n; i++) {
                        unsigned char* ap = rec->jobs[i]->aplane;
                        if (!ap) continue;
                        memcpy(rec->aplains + o, ap, (size_t)rec->H * rec->W);
                        o += (size_t)rec->H * rec->W;
                    }
                }
            }
            void** ptrs = (void**)malloc(sizeof(void*) * n);
            for (int i = 0; i < n; i++) ptrs[i] = rec->jobs[i]->rows;
            if (rec->bpp == 4) {
                // GPU defilter path; the DLL writes alpha flags into
                // rec->alpha (h_alpha_flags) during stage23
                int rid = p_submit_f(ptrs, n, rec->W, rec->H, 4, g_quality,
                                     rec->y_dc, rec->y_ac, rec->uv_lv,
                                     rec->is_i4, rec->i16m, rec->uvm,
                                     rec->i4m, rec->alpha);
                free(ptrs);
                if (rid <= 0) { g_err.store(rid ? rid : -500); }
                else {
                    rec->req_id = rid;
                    std::lock_guard<std::mutex> l2(g_omu);
                    OutRec o; o.req_id = rid; o.rec = rec;
                    g_out.push_back(o);
                }
                continue;
            }
            for (int i = 0; i < n; i++) rec->alpha[i] =
                (unsigned char)rec->jobs[i]->has_alpha;
            int rid = p_submit(ptrs, n, rec->W, rec->H, g_quality,
                               rec->y_dc, rec->y_ac, rec->uv_lv,
                               rec->is_i4, rec->i16m, rec->uvm, rec->i4m);
            free(ptrs);
            if (rid <= 0) {
                g_err.store(rid ? rid : -500);
                // mark jobs failed; move to done so Python can fall back
                for (auto* j : rec->jobs) { free(j->rows); j->rows = nullptr; j->ok = 0; }
                rec->req_id = 0;
                { std::lock_guard<std::mutex> l2(g_mu); g_done.push_back(rec); }
                g_cv_done.notify_one();
                continue;
            }
            rec->req_id = rid;
            {
                std::lock_guard<std::mutex> l2(g_omu);
                OutRec o; o.req_id = rid; o.rec = rec;
                g_out.push_back(o);
            }
        }
    }
}


static inline int paeth_c(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p;
    int pb = p > b ? p - b : b - p;
    int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

// exact port of gpuwebp.pngdec._defilter (bit-exact vs imagecodecs/Pillow)
// raw: h rows of (1+stride); out: h*W*4 RGBA (RGB expanded with 255)
static void defilter_rgba(const unsigned char* raw, int h, int stride,
                          int W, int bpp, unsigned char* out)
{
    for (int y = 0; y < h; ++y) {
        const unsigned char* rp = raw + (size_t)y * (stride + 1);
        int f = rp[0];
        const unsigned char* rv = rp + 1;
        unsigned char* r = out + (size_t)y * W * 4;
        unsigned char* up = y > 0 ? r - (size_t)W * 4 : nullptr;
        int st4 = W * 4;
        if (f == 0) {
            for (int p = 0; p < W; ++p) {
                r[p*4+0] = rv[p*bpp+0];
                r[p*4+1] = rv[p*bpp+1];
                r[p*4+2] = rv[p*bpp+2];
                r[p*4+3] = bpp == 4 ? rv[p*bpp+3] : 255;
            }
        } else if (f == 1) {
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < 4; ++c) {
                    int sc = c < bpp ? c : bpp - 1;
                    int left = p > 0 ? r[(p-1)*4 + sc] : 0;
                    int v = (c < bpp ? rv[p*bpp + c] : 255);
                    if (c == 3 && bpp == 4) left = p > 0 ? r[(p-1)*4+3] : 0;
                    r[p*4+c] = (unsigned char)(c == 3 && bpp == 3 ? 255 : v + left);
                }
        } else if (f == 2) {
            const unsigned char* up4 = up;
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < 4; ++c) {
                    int sc = c < bpp ? c : bpp - 1;
                    int u = up4 ? up4[p*4+sc] : 0;
                    int v = (c < bpp ? rv[p*bpp+c] : 255);
                    r[p*4+c] = (unsigned char)(c == 3 && bpp == 3 ? 255 : v + u);
                }
        } else if (f == 3) {
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < 4; ++c) {
                    int sc = c < bpp ? c : bpp - 1;
                    int left = p > 0 ? r[(p-1)*4+sc] : 0;
                    int u = up ? up[p*4+sc] : 0;
                    int v = (c < bpp ? rv[p*bpp+c] : 255);
                    r[p*4+c] = (unsigned char)(c == 3 && bpp == 3 ? 255 : v + ((left + u) >> 1));
                }
        } else {
            for (int p = 0; p < W; ++p)
                for (int c = 0; c < 4; ++c) {
                    int sc = c < bpp ? c : bpp - 1;
                    int left = p > 0 ? r[(p-1)*4+sc] : 0;
                    int u = up ? up[p*4+sc] : 0;
                    int ul = (up && p > 0) ? up[(p-1)*4+sc] : 0;
                    int v = (c < bpp ? rv[p*bpp+c] : 255);
                    r[p*4+c] = (unsigned char)(c == 3 && bpp == 3 ? 255 : v + paeth_c(left, u, ul));
                }
        }
        (void)st4;
    }
}

static void worker_loop() {
    while (g_run.load()) {
        size_t i = g_next.fetch_add(1);
        if (i >= g_order.size()) break;
        int fi = g_order[i];
        ImgJob& j = g_jobs[fi];
        // read file (utf-8 -> wide)
        int wl = MultiByteToWideChar(CP_UTF8, 0, j.path.c_str(), -1, nullptr, 0);
        std::wstring wp(wl, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, j.path.c_str(), -1, &wp[0], wl);
        HANDLE hf = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, 0, nullptr);
        if (hf == INVALID_HANDLE_VALUE) { j.ok = 0; continue; }
        size_t sz = GetFileSize(hf, nullptr);
        std::vector<unsigned char> buf(sz);
        DWORD rd = 0;
        ReadFile(hf, buf.data(), (DWORD)sz, &rd, nullptr);
        CloseHandle(hf);
        if (rd != sz) { j.ok = 0; continue; }
        // meta scan
        int ml = 0;
        if (p_metascan) {
            j.meta = (unsigned char*)malloc(4 << 20);
            if (p_metascan(buf.data(), (int)sz, j.meta, 4 << 20, &ml) == 0)
                j.meta_len = ml;
            else { free(j.meta); j.meta = nullptr; j.meta_len = 0; }
        }
        // inflate
        size_t rstride = (size_t)j.W * j.bpp + 1;
        size_t need = (size_t)j.H * rstride;
        j.rows = (unsigned char*)malloc(need);
        int o, W2, H2, B2;
        int r = p_inflate(buf.data(), (int)sz, j.rows, (int)need,
                          &o, &W2, &H2, &B2);
        if (r != 0) { free(j.rows); j.rows = nullptr; j.ok = 0; continue; }
        {
            // CPU defilter for ALL inputs (bit-exact; the GPU kernel is
            // one-thread-per-image and too slow, and bpp=3 has a defect)
            size_t rgba_need = (size_t)j.H * j.W * 4;
            unsigned char* rgba = (unsigned char*)malloc(rgba_need);
            defilter_rgba(j.rows, j.H, j.W * j.bpp, j.W, j.bpp, rgba);
            free(j.rows);
            j.rows = rgba;
            j.rows_len = (int)rgba_need;
            j.has_alpha = 0;
            if (0) { j.has_alpha = -1; }   // keep -1 semantics unused
        }
        j.ok = 1;
        // enqueue into its group
        {
            std::lock_guard<std::mutex> lk(g_gmu);
            for (size_t gi = 0; gi < g_keys.size(); gi++) {
                if (g_keys[gi].H == j.H && g_keys[gi].W == j.W
                    && g_keys[gi].bpp == j.bpp) {
                    g_groups[gi].pending.push_back(&j);
                    g_groups[gi].remaining--;
                    break;
                }
            }
        }
        try_submit_groups(false);
    }
    if (g_active.fetch_sub(1) == 1) {
        try_submit_groups(true);          // last thread: flush partial groups
        g_all_done.store(true);
        g_cv_done.notify_all();
    }
}

// register the completion callback with the gpu dll (dynamic)
typedef int (*setcb_t)(void (__stdcall*)(void*, int), void*);
static setcb_t p_setcb = nullptr;
typedef int (*loadgpu_t)(const char*);

extern "C" __declspec(dllexport)
int feeder_start_d(const char* files_blob, int n_files, int quality,
                   int batch_max, int nthreads,
                   const char* pngdec_path, const char* gpu_path, int device);

extern "C" __declspec(dllexport)
int feeder_start(const char* files_blob, int n_files, int quality,
                 int batch_max, int nthreads,
                 const char* pngdec_path, const char* gpu_path)
{
    return feeder_start_d(files_blob, n_files, quality, batch_max, nthreads,
                          pngdec_path, gpu_path, 0);
}

extern "C" __declspec(dllexport)
int feeder_start_d(const char* files_blob, int n_files, int quality,
                   int batch_max, int nthreads,
                   const char* pngdec_path, const char* gpu_path, int device)
{
    if (!load_deps(pngdec_path, gpu_path)) return -1;
    {
        typedef int (*setdev_t)(int);
        setdev_t psd = (setdev_t)GetProcAddress(h_gpu, "gpu_set_device");
        if (psd && psd(device) != 0) return -3;
    }
    p_setcb = (setcb_t)GetProcAddress(h_gpu, "set_completion_cb");
    if (p_setcb) p_setcb(completion_cb, nullptr);
    else return -2;   // gpu dll too old: no callback support
    g_quality = quality;
    g_batch_max = batch_max > 0 ? batch_max : 96;
    // split blob (NUL-separated utf-8)
    g_files.clear();
    const char* p = files_blob;
    for (int i = 0; i < n_files; i++) {
        g_files.push_back(std::string(p));
        p += strlen(p) + 1;
    }
    g_jobs.resize(g_files.size());
    g_keys.clear(); g_groups.clear(); g_order.clear();
    std::vector<int> unsupported;
    // header scan
    for (size_t i = 0; i < g_files.size(); i++) {
        ImgJob& j = g_jobs[i];
        j.path = g_files[i];
        int wl = MultiByteToWideChar(CP_UTF8, 0, j.path.c_str(), -1, nullptr, 0);
        std::wstring wp(wl, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, j.path.c_str(), -1, &wp[0], wl);
        HANDLE hf = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, 0, nullptr);
        if (hf == INVALID_HANDLE_VALUE) { unsupported.push_back((int)i); continue; }
        unsigned char head[33];
        DWORD rd = 0;
        ReadFile(hf, head, 33, &rd, nullptr);
        CloseHandle(hf);
        if (rd < 33) { unsupported.push_back((int)i); continue; }
        static const unsigned char magic[8] = {137,80,78,71,13,10,26,10};
        if (memcmp(head, magic, 8) != 0 || memcmp(head + 12, "IHDR", 4) != 0) {
            unsupported.push_back((int)i); continue;
        }
        unsigned w = (head[16]<<24)|(head[17]<<16)|(head[18]<<8)|head[19];
        unsigned h = (head[20]<<24)|(head[21]<<16)|(head[22]<<8)|head[23];
        unsigned bd = head[24], ct = head[25], il = head[28];
        if (bd != 8 || (ct != 2 && ct != 6) || il != 0
            || w % 16 || h % 16 || w > 30000 || h > 30000
            || (h % 2) || (w % 2)) {
            unsupported.push_back((int)i); continue;
        }
        j.W = (int)w; j.H = (int)h; j.bpp = ct == 6 ? 4 : 3;
        GroupKey k{(int)h, (int)w, j.bpp};
        bool found = false;
        for (size_t gi = 0; gi < g_keys.size(); gi++)
            if (g_keys[gi].H == k.H && g_keys[gi].W == k.W && g_keys[gi].bpp == k.bpp) {
                g_groups[gi].remaining++; found = true; break;
            }
        if (!found) {
            GroupKey nk{k.H, k.W, k.bpp};
            g_keys.push_back(nk);
            GroupState gs; gs.remaining = 1;
            g_groups.push_back(gs);
        }
    }
    // order: group-major, path order inside groups
    std::vector<std::vector<int>> byk(g_keys.size());
    for (size_t i = 0; i < g_jobs.size(); i++) {
        if (!g_jobs[i].W) continue;
        for (size_t gi = 0; gi < g_keys.size(); gi++)
            if (g_keys[gi].H == g_jobs[i].H && g_keys[gi].W == g_jobs[i].W
                && g_keys[gi].bpp == g_jobs[i].bpp) { byk[gi].push_back((int)i); break; }
    }
    for (auto& v : byk) for (int i : v) g_order.push_back(i);
    g_unsupported.swap(unsupported);
    g_next.store(0);
    g_done.clear();
    g_err.store(0);
    g_all_done.store(false);
    g_run.store(true);
    int nt = nthreads > 0 ? nthreads : 12;
    g_active.store(nt);
    for (int t = 0; t < nt; t++) std::thread(worker_loop).detach();
    return 0;
}

extern "C" __declspec(dllexport)
int feeder_unsupported_count() { return (int)g_unsupported.size(); }

extern "C" __declspec(dllexport)
void feeder_unsupported_get(int* out_ints) {
    for (size_t i = 0; i < g_unsupported.size(); i++) out_ints[i] = g_unsupported[i];
}

extern "C" __declspec(dllexport)
void* feeder_poll(int block_ms)   // returns BatchRec* handle or NULL
{
    std::unique_lock<std::mutex> lk(g_mu);
    if (g_done.empty()) {
        if (block_ms == 0) return nullptr;
        if (block_ms < 0)
            g_cv_done.wait(lk, [] { return !g_done.empty(); });
        else
            g_cv_done.wait_for(lk, std::chrono::milliseconds(block_ms),
                               [] { return !g_done.empty(); });
        if (g_done.empty()) return nullptr;
    }
    BatchRec* rec = g_done.front();
    g_done.erase(g_done.begin());
    return rec;
}

extern "C" __declspec(dllexport)
void feeder_batch_info(void* handle, int* n, int* W, int* H, int* bpp,
                       int* req_id)
{
    BatchRec* r = (BatchRec*)handle;
    *n = r->n; *W = r->W; *H = r->H; *bpp = r->bpp; *req_id = r->req_id;
}

extern "C" __declspec(dllexport)
void* feeder_batch_ptr(void* handle, int which)
{
    BatchRec* r = (BatchRec*)handle;
    switch (which) {
    case 0: return r->y_dc;
    case 1: return r->y_ac;
    case 2: return r->uv_lv;
    case 3: return r->is_i4;
    case 4: return r->i16m;
    case 5: return r->uvm;
    case 6: return r->i4m;
    case 7: return r->alpha;
    }
    return nullptr;
}

extern "C" __declspec(dllexport)
int feeder_batch_nok(void* handle)   // count of jobs that failed (rows null)
{
    BatchRec* r = (BatchRec*)handle;
    int c = 0;
    for (auto* j : r->jobs) if (!j->ok) c++;
    return c;
}

// paths blob: NUL-separated utf-8; returns total bytes copied into dst
extern "C" __declspec(dllexport)
int feeder_batch_paths(void* handle, char* dst, int cap)
{
    BatchRec* r = (BatchRec*)handle;
    int off = 0;
    for (auto* j : r->jobs) {
        int l = (int)j->path.size() + 1;
        if (off + l > cap) return -1;
        memcpy(dst + off, j->path.c_str(), l);
        off += l;
    }
    return off;
}

// meta blob: concatenated per-image (u32 len + bytes) of png_meta_scan output
extern "C" __declspec(dllexport)
int feeder_batch_meta(void* handle, unsigned char* dst, int cap)
{
    BatchRec* r = (BatchRec*)handle;
    int off = 0;
    for (auto* j : r->jobs) {
        int l = j->meta_len;
        int total = 4 + l;
        if (off + total > cap) return -1;
        unsigned lb[4] = {(unsigned)l, (unsigned)(l >> 8),
                          (unsigned)(l >> 16), (unsigned)(l >> 24)};
        memcpy(dst + off, lb, 4); off += 4;
        if (l) { memcpy(dst + off, j->meta, l); off += l; }
    }
    return off;
}

extern "C" __declspec(dllexport)
void feeder_batch_release(void* handle)
{
    BatchRec* r = (BatchRec*)handle;
    free(r->aplains);
    for (auto* j : r->jobs) {
        free(j->rows);
        free(j->aplane);
        free(j->meta);
        j->rows = nullptr; j->meta = nullptr;
    }
    free(r->y_dc); free(r->y_ac); free(r->uv_lv);
    free(r->is_i4); free(r->i16m); free(r->uvm); free(r->i4m);
    free(r->alpha);
    delete r;
}

extern "C" __declspec(dllexport)
int feeder_busy()      // 1 while inflating or batches outstanding
{
    size_t infl = g_next.load();
    std::lock_guard<std::mutex> lk(g_mu);
    std::lock_guard<std::mutex> lk2(g_omu);
    return (infl < g_order.size()) || !g_done.empty() || !g_out.empty() ? 1 : 0;
}

// debug state: returns order_size, next, done, out, unsupported, active
extern "C" __declspec(dllexport)
void feeder_dbg(int* o) {
    std::lock_guard<std::mutex> l1(g_mu);
    std::lock_guard<std::mutex> l2(g_omu);
    o[0] = (int)g_order.size();
    o[1] = (int)g_next.load();
    o[2] = (int)g_done.size();
    o[3] = (int)g_out.size();
    o[4] = (int)g_unsupported.size();
    o[5] = (int)g_active.load();
    o[6] = (int)g_groups.size();
}

// alpha plane pointer for image k (null if opaque); needs the job order
extern "C" __declspec(dllexport)
void* feeder_batch_aplane(void* handle, int k)
{
    BatchRec* r = (BatchRec*)handle;
    if (!r->aplains || !r->alpha || !r->alpha[k]) return nullptr;
    size_t o = 0;
    for (int i = 0; i < k; i++)
        if (r->alpha[i]) o += (size_t)r->H * r->W;
    return r->aplains + o;
}
