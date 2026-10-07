// Standalone Phase-1 gate benchmark for gpu_inflate.cuh.
// Usage: test_inflate.exe <dir>   (dir holds count.txt, idat_k.bin, raw_k.bin)
// Verifies GPU inflate output bit-exactly against the CPU reference, then
// measures per-stream (n=1) and batch (n=all) throughput.
#include <cstdio>
#include <windows.h>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "gpu_inflate.cuh"

static bool read_file(const char* p, std::vector<unsigned char>* out) {
    FILE* f = fopen(p, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out->resize((size_t)n);
    if (n && fread(out->data(), 1, (size_t)n, f) != (size_t)n) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void chk(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // never lose output to a crash
    if (argc < 2) { fprintf(stderr, "usage: %s <dir>\n", argv[0]); return 1; }
    std::string dir = argv[1];
    int count = 0;
    { FILE* f = fopen((dir + "/count.txt").c_str(), "r"); if (!f) { fprintf(stderr, "no count.txt\n"); return 1; } fscanf(f, "%d", &count); fclose(f); }

    cudaDeviceProp prop;
    chk(cudaGetDeviceProperties(&prop, 0), "props");
    printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    std::vector<std::vector<unsigned char>> idat(count), raw(count);
    for (int i = 0; i < count; i++) {
        if (!read_file((dir + "/idat_" + std::to_string(i) + ".bin").c_str(), &idat[i]) ||
            !read_file((dir + "/raw_" + std::to_string(i) + ".bin").c_str(), &raw[i])) {
            fprintf(stderr, "missing file %d\n", i); return 1;
        }
    }

    size_t idat_tot = 0, raw_tot = 0;
    std::vector<int> ioff(count), ilen(count), ooff(count), oexp(count);
    for (int i = 0; i < count; i++) {
        ioff[i] = (int)idat_tot; idat_tot += idat[i].size();
        ooff[i] = (int)raw_tot;  raw_tot += raw[i].size();
        ilen[i] = (int)idat[i].size();
        oexp[i] = (int)raw[i].size();
    }

    int rep = argc > 2 ? atoi(argv[2]) : 1;
    if (rep > 1) {
        std::vector<int> ioff2, ilen2, ooff2, oexp2;
        std::vector<std::vector<unsigned char>> idat2, raw2;
        for (int r = 0; r < rep; r++)
            for (int i = 0; i < count; i++) {
                ioff2.push_back(ioff[i]); ilen2.push_back(ilen[i]);
                ooff2.push_back(ooff[i]); oexp2.push_back(oexp[i]);
                idat2.push_back(idat[i]); raw2.push_back(raw[i]);
            }
        ioff = ioff2; ilen = ilen2; ooff = ooff2; oexp = oexp2;
        idat = idat2; raw = raw2; count *= rep;
    }
    unsigned char* d_idat; int* d_ioff; int* d_ilen; unsigned char* d_fraw;
    int* d_ooff; int* d_oexp; int* d_err; unsigned char* d_ws;
    chk(cudaMalloc(&d_idat, idat_tot), "malloc idat");
    chk(cudaMalloc(&d_fraw, raw_tot), "malloc fraw");
    chk(cudaMalloc(&d_ioff, count * 4), "malloc");
    chk(cudaMalloc(&d_ilen, count * 4), "malloc");
    chk(cudaMalloc(&d_ooff, count * 4), "malloc");
    chk(cudaMalloc(&d_oexp, count * 4), "malloc");
    chk(cudaMalloc(&d_err, count * 4), "malloc");
    chk(cudaMalloc(&d_ws, count * gpuinfl::HT_WS_STRIDE), "malloc ws");
    // flatten host blobs
    std::vector<unsigned char> fidat(idat_tot), fraw_h(raw_tot);
    for (int i = 0; i < count; i++) {
        memcpy(fidat.data() + ioff[i], idat[i].data(), idat[i].size());
    }
    chk(cudaMemcpy(d_idat, fidat.data(), idat_tot, cudaMemcpyHostToDevice), "upload");
    chk(cudaMemcpy(d_ioff, ioff.data(), count * 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_ilen, ilen.data(), count * 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_ooff, ooff.data(), count * 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_oexp, oexp.data(), count * 4, cudaMemcpyHostToDevice), "up");

    // ---- correctness: full batch ----
    int rc = gpuinfl::run_inflate_batch(0, d_idat, d_ioff, d_ilen, d_fraw, d_ooff, d_oexp, d_err, d_ws, count);
    printf("launched rc=%d, polling...\n", rc); fflush(stdout);
    {
        cudaStream_t ps; cudaStreamCreate(&ps);
        int hb = -1; cudaEvent_t pev; cudaEventCreate(&pev);
        for (int t = 0; t < 300; t++) {
            cudaEventRecord(pev, ps);
            cudaMemcpyAsync(&hb, d_err, 4, cudaMemcpyDeviceToHost, ps);
            cudaEventSynchronize(pev);
            printf("t=%dms err0=%d\n", t * 200, hb); fflush(stdout);
            cudaError_t pe = cudaStreamQuery(0);
            if (pe == cudaSuccess) { printf("kernel done\n"); fflush(stdout); break; }
            if (pe != cudaErrorNotReady) { printf("stream err %s\n", cudaGetErrorString(pe)); fflush(stdout); break; }
            Sleep(200);
        }
        cudaStreamDestroy(ps); cudaEventDestroy(pev);
    }
    printf("pre-sync\n"); fflush(stdout);
    chk(cudaDeviceSynchronize(), "sync");
    printf("post-sync\n"); fflush(stdout);
    std::vector<int> err(count);
    printf("pre-errdl\n"); fflush(stdout);
    chk(cudaMemcpy(err.data(), d_err, count * 4, cudaMemcpyDeviceToHost), "down err");
    printf("post-errdl err0=%d\n", err[0]); fflush(stdout);
    printf("PREFAWDL\n"); fflush(stdout);
    chk(cudaMemcpy(fraw_h.data(), d_fraw, raw_tot, cudaMemcpyDeviceToHost), "down fraw");
    printf("POSTFAWDL\n"); fflush(stdout);
    int bad = 0;
    for (int i = 0; i < count; i++) {
        if (err[i] != 0) { printf("img %d: err %d\n", i, err[i]); bad++; }
        else if (memcmp(fraw_h.data() + ooff[i], raw[i].data(), raw[i].size()) != 0) {
            printf("img %d: BYTES DIFFER\n", i); bad++;
        }
    }
    printf("correctness: %d/%d bit-exact\n", count - bad, count);
    if (bad) return 2;

    // ---- timing helpers ----
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int R = 5;
    auto time_run = [&](const int* pioff, const int* pilen,
                        const int* pooff, const int* poexp, int n, double* ms) {
        cudaEventRecord(e0);
        for (int r = 0; r < R; r++)
            gpuinfl::run_inflate_batch(0, d_idat, pioff, pilen, d_fraw,
                                       pooff, poexp, d_err, d_ws, n);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float el; cudaEventElapsedTime(&el, e0, e1);
        *ms = el / R;
    };

    // single stream: largest image only (one thread active)
    int big = 0;
    for (int i = 1; i < count; i++) if (oexp[i] > oexp[big]) big = i;
    // build a 1-image view
    std::vector<int> ioff1 = { ioff[big] }, ilen1 = { ilen[big] };
    std::vector<int> ooff1 = { ooff[big] }, oexp1 = { oexp[big] };
    int *d_ioff1, *d_ilen1, *d_ooff1, *d_oexp1;
    chk(cudaMalloc(&d_ioff1, 4), "m"); chk(cudaMalloc(&d_ilen1, 4), "m");
    chk(cudaMalloc(&d_ooff1, 4), "m"); chk(cudaMalloc(&d_oexp1, 4), "m");
    chk(cudaMemcpy(d_ioff1, ioff1.data(), 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_ilen1, ilen1.data(), 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_ooff1, ooff1.data(), 4, cudaMemcpyHostToDevice), "up");
    chk(cudaMemcpy(d_oexp1, oexp1.data(), 4, cudaMemcpyHostToDevice), "up");
    double ms1;
    time_run(d_ioff1, d_ilen1, d_ooff1, d_oexp1, 1, &ms1);
    double mbps1 = oexp[big] / 1e6 / (ms1 / 1e3);
    printf("single stream (largest, %.1fMB raw): %.1f ms -> %.0f MB/s\n",
           oexp[big] / 1e6, ms1, mbps1);

    // median across singles: run each image alone once
    std::vector<double> rates;
    bool quick = argc > 3 && argv[3][0] == 'q';
    if (quick) printf("(skip per-image loop)\n");
    for (int i = 0; !quick && i < count; i++) {
        std::vector<int> io = { ioff[i] }, il = { ilen[i] };
        std::vector<int> oo = { ooff[i] }, oe = { oexp[i] };
        cudaMemcpy(d_ioff1, io.data(), 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ilen1, il.data(), 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ooff1, oo.data(), 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_oexp1, oe.data(), 4, cudaMemcpyHostToDevice);
        double ms; time_run(d_ioff1, d_ilen1, d_ooff1, d_oexp1, 1, &ms);
        if (ms > 0.01) rates.push_back(oexp[i] / 1e6 / (ms / 1e3));
    }
    std::sort(rates.begin(), rates.end());
    if (!rates.empty())
        printf("per-stream rates: min %.0f  p50 %.0f  max %.0f MB/s  (n=%zu)\n",
               rates.front(), rates[rates.size() / 2], rates.back(), rates.size());

    double msn;
    time_run(d_ioff, d_ilen, d_ooff, d_oexp, count, &msn);
    printf("batch n=%d: %.1f ms -> aggregate %.0f MB/s (%.0f img/s equiv 1MP)\n",
           count, msn, raw_tot / 1e6 / (msn / 1e3), count / (msn / 1e3));
    return 0;
}
