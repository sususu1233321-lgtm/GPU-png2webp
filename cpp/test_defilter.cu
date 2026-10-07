// Standalone png_defilter_kernel test: idat-free, feeds CPU-inflated rows
// from _deflt/in.bin (raw rows) + ref RGBA (ref.bin), W H bpp argv.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.cuh"

static bool rf(const char* p, std::vector<unsigned char>* o) {
    FILE* f = fopen(p, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    o->resize((size_t)n);
    if (n && fread(o->data(), 1, (size_t)n, f) != (size_t)n) { fclose(f); return false; }
    fclose(f); return true;
}
static void chk(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    // argv: dir W H bpp
    const char* dir = argv[1];
    int W = atoi(argv[2]), H = atoi(argv[3]), bpp = atoi(argv[4]);
    std::vector<unsigned char> raw, ref;
    if (!rf((std::string(dir) + "/in.bin").c_str(), &raw)) { fprintf(stderr, "no in.bin\n"); return 1; }
    if (!rf((std::string(dir) + "/ref.bin").c_str(), &ref)) { fprintf(stderr, "no ref.bin\n"); return 1; }
    size_t rstride = (size_t)W * bpp + 1;
    size_t one = (size_t)H * rstride;
    if (raw.size() < one) { fprintf(stderr, "in.bin %zu < %zu\n", raw.size(), one); return 1; }
    if (ref.size() < (size_t)H * W * 4) { fprintf(stderr, "ref.bin too small\n"); return 1; }

    unsigned char *d_in, *d_out;
    chk(cudaMalloc(&d_in, one), "in");
    chk(cudaMalloc(&d_out, (size_t)H * W * 4), "out");
    chk(cudaMemcpy(d_in, raw.data(), one, cudaMemcpyHostToDevice), "up");
    chk(cudaMemset(d_out, 0xEE, (size_t)H * W * 4), "init");
    png_defilter_kernel<<<1, 128>>>(d_in, d_out, 1, H, (int)rstride, W, bpp);
    chk(cudaGetLastError(), "launch");
    chk(cudaDeviceSynchronize(), "sync");
    std::vector<unsigned char> out((size_t)H * W * 4);
    chk(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost), "down");
    long bad = 0;
    for (size_t i = 0; i < out.size(); i++) if (out[i] != ref[i]) bad++;
    printf("W=%d H=%d bpp=%d: %ld/%zu bytes differ\n", W, H, bpp, bad, out.size());
    if (bad) {
        // first diff pixel
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                size_t p = ((size_t)y * W + x) * 4;
                if (memcmp(&out[p], &ref[p], 4) != 0) {
                    printf("first diff pixel (%d,%d): gpu [%d %d %d %d] ref [%d %d %d %d]\n",
                           x, y, out[p], out[p+1], out[p+2], out[p+3],
                           ref[p], ref[p+1], ref[p+2], ref[p+3]);
                    y = H; break;
                }
            }
        return 2;
    }
    printf("bit-exact\n");
    return 0;
}
