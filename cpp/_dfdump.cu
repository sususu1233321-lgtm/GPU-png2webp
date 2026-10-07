#include <cstdio>
#include <vector>
#include "kernels.cuh"
int main() {
    unsigned char raw[] = {1, 10,20,30, 1,0,2, 5,5,5, 200,100,50};
    unsigned char* d_in; unsigned char* d_out;
    cudaMalloc(&d_in, 13); cudaMalloc(&d_out, 16);
    cudaMemcpy(d_in, raw, 13, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0xEE, 16);
    png_defilter_kernel<<<1, 32>>>(d_in, d_out, 1, 1, 13, 4, 3);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("err %s\n", cudaGetErrorString(e)); return 1; }
    unsigned char out[16];
    cudaMemcpy(out, d_out, 16, cudaMemcpyDeviceToHost);
    printf("out: ");
    for (int i = 0; i < 16; i++) printf("%d ", out[i]);
    printf("\n");
    return 0;
}
