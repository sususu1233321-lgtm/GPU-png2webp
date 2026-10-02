/* Standalone: libwebp v1.5.0 FTransform_C / FTransformWHT_C / QuantizeBlock_C
   copied verbatim. stdin: 16 ints (residual 4x4 raster). prints coeffs.
   With 2nd arg "wht": 256 ints (16 blocks x 16 coeffs raster) -> 16 wht coeffs.
   With 3rd arg quant q0 q1 bias0 bias1: also quantized levels (zigzag). */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#define WEBP_RESTRICT

static const uint8_t kZigzag[16] = {0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15};

static void FTransform_C(const uint8_t* WEBP_RESTRICT src,
                          const uint8_t* WEBP_RESTRICT ref,
                          int16_t* WEBP_RESTRICT out) {
  int i;
  int tmp[16];
  for (i = 0; i < 4; ++i, src += 4, ref += 4) {
    const int d0 = src[0] - ref[0];
    const int d1 = src[1] - ref[1];
    const int d2 = src[2] - ref[2];
    const int d3 = src[3] - ref[3];
    const int a0 = (d0 + d3);
    const int a1 = (d1 + d2);
    const int a2 = (d1 - d2);
    const int a3 = (d0 - d3);
    tmp[0 + i * 4] = (a0 + a1) * 8;
    tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
    tmp[2 + i * 4] = (a0 - a1) * 8;
    tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 +  937) >> 9;
  }
  for (i = 0; i < 4; ++i) {
    const int a0 = (tmp[0 + i] + tmp[12 + i]);
    const int a1 = (tmp[4 + i] + tmp[ 8 + i]);
    const int a2 = (tmp[4 + i] - tmp[ 8 + i]);
    const int a3 = (tmp[0 + i] - tmp[12 + i]);
    out[0 + i] = (a0 + a1 + 7) >> 4;
    out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0);
    out[8 + i] = (a0 - a1 + 7) >> 4;
    out[12+ i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16);
  }
}

static void FTransformWHT_C(const int16_t* WEBP_RESTRICT in,
                            int16_t* WEBP_RESTRICT out) {
  int32_t tmp[16];
  int i;
  for (i = 0; i < 4; ++i, in += 64) {
    const int a0 = (in[0 * 16] + in[2 * 16]);
    const int a1 = (in[1 * 16] + in[3 * 16]);
    const int a2 = (in[1 * 16] - in[3 * 16]);
    const int a3 = (in[0 * 16] - in[2 * 16]);
    tmp[0 + i * 4] = a0 + a1;
    tmp[1 + i * 4] = a3 + a2;
    tmp[2 + i * 4] = a3 - a2;
    tmp[3 + i * 4] = a0 - a1;
  }
  for (i = 0; i < 4; ++i) {
    const int a0 = (tmp[0 + i] + tmp[8 + i]);
    const int a1 = (tmp[4 + i] + tmp[12+ i]);
    const int a2 = (tmp[4 + i] - tmp[12+ i]);
    const int a3 = (tmp[0 + i] - tmp[8 + i]);
    const int b0 = a0 + a1;
    const int b1 = a3 + a2;
    const int b2 = a3 - a2;
    const int b3 = a0 - a1;
    out[ 0 + i] = b0 >> 1;
    out[ 4 + i] = b1 >> 1;
    out[ 8 + i] = b2 >> 1;
    out[12 + i] = b3 >> 1;
  }
}

#define QFIX 17
#define BIAS(b)  ((b) << (QFIX - 8))
static int QUANTDIV(uint32_t n, uint32_t iQ, uint32_t B) {
  return (int)((n * iQ + B) >> QFIX);
}
#define MAX_LEVEL 2047

static int QuantizeBlock_C(int16_t in[16], int16_t out[16],
                           long q0, long q1, long bias0, long bias1) {
  long q[16], iq[16], bias[16];
  int last = -1, n;
  for (n = 0; n < 16; ++n) {
    q[n] = n ? q1 : q0;
    iq[n] = (1L << QFIX) / q[n];
    bias[n] = (n ? bias1 : bias0) << (QFIX - 8);
  }
  for (n = 0; n < 16; ++n) {
    const int j = kZigzag[n];
    const int sign = (in[j] < 0);
    const uint32_t coeff = (sign ? -in[j] : in[j]);
    if (coeff > 0) {   /* no zthresh sharpening for simplicity */
      const uint32_t Q = q[j];
      const uint32_t iQ = iq[j];
      const uint32_t B = bias[j];
      int level = QUANTDIV(coeff, iQ, B);
      if (level > MAX_LEVEL) level = MAX_LEVEL;
      if (sign) level = -level;
      out[n] = level;
      if (level) last = n;
    } else {
      out[n] = 0;
    }
  }
  return (last >= 0);
}

int main(int argc, char** argv) {
  if (strcmp(argv[1], "fdct") == 0) {
    uint8_t src[16], ref[16] = {0};
    int16_t out[16];
    for (int i = 0; i < 16; ++i) { int v; scanf("%d", &v); src[i] = (uint8_t)v; }
    FTransform_C(src, ref, out);
    for (int i = 0; i < 16; ++i) printf("%d ", out[i]);
    printf("\n");
  } else if (strcmp(argv[1], "wht") == 0) {
    int16_t in[256], out[16];
    for (int i = 0; i < 256; ++i) { int v; scanf("%d", &v); in[i] = (int16_t)v; }
    FTransformWHT_C(in, out);
    for (int i = 0; i < 16; ++i) printf("%d ", out[i]);
    printf("\n");
  } else if (strcmp(argv[1], "quant") == 0) {
    int16_t in[16], out[16];
    long q0 = atol(argv[2]), q1 = atol(argv[3]);
    long b0 = atol(argv[4]), b1 = atol(argv[5]);
    for (int i = 0; i < 16; ++i) { int v; scanf("%d", &v); in[i] = (int16_t)v; }
    QuantizeBlock_C(in, out, q0, q1, b0, b1);
    for (int i = 0; i < 16; ++i) printf("%d ", out[i]);
    printf("\n");
  }
  return 0;
}
