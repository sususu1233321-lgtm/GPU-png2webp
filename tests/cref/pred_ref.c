/* libwebp v1.5.0 intra prediction (decoder versions = ground truth),
   verbatim from src/dsp/dec.c. argv: mode blocktype; stdin: 13 edge bytes
   for 4x4 (L0 L1 L2 L3 X T0..T7), or 33 for 16x16 (left16 top16 X), or 17+17
   for chroma 8x8. prints predicted block raster. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define BPS 32
#define WEBP_RESTRICT

static uint8_t clip_8b(int v) {
  return (!(v & ~0xff)) ? v : (v < 0) ? 0 : 255;
}
static uint8_t clip1[255 + 510 + 1];
static int tables_ok = 0;
static void InitTables(void) {
  if (!tables_ok) {
    int i;
    for (i = -255; i <= 255 + 255; ++i) clip1[255 + i] = clip_8b(i);
    tables_ok = 1;
  }
}

#define DST(x, y) dst[(x) + (y) * BPS]
#define AVG3(a, b, c) ((uint8_t)(((a) + 2 * (b) + (c) + 2) >> 2))
#define AVG2(a, b) (((a) + (b) + 1) >> 1)

static void VE4(uint8_t* dst, const uint8_t* top) {
  const uint8_t vals[4] = {
    AVG3(top[-1], top[0], top[1]), AVG3(top[0], top[1], top[2]),
    AVG3(top[1], top[2], top[3]), AVG3(top[2], top[3], top[4]) };
  for (int i = 0; i < 4; ++i) memcpy(dst + i * BPS, vals, 4);
}
static void HE4(uint8_t* dst, const uint8_t* top) {
  const int A = dst[-1 - BPS], B = dst[-1], C = dst[-1 + BPS],
            D = dst[-1 + 2 * BPS], E = dst[-1 + 3 * BPS];
  for (int i = 0; i < 4; ++i) {
    memset(dst + i * BPS, (AVG3(A + i * 0, 0, 0)), 0);  /* placeholder */
  }
  memset(dst + 0 * BPS, AVG3(A, B, C), 4);
  memset(dst + 1 * BPS, AVG3(B, C, D), 4);
  memset(dst + 2 * BPS, AVG3(C, D, E), 4);
  memset(dst + 3 * BPS, AVG3(D, E, E), 4);
}
static void DC4(uint8_t* dst, const uint8_t* top) {
  uint32_t dc = 4;
  for (int i = 0; i < 4; ++i) dc += top[i] + dst[-1 + i * BPS];
  dc >>= 3;
  for (int i = 0; i < 4; ++i) memset(dst + i * BPS, dc, 4);
}
static void RD4(uint8_t* dst, const uint8_t* top) {
  const int I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS], K = dst[-1 + 2 * BPS],
            L = dst[-1 + 3 * BPS], X = dst[-1 - BPS];
  const int A = top[0], B = top[1], C = top[2], D = top[3];
  DST(0, 3) = AVG3(J, K, L);
  DST(0, 2) = DST(1, 3) = AVG3(I, J, K);
  DST(0, 1) = DST(1, 2) = DST(2, 3) = AVG3(X, I, J);
  DST(0, 0) = DST(1, 1) = DST(2, 2) = DST(3, 3) = AVG3(A, X, I);
  DST(1, 0) = DST(2, 1) = DST(3, 2) = AVG3(B, A, X);
  DST(2, 0) = DST(3, 1) = AVG3(C, B, A);
  DST(3, 0) = AVG3(D, C, B);
}
static void LD4(uint8_t* dst, const uint8_t* top) {
  const int A = top[0], B = top[1], C = top[2], D = top[3],
            E = top[4], F = top[5], G = top[6], H = top[7];
  DST(0, 0) = AVG3(A, B, C);
  DST(1, 0) = DST(0, 1) = AVG3(B, C, D);
  DST(2, 0) = DST(1, 1) = DST(0, 2) = AVG3(C, D, E);
  DST(3, 0) = DST(2, 1) = DST(1, 2) = DST(0, 3) = AVG3(D, E, F);
  DST(3, 1) = DST(2, 2) = DST(1, 3) = AVG3(E, F, G);
  DST(3, 2) = DST(2, 3) = AVG3(F, G, H);
  DST(3, 3) = AVG3(G, H, H);
}
static void VR4(uint8_t* dst, const uint8_t* top) {
  const int I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS], K = dst[-1 + 2 * BPS];
  const int X = dst[-1 - BPS];
  const int A = top[0], B = top[1], C = top[2], D = top[3];
  DST(0, 0) = DST(1, 2) = AVG2(X, A);
  DST(1, 0) = DST(2, 2) = AVG2(A, B);
  DST(2, 0) = DST(3, 2) = AVG2(B, C);
  DST(3, 0) = AVG2(C, D);
  DST(0, 3) = AVG3(K, J, I);
  DST(0, 2) = AVG3(J, I, X);
  DST(0, 1) = DST(1, 3) = AVG3(I, X, A);
  DST(1, 1) = DST(2, 3) = AVG3(X, A, B);
  DST(2, 1) = DST(3, 3) = AVG3(A, B, C);
  DST(3, 1) = AVG3(B, C, D);
}
static void VL4(uint8_t* dst, const uint8_t* top) {
  const int A = top[0], B = top[1], C = top[2], D = top[3],
            E = top[4], F = top[5], G = top[6], H = top[7];
  DST(0, 0) = AVG2(A, B);
  DST(1, 0) = DST(0, 2) = AVG2(B, C);
  DST(2, 0) = DST(1, 2) = AVG2(C, D);
  DST(3, 0) = DST(2, 2) = AVG2(D, E);
  DST(0, 1) = AVG3(A, B, C);
  DST(1, 1) = DST(0, 3) = AVG3(B, C, D);
  DST(2, 1) = DST(1, 3) = AVG3(C, D, E);
  DST(3, 1) = DST(2, 3) = AVG3(D, E, F);
  DST(3, 2) = AVG3(E, F, G);
  DST(3, 3) = AVG3(F, G, H);
}
static void HU4(uint8_t* dst, const uint8_t* top) {
  const int I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS], K = dst[-1 + 2 * BPS],
            L = dst[-1 + 3 * BPS];
  DST(0, 0) = AVG2(I, J);
  DST(2, 0) = DST(0, 1) = AVG2(J, K);
  DST(2, 1) = DST(0, 2) = AVG2(K, L);
  DST(1, 0) = AVG3(I, J, K);
  DST(3, 0) = DST(1, 1) = AVG3(J, K, L);
  DST(3, 1) = DST(1, 2) = AVG3(K, L, L);
  DST(3, 2) = DST(2, 2) = DST(0, 3) = DST(1, 3) = DST(2, 3) = DST(3, 3) = L;
}
static void HD4(uint8_t* dst, const uint8_t* top) {
  const int I = dst[-1 + 0 * BPS], J = dst[-1 + 1 * BPS], K = dst[-1 + 2 * BPS],
            L = dst[-1 + 3 * BPS];
  const int X = dst[-1 - BPS];
  const int A = top[0], B = top[1], C = top[2];
  DST(0, 0) = DST(2, 1) = AVG2(I, X);
  DST(0, 1) = DST(2, 2) = AVG2(J, I);
  DST(0, 2) = DST(2, 3) = AVG2(K, J);
  DST(0, 3) = AVG2(L, K);
  DST(3, 0) = AVG3(A, B, C);
  DST(2, 0) = AVG3(X, A, B);
  DST(1, 0) = DST(3, 1) = AVG3(I, X, A);
  DST(1, 1) = DST(3, 2) = AVG3(J, I, X);
  DST(1, 2) = DST(3, 3) = AVG3(K, J, I);
  DST(1, 3) = AVG3(L, K, J);
}
static void TM4(uint8_t* dst, const uint8_t* top) {
  const uint8_t* clip = clip1 + 255 - top[-1];
  for (int y = 0; y < 4; ++y) {
    const uint8_t* clip_table = clip + dst[-2 - y + y * 0];
    /* left[y] is at dst[-1 + y*BPS] */
    clip_table = clip + dst[-1 + y * BPS];
    for (int x = 0; x < 4; ++x) dst[x + y * BPS] = clip_table[top[x]];
  }
}

/* 16x16 / 8x8 */
static void TM(uint8_t* dst, int size) {
  const uint8_t* top = dst - BPS;
  const uint8_t* clip0 = clip1 + 255 - top[-1];
  for (int y = 0; y < size; ++y) {
    const uint8_t* clip = clip0 + dst[-1 + y * BPS];
    for (int x = 0; x < size; ++x) dst[x + y * BPS] = clip[top[x]];
  }
}
static void VE(uint8_t* dst, int size) {
  for (int j = 0; j < size; ++j) memcpy(dst + j * BPS, dst - BPS, size);
}
static void HE(uint8_t* dst, int size) {
  for (int j = 16; j > 0; --j) { memset(dst, dst[-1], size); dst += BPS; }
}
static void Put(int v, uint8_t* dst, int size) {
  for (int j = 0; j < size; ++j) memset(dst + j * BPS, v, size);
}
static void DC(uint8_t* dst, int size, int round, int shift) {
  int DCv = round;
  for (int j = 0; j < size; ++j) DCv += dst[-1 + j * BPS] + dst[j - BPS];
  Put(DCv >> shift, dst, size);
}

int main(int argc, char** argv) {
  InitTables();
  static uint8_t buf[32 * 32];
  memset(buf, 0xAA, sizeof(buf));
  uint8_t* dst = buf + 2 * BPS + 2;
  int mode = atoi(argv[1]);
  const char* type = argv[2];   /* "i4", "i16", "c8" */
  int size = strcmp(type, "i4") == 0 ? 4 : (strcmp(type, "i16") == 0 ? 16 : 8);
  /* stdin: L rows (size), X, top (size + (i4? 4 : 0)) */
  int nleft = size, ntop = size + (size == 4 ? 4 : 0);
  for (int i = 0; i < nleft; ++i) { int v; scanf("%d", &v); dst[-1 + i * BPS] = (uint8_t)v; }
  { int v; scanf("%d", &v); dst[-1 - BPS] = (uint8_t)v; }
  for (int i = 0; i < ntop; ++i) { int v; scanf("%d", &v); dst[i - BPS] = (uint8_t)v; }
  if (size == 4) {
    /* T0..T3 are dst[-BPS..], TR at dst[4-BPS..7-BPS] — already placed */
    switch (mode) {
      case 0: DC4(dst, dst - BPS); break;
      case 1: TM4(dst, dst - BPS); break;
      case 2: VE4(dst, dst - BPS); break;
      case 3: HE4(dst, dst - BPS); break;
      case 4: RD4(dst, dst - BPS); break;
      case 5: VR4(dst, dst - BPS); break;
      case 6: LD4(dst, dst - BPS); break;
      case 7: VL4(dst, dst - BPS); break;
      case 8: HD4(dst, dst - BPS); break;
      case 9: HU4(dst, dst - BPS); break;
    }
  } else {
    switch (mode) {
      case 0: DC(dst, size, size, size == 16 ? 5 : 4); break;
      case 1: VE(dst, size); break;
      case 2: HE(dst, size); break;
      case 3: TM(dst, size); break;
    }
  }
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) printf("%d ", dst[x + y * BPS]);
  }
  printf("\n");
  return 0;
}
