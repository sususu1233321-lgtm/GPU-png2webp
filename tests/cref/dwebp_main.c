#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/webp/decode.h"

int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  if (!f) { printf("no file\n"); return 1; }
  static unsigned char buf[1 << 26];
  size_t n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  int w = 0, h = 0;
  uint8_t *u = NULL, *v = NULL;
  int stride = 0, uvstride = 0;
  uint8_t* yp = WebPDecodeYUV(buf, n, &w, &h, &u, &v, &stride, &uvstride);
  if (!yp) { printf("DECODE FAILED\n"); return 1; }
  printf("%d %d %d %d\n", w, h, stride, uvstride);
  FILE* o = fopen(argv[2], "wb");
  for (int y = 0; y < h; ++y) fwrite(yp + (size_t)y * stride, 1, w, o);
  for (int y = 0; y < (h + 1) / 2; ++y) fwrite(u + (size_t)y * uvstride, 1, (w + 1) / 2, o);
  for (int y = 0; y < (h + 1) / 2; ++y) fwrite(v + (size_t)y * uvstride, 1, (w + 1) / 2, o);
  fclose(o);
  WebPFree(yp);
  return 0;
}
