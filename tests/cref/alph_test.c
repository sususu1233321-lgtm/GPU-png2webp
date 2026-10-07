/* Test a raw VP8L alpha stream with libwebp's real decoder. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "src/dec/webp_dec.h"
#include "src/dec/vp8i_dec.h"

int main(int argc, char** argv) {
  /* argv[1] = file with: 2-byte LE W, 2-byte LE H, then ALPH payload (after 1 header byte) */
  FILE* f = fopen(argv[1], "rb");
  if (!f) return 1;
  unsigned char hdr[4];
  fread(hdr, 1, 4, f);
  int W = hdr[0] | (hdr[1] << 8);
  int H = hdr[2] | (hdr[3] << 8);
  static unsigned char buf[1 << 22];
  int n = fread(buf, 1, sizeof(buf), f);
  fclose(f);
  fprintf(stderr, "W=%d H=%d stream=%d bytes\n", W, H, n);
  /* build a full lossless webp around the stream: use alpha decode path */
  uint8_t* alpha = (uint8_t*)WebPDecodeAlpha? 
  return 0;
}
