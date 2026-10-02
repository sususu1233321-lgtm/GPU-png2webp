/* trace libwebp VP8GetBit to find divergence with Python port */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef struct { const uint8_t* buf_; size_t len_; const uint8_t* buf_end_;
                 const uint8_t* buf_max_; uint64_t value_; int bits_; int eof_; } VP8BitReader;

static void LoadNewBytes(VP8BitReader* br) {
  if (br->buf_ < br->buf_max_) {
    uint64_t in_bits; uint8_t* p = (uint8_t*)&in_bits;
    memcpy(p, br->buf_, 8);
    /* big-endian load of up to 8 bytes, zero-padded */
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | (br->buf_[i] < br->buf_end_ - br->buf_ + i ? 0 : 0);
    br->buf_ += 8;
    (void)v; (void)p;
  }
}
int main(){return 0;}
