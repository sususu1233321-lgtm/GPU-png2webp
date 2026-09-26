/* Standalone reference: libwebp v1.5.0 bool reader (VP8GetBit), from
   src/utils/bit_reader_inl_utils.h. argv[1]=stream file, stdin="prob bit"
   pairs. Prints per-op state; exits nonzero on mismatch. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define BITS 56
typedef uint64_t bit_t;

typedef struct {
  const uint8_t* buf_;
  const uint8_t* buf_end_;
  const uint8_t* buf_max_;
  bit_t value_;
  int bits_;
  int range_;   /* stored: true range - 1 */
  int eof_;
} BR;

static void LoadNewBytes(BR* const br) {
  if (br->buf_ < br->buf_max_) {
    uint64_t in_bits;
    memcpy(&in_bits, br->buf_, 8);
    br->buf_ += BITS >> 3;
    bit_t bits = 0;
    const uint8_t* p = (const uint8_t*)&in_bits;
    for (int i = 0; i < (int)sizeof(in_bits); ++i) bits = (bits << 8) | p[i];
    bits >>= 64 - BITS;
    br->value_ = bits | (br->value_ << BITS);
    br->bits_ += BITS;
  } else {
    /* final bytes: shift in remaining, then zero pad */
    while (br->buf_ < br->buf_end_ && br->bits_ < 48) {
      br->value_ = ((bit_t)(*br->buf_)) | (br->value_ << 8);
      br->bits_ += 8;
      ++br->buf_;
    }
    while (br->bits_ < 48) {
      br->value_ <<= 8;
      br->bits_ += 8;
    }
    if (br->bits_ < 0) br->eof_ = 1;
  }
}

static int BitsLog2Floor(uint32_t n) {
  int l = n ? 0 : -1;
  while (n > 1) { n >>= 1; ++l; }
  return l;
}

static int GetBit(BR* const br, int prob) {
  bit_t range = (bit_t)br->range_;
  if (br->bits_ < 0) LoadNewBytes(br);
  {
    const int pos = br->bits_;
    const bit_t split = (range * prob) >> 8;
    const bit_t value = (bit_t)(br->value_ >> pos);
    const int bit = (value > split);
    if (bit) {
      range -= split;
      br->value_ -= (split + 1) << pos;
    } else {
      range = split + 1;
    }
    {
      const int shift = 7 ^ BitsLog2Floor((uint32_t)range);
      range <<= shift;
      br->bits_ -= shift;
    }
    br->range_ = (int)(range - 1);
    return bit;
  }
}

int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb");
  static uint8_t data[1 << 20];
  size_t n = fread(data, 1, sizeof(data) - 8, f);
  fclose(f);

  BR br;
  br.buf_ = data;
  br.buf_end_ = data + n;
  br.buf_max_ = (n > 8) ? data + n - 8 : data;  /* keep 8 bytes readable in main path */
  br.value_ = 0;
  br.bits_ = -8;
  br.range_ = 255 - 1;
  br.eof_ = 0;
  LoadNewBytes(&br);

  int prob, want, i = 0, bad = 0;
  while (scanf("%d %d", &prob, &want) == 2) {
    int bit = GetBit(&br, prob);
    printf("op%d p=%3d want=%d got=%d value=%llx bits=%d range=%d buf_ofs=%lld\n",
           i, prob, want, bit, (unsigned long long)br.value_, br.bits_, br.range_,
           (long long)(br.buf_ - data));
    if (bit != want) bad = 1;
    ++i;
  }
  return bad;
}
