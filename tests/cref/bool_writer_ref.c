/* Standalone reference: libwebp v1.5.0 bool writer, copied verbatim from
   src/utils/bit_writer_utils.c (BSD). Reads ops from stdin as text lines
   "prob bit", prints hex of the encoded stream. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* ---- verbatim libwebp bool writer ---- */
typedef struct VP8BitWriter {
  int64_t range_;
  uint64_t value_;
  int run_;
  int nb_bits_;
  int done_;
  uint8_t* buf_;
  size_t pos_;
  size_t max_pos_;
  int error_;
} VP8BitWriter;

static const uint8_t kNorm[128] = {
     7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0
};

static const uint8_t kNewRange[128] = {
  127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
  127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
  247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
  183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
  243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
  151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
  181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
  211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
  241, 243, 245, 247, 249, 251, 253, 127
};

static void Flush(VP8BitWriter* const bw) {
  const int s = 8 + bw->nb_bits_;
  const int32_t bits = (int32_t)(bw->value_ >> s);
  assert(bw->nb_bits_ >= 0);
  bw->value_ -= (uint64_t)bits << s;
  bw->nb_bits_ -= 8;
  if ((bits & 0xff) != 0xff) {
    size_t pos = bw->pos_;
    if (bits & 0x100) {  // overflow -> propagate carry over pending 0xff's
      if (pos > 0) bw->buf_[pos - 1]++;
    }
    if (bw->run_ > 0) {
      const int value = (bits & 0x100) ? 0x00 : 0xff;
      for (; bw->run_ > 0; --bw->run_) bw->buf_[pos++] = value;
    }
    bw->buf_[pos++] = bits & 0xff;
    bw->pos_ = pos;
  } else {
    bw->run_++;   // delay writing of bytes 0xff, pending eventual carry.
  }
}

int VP8PutBit(VP8BitWriter* const bw, int bit, int prob) {
  const int split = (bw->range_ * prob) >> 8;
  if (bit) {
    bw->value_ += split + 1;
    bw->range_ -= split + 1;
  } else {
    bw->range_ = split;
  }
  if (bw->range_ < 127) {   // emit 'shift' bits out and renormalize
    const int shift = kNorm[bw->range_];
    bw->range_ = kNewRange[bw->range_];
    bw->value_ <<= shift;
    bw->nb_bits_ += shift;
    if (bw->nb_bits_ > 0) Flush(bw);
  }
  return bit;
}

int VP8PutBitUniform(VP8BitWriter* const bw, int bit) {
  const int split = bw->range_ >> 1;
  if (bit) {
    bw->value_ += split + 1;
    bw->range_ -= split + 1;
  } else {
    bw->range_ = split;
  }
  if (bw->range_ < 127) {
    bw->range_ = kNewRange[bw->range_];
    bw->value_ <<= 1;
    bw->nb_bits_ += 1;
    if (bw->nb_bits_ > 0) Flush(bw);
  }
  return bit;
}

void VP8PutBits(VP8BitWriter* const bw, uint32_t value, int nb_bits) {
  uint32_t mask;
  for (mask = 1u << (nb_bits - 1); mask; mask >>= 1) {
    VP8PutBitUniform(bw, value & mask);
  }
}



int main(void) {
  VP8BitWriter bw;
  int prob, bit;
  bw.range_ = 255 - 1;
  bw.value_ = 0;
  bw.run_ = 0;
  bw.nb_bits_ = -8;
  bw.pos_ = 0;
  bw.run_ = 0;
  bw.max_pos_ = 1 << 20;
  bw.buf_ = (uint8_t*)calloc(1, bw.max_pos_);
  bw.error_ = 0;

  while (scanf("%d %d", &prob, &bit) == 2) {
    VP8PutBit(&bw, bit, prob);
  }
  /* finish: VP8PutBits(bw, 0, 9 - nb_bits); nb_bits=0; Flush() */
  {
    int n = 9 - bw.nb_bits_;
    int i;
    for (i = 0; i < n; ++i) VP8PutBitUniform(&bw, 0);
    bw.nb_bits_ = 0;
    Flush(&bw);
  }
  for (size_t i = 0; i < bw.pos_; ++i) printf("%02x", bw.buf_[i]);
  printf("\n");
  return 0;
}
