"""VP8 boolean arithmetic (range) coder, ported 1:1 from libwebp's
bit_writer_utils.c. Numba-jitted so it runs at machine-code speed and can be
called from threads (nogil).

Ops encoding: each op is an int32  (prob << 1) | bit.
prob==0 is unused (VP8 probabilities are >= 1).
"""
import numpy as np
from numba import njit

# kNorm[i] = 8 - log2(i), for i in [1..127]; kNorm[0] unused sentinel
_K_NORM = np.array(
    [7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     0], dtype=np.int64)

_K_NEW_RANGE = np.array(
    [127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239,
     127, 135, 143, 151, 159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239,
     247, 127, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179,
     183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239,
     243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149,
     151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179,
     181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209,
     211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239,
     241, 243, 245, 247, 249, 251, 253, 127], dtype=np.int64)


@njit(cache=True, nogil=True)
def _flush(value, nb_bits, run, pos, out):
    s = 8 + nb_bits
    bits = value >> s
    value = value - ((bits << s))
    nb_bits -= 8
    if (bits & 0xff) != 0xff:
        if (bits & 0x100) != 0:      # carry
            if pos > 0:
                out[pos - 1] = out[pos - 1] + 1
        if run > 0:
            fill = 0
            if (bits & 0x100) == 0:
                fill = 0xff
            for _ in range(run):
                out[pos] = fill
                pos += 1
            run = 0
        out[pos] = bits & 0xff
        pos += 1
    else:
        run += 1
    return value, nb_bits, run, pos


@njit(cache=True, nogil=True)
def bool_encode(ops, out):
    """Encode a stream of (prob,bit) ops into out (uint8). Returns bytes written.
    out must have capacity >= len(ops)."""
    range_ = np.int64(254)
    value = np.int64(0)
    run = 0
    nb_bits = -8
    pos = 0
    n = ops.shape[0]
    for i in range(n):
        op = ops[i]
        prob = op >> 1
        bit = op & 1
        split = (range_ * prob) >> 8
        if bit != 0:
            value += split + 1
            range_ -= split + 1
        else:
            range_ = split
        if range_ < 127:
            shift = _K_NORM[range_]
            range_ = _K_NEW_RANGE[range_]
            value <<= shift
            nb_bits += shift
            if nb_bits > 0:
                value, nb_bits, run, pos = _flush(value, nb_bits, run, pos, out)
    # finish: VP8PutBits(bw, 0, 9 - nb_bits); nb_bits=0; Flush()
    n_zero = 9 - nb_bits
    for _ in range(n_zero):
        # uniform zero bit (prob 128): renorm shift is exactly 1
        split = range_ >> 1
        range_ = split
        if range_ < 127:
            range_ = _K_NEW_RANGE[range_]
            value <<= 1
            nb_bits += 1
            if nb_bits > 0:
                value, nb_bits, run, pos = _flush(value, nb_bits, run, pos, out)
    nb_bits = 0
    value, nb_bits, run, pos = _flush(value, nb_bits, run, pos, out)
    return pos


def encode_ops(ops):
    """Encode ops (int32 array of (prob<<1)|bit) into bytes."""
    ops = np.ascontiguousarray(ops, dtype=np.int32)
    out = np.empty(max(len(ops) + 16, 64), dtype=np.uint8)
    n = bool_encode(ops, out)
    return out[:n].tobytes()
