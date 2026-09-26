"""Minimal VP8L encoder for the WebP ALPH (alpha) chunk.

Encodes the alpha plane as a standard lossless (VP8L) stream:
  - GREEN tree: Huffman over {literals 0..255, length prefix symbols 0..23}
  - RED / BLUE / ALPHA trees: single-symbol simple codes (0 bits per pixel)
  - DIST tree: Huffman over the distance symbols actually used
  - LZ77: greedy matching of distance-1 runs (constant stretches) and
    distance-W copies (equal to the row above)

Semantics (verified against libwebp v1.5.0 dec/vp8l_dec.c):
  prefix symbol s -> value: s<4: s+1; else extra=(s-2)>>1,
      offset=(2+(s&1))<<extra, value=offset+extra_bits+1
  dist_code = prefix value; dist_code>120 -> distance = dist_code-120,
  else plane code via kCodeToPlane (dist_code 1 -> offset (1,0) -> dist W)
  so direct distance d encodes as dist_code d+120 (dist 1 -> code 121).

Header: [no transforms][no color cache][no meta huffman] + 5 trees (alpha
streams are parsed like level-0 streams; 3 leading bits, empirically
verified byte-exact against Pillow/libwebp decoding).
"""
import heapq
from collections import Counter

import numpy as np
from numba import njit

NUM_LITERAL = 256
NUM_LENGTH = 24
ALPHA_G = NUM_LITERAL + NUM_LENGTH          # 280
ALPHA_DIST = 40
K_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

MAX_COPY = 4096


class BitWriter:
    __slots__ = ("bits",)

    def __init__(self):
        self.bits = []

    def bit(self, b):
        self.bits.append(b & 1)

    def bits_(self, value, n):
        for i in range(n):
            self.bits.append((value >> i) & 1)

    def tobytes(self):
        bits = self.bits + [0] * ((-len(self.bits)) % 8)
        out = bytearray((len(bits) + 7) // 8)
        for i, b in enumerate(bits):
            if b:
                out[i >> 3] |= 1 << (i & 7)
        return bytes(out)


def prefix_code(value):
    """VP8L prefix encoding. Returns (symbol, n_extra, extra_value)."""
    v = value - 1
    if v < 4:
        return v, 0, 0
    hb = v.bit_length() - 1
    second = (v >> (hb - 1)) & 1
    n_extra = hb - 1
    sym = 2 * hb + second
    extra = v & ((1 << n_extra) - 1)
    return sym, n_extra, extra


def _huffman_lengths(freqs):
    heap = [(f, [s]) for s, f in freqs.items()]
    heapq.heapify(heap)
    depth = {s: 0 for s in freqs}
    while len(heap) > 1:
        f1, l1 = heapq.heappop(heap)
        f2, l2 = heapq.heappop(heap)
        for s in l1 + l2:
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, l1 + l2))
    if max(depth.values()) > 15:
        raise ValueError("code too long")
    return depth


def _canonical(depth):
    syms = sorted(depth, key=lambda s: (depth[s], s))
    codes = {}
    code = 0
    prev_len = 0
    for s in syms:
        code <<= (depth[s] - prev_len)
        codes[s] = (code, depth[s])
        code += 1
        prev_len = depth[s]
    return codes


def _reversed_code(code, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (code & 1)
        code >>= 1
    return r


def _write_simple1(bw, sym):
    bw.bit(1)                       # simple
    bw.bit(0)                       # 1 symbol
    if sym < 2:
        bw.bit(0)                   # 1-bit symbol
        bw.bit(sym)
    else:
        bw.bit(1)                   # 8-bit symbol
        bw.bits_(sym, 8)


def _write_simple2(bw, sym0, sym1):
    """Two-symbol simple code: 1 bit per read."""
    bw.bit(1)                       # simple
    bw.bit(1)                       # 2 symbols
    bw.bit(1)                       # symbols written as 8 bits
    bw.bits_(sym0, 8)
    bw.bits_(sym1, 8)


def _write_normal_code(bw, depth, alphabet_size):
    """Write a normal Huffman header for sym->len; returns stream codes."""
    lens = sorted(depth.items())
    seq = []                        # (clsym, extra_val, extra_nbits)
    prev = -1
    for sym, ln in lens:
        gap = sym - prev - 1
        while gap >= 11:
            run = min(gap, 138)
            seq.append((18, run - 11, 7))
            gap -= run
        while gap >= 3:
            run = min(gap, 10)
            seq.append((17, run - 3, 3))
            gap -= run
        for _ in range(gap):
            seq.append((0, 0, 0))
        seq.append((ln, 0, 0))
        prev = sym
    gap = alphabet_size - 1 - prev
    while gap >= 11:
        run = min(gap, 138)
        seq.append((18, run - 11, 7))
        gap -= run
    while gap >= 3:
        run = min(gap, 10)
        seq.append((17, run - 3, 3))
        gap -= run
    for _ in range(gap):
        seq.append((0, 0, 0))

    lhist = Counter(sym for sym, _, _ in seq)
    if len(lhist) > 1:
        ldepth = _huffman_lengths(dict(lhist))
    else:
        ldepth = {next(iter(lhist)): 1}
    lcodes = _canonical(ldepth)

    bw.bit(0)                       # not simple
    last_used = max(K_ORDER.index(k) for k in ldepth)
    write_count = max(4, last_used + 1)
    bw.bits_(write_count - 4, 4)
    for i in range(write_count):
        bw.bits_(ldepth.get(K_ORDER[i], 0), 3)
    bw.bit(1)                       # use_length
    max_sym = len(seq)
    k = 0
    while (1 << (2 + 2 * k)) <= max_sym - 2:
        k += 1
    nbits = 2 + 2 * k
    bw.bits_(k, 3)
    bw.bits_(max_sym - 2, nbits)
    for sym, extra, nx in seq:
        c, n = lcodes[sym]
        if n:
            bw.bits_(_reversed_code(c, n), n)
        if nx:
            bw.bits_(extra, nx)

    codes = _canonical(depth)
    return {s: (_reversed_code(c, n), n) for s, (c, n) in codes.items()}


# ---------------------------------------------------------------- LZ77 (numba)

@njit(cache=True, nogil=True)
def _prefix_sym(v):
    """value -> (sym, n_extra, extra) for prefix value v >= 1."""
    x = v - 1
    if x < 4:
        return x, 0, 0
    hb = 61
    while (1 << hb) > x:
        hb -= 1
    second = (x >> (hb - 1)) & 1
    n_extra = hb - 1
    sym = 2 * hb + second
    extra = x & ((1 << n_extra) - 1)
    return sym, n_extra, extra


@njit(cache=True, nogil=True)
def _alpha_ops(alpha, W, ops):
    """Greedy LZ77 over the flat alpha plane.
    ops rows: [kind, a, b]; kind 0 literal a; kind 1 copy len a dist_code b.
    Returns op count; also fills green/dist histograms."""
    n = alpha.shape[0]
    pos = 0
    count = 0
    while pos < n:
        best_len = 0
        best_dc = 0
        # distance-1 run (constant stretch)
        if pos >= 1:
            l = 0
            lim = n - pos
            if lim > MAX_COPY:
                lim = MAX_COPY
            while l < lim and alpha[pos + l] == alpha[pos - 1]:
                l += 1
            if l >= 3:
                best_len = l
                best_dc = 121          # direct distance 1
        # distance-W (copy the row above)
        if pos >= W:
            l = 0
            lim = n - pos
            if lim > MAX_COPY:
                lim = MAX_COPY
            while l < lim and alpha[pos + l] == alpha[pos - W + l]:
                l += 1
            if l > best_len and l >= 3:
                best_len = l
                best_dc = 1            # plane code 1 -> offset (1,0) -> dist W
        if best_len >= 3:
            ops[count, 0] = 1
            ops[count, 1] = best_len
            ops[count, 2] = best_dc
            count += 1
            pos += best_len
        else:
            ops[count, 0] = 0
            ops[count, 1] = alpha[pos]
            ops[count, 2] = 0
            count += 1
            pos += 1
    return count


@njit(cache=True, nogil=True)
def _emit_bits(ops, count, gtab, dtab, out, acc, nbits):
    """Emit the op stream with prebuilt (code, nbits) tables.
    gtab[280], dtab[40]: code | (nbits << 24).  acc/nbits seed the bit
    accumulator (continues mid-byte from the tree header).  Returns bytes."""
    outpos = 0
    for i in range(count):
        if ops[i, 0] == 0:
            v = ops[i, 1]
            c = gtab[v]
            acc |= np.uint64(c & 0xFFFFFF) << np.uint64(nbits)
            nbits += (c >> 24) & 0xFF
        else:
            length = ops[i, 1]
            lsym, lx, lev = _prefix_sym(length)
            c = gtab[256 + lsym]
            acc |= np.uint64(c & 0xFFFFFF) << np.uint64(nbits)
            nbits += (c >> 24) & 0xFF
            if lx:
                acc |= np.uint64(lev) << np.uint64(nbits)
                nbits += lx
            dsym, dx, dev = _prefix_sym(ops[i, 2])
            c = dtab[dsym]
            acc |= np.uint64(c & 0xFFFFFF) << np.uint64(nbits)
            nbits += (c >> 24) & 0xFF
            if dx:
                acc |= np.uint64(dev) << np.uint64(nbits)
                nbits += dx
        while nbits >= 8:
            out[outpos] = acc & np.uint64(0xFF)
            outpos += 1
            acc >>= np.uint64(8)
            nbits -= 8
    if nbits:
        out[outpos] = acc & np.uint64(0xFF)
        outpos += 1
    return outpos


def _pack_table(codes, size):
    tab = np.zeros(size, np.uint32)
    for s, (code, n) in codes.items():
        tab[s] = (code & 0xFFFFFF) | (n << 24)
    return tab


@njit(cache=True, nogil=True)
def _alpha_hist(ops, count, ghist, dhist):
    """Fill green (280) / dist (40) histograms from the op list."""
    ghist[:] = 0
    dhist[:] = 0
    for i in range(count):
        if ops[i, 0] == 0:
            ghist[ops[i, 1]] += 1
        else:
            lsym, _, _ = _prefix_sym(ops[i, 1])
            ghist[256 + lsym] += 1
            dsym, _, _ = _prefix_sym(ops[i, 2])
            dhist[dsym] += 1


def encode_alpha_stream(alpha, W):
    """alpha: 2-D uint8 array. Returns VP8L stream bytes."""
    alpha = np.ascontiguousarray(alpha).reshape(-1)
    n = alpha.shape[0]
    ops = np.empty((n + 8, 3), np.int32)
    count = _alpha_ops(alpha, W, ops)

    ghist = np.zeros(ALPHA_G, np.int64)
    dhist = np.zeros(ALPHA_DIST, np.int64)
    _alpha_hist(ops, count, ghist, dhist)
    gcount = {int(s): int(c) for s, c in enumerate(ghist) if c}
    dcount = {int(s): int(c) for s, c in enumerate(dhist) if c}

    bw = BitWriter()
    bw.bit(0)                       # no transforms
    bw.bit(0)                       # no color cache
    bw.bit(0)                       # no meta huffman

    if len(gcount) == 0:
        gcount = {0: 1}
    if len(gcount) == 1:
        sym = next(iter(gcount))
        _write_simple1(bw, sym)
        gcodes = {sym: (0, 0)}
    else:
        gcodes = _write_normal_code(bw, _huffman_lengths(gcount), ALPHA_G)
    _write_simple1(bw, 0)           # red
    _write_simple1(bw, 0)           # blue
    _write_simple1(bw, 255)         # alpha literal
    if len(dcount) == 0:
        _write_simple1(bw, 0)
        dcodes = {0: (0, 0)}
    elif len(dcount) == 1:
        sym = next(iter(dcount))
        _write_simple1(bw, sym)
        dcodes = {sym: (0, 0)}
    elif len(dcount) == 2:
        s0, s1 = sorted(dcount)
        _write_simple2(bw, s0, s1)
        dcodes = {s0: (0, 1), s1: (1, 1)}
    else:
        dcodes = _write_normal_code(bw, _huffman_lengths(dcount), ALPHA_DIST)

    header = bw.tobytes()
    gtab = _pack_table(gcodes, ALPHA_G)
    dtab = _pack_table(dcodes, ALPHA_DIST)
    out = np.empty(count * 24 + 64, np.uint8)
    rem = len(bw.bits) % 8
    if rem:
        # continue mid-byte: the emitter rebuilds the header's last byte
        acc = np.uint64(header[-1])
        nb = _emit_bits(ops[:count], count, gtab, dtab, out, acc, rem)
        stream = header[:-1] + out[:nb].tobytes()
    else:
        nb = _emit_bits(ops[:count], count, gtab, dtab, out, np.uint64(0), 0)
        stream = header + out[:nb].tobytes()
    return stream


def make_alph_chunk(alpha):
    """alpha uint8 2-D. Returns ALPH chunk payload (header + VP8L stream)."""
    stream = encode_alpha_stream(alpha, alpha.shape[1])
    header = (1 << 0)               # compression = 1 (lossless), filter none
    return bytes([header]) + stream
