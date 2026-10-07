"""Own minimal VP8L decoder for validating the alpha encoder."""
import numpy as np

K_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
K_CODE_TO_PLANE = [
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
    0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
    0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
    0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
    0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
    0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
    0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
    0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
    0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
    0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
    0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
    0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
]


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bit = 0

    def read(self, n):
        v = 0
        for i in range(n):
            byte = self.data[self.pos] if self.pos < len(self.data) else 0
            v |= ((byte >> self.bit) & 1) << i
            self.bit += 1
            if self.bit == 8:
                self.bit = 0
                self.pos += 1
        return v


class Tree:
    """Canonical prefix code; read() accumulates bits MSB-first == canonical.
    Single-symbol trees consume 0 bits (libwebp special case)."""

    def __init__(self, codes):          # codes: sym -> (canonical_code, len)
        self.map = {(c, n): s for s, (c, n) in codes.items()}
        self.single = next(iter(codes)) if len(codes) == 1 else None

    def read(self, br):
        if self.single is not None:
            return self.single
        acc = 0
        n = 0
        while n <= 15:
            acc = (acc << 1) | br.read(1)
            n += 1
            if (acc, n) in self.map:
                return self.map[(acc, n)]
        raise ValueError("bad huffman code")


def build_canonical(lengths):
    syms = sorted((s for s in lengths if lengths.get(s)), key=lambda s: (lengths[s], s))
    codes = {}
    code = 0
    prev = 0
    for s in syms:
        code <<= (lengths[s] - prev)
        codes[s] = (code, lengths[s])
        code += 1
        prev = lengths[s]
    return codes


def read_huffman(br, alphabet):
    simple = br.read(1)
    if simple:
        nsym = br.read(1) + 1
        first_8 = br.read(1)
        sym = br.read(8) if first_8 else br.read(1)
        lengths = {sym: 1}
        if nsym == 2:
            lengths[br.read(8)] = 1
        return Tree(build_canonical(lengths))
    ncodes = br.read(4) + 4
    cl_lengths = {}
    for i in range(ncodes):
        ln = br.read(3)
        if ln:
            cl_lengths[K_ORDER[i]] = ln
    cl_tree = Tree(build_canonical(cl_lengths))
    if br.read(1):
        nbits = 2 + 2 * br.read(3)
        max_sym = 2 + br.read(nbits)
    else:
        max_sym = alphabet
    lengths = {}
    symbol = 0
    prev_len = 8
    reads = 0
    while symbol < alphabet and reads < max_sym:
        reads += 1
        cl = cl_tree.read(br)
        if cl < 16:
            if cl:
                lengths[symbol] = cl
                prev_len = cl
            symbol += 1
        elif cl == 16:
            r = 3 + br.read(2)
            for _ in range(r):
                if symbol < alphabet and prev_len:
                    lengths[symbol] = prev_len
                symbol += 1
        elif cl == 17:
            symbol += 3 + br.read(3)
        else:
            symbol += 11 + br.read(7)
    return Tree(build_canonical(lengths))


def _copy_len(br, sym):
    if sym < 4:
        return sym + 1
    e = (sym - 2) >> 1
    return ((2 + (sym & 1)) << e) + br.read(e) + 1


def decode_alpha(stream, W, H):
    br = BitReader(stream)
    assert br.read(1) == 0        # no transforms
    assert br.read(1) == 0        # no color cache
    assert br.read(1) == 0        # no meta huffman
    green = read_huffman(br, 280)
    red = read_huffman(br, 256)
    blue = read_huffman(br, 256)
    alph = read_huffman(br, 256)
    dist = read_huffman(br, 40)
    out = np.zeros((H, W), np.uint8)
    pos = 0
    total = W * H
    while pos < total:
        code = green.read(br)
        if code < 256:
            red.read(br)
            blue.read(br)
            alph.read(br)
            out.flat[pos] = code
            pos += 1
        else:
            length = _copy_len(br, code - 256)
            ds = dist.read(br)
            dc = _copy_len(br, ds)
            if dc > 120:
                d = dc - 120
            else:
                kc = K_CODE_TO_PLANE[dc - 1]
                d = (kc >> 4) * W + (8 - (kc & 0xF))
                d = max(d, 1)
            for i in range(length):
                out.flat[pos + i] = out.flat[pos + i - d]
            pos += length
    return out
