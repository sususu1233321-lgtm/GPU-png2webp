"""Debug tool: Python port of libwebp's bool decoder + partition0 parser.
Parses our own VP8 streams symbol-by-symbol to find encoder bugs."""
import struct
import sys

sys.path.insert(0, ".")
from gpuwebp import vp8_tables as T
from gpuwebp.webp_container import parse


class BoolDec:
    BITS = 56

    def __init__(self, data):
        self.data = data + b"\x00" * 16
        self.pos = 0
        self.value = 0
        self.bits = -8
        self.range = 254

    def _load(self):
        chunk = self.data[self.pos:self.pos + 7]
        chunk = chunk + b"\x00" * (7 - len(chunk))
        v = int.from_bytes(chunk, "big")
        self.value = v | ((self.value << 56) & 0xFFFFFFFFFFFFFFFF)
        self.bits += 56
        self.pos += 7

    def get(self, prob):
        prob = int(prob)
        if self.bits < 0:
            self._load()
        r = self.range                      # stored = true range - 1
        split = (r * prob) >> 8
        v = self.value >> self.bits
        bit = 1 if v > split else 0
        if bit:
            r -= split
            self.value -= (split + 1) << self.bits
        else:
            r = split + 1
        shift = 7 ^ (r.bit_length() - 1)    # 7 ^ log2floor(r)
        self.range = min(r << shift, 256) - 1   # store true range - 1
        self.bits -= shift
        return bit

    def literal(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.get(128)
        return v

    def signed(self, n):
        if self.get(128) == 0:
            return 0
        v = self.literal(n + 1)
        return -(v >> 1) if v & 1 else v >> 1


def parse_vp8(filename):
    data = open(filename, "rb").read()
    chunks = parse(data)
    vp8 = chunks["VP8 "]
    tag = vp8[0] | (vp8[1] << 8) | (vp8[2] << 16)
    keyframe = not (tag & 1)
    size0 = tag >> 5
    sig = vp8[3:6]
    w = vp8[6] | (vp8[7] << 8) & 0x3FFF
    h = vp8[8] | (vp8[9] << 8) & 0x3FFF
    print(f"keyframe={keyframe} size0={size0} sig={sig.hex()} {w}x{h}")
    p0 = vp8[10:10 + size0]
    br = BoolDec(p0)
    cs = br.get(128); cl = br.get(128)
    seg = br.get(128)
    simple = br.get(128); level = br.literal(6); sharp = br.literal(3)
    lfdelta = br.get(128)
    parts_log2 = br.literal(2)
    yac_qi = br.literal(7)
    dq = [br.signed(4) for _ in range(5)]
    refresh = br.get(128)
    print(f"colorspace={cs} clamp={cl} seg={seg} simple={simple} level={level} "
          f"sharp={sharp} lfdelta={lfdelta} parts={1 << parts_log2}")
    print(f"y_ac_qi={yac_qi} dq={dq} refresh_entropy={refresh}")
    upd = T.COEFFS_UPDATE_PROBA
    n_upd = 0
    idx = 0
    for t in range(4):
        for b in range(8):
            for c in range(3):
                for p in range(11):
                    if br.get(upd[t][b][c][p]):
                        n_upd += 1
                        br.literal(8)
                    idx += 1
    print(f"coeff prob updates: {n_upd}")
    use_skip = br.get(128)
    skip_proba = br.literal(8) if use_skip else None
    print(f"mb_no_coeff_skip={use_skip} prob={skip_proba}")
    mb_w, mb_h = (w + 15) // 16, (h + 15) // 16
    kf = T.KF_BMODE_PROBA
    modes = []
    above = [0] * (4 * mb_w)
    for mby in range(mb_h):
        left = 0
        row = []
        for mbx in range(mb_w):
            mb = []
            if use_skip:
                mb.append(("skip", br.get(skip_proba)))
            is16 = br.get(145)
            if is16:
                b1 = br.get(156)
                if b1:
                    m = 3 if br.get(128) else 2
                else:
                    m = 1 if br.get(163) else 0
                mb.append(("i16", m))
            else:
                mm = []
                for y in range(4):
                    l = left if y == 0 else None
                    for x in range(4):
                        if y == 0:
                            top = above[mbx * 4 + x]
                            l = left
                        prob = kf[top][l]
                        # tree
                        if br.get(prob[0]) == 0:
                            m = 0
                        elif br.get(prob[1]) == 0:
                            m = 1
                        elif br.get(prob[2]) == 0:
                            m = 2
                        elif br.get(prob[3]) == 0:
                            if br.get(prob[4]) == 0:
                                m = 3
                            elif br.get(prob[5]) == 0:
                                m = 4
                            else:
                                m = 5
                        else:
                            if br.get(prob[6]) == 0:
                                m = 6
                            elif br.get(prob[7]) == 0:
                                m = 7
                            elif br.get(prob[8]) == 0:
                                m = 8
                            else:
                                m = 9
                        mm.append(m)
                        l = m
                    if y == 0:
                        left = l
                mb.append(("i4", mm))
                row_modes = mm
            # uv
            if br.get(142) == 0:
                uv = 0
            elif br.get(114) == 0:
                uv = 1
            elif br.get(183) == 0:
                uv = 2
            else:
                uv = 3
            mb.append(("uv", uv))
            row.append(mb)
            if not is16:
                for x in range(4):
                    above[mbx * 4 + x] = row_modes[12 + x]
        modes.append(row)
    for mby, row in enumerate(modes):
        s = ""
        for mbx, mb in enumerate(row):
            tagstr = "S" if (use_skip and mb[0][1]) else "."
            if mb[1][0] == "i16":
                s += f"{['D','V','H','T'][mb[1][1]]}{tagstr} "
            else:
                s += f"i4{tagstr} "
        print(f"row{mby}: {s}")


if __name__ == "__main__":
    parse_vp8(sys.argv[1] if len(sys.argv) > 1 else "dbg_flat128.webp")
