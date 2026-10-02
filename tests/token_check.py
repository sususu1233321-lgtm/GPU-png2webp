"""Token-level verification: parse our own stream with an exact Python port of
libwebp's ParseResiduals/GetCoeffsFast and compare with encoder levels."""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from dbg_decoder import BoolDec
from gpuwebp import vp8_encode as E
from gpuwebp import vp8_tables as T
from gpuwebp.webp_container import parse

BANDS16 = T.ENC_BANDS + [0]   # sentinel band = 0
CAT = {3: T.CAT3, 4: T.CAT4, 5: T.CAT5, 6: T.CAT6}
CP = np.array(T.COEFFS_PROBA0)   # [t][b][c][p]
ZIG = T.ZIGZAG


def get_large_value(br, p):
    if not br.get(p[3]):
        if not br.get(p[4]):
            return 2
        return 3 + br.get(p[5])
    if not br.get(p[6]):
        if not br.get(p[7]):
            return 5 + br.get(159)
        v = 7 + 2 * br.get(165)
        return v + br.get(145)
    bit1 = br.get(p[8])
    bit0 = br.get(p[9 + bit1])
    cat = 2 * bit1 + bit0
    v = 0
    for prob in [T.CAT3, T.CAT4, T.CAT5, T.CAT6][cat]:
        v = v + v + br.get(prob)
    return v + 3 + (8 << cat)


def get_coeffs(br, ctype, ctx, first):
    """Exact port of GetCoeffsFast. Returns (last_plus_one, out_in_zigzag_order)."""
    out = [0] * 16
    n = first
    p = CP[ctype][BANDS16[n]][ctx]
    while n < 16:
        if not br.get(p[0]):
            return n, out
        while not br.get(p[1]):
            n += 1
            if n == 16:
                return 16, out
            p = CP[ctype][BANDS16[n]][0]
        if not br.get(p[2]):
            v = 1
            p = CP[ctype][BANDS16[n + 1]][1]
        else:
            v = get_large_value(br, p)
            p = CP[ctype][BANDS16[n + 1]][2]
        out[n] = -v if br.get(128) else v
        n += 1
    return 16, out


def main(path):
    data = open(path, "rb").read()
    chunks = parse(data)
    vp8 = chunks["VP8 "]
    tag = vp8[0] | vp8[1] << 8 | vp8[2] << 16
    size0 = tag >> 5
    W = vp8[6] | vp8[7] << 8
    H = vp8[8] | vp8[9] << 8
    mb_w, mb_h = (W + 15) // 16, (H + 15) // 16
    p0 = vp8[10:10 + size0]
    rest = vp8[10 + size0:]

    br = BoolDec(p0)
    assert br.get(128) == 0 and br.get(128) == 0
    assert br.get(128) == 0  # segmentation
    br.get(128); [br.literal(6), br.literal(3)]; br.get(128)
    nparts_log = br.literal(2)
    br.literal(7)
    for _ in range(5):
        br.signed(4)
    br.get(128)
    for t in range(4):
        for b in range(8):
            for c in range(3):
                for pp in range(11):
                    if br.get(T.COEFFS_UPDATE_PROBA[t][b][c][pp]):
                        br.literal(8)
    use_skip = br.get(128)
    skip_p = br.literal(8) if use_skip else 255
    print(f"use_skip={use_skip} proba={skip_p} parts={1 << nparts_log}")

    kf = T.KF_BMODE_PROBA
    modes = []
    skips = []
    above = [0] * (4 * mb_w)
    for mby in range(mb_h):
        left0 = 0
        for mbx in range(mb_w):
            skips.append(br.get(skip_p) if use_skip else 0)
            is16 = br.get(145)
            i4m = None
            if is16:
                m = (3 if br.get(128) else 2) if br.get(156) else (1 if br.get(163) else 0)
                i16m = m
            else:
                i4m = []
                for y in range(4):
                    left = 0 if mbx == 0 else modes[-mb_w][2][y * 4 + 3] if False else 0
                # context: top from above MB row (or 0), left from prev subblock / left MB
                for y in range(4):
                    left = left0 if y == 0 else i4m[(y - 1) * 4 + 3]
                    # hmm: left for y>0 = mode of subblock (3, y-1)
                    for x in range(4):
                        top = 0 if mby == 0 else above[mbx * 4 + x]
                        prob = kf[top][left]
                        if not br.get(prob[0]):
                            m = 0
                        elif not br.get(prob[1]):
                            m = 1
                        elif not br.get(prob[2]):
                            m = 2
                        elif not br.get(prob[3]):
                            if not br.get(prob[4]):
                                m = 3
                            elif not br.get(prob[5]):
                                m = 4
                            else:
                                m = 5
                        elif not br.get(prob[6]):
                            m = 6
                        elif not br.get(prob[7]):
                            m = 7
                        elif not br.get(prob[8]):
                            m = 8
                        else:
                            m = 9
                        i4m.append(m)
                        left = m
                    left0 = left
                # update above from last row
            uv = 0 if not br.get(142) else (1 if not br.get(114) else (2 if not br.get(183) else 3))
            modes.append((i16m if is16 else None, i4m, uv))
            if is16:
                for x in range(4):
                    above[mbx * 4 + x] = 0  # i16 modes don't participate? decoder: top[]=ymode
            else:
                for x in range(4):
                    above[mbx * 4 + x] = i4m[12 + x]
    print("modes parsed:", modes[:8])
    print("NOTE: i16 above-mode handling simplified - check needed")
    print("token bytes:", rest.hex())

    # now parse tokens for partition 0
    tbr = BoolDec(rest)
    top_nz = np.zeros((mb_w, 9), int)
    left_nz = np.zeros(9, int)
    for mby in range(mb_h):
        left_nz[:] = 0
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            i16m, i4m, uv = modes[mb]
            if skips[mb]:
                top_nz[mbx] = 0
                left_nz[:] = 0
                continue
            is16 = i16m is not None
            if is16:
                ctx = top_nz[mbx, 8] + left_nz[8]
                nz, dc = get_coeffs(tbr, 1, ctx, 0)
                top_nz[mbx, 8] = left_nz[8] = int(nz > 0)
                first = 1
                ctype = 0
            else:
                first = 0
                ctype = 3
            for y in range(4):
                for x in range(4):
                    ctx = top_nz[mbx, x] + left_nz[y]
                    nz, blk = get_coeffs(tbr, ctype, ctx, first)
                    nzb = int(nz > first)
                    top_nz[mbx, x] = left_nz[y] = nzb
                    print(f"mb{mb} yblk({x},{y}) ctx={ctx} nz={nz} blk={blk}")
            for n in range(8):
                ch = n >> 2
                x = n & 1
                y = (n >> 1) & 1
                slot = 4 + ch * 2 + x
                lslot = 4 + ch * 2 + y
                ctx = top_nz[mbx, slot] + left_nz[lslot]
                nz, blk = get_coeffs(tbr, 2, ctx, 0)
                nzb = int(nz > 0)
                top_nz[mbx, slot] = left_nz[lslot] = nzb
                print(f"mb{mb} uvblk{n} ctx={ctx} nz={nz} blk={blk}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dbg_two_level.webp")
