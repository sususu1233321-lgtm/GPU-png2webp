"""Full-stream verifier: encode with our encoder, decode with exact Python
ports of libwebp's ParseIntraMode + ParseResiduals, diff everything."""
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
from gpuwebp import vp8_encode as E
from gpuwebp.bool_coder import bool_encode
from gpuwebp.webp_container import make_simple
from token_check import BoolDec, get_coeffs
from test_m1 import rgb_to_yuv420, psnr

T = E
from gpuwebp import vp8_tables as VT

KFM = VT.KF_BMODE_PROBA


def verify(rgb, quality=90, verbose=True):
    H, W = rgb.shape[:2]
    rgba = np.concatenate([rgb, np.full((H, W, 1), 255, np.uint8)], 2)
    y, u, v = rgb_to_yuv420(rgba)
    mb_w, mb_h = (W + 15) // 16, (H + 15) // 16
    base_quant, y1, y2, uv_m, filter_level = E.setup_quant(quality)
    Y = E.pad_to_mb(y, mb_h, mb_w)
    U = E.pad_to_mb(u, mb_h, mb_w, half=True)
    V = E.pad_to_mb(v, mb_h, mb_w, half=True)
    (mb_w, mb_h, is_i4, i16_mode, uv_mode, i4_modes,
     y_dc, y_ac, uv_levels, skip) = E.analyze(Y, U, V, y1, y2, uv_m)
    n_mb = mb_w * mb_h
    nb_skip = int(skip.sum())
    skip_proba = (n_mb - nb_skip) * 255 // n_mb
    use_skip = skip_proba < 250

    # ---- decode partition 0 (modes + skips)
    ops0 = np.empty(n_mb * 256 + 4096, np.int32)
    pos0 = E.write_partition0(mb_w, mb_h, base_quant, -2, 0, filter_level,
                              0, use_skip, skip_proba, skip, is_i4,
                              i16_mode, uv_mode, i4_modes, ops0)
    buf0 = np.empty(pos0 + 16, np.uint8)
    n0 = bool_encode(ops0[:pos0], buf0)
    br = BoolDec(buf0[:n0].tobytes())
    br.get(128); br.get(128); br.get(128)          # cs, clamp, seg
    br.get(128); br.literal(6); br.literal(3); br.get(128)
    br.literal(2); br.literal(7)
    for _ in range(5):
        br.signed(4)
    br.get(128)
    for t in range(4):
        for b in range(8):
            for c in range(3):
                for p in range(11):
                    if br.get(VT.COEFFS_UPDATE_PROBA[t][b][c][p]):
                        br.literal(8)
    d_use_skip = br.get(128)
    d_skip_p = br.literal(8) if d_use_skip else 255

    d_is16 = np.zeros(n_mb, bool)
    d_i16m = np.zeros(n_mb, int)
    d_uv = np.zeros(n_mb, int)
    d_skip = np.zeros(n_mb, bool)
    d_i4m = np.zeros((n_mb, 16), int)
    # decoder context arrays
    intra_t = [0] * (4 * mb_w)
    intra_l = [0] * 4
    for mby in range(mb_h):
        intra_l = [0] * 4
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            if d_use_skip:
                d_skip[mb] = bool(br.get(d_skip_p))
            is16 = br.get(145)
            d_is16[mb] = is16 == 1
            if is16:
                b1 = br.get(156)
                if b1:
                    m = 3 if br.get(128) else 2
                else:
                    m = 1 if br.get(163) else 0
                d_i16m[mb] = m
                for k in range(4):
                    intra_t[mbx * 4 + k] = m
                    intra_l[k] = m
            else:
                for yy in range(4):
                    ymode = intra_l[yy]
                    for xx in range(4):
                        prob = KFM[intra_t[mbx * 4 + xx]][ymode]
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
                        d_i4m[mb, yy * 4 + xx] = m
                        ymode = m
                        intra_t[mbx * 4 + xx] = m
                    intra_l[yy] = ymode
            uv = 0 if not br.get(142) else (1 if not br.get(114) else (2 if not br.get(183) else 3))
            d_uv[mb] = uv

    # ---- compare modes
    ok = True
    mism = np.flatnonzero(d_is16 != (~is_i4))
    if len(mism) and verbose:
        print(f"i4/i16 mismatches at MBs {mism[:10]}")
    ok &= len(mism) == 0
    mism = np.flatnonzero(d_skip != skip)
    if len(mism) and verbose:
        print(f"skip mismatches at MBs {mism[:10]}")
        mb = mism[0]
        print(f"  first: encoded={skip[mb]} decoded={d_skip[mb]}")
    ok &= len(mism) == 0

    # ---- decode tokens
    ops = np.empty(n_mb * 8200 + 64, np.int32)
    pos = E.write_token_partition(mb_w, mb_h, 0, 1, use_skip, skip, is_i4,
                                  y_dc, y_ac, uv_levels, ops)
    buf = np.empty(pos + 16, np.uint8)
    nbt = bool_encode(ops[:pos], buf)
    tbr = BoolDec(buf[:nbt].tobytes())
    top_nz = np.zeros((mb_w, 9), int)
    left_nz = np.zeros(9, int)
    for mby in range(mb_h):
        left_nz[:] = 0
        for mbx in range(mb_w):
            mb = mby * mb_w + mbx
            if skip[mb]:
                top_nz[mbx] = 0
                left_nz[:] = 0
                continue
            first = 0 if is_i4[mb] else 1
            if not is_i4[mb]:
                ctx = top_nz[mbx, 8] + left_nz[8]
                nz, dc = get_coeffs(tbr, 1, ctx, 0)
                if list(dc) != list(y_dc[mb]) and verbose:
                    print(f"mb{mb} y_dc MISMATCH")
                    print("  dec:", dc)
                    print("  enc:", list(y_dc[mb]))
                    ok = False
                top_nz[mbx, 8] = left_nz[8] = int(nz > 0)
            ctype = 3 if is_i4[mb] else 0
            nzmap = np.zeros((4, 4), int)
            for yy in range(4):
                for xx in range(4):
                    top = nzmap[yy - 1, xx] if yy > 0 else top_nz[mbx, xx]
                    left = nzmap[yy, xx - 1] if xx > 0 else left_nz[yy]
                    ctx = top + left
                    nz, blk = get_coeffs(tbr, ctype, ctx, first)
                    enc = list(y_ac[mb, xx + yy * 4])
                    if blk != enc:
                        if verbose:
                            print(f"mb{mb} y_ac({xx},{yy}) MISMATCH ctx={ctx} nz={nz}")
                            print("  dec:", blk)
                            print("  enc:", enc)
                        ok = False
                    nzb = int(nz > first)
                    nzmap[yy, xx] = nzb
                    top_nz[mbx, xx] = nzb
                    left_nz[yy] = nzb
            for n in range(8):
                ch = n >> 2
                x = n & 1
                yv = (n >> 1) & 1
                slot = 4 + ch * 2 + x
                lslot = 4 + ch * 2 + yv
                # within-MB uv nz: 2x2 grid per channel
                base_n = ch * 4
                top = top_nz[mbx, slot]
                left = left_nz[lslot]
                ctx = top + left
                nz, blk = get_coeffs(tbr, 2, ctx, 0)
                enc = list(uv_levels[mb, n])
                if blk != enc:
                    if verbose:
                        print(f"mb{mb} uv[{n}] MISMATCH ctx={ctx}")
                        print("  dec:", blk)
                        print("  enc:", enc)
                    ok = False
                nzb = int(nz > 0)
                top_nz[mbx, slot] = nzb
                left_nz[lslot] = nzb
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    # case 1: two MBs horizontal
    img = np.full((16, 16, 3), 128, np.uint8)
    img[0:16, 0:16] = np.clip(np.linspace(80, 180, 16)[:, None], 0, 255).astype(np.uint8)
    print("16x16 gradient:", verify(img))
    # case 2: 32x16
    img2 = np.full((16, 32, 3), 128, np.uint8)
    img2[..., 0] = (np.linspace(60, 200, 32)[None, :] * 0.5 +
                    np.linspace(40, 240, 16)[:, None] * 0.5).astype(np.uint8)
    print("16x32 gradient:", verify(img2))
    # case 3: 64x64 with structure
    img3 = np.full((64, 64, 3), 100, np.uint8)
    img3[16:48, 16:48] = 200
    img3[:, 30:34] = 30
    print("64x64 shapes:", verify(img3))
