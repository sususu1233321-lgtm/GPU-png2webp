"""Compare C++ GPU pipeline (gpu_pipeline_v2.dll) against the Python/cupy
reference, bit-exact, across images x qualities x batch sizes."""
import sys, ctypes, glob
sys.path.insert(0, '.')
import numpy as np, imagecodecs, cupy as cp
from gpuwebp import gpu_engine as GE
from gpuwebp import vp8_encode as E
from gpuwebp import vp8_tables as T
from gpuwebp.closed_loop_gpu import mode_search_batch_gpu, closed_loop_batch_gpu
from gpuwebp.vp8_encode import select_modes

lib = ctypes.CDLL(r'cpp\gpu_pipeline_v2.dll')
lib.process_batch.restype = ctypes.c_int
lib.process_batch.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7

def py_ref(arrs, W, H, quality):
    n = len(arrs); n_mb = (H//16)*(W//16)
    bq, y1, y2, uv_m, fl = E.setup_quant(quality)
    y2ac = max(8, int(T.AC_TABLE2[bq]))
    y1deq = np.array([T.DC_TABLE[bq]] + [T.AC_TABLE[bq]]*15, np.int64)
    y2deq = np.array([T.DC_TABLE[bq]*2] + [y2ac]*15, np.int64)
    uvdeq = np.array([T.DC_TABLE[max(0,min(117,bq-2))]] + [T.AC_TABLE[bq]]*15, np.int64)
    with cp.cuda.Device(0):
        rgb = cp.asarray(np.stack(arrs))
        ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb, int16_out=True)
        Ybs, Ubs, Vbs = cp.stack(list(ypl)), cp.stack(list(upl)), cp.stack(list(vpl))
        raw = mode_search_batch_gpu(Ybs, Ubs, Vbs, y1)
    penalty = 1000*y1.q_avg*y1.q_avg
    modes_list = []
    for i in range(n):
        sse = np.ascontiguousarray(raw['sse4'][i])
        i4_modes, is_i4, _ = select_modes(sse, raw['i16_score'][i*n_mb:(i+1)*n_mb],
            raw['i16_mode'][i*n_mb:(i+1)*n_mb], raw['mb_w'], raw['mb_h'], penalty)
        modes_list.append(dict(is_i4=is_i4.astype(bool),
            i16_mode=raw['i16_mode'][i*n_mb:(i+1)*n_mb],
            uv_mode=raw['uv_mode'][i*n_mb:(i+1)*n_mb], i4_modes=i4_modes))
    with cp.cuda.Device(0):
        py_dc, py_ac, py_uv = closed_loop_batch_gpu(Ybs, Ubs, Vbs, modes_list,
                                                    y1, y2, uv_m, y1deq, y2deq, uvdeq)
    return modes_list, py_dc, py_ac, py_uv

def cpp_run(arrs, W, H, quality):
    n = len(arrs); n_mb = (H//16)*(W//16)
    y_dc = np.zeros(n*n_mb*16, np.int16); y_ac = np.zeros(n*n_mb*256, np.int16)
    uv_lv = np.zeros(n*n_mb*128, np.int16)
    is_i4 = np.zeros(n*n_mb, np.uint8); i16m = np.zeros(n*n_mb, np.uint8)
    uvm = np.zeros(n*n_mb, np.uint8); i4m = np.zeros(n*n_mb*16, np.uint8)
    buf = b''.join(a.tobytes() for a in arrs)
    ret = lib.process_batch(buf, n, W, H, quality,
        y_dc.ctypes.data_as(ctypes.c_void_p), y_ac.ctypes.data_as(ctypes.c_void_p),
        uv_lv.ctypes.data_as(ctypes.c_void_p), is_i4.ctypes.data_as(ctypes.c_void_p),
        i16m.ctypes.data_as(ctypes.c_void_p), uvm.ctypes.data_as(ctypes.c_void_p),
        i4m.ctypes.data_as(ctypes.c_void_p))
    assert ret == 0, ret
    return (is_i4, i16m, uvm, i4m.reshape(n*n_mb, 16),
            y_dc.reshape(n*n_mb, 16), y_ac.reshape(n*n_mb, 16, 16), uv_lv.reshape(n*n_mb, 8, 16))

if __name__ == '__main__':
    files = sorted(glob.glob('L:/图片备份8/nai3_240531/*.png'))
    import random
    random.seed(7)
    # only 16-aligned images are supported by the C++ dll
    cands = []
    for f in random.sample(files, 300):
        try:
            a = imagecodecs.png_decode(open(f, 'rb').read())
        except Exception:
            continue
        if a.ndim == 2: a = np.stack([a]*3, -1)
        if a.shape[-1] == 3: a = np.concatenate([a, np.full(a.shape[:2]+(1,), 255, np.uint8)], -1)
        h, w = a.shape[:2]
        if h % 16 == 0 and w % 16 == 0:
            cands.append((a, w, h, f))
        if len(cands) >= 6: break

    total_mb = 0; fails = 0
    for quality in (90, 75, 95):
        for bs in (1, 4, 12):
            for start in range(0, len(cands), bs):
                grp = cands[start:start+bs]
                arrs = [g[0] for g in grp]; H, W = grp[0][1+1], grp[0][1]
                if any(g[1] != W or g[2] != H for g in grp): continue
                ml, pdc, pac, puv = py_ref(arrs, W, H, quality)
                c = cpp_run(arrs, W, H, quality)
                n = len(arrs); n_mb = (H//16)*(W//16)
                py_is4 = np.concatenate([m['is_i4'].astype(np.uint8) for m in ml])
                py_i16 = np.concatenate([m['i16_mode'] for m in ml])
                py_uvm = np.concatenate([m['uv_mode'] for m in ml])
                py_i4m = np.concatenate([m['i4_modes'] for m in ml])
                checks = [
                    (c[0], py_is4, 'is_i4'), (c[1], py_i16, 'i16_mode'),
                    (c[2], py_uvm, 'uv_mode'), (c[3], py_i4m, 'i4_modes'),
                    (c[4], pdc.reshape(n*n_mb, 16), 'y_dc'),
                    (c[5], pac.reshape(n*n_mb, 16, 16), 'y_ac'),
                    (c[6], puv.reshape(n*n_mb, 8, 16), 'uv_lv')]
                bad = [nm for cc, pp, nm in checks if not np.array_equal(cc, pp)]
                total_mb += n * n_mb
                if bad:
                    fails += 1
                    print(f'FAIL q={quality} bs={bs} {grp[0][3].split(chr(92))[-1]}: {bad}')
    print(f'done: {total_mb} MBs checked, {fails} group failures')
