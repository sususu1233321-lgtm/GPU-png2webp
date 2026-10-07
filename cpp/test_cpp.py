"""Test the C++ GPU pipeline DLL against the Python/cupy version."""
import ctypes
import sys
import time
import numpy as np

sys.path.insert(0, '.')

DLL_PATH = r'cpp\gpu_pipeline.dll'


def main():
    import imagecodecs
    import glob

    lib = ctypes.CDLL(DLL_PATH)

    # load test images
    files = sorted(glob.glob('test_batch/in/*.png'))
    datas = [imagecodecs.png_decode(open(f, 'rb').read()) for f in files[:12]]
    arrs = []
    for a in datas:
        if a.shape[-1] == 3:
            a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
        arrs.append(a)
    n = len(arrs)
    H, W = arrs[0].shape[:2]
    n_mb = (H // 16) * (W // 16)

    # concatenate into single buffer
    rgba = np.concatenate([a.reshape(-1) for a in arrs])
    rgba_pinned = np.ascontiguousbuf = rgba.copy()  # ensure contiguous

    # allocate output buffers
    y_dc = np.zeros(n * n_mb * 16, np.int16)
    y_ac = np.zeros(n * n_mb * 256, np.int16)
    uv_lv = np.zeros(n * n_mb * 128, np.int16)
    is_i4 = np.zeros(n * n_mb, np.uint8)
    i16_mode = np.zeros(n * n_mb, np.uint8)
    uv_mode = np.zeros(n * n_mb, np.uint8)
    i4_modes = np.zeros(n * n_mb * 16, np.uint8)

    # set up function signature
    lib.process_batch.restype = ctypes.c_int
    lib.process_batch.argtypes = [
        ctypes.c_char_p,  # rgba
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, W, H, quality
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # y_dc, y_ac, uv_lv
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # modes
    ]
    lib.benchmark_batch.restype = ctypes.c_double
    lib.benchmark_batch.argtypes = [
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]

    # benchmark
    rgba_ptr = rgba_pinned.tobytes()  # keep reference alive
    ms = lib.benchmark_batch(rgba_ptr, n, W, H, 90, 5)
    print(f'C++ GPU pipeline: {ms:.1f}ms per batch of {n} '
          f'({ms/n:.2f}ms/img → {1000/(ms/n):.0f} img/s)')

    # run once and compare with Python version
    ret = lib.process_batch(
        rgba_ptr, n, W, H, 90,
        y_dc.ctypes.data_as(ctypes.c_void_p),
        y_ac.ctypes.data_as(ctypes.c_void_p),
        uv_lv.ctypes.data_as(ctypes.c_void_p),
        is_i4.ctypes.data_as(ctypes.c_void_p),
        i16_mode.ctypes.data_as(ctypes.c_void_p),
        uv_mode.ctypes.data_as(ctypes.c_void_p),
        i4_modes.ctypes.data_as(ctypes.c_void_p))
    print(f'process_batch returned: {ret}')

    if ret == 0:
        # compare with Python/cupy pipeline
        from gpuwebp import gpu_engine as GE, vp8_encode as E
        from gpuwebp import vp8_tables as T
        from gpuwebp.closed_loop_gpu import mode_search_batch_gpu, closed_loop_batch_gpu
        import cupy as cp

        bq, y1, y2, uv_m, fl = E.setup_quant(90)
        y2ac = max(8, int(T.AC_TABLE2[bq]))
        y1deq = np.array([T.DC_TABLE[bq]] + [T.AC_TABLE[bq]]*15, np.int64)
        y2deq = np.array([T.DC_TABLE[bq]*2] + [y2ac]*15, np.int64)
        uvdeq = np.array([T.DC_TABLE[max(0,min(117,bq-2))]] + [T.AC_TABLE[bq]]*15, np.int64)

        with cp.cuda.Device(0):
            rgb = cp.asarray(np.stack(arrs))
            ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb, int16_out=True)
            Ybs = cp.stack(list(ypl)); Ubs = cp.stack(list(upl)); Vbs = cp.stack(list(vpl))
            raw = mode_search_batch_gpu(Ybs, Ubs, Vbs, y1)

            from gpuwebp.vp8_encode import select_modes
            n_mb2 = mb_h = mb_w = 0
            mb_h = H // 16; mb_w = W // 16; n_mb2 = mb_h * mb_w
            modes_list = []
            for i in range(n):
                sse = np.ascontiguousarray(raw['sse4'][i])
                i4m, isf, _ = select_modes(sse,
                    raw['i16_score'][i*n_mb2:(i+1)*n_mb2],
                    raw['i16_mode'][i*n_mb2:(i+1)*n_mb2], mb_w, mb_h, 1000)
                modes_list.append(dict(is_i4=isf.astype(bool),
                    i16_mode=raw['i16_mode'][i*n_mb2:(i+1)*n_mb2],
                    uv_mode=raw['uv_mode'][i*n_mb2:(i+1)*n_mb2], i4_modes=i4m))
            py_ydc, py_yac, py_uvlv = closed_loop_batch_gpu(
                Ybs, Ubs, Vbs, modes_list, y1, y2, uv_m, y1deq, y2deq, uvdeq)

        # compare
        print('\nPython vs C++ comparison:')
        py_dc = py_ydc.reshape(n, -1)
        py_ac = py_yac.reshape(n, -1)
        print(f'  is_i4 match: {np.array_equal(is_i4.reshape(n,-1), np.concatenate([m["is_i4"].astype(np.uint8) for m in modes_list]).reshape(n,-1))}')
        print(f'  i16_mode match: {np.array_equal(i16_mode.reshape(n,-1), np.concatenate([m["i16_mode"] for m in modes_list]).reshape(n,-1))}')
        print(f'  y_dc nonzero (C++): {(y_dc != 0).sum()}, (Python): {(py_dc != 0).sum()}')
        print(f'  y_ac nonzero (C++): {(y_ac != 0).sum()}, (Python): {(py_ac != 0).sum()}')


if __name__ == '__main__':
    main()
