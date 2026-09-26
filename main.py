"""GPU压图 入口:无参数启动图形界面,带参数运行命令行批处理。"""
import multiprocessing
import os
import sys


def _frozen_setup():
    appdir = os.path.dirname(sys.executable)
    # cupy 在冻结环境下把 CUDA bin 路径解析到程序目录并直接
    # os.add_dll_directory(不检查存在性) — 预建空目录避免 FileNotFoundError
    try:
        os.makedirs(os.path.join(appdir, "bin"), exist_ok=True)
    except OSError:
        pass
    # 常驻 cupy 内核缓存(首次运行现场编译,之后直接复用)
    try:
        cache = os.path.join(appdir, "cupy_cache")
        os.makedirs(cache, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", cache)
    except OSError:
        pass


if getattr(sys, "frozen", False):
    _frozen_setup()

    # cupy/cuda.pathfinder 会用 "python -m <module>" 子进程做 DLL 探测;
    # 冻结环境下 sys.executable 是本 exe,需要在这里代理执行模块。
    if len(sys.argv) > 2 and sys.argv[1] == "-m":
        import runpy
        _mod = sys.argv[2]
        sys.argv = [_mod] + sys.argv[3:]
        runpy.run_module(_mod, run_name="__main__")
        sys.exit(0)

def _prewarm():
    """加密字节码模块每次进程启动都要 numba JIT(约 20-25 秒),
    在后台线程提前触发,与启动/GUI 初始化重叠。"""
    try:
        import numpy as np
        from gpuwebp.closed_loop_jit import closed_loop_full
        from gpuwebp.vp8_encode import select_modes, write_token_partition
        from gpuwebp.bool_coder import bool_encode
        from gpuwebp.alpha_enc import make_alph_chunk
        from gpuwebp.pngdec import _defilter
        closed_loop_full(np.zeros((16, 16), np.int16),
                         np.zeros((8, 8), np.int16),
                         np.zeros((8, 8), np.int16),
                         np.zeros(1, bool), np.zeros(1, np.uint8),
                         np.zeros(1, np.uint8), np.zeros((1, 16), np.uint8),
                         *([np.ones(16, np.int64)] * 5) * 3,
                         np.ones(2, np.int64), np.ones(2, np.int64),
                         np.ones(2, np.int64))
        select_modes(np.zeros((16, 16, 10), np.int64), np.ones(16),
                     np.zeros(16, np.uint8), 4, 4, 1000)
        bool_encode(np.array([256, 0], np.int32), np.empty(64, np.uint8))
        make_alph_chunk(np.full((8, 8), 255, np.uint8))
        _defilter(np.zeros((2, 17), np.uint8), np.zeros(32, np.uint8),
                  2, 16, 4)
    except Exception:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    import threading
    threading.Thread(target=_prewarm, daemon=True).start()
    from gpuwebp.app import main
    main()
