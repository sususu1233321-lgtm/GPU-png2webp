
import time
t0 = time.time()
import numpy as np
import gpuwebp
from gpuwebp.vp8_encode import select_modes
from gpuwebp.closed_loop_jit import closed_loop_full
print("loaded in %.1fs" % (time.time()-t0))
sse = np.zeros((16,16,10), np.int64)
r = select_modes(sse, np.ones(16), np.zeros(16, np.uint8), 4, 4, 1000)
print("numba exec ok:", r[0].shape)
t0 = time.time()
y = np.zeros((16,16), np.int16)
closed_loop_full(y, np.zeros((8,8), np.int16), np.zeros((8,8), np.int16),
                 np.zeros(1, bool), np.zeros(1, np.uint8), np.zeros(1, np.uint8),
                 np.zeros((1,16), np.uint8), *([np.ones(16, np.int64)]*5)*3,
                 np.ones(2, np.int64), np.ones(2, np.int64), np.ones(2, np.int64))
print("closed_loop_full JIT: %.1fs" % (time.time()-t0))
print("CUDA src:", "closed_loop_kernel" in gpuwebp.closed_loop_gpu._CUDA_SRC if False else __import__("importlib").import_module("gpuwebp.closed_loop_gpu")._CUDA_SRC)
