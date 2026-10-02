"""async submit_batch (slot path) uniform-size sanity + bit-exact vs sync."""
import sys, ctypes, os, glob, random, struct
sys.path.insert(0, ".")
import numpy as np, imagecodecs
lib = ctypes.CDLL(os.path.abspath(r"cpp\gpu_pipeline_v2.dll"))
lib.gpu_set_device.restype = ctypes.c_int; lib.gpu_set_device.argtypes = [ctypes.c_int]
lib.submit_batch.restype = ctypes.c_int
lib.submit_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*7
lib.process_batch_ptrs.restype = ctypes.c_int
lib.process_batch_ptrs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*7
lib.poll_batch.restype = ctypes.c_int; lib.poll_batch.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.gpu_set_device(0)
P = lambda x: x.ctypes.data_as(ctypes.c_void_p)
random.seed(7)
arrs = []
while len(arrs) < 8:
    f = random.choice(glob.glob("D:/gpuimgtest3/*.png"))
    head = open(f, "rb").read(26)
    w, h = struct.unpack(">II", head[16:24])
    if (w, h) != (832, 1216):
        continue
    a = imagecodecs.png_decode(open(f, "rb").read())
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:2]+(1,), 255, np.uint8)], -1)
    arrs.append(a)
n = len(arrs); nmb = 76*52; tot = n*nmb
ref = [np.zeros(tot*16, np.int16), np.zeros(tot*256, np.int16), np.zeros(tot*128, np.int16),
       np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot*16, np.uint8)]
lib.process_batch_ptrs((ctypes.c_void_p*n)(*[P(a) for a in arrs]), n, 832, 1216, 90, *map(P, ref))
out = [np.zeros_like(x) for x in ref]
ipa = (ctypes.c_void_p*n)(*[P(a) for a in arrs])
rid = lib.submit_batch(ipa, n, 832, 1216, 90, *map(P, out))
err = ctypes.c_int()
got = lib.poll_batch(-1, ctypes.byref(err))
ok = all(np.array_equal(a, b) for a, b in zip(ref, out))
print(f"rid={rid} poll={got} err={err.value} ASYNC UNIFORM BIT-EXACT: {ok}")
