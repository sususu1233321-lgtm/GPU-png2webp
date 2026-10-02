import sys, ctypes, os, struct, glob, random
sys.path.insert(0, '.')
import numpy as np, imagecodecs
lib = ctypes.CDLL(os.path.abspath(r'cpp\gpu_pipeline_v2.dll'))
lib.gpu_set_device.restype = ctypes.c_int; lib.gpu_set_device.argtypes = [ctypes.c_int]
lib.submit_batch_padded.restype = ctypes.c_int
lib.submit_batch_padded.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7
lib.poll_batch.restype = ctypes.c_int; lib.poll_batch.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.gpu_set_device(0)
P = lambda x: x.ctypes.data_as(ctypes.c_void_p)
random.seed(1)
by_size = {}
for f in random.sample(sorted(glob.glob('D:/gpuimgtest3/*.png')), 1500):
    head = open(f, 'rb').read(26)
    w, h = struct.unpack('>II', head[16:24])
    if (w, h) == (832, 1216) or w % 16 or h % 16 or w > 1536 or h > 1920: continue
    a = imagecodecs.png_decode(open(f, 'rb').read())
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:2]+(1,), 255, np.uint8)], -1)
    by_size.setdefault((w, h), []).append(a)
sizes = sorted(by_size, key=lambda k: -len(by_size[k]))[:2]
arrs = by_size[sizes[0]][:1] + by_size[sizes[1]][:1]
Wp = (max(x.shape[1] for x in arrs)+15)//16*16; Hp = (max(x.shape[0] for x in arrs)+15)//16*16
wr = (ctypes.c_int*2)(arrs[0].shape[1], arrs[1].shape[1])
hr = (ctypes.c_int*2)(arrs[0].shape[0], arrs[1].shape[0])
nmb = [(x.shape[0]//16)*(x.shape[1]//16) for x in arrs]; tot = sum(nmb)
outs = (np.zeros(tot*16, np.int16), np.zeros(tot*256, np.int16), np.zeros(tot*128, np.int16),
        np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot*16, np.uint8))
ipa = (ctypes.c_void_p*2)(*[x.ctypes.data_as(ctypes.c_void_p) for x in arrs])
err = ctypes.c_int()
rid = lib.submit_batch_padded(ipa, 2, wr, hr, Wp, Hp, 90, *map(P, outs))
print('rid', rid, flush=True)
lib.poll_batch(-1, ctypes.byref(err))
print('err', err.value, flush=True)
