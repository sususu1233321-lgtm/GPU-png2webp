import sys, ctypes, os, struct, glob, random
sys.path.insert(0, '.')
import numpy as np, imagecodecs

lib = ctypes.CDLL(os.path.abspath(r'cpp\gpu_pipeline_v2.dll'))
lib.gpu_set_device.restype = ctypes.c_int
lib.gpu_set_device.argtypes = [ctypes.c_int]
lib.process_batch_ptrs.restype = ctypes.c_int
lib.process_batch_ptrs.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7
lib.submit_batch_padded.restype = ctypes.c_int
lib.submit_batch_padded.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7
lib.poll_batch.restype = ctypes.c_int
lib.poll_batch.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.gpu_set_device(0)

random.seed(11)
by_size = {}
for f in random.sample(sorted(glob.glob('D:/gpuimgtest3/*.png')), 1500):
    head = open(f, 'rb').read(26)
    w, h = struct.unpack('>II', head[16:24])
    if (w, h) == (832, 1216) or w % 16 or h % 16 or w > 1536 or h > 1920:
        continue
    a = imagecodecs.png_decode(open(f, 'rb').read())
    if a.shape[-1] == 3:
        a = np.concatenate([a, np.full(a.shape[:2]+(1,), 255, np.uint8)], -1)
    by_size.setdefault((w, h), []).append(a)
sizes = sorted(by_size, key=lambda k: -len(by_size[k]))[:8]
arrs = []
for k in sizes:
    arrs.extend(by_size[k][:3])
print(f'{len(arrs)} imgs, {len(sizes)} distinct real dims:', sorted(set((a.shape[1], a.shape[0]) for a in arrs)))

P = lambda x: x.ctypes.data_as(ctypes.c_void_p)
err = ctypes.c_int()
Wp = (max(a.shape[1] for a in arrs) + 15) // 16 * 16
Hp = (max(a.shape[0] for a in arrs) + 15) // 16 * 16
print('pad:', Wp, 'x', Hp)
wr = (ctypes.c_int * len(arrs))(*[a.shape[1] for a in arrs])
hr = (ctypes.c_int * len(arrs))(*[a.shape[0] for a in arrs])
nmb = [(a.shape[0]//16)*(a.shape[1]//16) for a in arrs]
tot = sum(nmb)
offs = [0]
for m in nmb: offs.append(offs[-1] + m)

ok_all = True
for q in (90, 75, 95):
    refs = []
    for a in arrs:
        h, w = a.shape[:2]
        n = (h//16)*(w//16)
        o = (np.zeros(n*16, np.int16), np.zeros(n*256, np.int16), np.zeros(n*128, np.int16),
             np.zeros(n, np.uint8), np.zeros(n, np.uint8), np.zeros(n, np.uint8), np.zeros(n*16, np.uint8))
        lib.process_batch_ptrs((ctypes.c_void_p*1)(a.ctypes.data_as(ctypes.c_void_p)), 1, w, h, q, *map(P, o))
        refs.append(o)
    outs = (np.zeros(tot*16, np.int16), np.zeros(tot*256, np.int16), np.zeros(tot*128, np.int16),
            np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot, np.uint8), np.zeros(tot*16, np.uint8))
    ipa = (ctypes.c_void_p*len(arrs))(*[a.ctypes.data_as(ctypes.c_void_p) for a in arrs])
    rid = lib.submit_batch_padded(ipa, len(arrs), wr, hr, Wp, Hp, q, *map(P, outs))
    if rid <= 0:
        print('submit failed', rid); sys.exit(1)
    lib.poll_batch(-1, ctypes.byref(err))
    per = [16, 256, 128, 1, 1, 1, 16]
    bad = 0
    for i in range(len(arrs)):
        for k in range(7):
            lo, hi = offs[i]*per[k], offs[i+1]*per[k]
            if not np.array_equal(outs[k][lo:hi], refs[i][k]):
                bad += 1
                if bad <= 2:
                    print(f'q={q} img{i} arr{k} DIFF {(outs[k][lo:hi]!=refs[i][k]).sum()}')
                break
    print(f'q={q}: {len(arrs)-bad}/{len(arrs)} images bit-exact')
    ok_all = ok_all and bad == 0
print('MIXED-DIMS PADDED BIT-EXACT:', ok_all)
