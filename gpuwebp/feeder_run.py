"""C++ feeder pipeline runner: decode+group+submit run entirely in C++
threads (zero GIL in the feed path); Python only drains completed GPU
batches, runs the batch entropy DLL call and assembles/writes outputs.

Outputs are byte-identical to the legacy Python feed path (verified by the
same bit-exact harness); the GPU defilter input path is bit-exact with the
RGBA path.
"""
import ctypes
import os
import threading

import numpy as np

from . import vp8_encode
from .encoder import _container
from .alpha_enc import make_alph_chunk
from .png_meta import extract_meta as _py_meta
from .pngdec_cpp import _parse as _parse_meta_blob


class Feeder:
    def __init__(self, root):
        self.lib = ctypes.CDLL(os.path.join(root, "cpp", "feeder.dll"))
        self.pngdec = os.path.join(root, "cpp", "pngdec.dll").encode("utf-8")
        self.gpu = os.path.join(root, "cpp", "gpu_pipeline_v2.dll").encode("utf-8")
        L = self.lib
        L.feeder_start.restype = ctypes.c_int
        L.feeder_start.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_char_p]
        L.feeder_poll.restype = ctypes.c_void_p
        L.feeder_poll.argtypes = [ctypes.c_int]
        L.feeder_busy.restype = ctypes.c_int
        L.feeder_unsupported_count.restype = ctypes.c_int
        L.feeder_unsupported_get.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.feeder_batch_info.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 5
        L.feeder_batch_ptr.restype = ctypes.c_void_p
        L.feeder_batch_ptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.feeder_batch_paths.restype = ctypes.c_int
        L.feeder_batch_paths.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        L.feeder_batch_meta.restype = ctypes.c_int
        L.feeder_batch_meta.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        L.feeder_batch_nok.restype = ctypes.c_int
        L.feeder_batch_nok.argtypes = [ctypes.c_void_p]
        L.feeder_batch_release.argtypes = [ctypes.c_void_p]

    def start(self, files, quality, batch=64, threads=6, device=0):
        blob = b"\0".join(f.encode("utf-8") for f in files) + b"\0"
        self.lib.feeder_start_d.restype = ctypes.c_int
        self.lib.feeder_start_d.argtypes = (
            [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int])
        return self.lib.feeder_start_d(blob, len(files), quality, batch,
                                       threads, self.pngdec, self.gpu,
                                       device)

    def unsupported(self):
        n = self.lib.feeder_unsupported_count()
        if n <= 0:
            return []
        arr = (ctypes.c_int * n)()
        self.lib.feeder_unsupported_get(arr)
        return list(arr)

    def poll(self, block_ms):
        return self.lib.feeder_poll(block_ms)

    def busy(self):
        return self.lib.feeder_busy()


_NP2CT = {np.int16: ctypes.c_int16, np.uint8: ctypes.c_uint8}

def _wrap(ptr, dtype, count):
    return np.ctypeslib.as_array(
        ctypes.cast(ptr, ctypes.POINTER(_NP2CT[dtype])), shape=(count,))


def n_mb_of(bn, W, H):
    return (H // 16) * (W // 16)


def run_feeder(files, dst, base, quality, min_psnr, verify_meta,
               stats, log, progress_cb, stop_event, verify_pool, batch=96,
               device=0):
    """Feed+encode the conforming subset via the C++ feeder. Returns the
    number processed; unsupported files are reported via stats by caller."""
    import imagecodecs
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fd = Feeder(root)
    bq, y1, y2, uv_m, fl = vp8_encode.setup_quant(quality)

    # entropy batch dll
    ent = ctypes.CDLL(os.path.join(root, "cpp", "entropy.dll"))
    ent.encode_vp8_batch.restype = ctypes.c_int
    ent.encode_vp8_batch.argtypes = (
        [ctypes.c_int] * 5 + [ctypes.c_void_p] * 8
        + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])

    import multiprocessing as mp
    from . import verifyworker as _vw
    verify_results = []
    P = ctypes.c_void_p
    if verify_pool is None and min_psnr > 0:
        # local subprocess pool (path-based payloads; zero big pickles)
        try:
            import sys as _sys
            _frozen = getattr(_sys, "frozen", False)
            _main = getattr(_sys.modules.get("__main__"), "__file__", None)
            if (not _frozen and (not _main or _main.endswith("<stdin>")
                                 or not os.path.isfile(_main))):
                raise RuntimeError("interactive")
            ctx = mp.get_context("spawn")
            nvp = max(2, min(8, (os.cpu_count() or 8) // 2))
            verify_pool = ctx.Pool(nvp)
        except Exception:                           # noqa: BLE001
            verify_pool = None

    if fd.start(files, quality, batch, device=device) != 0:
        log("C++供给器启动失败，回退Python路径")
        return -1

    def finish_one(path, W, H, meta, vp8, aplane):
        try:
            rel = os.path.relpath(path, base)
            out_path = os.path.join(dst, os.path.splitext(rel)[0] + ".webp")
            alpha = None
            if aplane:
                import imagecodecs as _ic
                arr = _ic.png_decode(open(path, "rb").read())
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, -1)
                if arr.shape[-1] == 3:
                    arr = np.concatenate(
                        [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)],
                        -1)
                a = arr[..., 3]
                alpha = None if bool((a == 255).all()) else make_alph_chunk(a)
            webp = _container(vp8, alpha, meta)
            _d = os.path.dirname(out_path)
            if _d and _d not in _made_dirs:
                os.makedirs(_d, exist_ok=True)
                _made_dirs.add(_d)
            with open(out_path, "wb") as f:
                f.write(webp)
            with stats.lock:
                stats.done += 1
                stats.dst_bytes += len(webp)
            if verify_pool is not None and min_psnr > 0:
                payload = (path, webp, verify_meta, min_psnr, meta)

                def _on(result, p=path):
                    ok, note = result
                    if ok:
                        return
                    log(f"校验未过 {os.path.basename(p)}: {note}")

                verify_results.append(
                    verify_pool.apply_async(_vw.verify_payload, (payload,),
                                            callback=_on))
        except Exception as e:                       # noqa: BLE001
            log(f"feeder finish 失败 {path}: {e}")
            with stats.lock:
                stats.failed += 1

    from concurrent.futures import ThreadPoolExecutor
    fin_ex = ThreadPoolExecutor(max_workers=12)
    fin_futs = []
    _made_dirs = set()
    processed = 0
    done_images = 0
    failed_paths = []
    n = len(files)
    n_expected = n - max(0, len(fd.unsupported()))

    def _process_batch(h, bn, W, H, n_mb, paths, metas,
                       y_dc, y_ac, uv_lv, is_i4, i16m, uvm, i4m, aflags):
        try:
            skip_b = ~(y_dc.view(np.uint8).reshape(bn, n_mb, 32).any(-1)
                       | y_ac.view(np.uint8).reshape(bn, n_mb, 512).any(-1)
                       | uv_lv.view(np.uint8).reshape(bn, n_mb, 256).any(-1))
            cap = n_mb * 2400 + 8192
            outs = np.empty(bn * cap, np.uint8)
            lens = np.empty(2 * bn, np.int32)
            ptrs = lambda arr, per: (P * bn)(
                *[arr[k * per:(k + 1) * per].ctypes.data_as(P)
                  for k in range(bn)])
            ret = ent.encode_vp8_batch(
                bn, W // 16, H // 16, bq, fl,
                ptrs(skip_b.reshape(-1), n_mb), ptrs(is_i4, n_mb),
                ptrs(i16m, n_mb), ptrs(uvm, n_mb), ptrs(i4m, n_mb * 16),
                ptrs(y_dc, n_mb * 16), ptrs(y_ac, n_mb * 256),
                ptrs(uv_lv, n_mb * 128),
                (P * bn)(*[outs[k * cap:(k + 1) * cap].ctypes.data_as(P)
                           for k in range(bn)]),
                cap, lens.ctypes.data_as(P))
            if ret != 0 or not (lens[:bn] >= 0).all():
                log(f"feeder熵编码失败 {ret}")
                failed_paths.extend(paths)
                return   # release happens in finally -- never double-free
            aptr = fd.lib.feeder_batch_aplane
            aptr.restype = ctypes.c_void_p
            aptr.argtypes = [ctypes.c_void_p, ctypes.c_int]
            for k in range(bn):
                p0 = int(lens[k]); tot = int(lens[bn + k])
                vp8 = (((1 << 4) | (p0 << 5)).to_bytes(3, "little")
                       + (0x9D012A).to_bytes(3, "big")
                       + (W & 0x3FFF).to_bytes(2, "little")
                       + (H & 0x3FFF).to_bytes(2, "little")
                       + outs[k * cap:k * cap + tot].tobytes())
                if len(vp8) & 1:
                    vp8 += bytes([0])
                fin_futs.append(fin_ex.submit(
                    finish_one, paths[k], W, H, metas[k], vp8,
                    int(aflags[k])))
        finally:
            fd.lib.feeder_batch_release(h)

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        h = fd.poll(200)
        if not h:
            if done_images + len(failed_paths) < n_expected:
                continue
            if fd.busy():
                continue
            h = fd.poll(1500)
            if not h:
                break
        ci = (ctypes.c_int * 5)()
        pinfos = [ctypes.cast(ctypes.byref(ci, k * 4),
                              ctypes.POINTER(ctypes.c_int))
                  for k in range(5)]
        fd.lib.feeder_batch_info(h, *pinfos)
        bn, W, H, bpp, req_id = list(ci)
        nok = fd.lib.feeder_batch_nok(h)
        pb = ctypes.create_string_buffer(bn * 1024)
        plen = fd.lib.feeder_batch_paths(h, pb, bn * 1024)
        paths = [q.decode('utf-8') for q in pb.raw[:plen].split(b'\0') if q]
        mb = np.empty(bn * (4 << 20), np.uint8)
        mlen = fd.lib.feeder_batch_meta(h, mb.ctypes.data_as(P), mb.size)
        metas = []
        off = 0
        for _k in range(bn):
            ln = int.from_bytes(mb[off:off + 4].tobytes(), "little"); off += 4
            metas.append(_parse_meta_blob(mb[off:off + ln]) if ln else {})
            off += ln
        y_dc = _wrap(fd.lib.feeder_batch_ptr(h, 0), np.int16, bn * n_mb_of(bn, W, H) * 16)
        nmb = n_mb_of(bn, W, H)
        y_ac = _wrap(fd.lib.feeder_batch_ptr(h, 1), np.int16, bn * nmb * 256)
        uv_lv = _wrap(fd.lib.feeder_batch_ptr(h, 2), np.int16, bn * nmb * 128)
        is_i4 = _wrap(fd.lib.feeder_batch_ptr(h, 3), np.uint8, bn * nmb)
        i16m = _wrap(fd.lib.feeder_batch_ptr(h, 4), np.uint8, bn * nmb)
        uvm = _wrap(fd.lib.feeder_batch_ptr(h, 5), np.uint8, bn * nmb)
        i4m = _wrap(fd.lib.feeder_batch_ptr(h, 6), np.uint8, bn * nmb * 16)
        aflags = _wrap(fd.lib.feeder_batch_ptr(h, 7), np.uint8, bn)
        if nok or req_id == 0:
            failed_paths.extend(paths)
            fd.lib.feeder_batch_release(h)
            log(f"feeder批失败(req={req_id} nok={nok})，回退 {bn} 张")
            done_images += bn
            continue
        fin_futs.append(fin_ex.submit(_process_batch, h, bn, W, H, nmb,
                                      paths, metas, y_dc, y_ac, uv_lv,
                                      is_i4, i16m, uvm, i4m, aflags))
        processed += bn
        done_images += bn
        progress_cb and progress_cb(stats)
    for f in fin_futs:
        try:
            f.result(timeout=120)
        except Exception:                           # noqa: BLE001
            pass
    fin_ex.shutdown(wait=True)
    for r in verify_results:
        try:
            r.wait(timeout=300)
        except Exception:                            # noqa: BLE001
            pass
    return processed, failed_paths
