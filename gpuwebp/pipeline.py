"""Batch PNG -> WebP conversion pipeline with verification + CPU fallback.

Each image is encoded with the GPU engine, then verified:
  - decoded dimensions match
  - alpha channel bit-exact
  - PNG metadata chunks byte-identical (XMP/EXIF/ICCP round-trip)
  - RGB PSNR >= threshold (default 34 dB)
Any failure triggers a Pillow (libwebp) CPU re-encode with the same metadata
carried over, which is itself re-verified; if that also fails the original
file is reported and kept untouched.
"""
import io
import os
import ctypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .encoder import encode_rgba
from .png_meta import extract_meta, build_xmp, verify


class BatchStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.fallbacks = 0
        self.src_bytes = 0
        self.dst_bytes = 0
        self.total = 0
        self.start = None
        self.cur_name = ""
        self.min_psnr = 1e9

    def note(self, src, dst):
        with self.lock:
            self.done += 1
            self.src_bytes += src
            self.dst_bytes += dst

    @property
    def elapsed(self):
        return (time.time() - self.start) if self.start else 0.0

    @property
    def speed(self):
        return self.done / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def eta(self):
        remaining = self.total - self.done - self.failed - self.skipped
        return remaining / self.speed if self.speed > 0 else 0.0


import numpy as np
from numba import njit
from PIL import Image

from .encoder import encode_rgba
from .png_meta import extract_meta, build_xmp, verify


@njit(cache=True, nogil=True)
def _mse_u8(a, b):
    s = np.int64(0)
    for i in range(a.shape[0]):
        d = np.int64(a[i]) - np.int64(b[i])
        s += d * d
    return s


def _psnr_rgba(a, b):
    if a.shape != b.shape:
        return -1.0
    n = a.size
    mse = _mse_u8(np.ascontiguousarray(a).reshape(-1).view(np.uint8),
                  np.ascontiguousarray(b).reshape(-1).view(np.uint8)) / n
    return 99.0 if mse == 0 else 10 * np.log10(255 * 255 / mse)




def _park_failed(path, dst, log):
    """失败/无法转换的文件原样复制到 输出目录/未转换/ 子文件夹。"""
    import shutil
    try:
        park = os.path.join(dst, "未转换")
        os.makedirs(park, exist_ok=True)
        shutil.copy2(path, os.path.join(park, os.path.basename(path)))
    except OSError as e:
        log(f"[归档失败] {path}: {e}")

def _pillow_encode(arr, quality, meta):
    """CPU fallback: Pillow (libwebp) with the same metadata packets."""
    img = Image.fromarray(arr[..., :3] if arr.shape[-1] == 4 else arr, "RGB")
    if arr.shape[-1] == 4 and not bool((arr[..., 3] == 255).all()):
        img = Image.fromarray(arr, "RGBA")
    kw = dict(quality=quality, method=6)
    if meta is not None:
        if meta.get("texts") or meta.get("phys_raw") is not None:
            kw["xmp"] = build_xmp(meta)
        if meta.get("exif_raw"):
            kw["exif"] = meta["exif_raw"]
        if meta.get("icc_raw"):
            kw["icc_profile"] = meta["icc_raw"]
    buf = io.BytesIO()
    img.save(buf, "WEBP", **kw)
    return buf.getvalue()


def convert_one(png_path, out_path, quality, engine, device,
                min_psnr, verify_meta, log):
    """Returns ('ok'|'fallback'|'fail', webp_bytes_or_None, note)."""
    png_data = open(png_path, "rb").read()
    try:
        import imagecodecs
        arr = imagecodecs.png_decode(png_data)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
    except Exception:                                   # noqa: BLE001
        arr = None
    if arr is None:
        img = Image.open(io.BytesIO(png_data))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.getbands()
                              or "transparency" in img.info
                              else "RGB")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.shape[-1] == 3:
            arr = np.concatenate(
                [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
    meta = extract_meta(png_data)

    webp = None
    used_fallback = False
    gpu_reason = ""
    H, W = arr.shape[:2]
    if H % 2 == 0 and W % 2 == 0:
        try:
            webp = encode_rgba(arr, quality, engine=engine, device=device,
                               meta=meta)
        except Exception as e:                              # noqa: BLE001
            gpu_reason = f"GPU编码异常 {e}"
            log(f"GPU编码失败 {png_path}: {e}")
    else:
        gpu_reason = f"奇数尺寸 {W}x{H}"
        log(f"奇数尺寸 {W}x{H}，使用CPU引擎: {png_path}")

    def check(webp_bytes):
        dimg = Image.open(io.BytesIO(webp_bytes))
        darr = np.asarray(dimg.convert("RGBA"))
        if darr.shape[:2] != (H, W):
            return f"尺寸不符 {darr.shape[:2]}"
        if not np.array_equal(darr[..., 3], arr[..., 3]):
            return "alpha不一致"
        if verify_meta:
            ok, problems = verify(png_data, webp_bytes)
            if not ok:
                return "元数据校验失败: " + ";".join(problems)
        p = _psnr_rgba(arr, darr)
        if p < min_psnr:
            return f"PSNR {p:.2f} < {min_psnr}"
        return None

    note = None
    if webp is not None:
        note = check(webp)
    if webp is None or note is not None:
        reason = note or gpu_reason or "GPU不可用"
        try:
            webp = _pillow_encode(arr, quality, meta)
            note2 = check(webp)
            if note2 and not note2.startswith("PSNR"):
                return "fail", None, f"CPU兜底也失败: {note2} (先: {reason})"
            if note2:
                # Pillow(libwebp) itself can't reach the threshold on this
                # content — accept the reference encoder's output with a note
                used_fallback = True
                note = f"CPU兜底 ({reason}; 注意: {note2})"
            else:
                used_fallback = True
                note = f"CPU兜底 ({reason})"
        except Exception as e:                              # noqa: BLE001
            return "fail", None, f"{reason}; CPU兜底异常 {e}"
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(webp)
    return ("fallback" if used_fallback else "ok"), webp, note


def collect_pngs(src, recursive):
    out = []
    if recursive:
        for root, _dirs, files in os.walk(src):
            for fn in files:
                if fn.lower().endswith(".png"):
                    out.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(src):
            if fn.lower().endswith(".png"):
                out.append(os.path.join(src, fn))
    out.sort()
    return out


def run_batch(src, dst, quality=90, engine="gpu", device=1, recursive=False,
              skip_existing=True, min_psnr=34.0, verify_meta=True,
              workers=3, progress_cb=None, log_cb=None, stop_event=None,
              max_cores=None):
    """Convert all PNGs under src into dst (mirrors subfolder structure).
    progress_cb(stats) is called after every file; log_cb(str) for messages.
    Returns BatchStats."""
    if engine == "gpu-batch":
        if os.environ.get("FEEDER") == "1":   # opt-in; legacy is faster
                                                 # FEEDER=0 -> legacy path
            try:
                from .feeder_run import run_feeder
                from .pipeline import collect_pngs as _cp
                files = _cp(src, recursive)
                if files:
                    base = os.path.commonpath(files)
                    st = BatchStats()
                    st.total = len(files)
                    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
                    _dev = device
                    n_done, failed = run_feeder(
                        files, dst, base, quality, min_psnr, verify_meta,
                        st, log_cb or (lambda m: None), progress_cb,
                        stop_event, None, batch=64, device=_dev)
                    # leftovers: unsupported formats, batch failures, or any
                    # file whose output is missing -> legacy path
                    import os as _os
                    def _left(f):
                        rel = _os.path.relpath(f, base)
                        return not _os.path.exists(
                            _os.path.join(dst, _os.path.splitext(rel)[0] + ".webp"))
                    rest = [f for f in files if _left(f)]
                    if rest:
                        log_cb and log_cb(f"C++供给器完成 {n_done} 张，"
                                          f"{len(rest)} 张走回退路径")
                        st2 = run_batch_files_fast(
                            rest, dst, base=base, quality=quality,
                            device=device, skip_existing=skip_existing,
                            min_psnr=min_psnr, verify_meta=verify_meta,
                            batch=64, decode_workers=10, finish_workers=10,
                            progress_cb=progress_cb, log_cb=log_cb,
                            stop_event=stop_event)
                        with st.lock:
                            st.done += st2.done
                            st.failed += st2.failed
                            st.skipped += st2.skipped
                            st.fallbacks += st2.fallbacks
                            st.src_bytes += st2.src_bytes
                            st.dst_bytes += st2.dst_bytes
                        return st
                return st
            except ImportError:
                pass
        return run_batch_fast(src, dst, quality=quality, device=device,
                              recursive=recursive, skip_existing=skip_existing,
                              min_psnr=min_psnr, verify_meta=verify_meta,
                              batch=64, decode_workers=10, finish_workers=10,
                              progress_cb=progress_cb, log_cb=log_cb,
                              stop_event=stop_event, max_cores=max_cores)
    stats = BatchStats()
    log = log_cb or (lambda s: None)
    from .extfmt import collect_images
    files = collect_images(src, recursive)
    stats.total = len(files)
    stats.start = time.time()
    if not files:
        log("未找到PNG文件")
        return stats

    def work(png_path):
        if stop_event is not None and stop_event.is_set():
            return
        rel = os.path.relpath(png_path, src)
        out_path = os.path.join(dst, os.path.splitext(rel)[0] + ".webp")
        if skip_existing and os.path.exists(out_path):
            with stats.lock:
                stats.skipped += 1
            progress_cb and progress_cb(stats)
            return
        try:
            with open(png_path, "rb") as f:
                if f.read(8) != b"\x89PNG\r\n\x1a\n":
                    with stats.lock:
                        stats.skipped += 1
                    _park_failed(png_path, dst, log)
                    log(f"[跳过] 非PNG文件: {rel} (已复制到 未转换/)")
                    progress_cb and progress_cb(stats)
                    return
        except OSError as e:
            with stats.lock:
                stats.failed += 1
            _park_failed(png_path, dst, log)
            log(f"[失败] {rel}: 无法读取 {e} (原图已复制到 未转换/)")
            progress_cb and progress_cb(stats)
            return
        with stats.lock:
            stats.cur_name = os.path.basename(png_path)
        try:
            status, webp, note = convert_one(png_path, out_path, quality,
                                             engine, device, min_psnr,
                                             verify_meta, log)
            if status == "fail":
                with stats.lock:
                    stats.failed += 1
                _park_failed(png_path, dst, log)
                log(f"[失败] {rel}: {note} (原图已复制到 未转换/)")
            else:
                if status == "fallback":
                    with stats.lock:
                        stats.fallbacks += 1
                    log(f"[CPU兜底] {rel}: {note}")
                stats.note(os.path.getsize(png_path), len(webp))
        except Exception as e:                              # noqa: BLE001
            with stats.lock:
                stats.failed += 1
            _park_failed(png_path, dst, log)
            log(f"[异常] {rel}: {e} (原图已复制到 未转换/)")
        progress_cb and progress_cb(stats)

    if workers > 1 and engine == "gpu":
        # GPU kernels serialize on the device; extra threads mainly overlap
        # PNG decode / verification
        workers = min(workers, 4)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(work, files))
    return stats


# ---------------------------------------------------------------- fast batch
#
# 高吞吐模式:解码线程池 -> GPU 批量模式搜索(一次内核处理多张图) ->
# 收尾线程池(闭环量化+熵编码+校验+落盘),三段全并行,把 GPU 喂满。

def run_batch_fast(src, dst, quality=90, device=1, recursive=False,
                   skip_existing=True, min_psnr=34.0, verify_meta=True,
                   batch=16, decode_workers=4, finish_workers=8,
                   progress_cb=None, log_cb=None, stop_event=None,
                   max_cores=None):
    from .extfmt import collect_images
    files = collect_images(src, recursive)
    return run_batch_files_fast(files, dst, base=src, quality=quality, device=device,
                                skip_existing=skip_existing, min_psnr=min_psnr,
                                verify_meta=verify_meta, batch=batch,
                                max_cores=max_cores,
                                decode_workers=decode_workers,
                                finish_workers=finish_workers,
                                progress_cb=progress_cb, log_cb=log_cb,
                                stop_event=stop_event)


def run_batch_files_fast(files, dst, base=None, quality=90, device=1,
                         skip_existing=True, min_psnr=34.0, verify_meta=True,
                         batch=16, decode_workers=4, finish_workers=8,
                         progress_cb=None, log_cb=None, stop_event=None,
                         nvp=None, max_cores=None):
    import queue as _queue
    # Multi-GPU: indices follow nvidia-smi order (PCI_BUS_ID); the caller
    # picks the device (UI dropdown / --device). Users may still restrict
    # via CUDA_VISIBLE_DEVICES themselves. device<0 = auto (most VRAM).
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if device is None or device < 0:
        try:
            import cupy as _cp
            _best, _key = 0, (-1, -1)
            for _i in range(_cp.cuda.runtime.getDeviceCount()):
                _p = _cp.cuda.runtime.getDeviceProperties(_i)
                _k = (_p.get("totalGlobalMem", 0),
                      _p.get("multiProcessorCount", 0))
                if _k > _key:
                    _best, _key = _i, _k
            device = _best
        except Exception:
            device = 0
    import cupy as cp
    # keep the OS responsive: the encoder may eat every core, the shell
    # must never starve (three full-machine freezes taught us this)
    try:
        import ctypes as _ct
        _ct.windll.kernel32.SetPriorityClass(
            _ct.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass
    from .reswatch import Governor, mem_status as _memstat
    _gov = Governor(log=log_cb or (lambda m: None),
                    low_gb=8.0, crit_gb=3.0).start()
    _fin_sem = threading.Semaphore(16)   # max in-flight entropy batches
                                             # (8x230MB=1.8GB; input leak fixed)
    # pin cupy's current device too: without this its default-device init
    # grabs a context on the DISPLAY GPU (2080) even when we never use it
    try:
        cp.cuda.Device(device).use()
    except Exception:
        pass
    from . import gpu_engine as GE
    from . import vp8_encode, vp8_tables as T
    from .closed_loop_gpu import closed_loop_batch_gpu
    from .alpha_enc import make_alph_chunk
    from .encoder import _assemble, _container
    from .pngdec_cpp import extract_meta_cpp

    stats = BatchStats()
    log = log_cb or (lambda s: None)
    # bind proof: log the physical card this run actually uses, so a wrong
    # CUDA_VISIBLE_DEVICES mapping can never hide again
    try:
        _nm = cp.cuda.runtime.getDeviceProperties(device)["name"]
        _nm = _nm.decode("latin1", "replace") if isinstance(_nm, bytes) else str(_nm)
        _pci = cp.cuda.runtime.deviceGetPCIBusId(device)
        log(f"GPU绑定: {_nm.strip()} (device {device}, PCI {_pci})")
    except Exception:                           # noqa: BLE001
        pass
    stats.total = len(files)
    stats.start = time.time()
    if not files:
        log("未找到PNG文件")
        return stats
    if base is None:
        base = os.path.commonpath(files)

    # pre-group files by padded macroblock size and decode size-grouped:
    # each size's partial batch flushes the moment its last file is queued
    # (no starvation-timer idle while the GPU waits for stragglers)
    import struct as _struct
    _PNG_MAGIC = bytes((137, 80, 78, 71, 13, 10, 26, 10))

    def _padded_key(path):
        try:
            with open(path, "rb") as f:
                head = f.read(26)
            if head[:8] != _PNG_MAGIC or len(head) < 26:
                return None
            w, h = _struct.unpack(">II", head[16:24])
            return ((h + 15) // 16, (w + 15) // 16)
        except OSError:
            return None
    # single header pass: network/backup-folder corpora pay ~3ms per random
    # open, and the old code read each header three times (group_total,
    # real_total, sort key) -- minutes on a 30k-file slow drive
    group_total = {}
    real_total = {}               # exact (h, w) from headers -> count
    key_of = {}
    for p in files:
        try:
            with open(p, "rb") as f:
                _hd = f.read(26)
        except OSError:
            _hd = b""
        if _hd[:8] == _PNG_MAGIC and len(_hd) >= 24:
            _w, _h = _struct.unpack(">II", _hd[16:24])
            real_total[(_h, _w)] = real_total.get((_h, _w), 0) + 1
            k = ((_h + 15) // 16, (_w + 15) // 16)
        else:
            k = (-1, -1)          # unreadable: decode path will park it
        key_of[p] = k
        group_total[k] = group_total.get(k, 0) + 1
    files = sorted(files, key=lambda p: (key_of[p], p))
    group_queued = dict.fromkeys(group_total, 0)

    # ---- 2D-padding buckets for the mixed-size tail ----
    # Small sizes each form short batches whose wavefront latency caps the
    # GPU far below the main-size rate. Padding real content top-left into a
    # shared zero grid (portrait/landscape) lets the whole tail run as big
    # batches. The pad region never affects real MBs (only cross-MB read is
    # the i4 top-right, clamped at the real W inside the DLL), and outputs
    # are gathered back per real MB — bit-exact with the exact-size path.
    _real_bucket = {}             # (h, w) 16-mult -> ("__pad__", Hp, Wp)
    _bucket_total = {}            # bucket key -> file count (flush hint)
    _bucket_queued = {}
    _buckets_on = (os.environ.get("PADBUCKET") != "0")
    if _buckets_on:
        _EXACT_MIN = max(24, 3 * batch // 4)   # self-batching sizes stay exact
        _CAP_W, _CAP_H, _KMAX = 1536, 1920, 3  # bucket dims cap: one giant
        # image (4864x3328 in this corpus) would inflate a shared grid and
        # multiply everyone's padded work a hundredfold
        _tail = {k: c for k, c in real_total.items()
                 if c < _EXACT_MIN and k[0] % 16 == 0 and k[1] % 16 == 0
                 and k[1] <= _CAP_W and k[0] <= _CAP_H
                 and k[0] * k[1] <= 24_000_000}
        if real_total:                 # the dominant size always stays exact
            _tail.pop(max(real_total, key=lambda k: real_total[k]), None)

        def _bcost(mem):
            hp = max(k[0] for k in mem)
            wp = max(k[1] for k in mem)
            return sum(_tail[k] for k in mem) * hp * wp, hp, wp

        def _greedy_split(members, K):
            """Split into <=K groups minimizing total padded pixels; a split
            must cut cost by 15% to be worth another partial batch."""
            buckets = [sorted(members)]
            while len(buckets) < K:
                best = None
                for bi, b in enumerate(buckets):
                    if len(b) < 2:
                        continue
                    c0, _, _ = _bcost(b)
                    for key in (0, 1):
                        s = sorted(b, key=lambda k: (k[key], k[1 - key]))
                        for cut in range(1, len(s)):
                            c1, _, _ = _bcost(s[:cut])
                            c2, _, _ = _bcost(s[cut:])
                            red = c0 - c1 - c2
                            if best is None or red > best[0]:
                                best = (red, bi, s[:cut], s[cut:], c0)
                if best is None or best[0] < 0.15 * best[4]:
                    break
                _, bi, left, right, _ = best
                buckets[bi:bi + 1] = [left, right]
            return buckets

        for _orient in (lambda k: k[0] > k[1],      # portrait H > W
                        lambda k: k[0] <= k[1]):    # landscape
            _mem = [k for k in _tail if _orient(k)]
            if not _mem:
                continue
            for _b in _greedy_split(_mem, _KMAX):
                _c, _Hp, _Wp = _bcost(_b)
                _bk = ("__pad__", _Hp, _Wp)
                for k in _b:
                    _real_bucket[k] = _bk
                _bucket_total[_bk] = sum(_tail[k] for k in _b)
                _bucket_queued[_bk] = 0
        if _bucket_total:
            log("尾部混合尺寸入padding桶: "
                + ", ".join(f"{v}张→{k[2]}x{k[1]}"
                            for k, v in sorted(_bucket_total.items(),
                                               key=lambda kv: -kv[1])))

    # ---- tiered verification ----
    # every image: GPU reconstruction-PSNR gate (quantisation error of what
    # the decoder will rebuild, computed on-device for free). A deterministic
    # subset additionally gets the full subprocess decode-verify (webp
    # decode + alpha + metadata + RGBA PSNR). VERIFYSAMPLE=100 restores the
    # verify-everything behaviour.
    # default 100 = full decode-verify of every image (product semantics);
    # VERIFYSAMPLE=25 opts into tiered verification for speed
    _sample_pct = max(0, min(100, int(os.environ.get("VERIFYSAMPLE") or 100)))
    _gpu_psnr_min = float(os.environ.get("GPUPSNR_MIN") or 0)   # 0 = log-only
    _gate_stats = {"gate": 0, "sampled": 0, "sample_fail": 0}
    _psnrlog = open(os.environ["PSNRLOG"], "a", buffering=1)         if os.environ.get("PSNRLOG") else None

    import zlib as _zlib

    def _is_sampled(rel):
        return _zlib.crc32(rel.encode("utf-8")) % 100 < _sample_pct

    def _gpu_psnr(sse3, h, w):
        # YUV-domain PSNR: equal weight per sample, 1.5 samples/pixel
        n_y = h * w
        n_uv = (h // 2) * (w // 2)
        tot = sse3[0] + sse3[1] + sse3[2]
        mse = tot / (n_y + 2 * n_uv) if (n_y + 2 * n_uv) else 1e18
        return 99.0 if mse <= 0 else 10 * np.log10(255 * 255 / mse)

    # CPU core cap (UI spinbox / --cores / CPU_MAXCORES env): scale every
    # worker pool so total threads ~stay under the budget. 0/None = uncapped.
    _mc = max_cores if max_cores is not None else os.environ.get("CPU_MAXCORES")
    try:
        _mc = int(_mc) if _mc is not None else 0
    except (TypeError, ValueError):
        _mc = 0
    if _mc and _mc > 0:
        _nc = os.cpu_count() or 8
        _mc = min(_mc, _nc)
        _frac = _mc / _nc
        decode_workers = max(1, min(decode_workers,
                                     int(round(decode_workers * _frac)) or 1))
        finish_workers = max(1, min(finish_workers,
                                     int(round(finish_workers * _frac)) or 1))
        nvp = max(1, min(nvp or 12, max(2, int(round(12 * _frac))))
                   ) if nvp is None else max(1, min(nvp, _mc))

    bq, y1, y2, uv_m, fl = vp8_encode.setup_quant(quality)
    y2ac = max(8, int(T.AC_TABLE2[bq]))
    y1deq = np.array([T.DC_TABLE[bq]] + [T.AC_TABLE[bq]] * 15, np.int64)
    y2deq = np.array([T.DC_TABLE[bq] * 2] + [y2ac] * 15, np.int64)
    uvdeq = np.array([T.DC_TABLE[max(0, min(117, bq - 2))]]
                     + [T.AC_TABLE[bq]] * 15, np.int64)

    # ---- optional C++ GPU pipeline (bit-exact with the cupy path) ----
    # one blocking DLL call replaces upload+YUV+mode search+select+closed
    # loop; ctypes releases the GIL so decode/finish threads keep running
    _cpp = None
    _entb = None
    if os.environ.get("GPUPIPE_DISABLE") != "1":
        try:
            _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _cand = [os.path.join(_root, "cpp", "gpu_pipeline_v2.dll"),
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "gpu_pipeline_v2.dll")]
            for _p in _cand:
                if os.path.isfile(_p):
                    _cpp = ctypes.CDLL(_p)
                    _cpp.process_batch.restype = ctypes.c_int
                    _cpp.process_batch.argtypes = (
                        [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 7)
                    try:
                        _cpp.process_batch_ptrs.restype = ctypes.c_int
                        _cpp.process_batch_ptrs.argtypes = (
                            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 7)
                    except AttributeError:
                        pass
                    try:
                        _cpp.gpu_set_device.restype = ctypes.c_int
                        _cpp.gpu_set_device.argtypes = [ctypes.c_int]
                        # NOTE: `device` was already remapped above — under
                        # CUDA_VISIBLE_DEVICES=1 + PCI_BUS_ID order the
                        # visible index 0 is the physical V100; the 2080 is
                        # invisible to this process entirely
                        _rc = _cpp.gpu_set_device(device)
                        if _rc != 0:
                            raise RuntimeError(
                                f"gpu_set_device({device}) -> {_rc}")
                    except AttributeError:
                        pass
                    try:
                        _cpp.submit_batch.restype = ctypes.c_int
                        _cpp.submit_batch.argtypes = (
                            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 7)
                        _cpp.poll_batch.restype = ctypes.c_int
                        _cpp.poll_batch.argtypes = [ctypes.c_int,
                                                    ctypes.c_void_p]
                    except AttributeError:
                        pass
                    try:
                        # pre-grow slot caches for every flush shape: the
                        # first batch of each new (n, W, H) otherwise pays
                        # its cudaMalloc bill (~1s for GB-size buffers)
                        # mid-run and stalls the main-size stream
                        if not os.environ.get("GPUPIPE_NOWARM"):
                            _cpp.gpu_warm.restype = ctypes.c_int
                            _cpp.gpu_warm.argtypes = [ctypes.c_int,
                                                      ctypes.c_int,
                                                      ctypes.c_int]
                            # caches are flat n*W*H buffers (grow-only),
                            # so one warm at the largest-product shape
                            # covers every smaller batch shape
                            _best = (0, 0, 64)     # n*W*H, W, H
                            for (_h, _w), _c in real_total.items():
                                if (_h % 16 == 0 and _w % 16 == 0
                                        and _h * _w <= 24_000_000):
                                    _n = min(batch, _c)
                                    if _n * _w * _h > _best[0]:
                                        _best = (_n * _w * _h, _w, _h)
                            for _bk in _bucket_total:
                                _n = min(batch, _bucket_total[_bk])
                                if _n * _bk[2] * _bk[1] > _best[0]:
                                    _best = (_n * _bk[2] * _bk[1],
                                             _bk[2], _bk[1])
                            if _best[0]:
                                _cpp.gpu_warm(batch, _best[1], _best[2])
                    except AttributeError:
                        pass
                    try:
                        # trellis RD quantization (default on; TRELLIS=0 off)
                        _cpp.set_trellis.restype = ctypes.c_int
                        _cpp.set_trellis.argtypes = [ctypes.c_int]
                        _cpp.set_trellis(
                            0 if os.environ.get("TRELLIS") == "0" else 1)
                    except AttributeError:
                        pass
                    try:
                        # zero-copy async staging: submit_batch stores the
                        # per-image pointers and W1 stages from them; the
                        # group (t["arr"]) stays alive in _pend until the
                        # batch completes
                        _cpp.set_zc_submit.restype = ctypes.c_int
                        _cpp.set_zc_submit.argtypes = [ctypes.c_int]
                        _cpp.set_zc_submit(1)
                    except AttributeError:
                        pass
                    try:
                        # 8-output variants (adds per-image recon SSE)
                        _cpp.submit_batch2.restype = ctypes.c_int
                        _cpp.submit_batch2.argtypes = (
                            [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 8)
                        _cpp.submit_batch_padded2.restype = ctypes.c_int
                        _cpp.submit_batch_padded2.argtypes = (
                            [ctypes.c_void_p, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 8)
                    except AttributeError:
                        pass
                    try:
                        # padded mixed-size variant of the GPU PNG decoder
                        _cpp.submit_batch_pngv_padded2.restype = (
                            ctypes.c_int)
                        _cpp.submit_batch_pngv_padded2.argtypes = (
                            [ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 7
                            + [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p]
                            + [ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_int])
                    except AttributeError:
                        pass
                    try:
                        # GPU PNG decode: zlib streams in, decoded RGBA back
                        # with the batch result (all bit depths / colour
                        # types / interlace / tRNS)
                        _cpp.submit_batch_pngv.restype = ctypes.c_int
                        _cpp.submit_batch_pngv.argtypes = (
                            [ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int]
                            + [ctypes.c_void_p] * 7
                            + [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_void_p]
                            + [ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_int, ctypes.c_int])
                    except AttributeError:
                        pass
                    try:
                        _cpp.submit_batch_padded.restype = ctypes.c_int
                        _cpp.submit_batch_padded.argtypes = (
                            [ctypes.c_void_p, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
                            + [ctypes.c_void_p] * 7)
                    except AttributeError:
                        pass
                    break
        except OSError:
            _cpp = None
    # PNG streams CAN go to the DLL decoder (all variants, bit-exact vs
    # libpng); opt-in while the inflate aggregate ceiling is below the CPU
    # supply rate (PNG_GPU=1 enables)
    _png_gpu_ok = (_cpp is not None
                   and hasattr(_cpp, "submit_batch_pngv")
                   and os.environ.get("PNG_GPU") == "1")
    if _cpp is not None and os.environ.get("ENTROPY_DISABLE") != "1":
        try:
            _ep = os.path.join(_root, "cpp", "entropy.dll")
            if not os.path.isfile(_ep):
                _ep = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "entropy.dll")
            if os.path.isfile(_ep):
                _entb = ctypes.CDLL(_ep)
                _entb.encode_vp8_batch.restype = ctypes.c_int
                _entb.encode_vp8_batch.argtypes = (
                    [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.c_int] + [ctypes.c_void_p] * 8 +
                    [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p])
        except OSError:
            _entb = None

    def _cpp_entropy_finish(group, W2, H2, quality, y_dc, y_ac, uv_lv,
                            is_i4, i16m, uvm, i4m, sampled=None,
                            hdr_w=None, hdr_h=None,
                            _on_done=None):
        """Batch entropy on a finish worker (keeps the GPU thread free) then
        per-image pre-encoded submits. Arrays are batch-fresh: no races."""
        try:
            n = len(group)
            import hashlib as _hl, os as _os
            if _os.environ.get('PIPE_CHK2'):
                h1 = _hl.md5(y_dc.tobytes() + y_ac.tobytes() + uv_lv.tobytes()
                             + is_i4.tobytes() + i16m.tobytes() + uvm.tobytes()
                             + i4m.tobytes()).hexdigest()
                _ENT_IN_HASH.append(h1)
            mb_h2, mb_w2 = H2 // 16, W2 // 16
            n_mb = mb_h2 * mb_w2
            skip_b = ~(y_dc.reshape(n, n_mb, 16).any(-1)
                       | y_ac.reshape(n, n_mb, 16, 16).any(-1).any(-1)
                       | uv_lv.reshape(n, n_mb, 8, 16).any(-1).any(-1))
            cap = n_mb * 2400 + 8192
            outs = np.empty(n * cap, np.uint8)
            lens = np.empty(2 * n, np.int32)
            P = ctypes.c_void_p
            # NOTE: skip_b is 2D -- flatten before flat-index pointer slicing
            ptrs = lambda arr, per: (P * n)(
                *[arr[k * per:(k + 1) * per].ctypes.data_as(P) for k in range(n)])
            ret = _entb.encode_vp8_batch(
                n, mb_w2, mb_h2, bq, fl,
                ptrs(skip_b.reshape(-1), n_mb), ptrs(is_i4, n_mb),
                ptrs(i16m, n_mb), ptrs(uvm, n_mb), ptrs(i4m, n_mb * 16),
                ptrs(y_dc, n_mb * 16), ptrs(y_ac, n_mb * 256),
                ptrs(uv_lv, n_mb * 128),
                (P * n)(*[outs[k * cap:(k + 1) * cap].ctypes.data_as(P)
                          for k in range(n)]),
                cap, lens.ctypes.data_as(ctypes.c_void_p))
            if ret != 0 or not (lens[:n] >= 0).all():
                raise RuntimeError(f"C++批式熵编码返回 {ret}")
            for k, t in enumerate(group):
                if stop_event is not None and stop_event.is_set():
                    return
                p0 = int(lens[k]); tot = int(lens[n + k])
                hw = hdr_w if hdr_w is not None else W2
                hh = hdr_h if hdr_h is not None else H2
                vp8 = (((1 << 4) | (p0 << 5)).to_bytes(3, "little")
                       + (0x9D012A).to_bytes(3, "big")
                       + (hw & 0x3FFF).to_bytes(2, "little")
                       + (hh & 0x3FFF).to_bytes(2, "little")
                       + outs[k * cap:k * cap + tot].tobytes())
                if len(vp8) & 1:
                    vp8 += bytes([0])
                finish_gpu_pre(t, vp8, sampled[k] if sampled else True)   # inline: executor may be draining
            if _on_done is not None:      # success path: release the permit
                try: _on_done()
                except Exception: pass
        except Exception as e:                          # noqa: BLE001
            log(f"C++批式熵编码失败({e})，回退逐张路径")
            if _on_done is not None:
                try: _on_done()
                except Exception: pass
            for k, t in enumerate(group):
                sl = slice(k * n_mb, (k + 1) * n_mb)
                modes = dict(is_i4=is_i4[sl].astype(bool), i16_mode=i16m[sl],
                             uv_mode=uvm[sl],
                             i4_modes=i4m[k * n_mb * 16:(k + 1) * n_mb * 16]
                             .reshape(n_mb, 16))
                finish_ex.submit(
                    finish_gpu, t, modes,
                    y_dc[k * n_mb * 16:(k + 1) * n_mb * 16].reshape(n_mb, 16),
                    y_ac[k * n_mb * 256:(k + 1) * n_mb * 256].reshape(n_mb, 16, 16),
                    uv_lv[k * n_mb * 128:(k + 1) * n_mb * 128].reshape(n_mb, 8, 16),
                    sampled[k] if sampled else True)

    _pend = []            # FIFO of in-flight async GPU batches
    _PEND_MAX = 3         # each in-flight batch pins host output buffers and
                          # device memory in the DLL queue; unbounded _pend
                          # once filled the 16GB V100 and then host RAM
                          # (page thrash -> freeze/segfault). 3 (not 2) so a
                          # full batch never waits on the previous drain

    def _drain_pend(block):
        # dispatch completed GPU batches to the entropy/finish path.
        # poll returns completions in FINISH order, which differs from
        # submission order once mixed exact/padded batches coexist (their
        # stage1 durations differ) -- match by rid, never blind-pop FIFO
        while _pend:
            err = ctypes.c_int()
            _tw0 = time.time()
            got = _cpp.poll_batch(-1 if block else 0, ctypes.byref(err))
            if _gpulog and got:
                _gpuline("poll", 0, 0, 0, _tw0)
            if got == 0:
                return
            ent = None
            for e in _pend:
                if e[0] == got:
                    ent = e
                    break
            if ent is None:
                # stale/duplicate completion for an already-drained entry
                continue
            _pend.remove(ent)
            (rid, group, W2, H2, y_dc, y_ac, uv_lv,
             is_i4, i16m, uvm, i4m, padded, sse, png_rgba,
             png_ierr) = ent
            if err.value != 0:
                log(f"C++ GPU管线批错误 {err.value}，CPU兜底")
                for t in group:
                    _submit_fallback(t, f"GPU管线错误 {err.value}")
                continue
            if padded:
                _dispatch_padded(group, y_dc, y_ac, uv_lv,
                                 is_i4, i16m, uvm, i4m, sse,
                                 png_rgba, png_ierr)
            else:
                _dispatch_batch(group, W2, H2, y_dc, y_ac, uv_lv,
                                is_i4, i16m, uvm, i4m, sse,
                                base=0, png_rgba=png_rgba,
                                png_ierr=png_ierr)

    _purged = [0]

    def _dev_low_purge():
        """Free the DLL's grow-only caches when device memory runs low.
        A many-sizes corpus walks every cached buffer up to its global max
        and fragments the device (observed: 16GB V100 at 0 bytes free,
        34s per single-image batch). Purge costs one re-grow (~100ms)."""
        try:
            if not hasattr(_cpp, "gpu_purge"):
                return
            with cp.cuda.Device(device):
                free, _t = cp.cuda.runtime.memGetInfo()
            if free / 1e9 < 2.5:
                _cpp.gpu_purge()
                cp.get_default_memory_pool().free_all_blocks()
                _purged[0] += 1
                log(f"显存缓存清理#{_purged[0]}(空闲{free / 1e9:.1f}GB)")
        except Exception:                           # noqa: BLE001
            pass

    def _cpp_flush(group, W2, H2):
        n = len(group)
        mb_h, mb_w = H2 // 16, W2 // 16
        n_mb = mb_h * mb_w
        # fresh arrays per batch: np.empty is page-lazy (fast) and guarantees
        # the async finish threads can never see another batch's data, so no
        # defensive per-image copies are needed
        y_dc = np.empty(n * n_mb * 16, np.int16)
        y_ac = np.empty(n * n_mb * 256, np.int16)
        uv_lv = np.empty(n * n_mb * 128, np.int16)
        is_i4 = np.empty(n * n_mb, np.uint8)
        i16m = np.empty(n * n_mb, np.uint8)
        uvm = np.empty(n * n_mb, np.uint8)
        i4m = np.empty(n * n_mb * 16, np.uint8)
        _ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        if hasattr(_cpp, "submit_batch") and not os.environ.get("GPUPIPE_SYNC"):
            # async double-buffered pipeline: submit and dispatch completions
            # FIFO; the arr arrays stay alive via the queued entry until the
            # batch completes (the DLL stages inputs in its worker)
            while len(_pend) >= _PEND_MAX:
                _drain_pend(True)          # block until one completes
            _dev_low_purge()
            ipa = (ctypes.c_void_p * n)(
                *[t["arr"].ctypes.data_as(ctypes.c_void_p) for t in group])
            sse = np.empty(n * 3, np.int64)
            _sse_ptr = _ptr(sse) if hasattr(_cpp, "submit_batch2") else None
            if _sse_ptr is not None:
                rid = _cpp.submit_batch2(ipa, n, W2, H2, quality,
                                         _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                                         _ptr(is_i4), _ptr(i16m), _ptr(uvm),
                                         _ptr(i4m), _sse_ptr)
            else:
                rid = _cpp.submit_batch(ipa, n, W2, H2, quality,
                                        _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                                        _ptr(is_i4), _ptr(i16m), _ptr(uvm),
                                        _ptr(i4m))
            if rid <= 0:
                raise RuntimeError(f"C++ GPU管线提交失败 {rid}")
            _pend.append((rid, group, W2, H2, y_dc, y_ac, uv_lv,
                          is_i4, i16m, uvm, i4m, False, sse, None, None))
            _drain_pend(False)
            return
        # GPUPIPE_SYNC debug path: synchronous DLL call
        if hasattr(_cpp, "process_batch_ptrs"):
            ipa = (ctypes.c_void_p * n)(
                *[t["arr"].ctypes.data_as(ctypes.c_void_p) for t in group])
            ret = _cpp.process_batch_ptrs(ipa, n, W2, H2, quality,
                                          _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                                          _ptr(is_i4), _ptr(i16m), _ptr(uvm),
                                          _ptr(i4m))
        else:
            buf = b"".join(t["arr"].tobytes() for t in group)
            ret = _cpp.process_batch(buf, n, W2, H2, quality,
                                     _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                                     _ptr(is_i4), _ptr(i16m), _ptr(uvm),
                                     _ptr(i4m))
        if ret != 0:
            raise RuntimeError(f"C++ GPU管线返回 {ret}")
        _dispatch_batch(group, W2, H2, y_dc, y_ac, uv_lv,
                        is_i4, i16m, uvm, i4m)

    def _png_raw_size_py(h, w, bd, ct, inter):
        """Python twin of the kernel's png_raw_size (raw scanline bytes)."""
        C = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
        X0 = (0, 4, 0, 2, 0, 1, 0)
        Y0 = (0, 0, 4, 0, 2, 0, 1)
        DX = (8, 8, 4, 4, 2, 2, 1)
        DY = (8, 8, 8, 4, 4, 2, 2)
        tot = 0
        for p in range(7 if inter else 1):
            if inter:
                wp = (w - X0[p] + DX[p] - 1) // DX[p]
                hp = (h - Y0[p] + DY[p] - 1) // DY[p]
            else:
                wp, hp = w, h
            if wp <= 0 or hp <= 0:
                continue
            rb = (wp * C * bd + 7) >> 3
            tot += hp * (rb + 1)
        return tot

    _raw_cap = int(os.environ.get("PNG_RAWCAP") or 0) or (1 << 30)

    def _png_chunk_group(group):
        """Split a png group so cumulative fraw/compressed offsets stay
        below the DLL's int32 offsets (1<<30 with headroom)."""
        out, cur, acc = [], [], 0
        for t in group:
            h, w, bd, ct, inter = t["ihdr"]
            per = max(_png_raw_size_py(h, w, bd, ct, inter),
                      len(t["idat"])) + 65536
            if cur and acc + per > _raw_cap:
                out.append(cur)
                cur, acc = [], 0
            cur.append(t)
            acc += per
        if cur:
            out.append(cur)
        return out

    def _cpp_flush_png(group, H2, W2):
        """GPU-decoded batch: zlib streams in, decoded RGBA (for alpha encode +
        the verify reference) comes back with the coefficients. The group is
        chunked so cumulative raw offsets stay int32-safe."""
        for chunk in _png_chunk_group(group):
            _cpp_flush_png_one(chunk, H2, W2)

    def _cpp_flush_png_one(group, H2, W2):
        n = len(group)
        mb_h, mb_w = H2 // 16, W2 // 16
        n_mb = mb_h * mb_w
        y_dc = np.empty(n * n_mb * 16, np.int16)
        y_ac = np.empty(n * n_mb * 256, np.int16)
        uv_lv = np.empty(n * n_mb * 128, np.int16)
        is_i4 = np.empty(n * n_mb, np.uint8)
        i16m = np.empty(n * n_mb, np.uint8)
        uvm = np.empty(n * n_mb, np.uint8)
        i4m = np.empty(n * n_mb * 16, np.uint8)
        _ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        P = ctypes.c_void_p
        h, w, bd, ct, inter = group[0]["ihdr"]
        while len(_pend) >= _PEND_MAX:
            _drain_pend(True)
        _dev_low_purge()
        ipa = (P * n)(*[ctypes.cast(ctypes.c_char_p(t["idat"]), P)
                        for t in group])
        lens = (ctypes.c_int * n)(*[len(t["idat"]) for t in group])
        plte = group[0]["plte"]
        trns256 = group[0]["trns256"]
        sse = np.empty(n * 3, np.int64)
        rgba = np.empty(n * H2 * W2 * 4, np.uint8)
        ierr = np.empty(n, np.int32)
        rid = _cpp.submit_batch_pngv(
            ipa, lens, n, w, h, bd, ct, inter, quality,
            _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
            _ptr(is_i4), _ptr(i16m), _ptr(uvm), _ptr(i4m),
            _ptr(sse), _ptr(ierr), _ptr(rgba),
            ctypes.cast(plte, P) if plte else P(0),
            len(plte) if plte else 0,
            ctypes.cast(trns256, P) if trns256 else P(0),
            len(trns256) if trns256 else 0,
            group[0]["trnsmode"])
        if rid <= 0:
            raise RuntimeError(f"C++ GPU管线提交失败 {rid}")
        _pend.append((rid, group, W2, H2, y_dc, y_ac, uv_lv,
                      is_i4, i16m, uvm, i4m, False, sse, rgba, ierr))
        _drain_pend(False)

    def _padded_flush_png(group, Hp, Wp):
        """Mixed-size GPU-decoded batch on one shared padded grid; chunked
        so cumulative raw offsets stay int32-safe."""
        for chunk in _png_chunk_group(group):
            _padded_flush_png_one(chunk, Hp, Wp)

    def _padded_flush_png_one(group, Hp, Wp):
        n = len(group)
        tot = sum((t["ihdr"][0] // 16) * (t["ihdr"][1] // 16) for t in group)
        y_dc = np.empty(tot * 16, np.int16)
        y_ac = np.empty(tot * 256, np.int16)
        uv_lv = np.empty(tot * 128, np.int16)
        is_i4 = np.empty(tot, np.uint8)
        i16m = np.empty(tot, np.uint8)
        uvm = np.empty(tot, np.uint8)
        i4m = np.empty(tot * 16, np.uint8)
        _ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
        P = ctypes.c_void_p
        while len(_pend) >= _PEND_MAX:
            _drain_pend(True)
        _dev_low_purge()
        ipa = (P * n)(*[ctypes.cast(ctypes.c_char_p(t["idat"]), P)
                        for t in group])
        lens = (ctypes.c_int * n)(*[len(t["idat"]) for t in group])
        wr = (ctypes.c_int * n)(*[t["ihdr"][1] for t in group])
        hr = (ctypes.c_int * n)(*[t["ihdr"][0] for t in group])
        h0, w0, bd, ct, inter = group[0]["ihdr"]
        plte = group[0]["plte"]
        trns256 = group[0]["trns256"]
        sse = np.empty(n * 3, np.int64)
        rgba = np.empty(n * Hp * Wp * 4, np.uint8)
        ierr = np.empty(n, np.int32)
        if os.environ.get("PPDBG2A"):
            print(f"[PP2] submit n={n} grid={Wp}x{Hp} bd={bd} ct={ct}",
                  flush=True)
        rid = _cpp.submit_batch_pngv_padded2(
            ipa, lens, n, wr, hr, Wp, Hp, bd, ct, inter, quality,
            _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
            _ptr(is_i4), _ptr(i16m), _ptr(uvm), _ptr(i4m),
            _ptr(sse), _ptr(ierr), _ptr(rgba),
            ctypes.cast(plte, P) if plte else P(0),
            len(plte) if plte else 0,
            ctypes.cast(trns256, P) if trns256 else P(0),
            len(trns256) if trns256 else 0,
            group[0]["trnsmode"])
        if rid <= 0:
            raise RuntimeError(f"png padded提交失败 {rid}")
        _pend.append((rid, group, Wp, Hp, y_dc, y_ac, uv_lv,
                      is_i4, i16m, uvm, i4m, True, sse, rgba, ierr))
        if os.environ.get("PPDBG2B"):
            print(f"[PP2] queued rid={rid}", flush=True)
        _drain_pend(False)

    def _dispatch_batch(group, W2, H2, y_dc, y_ac, uv_lv,
                        is_i4, i16m, uvm, i4m, sse=None, base=0,
                        png_rgba=None, png_ierr=None):
        if png_rgba is not None:
            # GPU-decoded batch: materialise per-image arr from the batch
            # download (copy() so tasks don't pin the whole batch buffer);
            # failed streams route through the existing gate/fallback path
            one = H2 * W2 * 4
            for i, t in enumerate(group):
                if png_ierr is not None and png_ierr[i] != 0:
                    # CRC-verified stream still failed to inflate: decode
                    # on demand so the fallback encoder sees true pixels
                    try:
                        t["arr"] = _decode_png_cpu(t["png"])
                    except Exception:                  # noqa: BLE001
                        t["arr"] = np.zeros((H2, W2, 4), np.uint8)
                    t["_gate_fail"] = f"GPU解码错误 {png_ierr[i]}"
                    continue
                t["arr"] = png_rgba[i * one:(i + 1) * one].reshape(
                    H2, W2, 4).copy()
        # GPU reconstruction gate: flag images whose quantisation error is
        # already too large; the coefficient arrays stay index-aligned (the
        # gate is honoured in finish_gpu_pre, which drops them before the
        # container assembly)
        if sse is not None:
            for i, t in enumerate(group):
                h2, w2 = t["arr"].shape[:2]
                gp = _gpu_psnr(sse[base + 3 * i: base + 3 * i + 3], h2, w2)
                if _psnrlog is not None:
                    _psnrlog.write(f"{t['rel']} {gp:.3f}{chr(10)}")
                if _gpu_psnr_min > 0 and gp < _gpu_psnr_min:
                    with stats.lock:
                        _gate_stats["gate"] += 1
                    log(f"GPU重建PSNR {gp:.2f} < {_gpu_psnr_min} "
                        f"{t['rel']}，CPU兜底")
                    t["_gate_fail"] = f"GPU重建PSNR {gp:.2f} < {_gpu_psnr_min}"
        sampled = [(_is_sampled(t["rel"]) if _sample_pct < 100 else True)
                   for t in group]
        if _verify_sem is not None:
            # verify backpressure on the GPU thread: images dispatched but
            # not yet verified hold their task (RGBA+PNG ~5.5MB) alive via
            # the result callbacks; without a cap the whole corpus (~37GB)
            # piles up whenever the pool lags the encoder
            for _s, _t in zip(sampled, group):
                if _s and "_gate_fail" not in _t:
                    _verify_sem.acquire()
        if _entb is not None:
            _fin_sem.acquire()          # bounds pinned batch arrays (~230MB ea)
            _hw = group[0].get("odd_wh")
            finish_ex.submit(_cpp_entropy_finish, group, W2, H2, quality,
                             y_dc, y_ac, uv_lv, is_i4, i16m, uvm, i4m,
                             sampled,
                             _hw[1] if _hw else None,
                             _hw[0] if _hw else None,
                             _on_done=_fin_sem.release)
            return
        _fin_sem.acquire()
        try:
            for i, t in enumerate(group):
                if stop_event is not None and stop_event.is_set():
                    return
                sl = slice(i * n_mb, (i + 1) * n_mb)
                modes = dict(is_i4=is_i4[sl].astype(bool), i16_mode=i16m[sl],
                             uv_mode=uvm[sl],
                             i4_modes=i4m[i * n_mb * 16:(i + 1) * n_mb * 16]
                             .reshape(n_mb, 16))
                finish_ex.submit(
                    finish_gpu, t, modes,
                    y_dc[i * n_mb * 16:(i + 1) * n_mb * 16].reshape(n_mb, 16),
                    y_ac[i * n_mb * 256:(i + 1) * n_mb * 256].reshape(n_mb, 16, 16),
                    uv_lv[i * n_mb * 128:(i + 1) * n_mb * 128].reshape(n_mb, 8, 16),
                    sampled[i])
        finally:
            _fin_sem.release()

    def _padded_flush(group, Hp, Wp):
        """Mixed-size batch on a zero-padded grid. The DLL stages each image
        top-left, encodes with W clamped to the real width, and gathers
        per-real-MB outputs packed in arrival order. Any failure falls back
        to the exact-size path (bit-identical outputs)."""
        try:
            if (_cpp is None or not hasattr(_cpp, "submit_batch_padded")
                    or os.environ.get("GPUPIPE_SYNC")):
                raise RuntimeError("padded路径不可用")
            _t0 = time.time()
            n = len(group)
            nmb_r = [(t["arr"].shape[0] // 16) * (t["arr"].shape[1] // 16)
                     for t in group]
            tot = sum(nmb_r)
            y_dc = np.empty(tot * 16, np.int16)
            y_ac = np.empty(tot * 256, np.int16)
            uv_lv = np.empty(tot * 128, np.int16)
            is_i4 = np.empty(tot, np.uint8)
            i16m = np.empty(tot, np.uint8)
            uvm = np.empty(tot, np.uint8)
            i4m = np.empty(tot * 16, np.uint8)
            _ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
            while len(_pend) >= _PEND_MAX:
                _drain_pend(True)          # same host/device memory bound
            _dev_low_purge()
            ipa = (ctypes.c_void_p * n)(
                *[t["arr"].ctypes.data_as(ctypes.c_void_p) for t in group])
            wr = (ctypes.c_int * n)(*[t["arr"].shape[1] for t in group])
            hr = (ctypes.c_int * n)(*[t["arr"].shape[0] for t in group])
            sse = np.empty(n * 3, np.int64)
            if hasattr(_cpp, "submit_batch_padded2"):
                rid = _cpp.submit_batch_padded2(
                    ipa, n, wr, hr, Wp, Hp, quality,
                    _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                    _ptr(is_i4), _ptr(i16m), _ptr(uvm), _ptr(i4m),
                    _ptr(sse))
            else:
                rid = _cpp.submit_batch_padded(
                    ipa, n, wr, hr, Wp, Hp, quality,
                    _ptr(y_dc), _ptr(y_ac), _ptr(uv_lv),
                    _ptr(is_i4), _ptr(i16m), _ptr(uvm), _ptr(i4m))
            if rid <= 0:
                raise RuntimeError(f"padded提交失败 {rid}")
            # group keeps t["arr"] alive: the DLL stages from these pointers
            # in its worker until stage1 copies them
            _pend.append((rid, group, Wp, Hp, y_dc, y_ac, uv_lv,
                          is_i4, i16m, uvm, i4m, True, sse, None, None))
            _gpuline("pad", Wp, Hp, n, _t0)
            _drain_pend(False)
            return
        except Exception as e:                          # noqa: BLE001
            # includes MemoryError: the uniform-dims OOM splitter in
            # flush_group must never see a mixed group, so split here
            log(f"padded批异常({len(group)}张): {e}，回退精确分组")

        by = {}
        for t in group:
            by.setdefault(t["arr"].shape[:2], []).append(t)

        def _try_exact(g2, depth=0):
            try:
                _flush_one(g2)
            except MemoryError:
                mid = len(g2) // 2
                if mid == 0 or depth > 3:
                    for t in g2:
                        _submit_fallback(t, "GPU显存不足")
                    return
                log(f"显存不足,padded回退拆批 {len(g2)} -> {mid}+{len(g2)-mid}")
                _try_exact(g2[:mid], depth + 1)
                _try_exact(g2[mid:], depth + 1)

        for _k2, g2 in by.items():
            _try_exact(g2)

    def _dispatch_padded(group, y_dc, y_ac, uv_lv, is_i4, i16m, uvm, i4m,
                         sse=None, png_rgba=None, png_ierr=None):
        """Split packed per-real-MB outputs into consecutive same-size runs
        and feed each through the uniform-dims dispatch/entropy path."""
        if png_rgba is not None:
            if os.environ.get("PPDBG2C"):
                print(f"[PP2] dispatch n={len(group)}", flush=True)
            Hp = group and max(t["padgrid"][1] for t in group)
            Wp = group and max(t["padgrid"][2] for t in group)
            one = Hp * Wp * 4
            for i, t in enumerate(group):
                h_r, w_r = t["ihdr"][0], t["ihdr"][1]
                if png_ierr is not None and png_ierr[i] != 0:
                    try:
                        t["arr"] = _decode_png_cpu(t["png"])
                    except Exception:              # noqa: BLE001
                        t["arr"] = np.zeros((h_r, w_r, 4), np.uint8)
                    t["_gate_fail"] = f"GPU解码错误 {png_ierr[i]}"
                    continue
                t["arr"] = png_rgba[i * one:(i + 1) * one].reshape(
                    Hp, Wp, 4)[:h_r, :w_r].copy()
        offs = [0]
        for t in group:
            s = t["arr"].shape[:2]
            offs.append(offs[-1] + (s[0] // 16) * (s[1] // 16))
        i = 0
        while i < len(group):
            j = i + 1
            while (j < len(group)
                   and group[j]["arr"].shape[:2] == group[i]["arr"].shape[:2]):
                j += 1
            H2r, W2r = group[i]["arr"].shape[:2]
            a, b = offs[i], offs[j]
            _dispatch_batch(group[i:j], W2r, H2r,
                            y_dc[a * 16:b * 16], y_ac[a * 256:b * 256],
                            uv_lv[a * 128:b * 128], is_i4[a:b], i16m[a:b],
                            uvm[a:b], i4m[a * 16:b * 16],
                            # coefficient arrays are packed per-MB (offsets
                            # a/b), the SSE array per-image (3 each)
                            sse, base=3 * i)
            i = j

    def _inline_verify(task, webp):
        """In-process fallback when the verify pool cannot spawn: same
        checks as the subprocess worker, on a finish thread."""
        try:
            note = check(webp, task["png"], task["arr"])
            if note:
                log(f"GPU输出校验未过 {task['rel']}: {note}，CPU兜底重压")
                _fb_sem.acquire()
                finish_pillow(task, note)
        except Exception as e:                      # noqa: BLE001
            log(f"校验异常 {task['rel']}: {e}")
            _fb_sem.acquire()
            finish_pillow(task, f"校验异常 {e}")

    def check(webp_bytes, png_data, arr):
        dimg = Image.open(io.BytesIO(webp_bytes))
        darr = np.asarray(dimg.convert("RGBA"))
        H, W = arr.shape[:2]
        if darr.shape[:2] != (H, W):
            return f"尺寸不符 {darr.shape[:2]}"
        if not np.array_equal(darr[..., 3], arr[..., 3]):
            return "alpha不一致"
        if verify_meta:
            ok, problems = verify(png_data, webp_bytes)
            if not ok:
                return "元数据校验失败: " + ";".join(problems)
        p = _psnr_rgba(arr, darr)
        if p < min_psnr:
            return f"PSNR {p:.2f} < {min_psnr}"
        return None

    def finish_common(task, webp, used_fallback, reason):
        rel, out_path = task["rel"], task["out"]
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(webp)
        if used_fallback:
            with stats.lock:
                stats.fallbacks += 1
            log(f"[CPU兜底] {rel}: {reason}")
        stats.note(task["size"], len(webp))
        progress_cb and progress_cb(stats)

    _fb_sem = threading.Semaphore(
        int(os.environ.get("FALLBACK_MAXFLIGHT") or 24))

    def _submit_fallback(task, reason):
        # GPU-decoded tasks carry no arr: the CPU fallback encoder needs
        # pixels, decode on demand (rare error path only)
        if "arr" not in task and "idat" in task:
            try:
                task["arr"] = _decode_png_cpu(task["png"])
            except Exception as e:                      # noqa: BLE001
                log(f"错误路径解码失败 {task.get('rel')}: {e}")
                task["arr"] = np.zeros(
                    (task["ihdr"][0], task["ihdr"][1], 4), np.uint8)
        """Bounded Pillow-fallback lane: each queued task holds its full
        RGBA+PNG dict alive until the (slow) re-encode completes. The
        permit is acquired here (caller side) and released by finish_pillow
        itself, so direct finish_pillow callers need their own acquire."""
        _fb_sem.acquire()
        try:
            finish_ex.submit(finish_pillow, task, reason)
        except Exception:
            _fb_sem.release()      # never submitted: finish_pillow won't fire
            raise

    def finish_pillow(task, reason):
        try:
            webp = _pillow_encode(task["arr"], quality, task["meta"])
            note2 = check(webp, task["png"], task["arr"])
            if note2 and not note2.startswith("PSNR"):
                with stats.lock:
                    stats.failed += 1
                _park_failed(os.path.join(base, task["rel"]), dst, log)
                log(f"[失败] {task['rel']}: CPU兜底也失败: {note2} "
                    f"(原图已复制到 未转换/)")
                progress_cb and progress_cb(stats)
                return
            finish_common(task, webp, True,
                          reason if not note2 else f"{reason}; 注意: {note2}")
        except Exception as e:                          # noqa: BLE001
            with stats.lock:
                stats.failed += 1
            _park_failed(os.path.join(base, task["rel"]), dst, log)
            log(f"[失败] {task['rel']}: {reason}; CPU兜底异常 {e} "
                f"(原图已复制到 未转换/)")
            progress_cb and progress_cb(stats)
        finally:
            _fb_sem.release()

    def _submit_verify(payload, task, slot=None):
        """Dispatch one verify job; releases the backpressure permit (and
        the shm slot) on BOTH success and worker-exception paths."""
        def _fin():
            if slot is not None:
                with _shm_lock:
                    _shm_slots.append(slot)
            if _verify_sem is not None:
                _verify_sem.release()

        def _done(result):
            _fin()
            ok, note = result
            if _psnrlog is not None and note:
                _psnrlog.write(f"V {task['rel']} {note}{chr(10)}")
            if ok:
                return
            with stats.lock:
                _gate_stats["sample_fail"] += 1
            log(f"GPU输出校验未过 {task['rel']}: {note}，CPU兜底重压")
            _submit_fallback(task, note)

        def _err(e):
            _fin()
            log(f"校验异常 {task['rel']}: {e}")
            _submit_fallback(task, f"校验异常 {e}")

        with stats.lock:
            _gate_stats["sampled"] += 1
        verify_results.append(
            verify_pool.apply_async(_vw.verify_payload, (payload,),
                                    callback=_done, error_callback=_err))

    def finish_gpu_pre(task, vp8, sampled=True):
        try:
            if "odd_wh" in task:
                _oh, _ow = task.pop("odd_wh")
                task["arr"] = np.ascontiguousarray(
                    task["arr"][:_oh, :_ow])
            if "_gate_fail" in task:
                _fb_sem.acquire()
                finish_pillow(task, task.pop("_gate_fail"))
                return
            a = task["arr"][..., 3]
            alpha = None if bool((a == 255).all()) else make_alph_chunk(a)
            webp = _container(vp8, alpha, task["meta"])
            finish_common(task, webp, False, "")
            if verify_pool is None or min_psnr <= 0:
                return
            if _vw is None:
                # pool could not spawn: verify in-process (never silently
                # skip a "verified" run)
                finish_ex.submit(_inline_verify, task, webp)
                return
            if not sampled:
                with stats.lock:
                    _gate_stats["sampled_out"] = _gate_stats.get("sampled_out", 0) + 1
                return
            # zero-copy shm ring (same as finish_gpu): the worker reads the
            # decoded RGBA straight from shared memory instead of re-decoding
            # the source PNG — the path payload costs a full PNG decode per
            # image in the worker and caps the whole pipeline's verify rate
            H2, W2 = task["arr"].shape[:2]
            payload = None
            slot = None
            _fits = (_shmring is not None
                     and H2 * W2 * 4 <= _shmring.capacity)
            if _fits:
                with _shm_lock:
                    slot = _shm_slots.pop() if _shm_slots else None
                if slot is not None:
                    # slot buffer is a flat 1-D 'B' view; assign a flat
                    # prefix (memoryview assignment demands matching
                    # structure, and slots may exceed this image's bytes)
                    _flat = np.ascontiguousarray(task["arr"], np.uint8)                         .reshape(-1)
                    _shmring.buf(slot)[:_flat.size] = memoryview(_flat)
                    payload = ("shm", _shmring.shm.name, _shmring.count,
                               _shmring.capacity, slot, H2, W2,
                               webp, verify_meta, min_psnr,
                               task["meta"])
            if payload is None:       # ring exhausted: path fallback
                payload = (task["src"], webp, verify_meta,
                           min_psnr, task["meta"])
            _submit_verify(payload, task, slot)
        except Exception as e:                          # noqa: BLE001
            log(f"GPU编码失败 {task['rel']}: {e}")
            finish_pillow(task, f"GPU编码异常 {e}")

    def finish_gpu(task, modes, y_dc, y_ac, uv_lv, sampled=True):
        try:
            if "odd_wh" in task:
                _oh, _ow = task.pop("odd_wh")
                task["arr"] = np.ascontiguousarray(
                    task["arr"][:_oh, :_ow])
            if "_gate_fail" in task:
                _fb_sem.acquire()
                finish_pillow(task, task.pop("_gate_fail"))
                return
            a = task["arr"][..., 3]
            alpha = None if bool((a == 255).all()) else make_alph_chunk(a)
            skip = ~(y_dc.any(-1) | y_ac.any(-1).any(-1) | uv_lv.any(-1).any(-1))
            H2, W2 = task["arr"].shape[:2]
            webp = _assemble(W2, H2, (W2 + 15) // 16, (H2 + 15) // 16,
                             bq, fl, modes["is_i4"], modes["i16_mode"],
                             modes["uv_mode"], modes["i4_modes"],
                             y_dc, y_ac, uv_lv, skip, alpha, task["meta"])
            if verify_pool is None or min_psnr <= 0:
                # explicit no-verify speed mode
                finish_common(task, webp, False, "")
                return
            if not sampled:
                # tiered verification: this image passed the GPU gate and
                # was not drawn into the decode-verify sample
                with stats.lock:
                    _gate_stats["sampled_out"] = _gate_stats.get("sampled_out", 0) + 1
                finish_common(task, webp, False, "")
                return
            if _vw is None:
                # pool could not spawn (e.g. interactive/`<stdin>` driver):
                # verify in-process so a "verified" run can never silently
                # skip verification again (the phantom 188.8/s benchmark)
                finish_ex.submit(_inline_verify, task, webp)
                return
            # write first, verify in a subprocess (true parallelism), then
            # either finalise or re-encode via the Pillow fallback
            finish_common(task, webp, False, "")
            H2, W2 = task["arr"].shape[:2]
            payload = None
            slot = None
            _fits = (_shmring is not None
                     and H2 * W2 * 4 <= _shmring.capacity)
            if _fits:
                with _shm_lock:
                    slot = _shm_slots.pop() if _shm_slots else None
                if slot is not None:
                    # slot buffer is a flat 1-D 'B' view; assign a flat
                    # prefix (memoryview assignment demands matching
                    # structure, and slots may exceed this image's bytes)
                    _flat = np.ascontiguousarray(task["arr"], np.uint8)                         .reshape(-1)
                    _shmring.buf(slot)[:_flat.size] = memoryview(_flat)
                    payload = ("shm", _shmring.shm.name, _shmring.count,
                               _shmring.capacity, slot, H2, W2,
                               webp, verify_meta, min_psnr,
                               task["meta"])
            if payload is None:   # ring exhausted: path fallback
                slot = None
                payload = (task["src"], webp, verify_meta,
                           min_psnr, task["meta"])
            _submit_verify(payload, task, slot)
        except Exception as e:                          # noqa: BLE001
            log(f"GPU编码失败 {task['rel']}: {e}")
            finish_pillow(task, f"GPU编码异常 {e}")

    class PngScanError(Exception):
        pass

    _PNG_LEGAL = {1: (0, 3), 2: (0, 3), 4: (0, 3), 8: (0, 2, 3, 4, 6),
                  16: (0, 2, 4, 6)}

    def png_idat_scan(data):
        """One chunk walk: IHDR geometry + IDAT concat + PLTE/tRNS bytes,
        CRC32-verified. Returns (w, h, bd, ct, inter, idat, plte, trns256,
        trnsmode); trns256 pre-expanded to the kernel layout."""
        import struct
        import zlib
        pos = 8
        n = len(data)
        w = h = bd = ct = comp = filt = inter = None
        idat = []
        plte = None
        trns = None
        while pos + 12 <= n:
            ln = int.from_bytes(data[pos:pos + 4], "big")
            typ = data[pos + 4:pos + 8]
            body = data[pos + 8:pos + 8 + ln]
            if len(body) != ln:
                raise PngScanError("chunk截断")
            crc = int.from_bytes(data[pos + 8 + ln:pos + 12 + ln], "big")
            if zlib.crc32(typ + body) & 0xFFFFFFFF != crc:
                raise PngScanError(f"CRC错误 {typ}")
            if typ == b"IHDR":
                if ln != 13:
                    raise PngScanError("IHDR长度")
                (w, h, bd, ct, comp, filt, inter) = struct.unpack(
                    ">IIBBBBB", body)
            elif typ == b"IDAT":
                idat.append(body)
            elif typ == b"PLTE":
                plte = body
            elif typ == b"tRNS":
                trns = body
            elif typ == b"IEND":
                break
            pos += 12 + ln
        if w is None:
            raise PngScanError("无IHDR")
        if comp != 0 or filt != 0 or inter not in (0, 1):
            raise PngScanError(f"非法头 comp={comp} filt={filt} int={inter}")
        if bd not in _PNG_LEGAL or ct not in _PNG_LEGAL[bd]:
            raise PngScanError(f"非法 bd={bd} ct={ct}")
        if w <= 0 or h <= 0:
            raise PngScanError("尺寸")
        trnsmode = 0
        trns256 = None
        if trns is not None:
            if ct == 3:
                trnsmode = 1
                t = bytearray(b"\xff" * 256)
                for i in range(min(len(trns), 256)):
                    t[i] = trns[i]
                trns256 = bytes(t)
            elif ct == 0 and len(trns) >= 2:
                trnsmode = 2
                k16 = int.from_bytes(trns[:2], "big")
                trns256 = bytes([(k16 >> 8) if bd == 16 else (k16 & 0xFF)])
            elif ct == 2 and len(trns) >= 6:
                trnsmode = 3
                ks = struct.unpack(">HHH", trns[:6])
                trns256 = bytes([(v >> 8) if bd == 16 else (v & 0xFF)
                                 for v in ks])
        return (w, h, bd, ct, inter, b"".join(idat), plte, trns256, trnsmode)

    def _decode_png_cpu(png_data):
        """On-demand CPU decode for error paths / non-16 sizes (GPU-decode
        twin semantics: 16-bit >>8, GA/gray normalisation)."""
        import imagecodecs
        a = imagecodecs.png_decode(png_data)
        if a.dtype == np.uint16:
            a = (a >> 8).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[-1] == 2:
            a = np.concatenate([a[..., :1]] * 3 + [a[..., 1:]], -1)
        if a.shape[-1] == 3:
            a = np.concatenate(
                [a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
        return a

    def _route_decoded(t, path, rel, out_path, H, W, stats, progress_cb,
                        log):
        """Shared post-decode routing: size cap, odd-edge padding, queue."""
        arr = t["arr"]
        if H * W > 24_000_000:
            _submit_fallback(t, f"超大尺寸 {W}x{H}")
            progress_cb and progress_cb(stats)
            return
        if H % 2 or W % 2:
            Hp, Wp = H + (H & 1), W + (W & 1)
            pad = np.empty((Hp, Wp, 4), arr.dtype)
            pad[:H, :W] = arr
            if Wp != W:
                pad[:, W:] = pad[:, W:W + 1]
            if Hp != H:
                pad[H:] = pad[H:H + 1]
            t["arr"] = np.ascontiguousarray(pad)
            t["odd_wh"] = (H, W)
        gpu_q.put(t)
        if "odd_wh" not in t:
            bk = _real_bucket.get((H, W))
            if bk is not None:
                with stats.lock:
                    _bucket_queued[bk] += 1
                    _last = (_bucket_queued[bk] == _bucket_total.get(bk, -1))
                if _last:
                    gpu_q.put(("__flush__", bk))
            key = ((H + 15) // 16, (W + 15) // 16)
            group_queued[key] = group_queued.get(key, 0) + 1
            if group_queued[key] == group_total.get(key):
                gpu_q.put(("__flush__", key))
        return

    def decode_one(path):
        if stop_event is not None and stop_event.is_set():
            return None
        rel = os.path.relpath(path, base)
        out_path = os.path.join(dst, os.path.splitext(rel)[0] + ".webp")
        if skip_existing and os.path.exists(out_path)                 and os.path.abspath(out_path) != os.path.abspath(path):
            # webp→webp 同目录压缩时输出就是输入本身, 不能跳过
            with stats.lock:
                stats.skipped += 1
            progress_cb and progress_cb(stats)
            return None
        try:
            _gov.checkpoint("decode")
            png_data = open(path, "rb").read()
            if png_data[:8] != b"\x89PNG\r\n\x1a\n":
                from .extfmt import decode_any_meta
                arr, _xmeta = decode_any_meta(png_data)
                meta = _xmeta or None
                t = dict(rel=rel, out=out_path, png=png_data,
                         arr=arr, src=path, meta=meta,
                         size=len(png_data))
                H, W = arr.shape[:2]
                _route_decoded(t, path, rel, out_path, H, W,
                               stats, progress_cb, log)
                return None
            if _png_gpu_ok:
                # GPU decode path: hand the zlib stream to the DLL; pixels
                # come back with the batch result (arr materialised at
                # dispatch). CPU libpng never runs.
                try:
                    (w, h, bd, ct, inter, idat_b, plte, trns256,
                     trnsmode) = png_idat_scan(png_data)
                except PngScanError as e:
                    raise ValueError(f"PNG块扫描失败: {e}")
                if h % 16 or w % 16:
                    # 奇数/非16倍尺寸走 CPU 旧路径(桶/odd 语义, 罕见)
                    arr = _decode_png_cpu(png_data)
                    meta = extract_meta_cpp(png_data) or extract_meta(
                        png_data)
                    t = dict(rel=rel, out=out_path, png=png_data, arr=arr,
                             src=path, meta=meta, size=len(png_data))
                    _route_decoded(t, path, rel, out_path, h, w,
                                   stats, progress_cb, log)
                    return None
                meta = extract_meta_cpp(png_data) or extract_meta(png_data)
                # variant identity: the DLL uploads ONE palette/tRNS set
                # per batch, so images differing in tRNS/palette must not
                # share a group even when (h, w, bd, ct, inter) match
                import zlib as _zl
                t = dict(rel=rel, out=out_path, png=png_data,
                         idat=idat_b, ihdr=(h, w, bd, ct, inter),
                         plte=plte, trns256=trns256, trnsmode=trnsmode,
                         pvar=(trnsmode,
                               _zl.crc32((plte or b"")
                                         + (trns256 or b"")) & 0xFFFFFFFF),
                         src=path, meta=meta, size=len(png_data))
                if h * w > 24_000_000:
                    _submit_fallback(t, f"超大尺寸 {w}x{h}")
                    progress_cb and progress_cb(stats)
                    return None
                gpu_q.put(t)
                bk = _real_bucket.get((h, w))
                if bk is not None:
                    # bucket member: mixed-size padded batch keeps the
                    # stream count high for the latency-bound inflate
                    t["padgrid"] = bk
                    with stats.lock:
                        _bucket_queued[bk] += 1
                        last = (_bucket_queued[bk]
                                == _bucket_total.get(bk, -1))
                    if last:
                        gpu_q.put(("__flush__",
                                   ("pnganybucket", bk[1], bk[2])))
                else:
                    key = ((h + 15) // 16, (w + 15) // 16)
                    group_queued[key] = group_queued.get(key, 0) + 1
                    if group_queued[key] == group_total.get(key):
                        gpu_q.put(("__flush__", ("pngany", h, w)))
                return None
            # decode: imagecodecs (libpng, GIL-free) -> Pillow fallback;
            # metadata: C++ chunk scanner (the Python parser slices every
            # multi-MB IDAT chunk)
            try:
                import imagecodecs
                arr = imagecodecs.png_decode(png_data)
                if (arr.dtype != np.uint8
                        or (arr.ndim == 3 and arr.shape[-1] == 2)):
                    # 16-bit / gray+alpha: normalise with the GPU-path
                    # semantics (>>8, GA->RGBA); the old concat silently
                    # promoted uint16 and fed garbage to the encoder
                    arr = _decode_png_cpu(png_data)
                else:
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, -1)
                    if arr.shape[-1] == 3:
                        arr = np.concatenate(
                            [arr, np.full(arr.shape[:2] + (1,), 255,
                                          np.uint8)], -1)
            except Exception:                       # noqa: BLE001
                img = Image.open(io.BytesIO(png_data))
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert(
                        "RGBA" if "A" in img.getbands()
                        or "transparency" in img.info else "RGB")
                arr = np.asarray(img)
                if (arr.dtype != np.uint8
                        or (arr.ndim == 3 and arr.shape[-1] == 2)):
                    arr = _decode_png_cpu(png_data)
                else:
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, -1)
                    if arr.shape[-1] == 3:
                        arr = np.concatenate(
                            [arr, np.full(arr.shape[:2] + (1,), 255,
                                          np.uint8)], -1)
            meta = extract_meta_cpp(png_data)
            if meta is None:
                meta = extract_meta(png_data)
            t = dict(rel=rel, out=out_path, png=png_data, arr=arr,
                     src=path,
                     meta=meta, size=len(png_data))
            H, W = arr.shape[:2]
            with stats.lock:
                stats.cur_name = os.path.basename(path)
            _route_decoded(t, path, rel, out_path, H, W,
                           stats, progress_cb, log)
            return None
        except Exception as e:                          # noqa: BLE001
            with stats.lock:
                stats.failed += 1
            _park_failed(path, dst, log)
            log(f"[失败] {rel}: 解码异常 {e} (原图已复制到 未转换/)")
            progress_cb and progress_cb(stats)
            return None

    finish_ex = ThreadPoolExecutor(max_workers=finish_workers)
    select_ex = ThreadPoolExecutor(max_workers=4)   # dedicated: never queues
                                                   # behind finish work
    # subprocess verification pool: Pillow webp-decode + PSNR in independent
    # interpreters so their GIL-held glue cannot contend with the pipeline
    verify_pool = None
    verify_results = []
    _vw = None                  # verifyworker import hoisted: the pool
                                # branch below can raise before reaching it
    if min_psnr <= 0 and not verify_meta:
        pass                    # verification fully disabled: no spawn cost
    else:
        try:
            from . import verifyworker as _vw
            import multiprocessing as _mp
            import sys as _sys
            # spawn needs a real entry file; interactive/<stdin> runs must not
            # create the pool (children would fail to re-import __main__)
            _frozen = getattr(_sys, "frozen", False)
            _main = getattr(_sys.modules.get("__main__"), "__file__", None)
            if (not _frozen and (not _main or _main.endswith("<stdin>")
                                 or not os.path.isfile(_main))):
                raise RuntimeError("交互式环境无法 spawn 子进程")
            ctx = _mp.get_context("spawn")
            if nvp is None:
                nvp = max(2, min(12, (os.cpu_count() or 8) - 4))
            # module-level entry (spawn pickles by reference; nested defs fail)
            # NOTE: maxtasksperchild here once cost 2.5s spawn per 32 tasks and
            # re-capped verify at ~80/s; the real leak was the DLL input
            # vectors (fixed), so no recycling by default. VERIFY_RECYCLE=N
            # opts in (large-photo corpora ratchet worker RSS to GBs and
            # trip the RAM governor without it)
            verify_pool = ctx.Pool(
                nvp, maxtasksperchild=(int(os.environ["VERIFY_RECYCLE"])
                                       if os.environ.get("VERIFY_RECYCLE")
                                       else 0))
        except Exception as _e:                        # noqa: BLE001
            log(f"子进程校验不可用({type(_e).__name__}: {_e})，改为进程内校验(较慢) -- 注意: 主脚本缺 if __name__=='__main__' 保护时spawn子进程会递归重跑")
            verify_pool = None
    # verify backpressure: caps dispatched-but-unverified images; each holds
    # its task (~5.5MB RGBA+PNG) alive until the pool returns, so an uncapped
    # backlog equals the whole corpus in RAM (~37GB for 6722 images)
    _verify_sem = None
    if verify_pool is not None and min_psnr > 0:
        # decouple encode from verify: at 192 permits the GPU thread blocks
        # on acquire and encode drops to the verify rate (~110/s observed,
        # vs 154/s encode-only). 512 permits peak ~2.8GB of held tasks —
        # the RAM governor still watches the real limit
        _verify_sem = threading.Semaphore(
            int(os.environ.get("VERIFY_MAXFLIGHT") or 512))
    # shared-memory ring for zero-pickle verify payloads
    _shmring = None
    _shm_slots = []          # free slot stack
    _shm_lock = threading.Lock()
    if verify_pool is not None:
        try:
            from . import shmr as _shmr
            # capacity = largest image among the corpus (from the pre-scan)
            _maxpx = 0
            for k in group_total:
                if k[0] > 0:
                    px = (k[0] * 16) * (k[1] * 16)
                    # ignore bomb-sized corrupt headers (a single file
                    # claiming 4.7M x 4.3B px once made the ring request
                    # exabytes and disabled the zero-copy path entirely)
                    if px <= 100_000_000:
                        _maxpx = max(_maxpx, px)
            if _maxpx:
                # slot cap 1536x1920 RGBA (11.8MB): sizing by the corpus max
                # once built a 4.1GB ring for one 4864x3328 image — every
                # slot page-commits on first write and evicts useful cache;
                # oversize images take the path-payload fallback instead
                _cap = min(_maxpx, 1536 * 1920) * 4
                _ring_n = 64       # >= verify window (192 permits): excess
                                   # falls back to path payloads
                _shmring = _shmr.create_ring(_ring_n, _cap)
                _shm_slots = list(range(_ring_n - 1, -1, -1))
        except Exception as _e:                       # noqa: BLE001
            log(f"共享内存环不可用({_e})，校验走路径回退")
            _shmring = None

    decode_ex = ThreadPoolExecutor(max_workers=decode_workers)
    # bounded queue: decode threads block when the GPU falls behind, so RAM
    # stays flat no matter the corpus size (31GB spike without this)
    gpu_q = _queue.Queue(maxsize=3 * batch)
    groups = {}
    decode_done = threading.Event()


    def flush_group(key):
        group = groups.pop(key, [])
        if not group:
            return
        if isinstance(key, tuple) and key and key[0] == "__pad__":
            _padded_flush(group, key[1], key[2])
            return
        if (isinstance(key, tuple) and key and key[0] == "pngpad"):
            Hp2, Wp2 = key[1], key[2]
            try:
                _padded_flush_png(group, Hp2, Wp2)
            except MemoryError:
                raise
            except Exception as e:                  # noqa: BLE001
                log(f"GPU解码桶批异常({len(group)}张): {e}")
                for t in group:
                    _submit_fallback(t, f"GPU解码桶批异常 {e}")
            return
        if isinstance(key, tuple) and key and key[0] == "png":
            H2, W2 = key[1], key[2]
            try:
                _cpp_flush_png(group, H2, W2)
            except MemoryError:
                raise
            except Exception as e:                      # noqa: BLE001
                log(f"GPU解码批异常({len(group)}张): {e}")
                for t in group:
                    _submit_fallback(t, f"GPU解码批异常 {e}")
            return
        oom = False
        try:
            _flush_one(group)
        except MemoryError:
            oom = True                   # handled below, outside the except
        if oom:
            # GPU OOM: release references (traceback frames keep arrays
            # alive inside the except block!), then split and retry
            import gc
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            mid = len(group) // 2
            log(f"显存不足,拆批 {len(group)} -> {mid}+{len(group)-mid}")
            if mid == 0:
                for t in group:
                    _submit_fallback(t, "GPU显存不足")
                return
            try:
                for sub in (group[:mid], group[mid:]):
                    _flush_one(sub)
            except Exception as e:                      # noqa: BLE001
                log(f"拆批后仍失败: {e}")
                for t in group:
                    _submit_fallback(t, f"GPU批量异常 {e}")

    _prev_stage = [None]    # previous batch's planes+select, awaiting closed_loop

    def _prepare(group):
        """Stage 1: upload + YUV + mode search; submit select async.
        Returns dict(planes, futures, group) — does NOT run closed loop."""
        with cp.cuda.Device(device):
            H2, W2 = group[0]["arr"].shape[:2]
            rgb = cp.empty((len(group), H2, W2, 4), cp.uint8)
            for i, t in enumerate(group):
                rgb[i].set(t["arr"])
            mb_h, mb_w = (H2 + 15) // 16, (W2 + 15) // 16
            if H2 % 16 == 0 and W2 % 16 == 0:
                ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb, int16_out=True)
                ys, us, vs = list(ypl), list(upl), list(vpl)
            else:
                ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb)
                ys = [GE.pad_to_mb_gpu(ypl[i], mb_h, mb_w) for i in range(len(group))]
                us = [GE.pad_to_mb_gpu(upl[i], mb_h, mb_w, half=True) for i in range(len(group))]
                vs = [GE.pad_to_mb_gpu(vpl[i], mb_h, mb_w, half=True) for i in range(len(group))]
            Ybs, Ubs, Vbs = cp.stack(ys), cp.stack(us), cp.stack(vs)
            del rgb, ypl, upl, vpl, ys, us, vs
            raw = GE.gpu_modes_pass_batch(Ybs, Ubs, Vbs, y1, select=False)

        from .vp8_encode import select_modes as _sel
        B = len(group)
        n_mb = raw["i16_mode"].shape[0] // B
        penalty = 1000 * y1.q_avg * y1.q_avg

        def _pick(i):
            sse = np.ascontiguousarray(raw["sse4"][i])
            i4_modes, is_i4, _ = _sel(
                sse, raw["i16_score"][i * n_mb:(i + 1) * n_mb],
                raw["i16_mode"][i * n_mb:(i + 1) * n_mb],
                raw["mb_w"], raw["mb_h"], penalty)
            return dict(is_i4=is_i4.astype(bool),
                        i16_mode=raw["i16_mode"][i * n_mb:(i + 1) * n_mb],
                        uv_mode=raw["uv_mode"][i * n_mb:(i + 1) * n_mb],
                        i4_modes=i4_modes)

        futs = [select_ex.submit(_pick, i) for i in range(B)]
        return dict(planes=(Ybs, Ubs, Vbs), futures=futs, group=group)

    def _closed_and_finish(stage):
        """Stage 2: closed_loop on a prepared stage + D2H + submit finish."""
        modes_list = [f.result() for f in stage["futures"]]
        Ybs, Ubs, Vbs = stage["planes"]
        with cp.cuda.Device(device):
            y_dc, y_ac, uv_lv = closed_loop_batch_gpu(
                Ybs, Ubs, Vbs, modes_list, y1, y2, uv_m,
                y1deq, y2deq, uvdeq)
        for i, t in enumerate(stage["group"]):
            if stop_event is not None and stop_event.is_set():
                return
            finish_ex.submit(finish_gpu, t, modes_list[i],
                             y_dc[i], y_ac[i], uv_lv[i])
        del stage  # release GPU plane refs so pool can reclaim


    _gpulog = None
    if os.environ.get("GPULOG"):
        _gpulog = open(os.environ["GPULOG"], "a", buffering=1)

    def _gpuline(stage, W, H, n, t0):
        if _gpulog is None:
            return
        try:
            with cp.cuda.Device(device):
                free, _tot = cp.cuda.runtime.memGetInfo()
            avail, _load = _memstat()
            _gpulog.write(
                f"{time.time():.2f},{stage},{W},{H},{n},"
                f"{(time.time() - t0) * 1000:.0f},{free / 1e9:.2f},"
                f"{avail:.1f}\n")
        except Exception:                           # noqa: BLE001
            pass

    def _flush_one(group):
        """Pipelined: prepare CURRENT batch, then closed_loop on PREVIOUS.
        GPU never idles between mode_search and closed_loop."""
        try:
            H2, W2 = group[0]["arr"].shape[:2]
            if _cpp is not None:
                if H2 % 16 == 0 and W2 % 16 == 0:
                    _t0 = time.time()
                    _cpp_flush(group, W2, H2)
                    _gpuline("dll", W2, H2, len(group), _t0)
                    return
            _t0 = time.time()
            _dev_low_purge()
            cur = _prepare(group)
            if _prev_stage[0] is not None:
                _closed_and_finish(_prev_stage[0])
                _prev_stage[0] = None
            _prev_stage[0] = cur
            cp.get_default_memory_pool().free_all_blocks()
            _gpuline("cupy", W2, H2, len(group), _t0)
        except MemoryError:
            raise
        except Exception as e:                          # noqa: BLE001
            log(f"GPU批量异常({len(group)}张): {e}")
            for t in group:
                _submit_fallback(t, f"GPU批量异常 {e}")

    # work-conserving dispatch: when no batch is in flight and the largest
    # partial group already holds >= _eager_min images, flush it instead of
    # idling the GPU. Mixed-size corpora spread arrivals over several
    # size-groups, so the batch-full and 2.5s-starvation triggers rarely
    # fire mid-run and the GPU sat idle between partial-group flushes.
    _em = os.environ.get("GPU_EAGERMIN")
    _eager_min = int(_em) if _em is not None else 24
    _pb = os.environ.get("PNG_BATCH")
    _png_batch = int(_pb) if _pb is not None else 192

    def _eager_flush():
        """Flush the biggest partial group when the GPU would idle anyway."""
        if not (_eager_min and groups and not _pend):
            return False
        big = max(groups, key=lambda _k: len(groups[_k]))
        if len(groups[big]) < _eager_min:
            return False
        flush_group(big)
        return True

    def gpu_worker():
        last_arrival = time.time()
        while True:
            try:
                t = gpu_q.get(timeout=0.05)
            except _queue.Empty:
                if _pend:
                    _drain_pend(True)      # wait for an in-flight batch
                if decode_done.is_set() and gpu_q.empty():
                    break
                if _eager_flush():
                    last_arrival = time.time()
                    continue
                # starvation: flush partial groups so tail images do not wait
                if groups and time.time() - last_arrival > 2.5:
                    for key in list(groups):
                        flush_group(key)
                    last_arrival = time.time()
                continue
            if t is None:
                break
            if isinstance(t, tuple) and t and t[0] == "__flush__":
                fk = t[1]
                if isinstance(fk, tuple) and fk and fk[0] == "pngany":
                    for k in list(groups):
                        if (k and k[0] == "png" and k[1] == fk[1]
                                and k[2] == fk[2]):
                            flush_group(k)
                elif (isinstance(fk, tuple) and fk
                        and fk[0] == "pnganybucket"):
                    # fk = ("pnganybucket", Hp, Wp); group key =
                    # ("pngpad", Hp, Wp, bd, ct, inter) -- compare k[1]/k[2]
                    for k in list(groups):
                        if (k and k[0] == "pngpad" and k[1] == fk[1]
                                and k[2] == fk[2]):
                            flush_group(k)
                else:
                    flush_group(fk)
                last_arrival = time.time()
                continue
            last_arrival = time.time()
            _gov.checkpoint("gpu")
            if "idat" in t:
                # GPU-decoded PNG task: group by full geometry (size groups
                # can mix bit depths / colour types); bucket members share
                # one padded grid keyed on (Hp, Wp, variant)
                h, w, bd, ct, inter = t["ihdr"]
                pg = t.get("padgrid")
                tail = (bd, ct, inter) + t.get("pvar", ())
                if pg is not None:
                    key = ("pngpad", pg[1], pg[2]) + tail
                else:
                    key = ("png", h, w) + tail
            else:
                shape = t["arr"].shape[:2]
                if "odd_wh" in t:
                    key = ("odd",) + shape  # true odd dims in the header
                else:
                    key = (_real_bucket.get(shape) or shape)
                                                # bucket members share one grid
            groups.setdefault(key, []).append(t)
            # PNG batches want MANY concurrent streams on the GPU: the
            # inflate kernel is latency-bound per stream, aggregate scales
            # with streams (measured 89 -> 304MB/s going 64 -> 256)
            _cap = _png_batch if "idat" in t else batch
            if len(groups[key]) >= _cap:
                flush_group(key)
            elif _eager_flush():
                last_arrival = time.time()
        for key in list(groups):
            if stop_event is not None and stop_event.is_set():
                break
            flush_group(key)
        # drain the async GPU pipeline completely before finishing
        _drain_pend(True)
        # process the final pipelined batch (no next batch to overlap with)
        if _prev_stage[0] is not None:
            try:
                _closed_and_finish(_prev_stage[0])
            except Exception as e:                      # noqa: BLE001
                log(f"GPU闭环异常(末批): {e}")
                for t in _prev_stage[0]["group"]:
                    _submit_fallback(t, f"GPU闭环异常 {e}")
            _prev_stage[0] = None


    def _res_logger():
        last = 0.0
        while not decode_done.is_set() or not gpu_q.empty():
            now = time.time()
            if now - last >= 5.0:
                last = now
                log(f"[资源] RAM可用{_gov.avail_gb:.1f}GB "
                    f"(最低{_gov.min_avail:.1f}GB) 内存占用{_gov.load_pct}%")
            time.sleep(0.5)
    threading.Thread(target=_res_logger, daemon=True).start()

    gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
    gpu_thread.start()

    def decode_driver():
        # parallel decode across decode_workers threads
        list(decode_ex.map(decode_one, files))
        decode_done.set()

    try:
        decode_driver()           # maps decode_one over decode_ex and waits
        decode_ex.shutdown(wait=False)
        decode_done.set()
        gpu_q.put(None)
        gpu_thread.join()
        select_ex.shutdown(wait=False)
        if verify_pool is not None:
            # drain verification BEFORE closing finish_ex: a failing verify
            # may still submit a Pillow fallback task
            for r in verify_results:
                try:
                    r.wait(timeout=60)
                except Exception:                      # noqa: BLE001
                    pass
        finish_ex.shutdown(wait=True)
    finally:
        # governor aborts (and any crash) must not leak the spawn pool:
        # orphaned children hold ~0.5GB each and starve the machine
        if verify_pool is not None:
            verify_pool.terminate()
        _gov.stop()
        if _psnrlog is not None:
            _psnrlog.close()
        log(f"分层校验: GPU门拦截{_gate_stats['gate']}张, "
            f"抽样{_gate_stats['sampled']}张(失败{_gate_stats['sample_fail']}张), "
            f"跳过{_gate_stats.get('sampled_out', 0)}张 "
            f"(抽样率{_sample_pct}%, GPU门限{_gpu_psnr_min or 'off'})")
    return stats
          


# ---------------------------------------------------------- multi-process
#
# 多进程版:每个子进程独立 GIL,各自喂同一块 GPU。把 16 核 CPU 的
# 解码/闭环/校验全部吃满,GPU 得到多路供给,占用率随进程数上升。

def run_batch_multiproc(src, dst, quality=90, device=1, nproc=0,
                        recursive=False, skip_existing=True, min_psnr=34.0,
                        verify_meta=True, batch=24, progress_cb=None,
                        log_cb=None, stop_event=None):
    import json
    import subprocess
    import sys
    import tempfile

    log = log_cb or (lambda s: None)
    from .extfmt import collect_images
    files = collect_images(src, recursive)
    stats = BatchStats()
    stats.total = len(files)
    stats.start = time.time()
    if not files:
        log("未找到PNG文件")
        return stats
    # one worker process per GPU (no CUDA-context clash across devices)
    try:
        import cupy as _cp
        ndev = _cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # noqa: BLE001
        ndev = 1
    nproc = max(1, min(nproc or ndev, ndev, (len(files) + 39) // 40))
    chunks = [files[i::nproc] for i in range(nproc)]
    tmp = tempfile.mkdtemp(prefix="gpupic_")
    jobs, ress, procs = [], [], []
    for i, chunk in enumerate(chunks):
        jf = os.path.join(tmp, f"job{i}.json")
        rf = os.path.join(tmp, f"res{i}.json")
        with open(jf, "w", encoding="utf-8") as f:
            json.dump(dict(files=chunk, dst=dst, base=src, quality=quality,
                           device=i % ndev, batch=batch,
                           decode_workers=6, finish_workers=6), f,
                      ensure_ascii=False)
        jobs.append(jf)
        ress.append(rf)
        import copy as _copy
        env = dict(os.environ)
        env["GPUPIC_DEVICE"] = str(i % ndev)
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "gpuwebp.worker", jf, rf], env=env))

    def read_prog(rf):
        try:
            with open(rf + ".prog", "r", encoding="utf-8") as f:
                parts = f.read().split()
                return tuple(int(x) for x in parts[:4]) + \
                    (int(parts[4]), int(parts[5]))
        except (OSError, ValueError, IndexError):
            return (0, 0, 0, 0, 0, 0)

    try:
        while any(p.poll() is None for p in procs):
            if stop_event is not None and stop_event.is_set():
                for p in procs:
                    p.terminate()
                break
            agg = [0] * 6
            for rf in ress:
                pr = read_prog(rf)
                for k in range(6):
                    agg[k] += pr[k]
            stats.done, stats.failed, stats.skipped, stats.fallbacks = agg[:4]
            stats.src_bytes, stats.dst_bytes = agg[4], agg[5]
            progress_cb and progress_cb(stats)
            time.sleep(0.5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.wait()
        # fold in final results
        agg = [0] * 6
        for rf in ress:
            pr = read_prog(rf)
            for k in range(6):
                agg[k] += pr[k]
        stats.done, stats.failed, stats.skipped, stats.fallbacks = agg[:4]
        stats.src_bytes, stats.dst_bytes = agg[4], agg[5]
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
    progress_cb and progress_cb(stats)
    return stats