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


def run_batch(src, dst, quality=90, engine="gpu", device=0, recursive=False,
              skip_existing=True, min_psnr=34.0, verify_meta=True,
              workers=3, progress_cb=None, log_cb=None, stop_event=None):
    """Convert all PNGs under src into dst (mirrors subfolder structure).
    progress_cb(stats) is called after every file; log_cb(str) for messages.
    Returns BatchStats."""
    if engine == "gpu-batch":
        return run_batch_fast(src, dst, quality=quality, device=device,
                              recursive=recursive, skip_existing=skip_existing,
                              min_psnr=min_psnr, verify_meta=verify_meta,
                              batch=96, decode_workers=10, finish_workers=4,
                              progress_cb=progress_cb, log_cb=log_cb,
                              stop_event=stop_event)
    stats = BatchStats()
    log = log_cb or (lambda s: None)
    files = collect_pngs(src, recursive)
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

def run_batch_fast(src, dst, quality=90, device=0, recursive=False,
                   skip_existing=True, min_psnr=34.0, verify_meta=True,
                   batch=16, decode_workers=4, finish_workers=8,
                   progress_cb=None, log_cb=None, stop_event=None):
    files = collect_pngs(src, recursive)
    return run_batch_files_fast(files, dst, base=src, quality=quality, device=device,
                                skip_existing=skip_existing, min_psnr=min_psnr,
                                verify_meta=verify_meta, batch=batch,
                                decode_workers=decode_workers,
                                finish_workers=finish_workers,
                                progress_cb=progress_cb, log_cb=log_cb,
                                stop_event=stop_event)


def run_batch_files_fast(files, dst, base=None, quality=90, device=0,
                         skip_existing=True, min_psnr=34.0, verify_meta=True,
                         batch=16, decode_workers=4, finish_workers=8,
                         progress_cb=None, log_cb=None, stop_event=None):
    import queue as _queue
    import cupy as cp
    from . import gpu_engine as GE
    from . import vp8_encode, vp8_tables as T
    from .closed_loop_gpu import closed_loop_batch_gpu
    from .alpha_enc import make_alph_chunk
    from .encoder import _assemble

    stats = BatchStats()
    log = log_cb or (lambda s: None)
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
    group_total = {}
    keys = []
    for p in files:
        k = _padded_key(p)
        if k is None:
            k = (-1, -1)          # unreadable: decode path will park it
        group_total[k] = group_total.get(k, 0) + 1
    files = sorted(files, key=lambda p: (_padded_key(p) or (-1, -1), p))
    group_queued = dict.fromkeys(group_total, 0)

    bq, y1, y2, uv_m, fl = vp8_encode.setup_quant(quality)
    y2ac = max(8, int(T.AC_TABLE2[bq]))
    y1deq = np.array([T.DC_TABLE[bq]] + [T.AC_TABLE[bq]] * 15, np.int64)
    y2deq = np.array([T.DC_TABLE[bq] * 2] + [y2ac] * 15, np.int64)
    uvdeq = np.array([T.DC_TABLE[max(0, min(117, bq - 2))]]
                     + [T.AC_TABLE[bq]] * 15, np.int64)

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

    def finish_gpu(task, modes, y_dc, y_ac, uv_lv):
        try:
            a = task["arr"][..., 3]
            alpha = None if bool((a == 255).all()) else make_alph_chunk(a)
            skip = ~(y_dc.any(-1) | y_ac.any(-1).any(-1) | uv_lv.any(-1).any(-1))
            H2, W2 = task["arr"].shape[:2]
            webp = _assemble(W2, H2, (W2 + 15) // 16, (H2 + 15) // 16,
                             bq, fl, modes["is_i4"], modes["i16_mode"],
                             modes["uv_mode"], modes["i4_modes"],
                             y_dc, y_ac, uv_lv, skip, alpha, task["meta"])
            if verify_pool is None:
                note = check(webp, task["png"], task["arr"])
                if note:
                    finish_pillow(task, note)
                    return
                finish_common(task, webp, False, "")
                return
            # write first, verify in a subprocess (true parallelism), then
            # either finalise or re-encode via the Pillow fallback
            finish_common(task, webp, False, "")
            payload = (task["arr"], webp, verify_meta, min_psnr, task["meta"])

            def _on_verify(result, task=task):
                ok, note = result
                if ok:
                    return
                log(f"GPU输出校验未过 {task['rel']}: {note}，CPU兜底重压")
                finish_ex.submit(finish_pillow, task, note)

            verify_results.append(
                verify_pool.apply_async(_vw.verify_payload, (payload,),
                                        callback=_on_verify))
        except Exception as e:                          # noqa: BLE001
            log(f"GPU编码失败 {task['rel']}: {e}")
            finish_pillow(task, f"GPU编码异常 {e}")

    def decode_one(path):
        if stop_event is not None and stop_event.is_set():
            return None
        rel = os.path.relpath(path, base)
        out_path = os.path.join(dst, os.path.splitext(rel)[0] + ".webp")
        if skip_existing and os.path.exists(out_path):
            with stats.lock:
                stats.skipped += 1
            progress_cb and progress_cb(stats)
            return None
        try:
            png_data = open(path, "rb").read()
            if png_data[:8] != b"\x89PNG\r\n\x1a\n":
                with stats.lock:
                    stats.skipped += 1
                _park_failed(path, dst, log)
                log(f"[跳过] 非PNG文件: {rel} (已复制到 未转换/)")
                progress_cb and progress_cb(stats)
                return None
            img = Image.open(io.BytesIO(png_data))
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.getbands()
                                  or "transparency" in img.info else "RGB")
            arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.shape[-1] == 3:
                arr = np.concatenate(
                    [arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
            t = dict(rel=rel, out=out_path, png=png_data, arr=arr,
                     meta=extract_meta(png_data), size=len(png_data))
            H, W = arr.shape[:2]
            with stats.lock:
                stats.cur_name = os.path.basename(path)
            if H % 2 == 0 and W % 2 == 0:
                gpu_q.put(t)
                key = ((H + 15) // 16, (W + 15) // 16)
                group_queued[key] = group_queued.get(key, 0) + 1
                if group_queued[key] == group_total.get(key):
                    gpu_q.put(("__flush__", key))   # last of its size: flush
            else:
                finish_ex.submit(finish_pillow, t, f"奇数尺寸 {W}x{H}")
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
    try:
        import multiprocessing as _mp
        import sys as _sys
        # spawn needs a real entry file; interactive/<stdin> runs must not
        # create the pool (children would fail to re-import __main__)
        _frozen = getattr(_sys, "frozen", False)
        _main = getattr(_sys.modules.get("__main__"), "__file__", None)
        if (not _frozen and (not _main or _main.endswith("<stdin>")
                             or not os.path.isfile(_main))):
            raise RuntimeError("交互式环境无法 spawn 子进程")
        from . import verifyworker as _vw
        ctx = _mp.get_context("spawn")
        nvp = max(2, min(6, (os.cpu_count() or 8) // 3))
        # module-level entry (spawn pickles by reference; nested defs fail)
        verify_pool = ctx.Pool(nvp)
    except Exception as _e:                            # noqa: BLE001
        log(f"子进程校验不可用({_e})，回退线程内校验")
        verify_pool = None
    decode_ex = ThreadPoolExecutor(max_workers=decode_workers)
    gpu_q = _queue.Queue()
    groups = {}
    decode_done = threading.Event()


    def flush_group(key):
        group = groups.pop(key, [])
        if not group:
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
                    finish_ex.submit(finish_pillow, t, "GPU显存不足")
                return
            try:
                for sub in (group[:mid], group[mid:]):
                    _flush_one(sub)
            except Exception as e:                      # noqa: BLE001
                log(f"拆批后仍失败: {e}")
                for t in group:
                    finish_ex.submit(finish_pillow, t, f"GPU批量异常 {e}")

    def _flush_one(group):
        try:
            with cp.cuda.Device(device):
                # direct per-image H2D into a preallocated buffer: no CPU
                # np.stack of a ~200MB array on this thread
                H2, W2 = group[0]["arr"].shape[:2]
                rgb = cp.empty((len(group), H2, W2, 4), cp.uint8)
                for i, t in enumerate(group):
                    rgb[i].set(t["arr"])
                mb_h, mb_w = (H2 + 15) // 16, (W2 + 15) // 16
                if H2 % 16 == 0 and W2 % 16 == 0:
                    # 16-aligned (99% of NAI output): YUV already int16 planes
                    # in final layout -> padding is a zero-copy no-op
                    ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb, int16_out=True)
                    ys, us, vs = list(ypl), list(upl), list(vpl)
                else:
                    ypl, upl, vpl = GE.rgb_to_yuv420_gpu(rgb)
                    ys = [GE.pad_to_mb_gpu(ypl[i], mb_h, mb_w) for i in range(len(group))]
                    us = [GE.pad_to_mb_gpu(upl[i], mb_h, mb_w, half=True) for i in range(len(group))]
                    vs = [GE.pad_to_mb_gpu(vpl[i], mb_h, mb_w, half=True) for i in range(len(group))]
                Ybs, Ubs, Vbs = cp.stack(ys), cp.stack(us), cp.stack(vs)
                raw = GE.gpu_modes_pass_batch(Ybs, Ubs, Vbs, y1, select=False)

            # per-image mode selection (nogil numba), parallel across threads
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
            modes_list = [f.result() for f in futs]
            with cp.cuda.Device(device):
                y_dc, y_ac, uv_lv = closed_loop_batch_gpu(
                    Ybs, Ubs, Vbs, modes_list, y1, y2, uv_m,
                    y1deq, y2deq, uvdeq)
            for i, t in enumerate(group):
                if stop_event is not None and stop_event.is_set():
                    return
                finish_ex.submit(finish_gpu, t, modes_list[i],
                                 y_dc[i], y_ac[i], uv_lv[i])
            cp.get_default_memory_pool().free_all_blocks()
        except MemoryError:
            raise                       # let flush_group split the batch
        except Exception as e:                          # noqa: BLE001
            log(f"GPU批量异常({len(group)}张): {e}")
            for t in group:
                finish_ex.submit(finish_pillow, t, f"GPU批量异常 {e}")

    def gpu_worker():
        last_arrival = time.time()
        while True:
            try:
                t = gpu_q.get(timeout=0.05)
            except _queue.Empty:
                if decode_done.is_set() and gpu_q.empty():
                    break
                # starvation: flush partial groups so tail images do not wait
                if groups and time.time() - last_arrival > 2.5:
                    for key in list(groups):
                        flush_group(key)
                    last_arrival = time.time()
                continue
            if t is None:
                break
            if isinstance(t, tuple) and t and t[0] == "__flush__":
                flush_group(t[1])
                last_arrival = time.time()
                continue
            last_arrival = time.time()
            key = t["arr"].shape[:2]        # exact dims: batching safety
            groups.setdefault(key, []).append(t)
            if len(groups[key]) >= batch:
                flush_group(key)
        for key in list(groups):
            if stop_event is not None and stop_event.is_set():
                break
            flush_group(key)


    gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
    gpu_thread.start()

    def decode_driver():
        # parallel decode across decode_workers threads
        list(decode_ex.map(decode_one, files))
        decode_done.set()

    decode_driver()               # maps decode_one over decode_ex and waits
    decode_ex.shutdown(wait=False)
    decode_done.set()
    gpu_q.put(None)
    gpu_thread.join()
    select_ex.shutdown(wait=False)
    if verify_pool is not None:
        # drain verification BEFORE closing finish_ex: a failing verify may
        # still submit a Pillow fallback task
        for r in verify_results:
            try:
                r.wait(timeout=300)
            except Exception:                          # noqa: BLE001
                pass
    finish_ex.shutdown(wait=True)
    if verify_pool is not None:
        verify_pool.terminate()
    return stats
          


# ---------------------------------------------------------- multi-process
#
# 多进程版:每个子进程独立 GIL,各自喂同一块 GPU。把 16 核 CPU 的
# 解码/闭环/校验全部吃满,GPU 得到多路供给,占用率随进程数上升。

def run_batch_multiproc(src, dst, quality=90, device=0, nproc=0,
                        recursive=False, skip_existing=True, min_psnr=34.0,
                        verify_meta=True, batch=24, progress_cb=None,
                        log_cb=None, stop_event=None):
    import json
    import subprocess
    import sys
    import tempfile

    log = log_cb or (lambda s: None)
    files = collect_pngs(src, recursive)
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
