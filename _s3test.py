"""Step 3 integration: mixed-size corpus through the pipeline twice.
Run A: PADBUCKET=0 (exact-size batches only) — baseline.
Run B: buckets on (tail sizes merged onto padded grids).
Every output .webp must be byte-identical.
"""
import os, sys, glob, random, struct, hashlib, shutil, time

MODE = sys.argv[1] if len(sys.argv) > 1 else "A"
DST = sys.argv[2] if len(sys.argv) > 2 else f"_s3out{MODE}"
os.environ["PADBUCKET"] = "0" if MODE == "A" else "1"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

def main():
    random.seed(42)
    # corpus: mostly main size + a spread of tail sizes (mix of orientations)
    main = []
    tails = {}
    for f in sorted(glob.glob("D:/gpuimgtest3/*.png")):
        try:
            head = open(f, "rb").read(26)
            w, h = struct.unpack(">II", head[16:24])
        except OSError:
            continue
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            continue
        if (w, h) == (832, 1216):
            main.append(f)
        elif w % 16 == 0 and h % 16 == 0 and w <= 1536 and h <= 1920 and w % 2 == 0:
            tails.setdefault((w, h), []).append(f)
    
    random.shuffle(main)
    sel = main[:80]
    tail_keys = sorted(tails, key=lambda k: -len(tails[k]))[:8]
    for k in tail_keys:
        sel.extend(tails[k][:12])
    print(f"{MODE}: {len(sel)} files, main=80, tail dims={len(tail_keys)}")
    
    shutil.rmtree(DST, ignore_errors=True)
    sys.path.insert(0, ".")
    from gpuwebp.pipeline import run_batch_files_fast  # noqa: E402
    
    t0 = time.time()
    stats = run_batch_files_fast(
        sel, DST, base="D:/gpuimgtest3", quality=90, device=1,
        skip_existing=False, min_psnr=34.0, verify_meta=True,
        batch=64, decode_workers=10, finish_workers=10,
        log_cb=lambda s: print(f"[{MODE}]", s) if ("桶" in s or "异常" in s or "错误" in s or "资源" in s) else None,
        progress_cb=None)
    el = time.time() - t0
    n = stats.done + stats.failed + stats.skipped
    print(f"{MODE}: done={stats.done} failed={stats.failed} fallbacks={stats.fallbacks} "
          f"{n / el:.1f}/s ({el:.1f}s)")
    
    # digest every output
    dig = {}
    for p in glob.glob(DST + "/**/*.webp", recursive=True):
        rel = os.path.relpath(p, DST)
        dig[rel] = hashlib.md5(open(p, "rb").read()).hexdigest()
    with open(f"_s3dig{MODE}.txt", "w") as f:
        for k in sorted(dig):
            f.write(f"{k} {dig[k]}\n")
    print(f"{MODE}: {len(dig)} outputs digested")


if __name__ == "__main__":
    main()
