"""Step 4 ladder: N-file corpus-proportional run. Guarded for spawn pool."""
import os, sys, glob, random, struct, hashlib, shutil, time


def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else "B"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    DST = sys.argv[2] if len(sys.argv) > 2 else f"_s4out{MODE}"
    os.environ["PADBUCKET"] = "0" if MODE == "A" else "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

    random.seed(1234)
    files = sorted(glob.glob("D:/gpuimgtest3/*.png"))
    sel = random.sample(files, min(N, len(files)))
    dims = {}
    for f in sel:
        head = open(f, "rb").read(26)
        w, h = struct.unpack(">II", head[16:24])
        dims[(w, h)] = dims.get((w, h), 0) + 1
    tail = sum(v for k, v in dims.items() if k != (832, 1216))
    print(f"{MODE}: {len(sel)} files, {len(dims)} sizes, tail={tail}")

    shutil.rmtree(DST, ignore_errors=True)
    sys.path.insert(0, ".")
    from gpuwebp.pipeline import run_batch_files_fast  # noqa: E402

    t0 = time.time()
    stats = run_batch_files_fast(
        sel, DST, base="D:/gpuimgtest3", quality=90, device=1,
        skip_existing=False, min_psnr=34.0, verify_meta=True,
        batch=64, decode_workers=10, finish_workers=10,
        log_cb=lambda s: print(f"[{MODE}]", s, flush=True)
        if any(x in s for x in ("桶", "异常", "错误", "兜底", "显存", "资源")) else None,
        progress_cb=None)
    el = time.time() - t0
    n = stats.done + stats.failed + stats.skipped
    print(f"{MODE}: done={stats.done} failed={stats.failed} "
          f"fallbacks={stats.fallbacks} {n / el:.1f}/s ({el:.1f}s)")

    # settle: verify callbacks can still be finalising right after return
    deadline = time.time() + 10
    while time.time() < deadline:
        outs = glob.glob(DST + "/**/*.webp", recursive=True)
        if len(outs) >= stats.done:
            break
        time.sleep(0.5)
    dig = {}
    for p in outs:
        dig[os.path.relpath(p, DST)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    with open(f"_s4dig{MODE}{N}.txt", "w") as f:
        for k in sorted(dig):
            f.write(f"{k} {dig[k]}\n")
    print(f"{MODE}: {len(dig)} outputs digested (expected {stats.done})")


if __name__ == "__main__":
    main()
