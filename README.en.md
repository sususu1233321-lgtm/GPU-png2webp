# GPU Image Compressor — a from-scratch GPU WebP encoder + batch tool

[简体中文](README.md) | **[English](README.en.md)** | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Español](README.es.md)

Compress large image collections to WebP: **quality and resolution essentially
unchanged, file size down to ~10–15% of the original, with metadata preserved
byte-for-byte inside the output files.**

## Highlights

- **Real GPU encoding**: a video-codec-grade VP8 keyframe encoder runs the
  macroblock intra-prediction mode search and the wavefront closed-loop
  quantisation on your NVIDIA GPU (CUDA / CuPy / NVRTC) — the heaviest part
  of the pipeline; entropy coding runs as Numba-compiled machine code on the
  CPU.
- **40+ input formats**: PNG / JPEG / WebP (re-compression) / TIFF / GIF /
  JP2 / JXL / AVIF / HEIC / QOI / DDS / BMP …; native imagecodecs dispatch
  with a full Pillow-plugin fallback, so exotic files still convert.
- **Quality on par with libwebp**: within ±0.3 dB PSNR at the same quality
  setting (36–44 dB measured at q90, depending on content).
- **Metadata preserved**: PNG `tEXt` / `pHYs` / `eXIf` / `iCCP` and JPEG
  EXIF / XMP are repacked into the WebP XMP/EXIF/ICCP chunks; every output
  is **read back and compared byte-for-byte**, with automatic re-encoding
  through the CPU engine on any mismatch.
- **Bit-exact alpha channel**: a minimal hand-written VP8L lossless encoder
  (LZ77 + Huffman) — not a single alpha bit differs.
- **Continuously improved compression**: per-frame coefficient probability
  adaptation plus GPU trellis rate-distortion quantisation shave another
  9.0% at identical pixels (6,722-image corpus: 1073→977 MB); mixed sizes
  are merged into padded batches.
- **Per-image verification + CPU fallback**: every output is re-decoded and
  checked (dimensions / alpha / metadata / PSNR ≥ 34 dB); anything that
  fails is automatically re-encoded with Pillow (libwebp).
- **Resource control**: CPU core cap from the UI (`--cores N`, 0 =
  unlimited); a live memory governor throttles under low RAM to keep the
  machine responsive.
- **Multi-GPU**: pick the card in the UI or via `--device`; GTX 10-series
  through RTX 40-series and Tesla V100 all work, with bit-identical output
  across cards. Throughput: ~**150 images/s** with full verification on a
  V100 (the 6,722-image corpus takes 45 s); 250+ images/s without
  verification.
- **Experimental: GPU PNG decoding** (`PNG_GPU=1`): inflate + defilter for
  every PNG variant (bit depths 1–16, gray/RGB/palette/gray+A/RGBA, Adam7
  interlace, tRNS) entirely on the GPU, double-checked via the zlib adler32
  and per-chunk CRC32, bit-exact against libpng (exhaustively verified over
  1,512 combinations). Off by default while throughput is being tuned.

## Usage

### GUI (double-click `dist/GPU压图/GPU压图.exe`)

1. Pick the source folder (remembered between runs); output defaults to
   `source\webp` — originals are untouched.
2. Drag the quality slider (default 90), choose engine (GPU/CPU), GPU card
   and the CPU core cap.
3. Press “开始压缩”. Progress, speed, ETA and saved-percentage update live.
4. Optional “delete original PNGs after completion” with a confirmation
   dialog.
5. Files that fail to convert are copied verbatim into an `未转换/`
   subfolder, so the output directory is always a complete set.

> To deploy on another machine, copy the **whole `GPU压图` folder** (not just
> the exe). Windows 10/11 64-bit; an NVIDIA GPU (driver 2023+) uses the GPU
> engine, otherwise it falls back to the identical CPU engine.

### Command line

```
GPU压图.exe --src D:\MyPictures --quality 90
Options: --dst OUT_DIR   --cpu CPU-only   --device 0 (nvidia-smi index)
         --cores N CPU core cap (0 = unlimited)   --recursive
         --no-verify disable per-image verification
GPU压图.exe --diag    environment self-check (GPU/dependencies)
```

### Restore original PNG metadata from a WebP

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...]
```

## Measured results (V100, 6,722-image real-world corpus)

| Metric | Value |
|---|---|
| Throughput | **~150 images/s** (full verification, 16 cores); 250+ without |
| Size | 8–18% of the original PNG |
| PSNR (q90) | 36–44 dB (same tier as Pillow/libwebp) |
| Alpha | bit-identical |
| Metadata | 100% byte-level identical (read-back comparison) |
| Determinism | byte-identical across runs, batch sizes and GPUs |
| Failures | 0 (odd sizes fall back to CPU; OOM auto-splits batches) |
| Package | ~410 MB folder / 141 MB installer, zero dependencies |

## Building from source

```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
cpp\build.bat          # CUDA pipeline DLL (CUDA 12.x + MSVC 2022)
cpp\build_entropy.bat  # entropy coding DLL
python tools/protect_build.py   # optional: encrypted release build
python -m PyInstaller --noconfirm GPU压图.spec
```

## Repository layout

```
gpuwebp/            encoder Python sources
cpp/                CUDA/C++ pipeline sources (gpu_pipeline.cu, kernels.cuh,
                    gpu_inflate.cuh, entropy.cpp, pngdec.cpp, build*.bat)
vendor/libwebp-1.5.0  reference sources (BSD)
tests/              byte-exact verification tools against C references
tests/gpu_decode/   GPU decode/encode bit-exactness and perf gates
main.py             exe entry point
```

## Honest notes

- The encoder is home-grown (not libwebp); after probability adaptation and
  trellis it lands ~35% larger than libwebp method=6 at equal PSNR, while
  being 5–8× faster. Pixel-exact determinism holds across GPU
  architectures.
- VP8 lossy coding uses 4:2:0 chroma; highly saturated fine lines may show
  slight colour softening at q90 — raise quality to 95 to eliminate it.
- Requires an NVIDIA GPU (CUDA 12.x); without one everything runs on the
  CPU (Pillow).
- GPU PNG decoding is experimental (off by default): correctness is
  exhaustively verified, throughput tuning is ongoing.
