# GPU压图 — 自研 GPU WebP 编码器 + 批量压缩工具

**[简体中文](README.md)** | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | [Español](README.es.md)

把大批量图片压缩为 WebP:**质量、分辨率基本不变,体积压到约 10~15%,
图片元数据原样保留在文件里(字节级校验)。**

## 核心特点

- **真正的 GPU 编码**:视频编码器级别的 VP8 关键帧编码器在本机 GPU
  (经 CUDA / CuPy / NVRTC)完成宏块帧内预测模式搜索与波前闭环量化
  —— 全流程中计算量最大的部分;熵编码用 Numba 编译为机器码在 CPU 闭环执行。
- **输入格式 40+**:PNG / JPEG / WebP(再压缩)/ TIFF / GIF / JP2 / JXL /
  AVIF / HEIC / QOI / DDS / BMP 等;imagecodecs 原生魔数分发 + Pillow
  全插件回退,异常格式自动降级,结果永远可用。
- **质量与官方 libwebp 持平**:同质量档位 PSNR 相差 ±0.3dB 以内
  (90 档实测 36~44dB,视图片内容而定)。
- **元数据完整保留**:PNG 的 `tEXt` / `pHYs` / `eXIf` / `iCCP`、JPEG 的
  EXIF / XMP 全部打包进 WebP 的 XMP/EXIF/ICCP 块;每张输出后
  **逐字节回读比对**,不一致自动改用 CPU 引擎重压。
- **alpha 通道逐位精确**:自研最小 VP8L 无损编码器(LZ77 + Huffman),
  透明像素一个 bit 都不差。
- **压缩率持续优化**:帧级系数概率自适应 + GPU Trellis 率失真量化,同质量
  逐像素不变,体积再降 9.0%(6722 张实测 1073→977MB);混合尺寸 2D 补边合批。
- **逐张校验 + CPU 兜底**:每张输出都重新解码验证(尺寸/alpha/元数据/PSNR≥34dB),
  任何一项不过就自动用 Pillow(libwebp)按同样的元数据重压。
- **资源可控**:UI 可设 **CPU 核心上限**(`--cores N`),0=不限制;实时内存
  治理器低内存自动节流,防死机。
- **多显卡**:UI 下拉框 / `--device` 选卡,GTX 10 系至 RTX 40 系 + Tesla V100
  均可,跨卡输出逐位一致;速度 V100 完整校验约 **150 张/秒**(6722 张全集
  45 秒),不校验稳态 250+ 张/秒。
- **实验:GPU PNG 解码**(`PNG_GPU=1` 开启):inflate + 反滤波全部变体
  (位深 1~16、灰度/RGB/调色板/灰度+A/RGBA、Adam7 隔行、tRNS)都在 GPU
  完成,zlib adler32 + 逐块 CRC32 双校验,与 libpng 逐字节一致
  (1512 组合穷举位准)。默认关闭,性能调优中。

## 使用

### 图形界面(双击 `dist/GPU压图/GPU压图.exe`)

1. 源文件夹(打开后自动记住上次目录),输出默认 `源文件夹\webp`(原图不动)。
2. 拖动质量滑条(默认 90),选择引擎(GPU/CPU)、GPU 卡、CPU 核心上限。
3. 点"开始压缩"。实时显示进度、速度、剩余时间、已节省百分比。
4. "完成后删除原PNG"可选,勾选后结束时还有二次确认。
5. 转换失败或无法识别的文件原样复制到输出目录 `未转换/` 子文件夹,
   保证输出目录永远是一份完整的集合。

> 分发到别的电脑:拷贝**整个 `GPU压图` 文件夹**(不能只拷 exe)。
> 要求 Windows 10/11 64 位;有 NVIDIA 显卡(驱动 2023 年以后)走 GPU,
> 没有则自动切换 CPU 引擎,功能完全一致。

### 命令行

```
GPU压图.exe --src D:\我的图片 --quality 90
可选项: --dst 输出目录   --cpu 纯CPU   --device 0 选卡(=nvidia-smi序号)
        --cores N CPU核心上限(0=不限)   --recursive 含子目录
        --no-verify 关闭逐张校验
GPU压图.exe --diag    环境自检(检查GPU/依赖)
```

### 从 WebP 恢复原 PNG 元数据

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...] 原始字节
```

## 实测数据(V100,6722 张真实图片语料)

| 指标 | 数值 |
|---|---|
| 吞吐 | **~150 张/秒**(完整校验,16 核);不校验稳态 250+ 张/秒 |
| 体积 | 压到原 PNG 的 8~18% |
| PSNR(q90) | 36~44 dB(与 Pillow/libwebp 同档) |
| alpha | 逐位一致 |
| 元数据 | 100% 字节级一致(回读比对) |
| 确定性 | 同一文件多次编码/不同批量大小/不同 GPU,输出逐字节一致 |
| 失败率 | 0(奇数尺寸自动 CPU 兜底;显存不足自动拆批) |
| exe 体积 | ~410MB 文件夹 / 141MB 安装包(免装任何依赖) |

## 从源码构建

```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
cpp\build.bat          # CUDA 管线 DLL (需要 CUDA 12.x + MSVC 2022)
cpp\build_entropy.bat  # 熵编码 DLL
python tools/protect_build.py   # 可选: 加密 + Nuitka 编译发行包
python -m PyInstaller --noconfirm GPU压图.spec
```

## 目录结构

```
gpuwebp/            编码器 Python 源码
  vp8_encode.py     VP8 帧头/模式/系数布尔编码(Numba)
  closed_loop_jit.py 闭环重建+量化(与解码器逐位一致,Numba)
  gpu_engine.py     CuPy 批量内核:预测/模式搜索/变换
  alpha_enc.py      ALPH 通道 VP8L 最小编码器
  png_meta.py       PNG 元数据解析/XMP 打包/回读校验
  extfmt.py         40+ 输入格式分发(imagecodecs + Pillow 回退)
  webp_container.py RIFF/VP8X 容器
  encoder.py        高层 API(GPU/CPU 引擎)
  pipeline.py       批量流水线+校验+兜底+多卡选择+核心上限
  app.py            中文界面 + CLI
cpp/                CUDA/C++ 管线源码
  gpu_pipeline.cu   异步批处理管线(模式搜索/闭环量化/多槽双缓冲)
  kernels.cuh       GPU 内核(波前闭环/Trellis/PNG 反滤波全变体)
  gpu_inflate.cuh   GPU zlib inflate 内核(实验 GPU 解码)
  entropy.cpp       批式布尔熵编码 + 概率自适应
  pngdec.cpp        PNG 块扫描/快速解码
  build*.bat        构建脚本
vendor/libwebp-1.5.0  参照源码(BSD,常量与语义对照)
tests/              各组件与 C 参照的逐字节验证工具
tests/gpu_decode/   GPU 解码/编码位准与性能门测试
main.py             exe 入口
```

## 技术说明(诚实预期)

- 编码器为自研(非 libwebp),加入概率自适应与 Trellis 后,同 PSNR 下与
  libwebp method=6 的体积差约 35%(快 5-8 倍);逐像素确定性跨 GPU 架构不变。
- VP8 有损编码按 4:2:0 处理色彩,极高饱和度细线条的图片在 90 档可能出现
  轻微色彩柔化;把质量滑条调到 95 即可基本消除。
- 需要 NVIDIA GPU(CUDA 12.x);无 GPU 时自动全程 CPU(Pillow)。
- GPU PNG 解码为实验特性(默认关闭):正确性已穷举验证,吞吐调优中。
