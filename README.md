# GPU压图 — 自研 GPU WebP 编码器 + 批量压缩工具

把带完整 png info 的 PNG 图片批量压缩为 WebP:
**质量、分辨率基本不变,体积压到约 10~15%,图片 info 原样保留在文件里(字节级校验)。**

## 核心特点

- **真正的 GPU 编码**:视频编码器级别的 VP8 关键帧编码器在本机 GPU(RTX 2080 / V100,
  经 CuPy/NVRTC)上完成宏块帧内预测模式搜索 —— 全流程中计算量最大的部分;
  量化系数与布尔算术编码用 Numba 编译为机器码在 CPU 上闭环执行。
- **质量与官方 libwebp 持平**:同质量档位 PSNR 相差 ±0.3dB 以内
  (90 档实测 36~44dB,视图片内容而定)。
- **info 完整保留**:PNG 的 `tEXt`(Title/Description/Software/Source/
  Generation_time/Comment 等)、`pHYs`、`eXIf`、`iCCP` 全部打包进 WebP 的
  XMP/EXIF/ICCP 块;每张输出后**逐字节回读比对**,不一致自动改用 CPU 引擎重压。
- **alpha 通道逐位精确**:自研最小 VP8L 无损编码器(LZ77 + Huffman),
  透明像素一个 bit 都不差;近全透明 alpha 通道从 126KB 压到 ~9KB。
- **逐张校验 + CPU 兜底**:每张输出都重新解码验证(尺寸/alpha/元数据/PSNR≥34dB),
  任何一项不过就自动用 Pillow(libwebp)按同样的元数据重压,保证结果永远可用。
- **速度**:RTX 2080 单卡约 **5 张/秒**(832×1216);6700 张全集约 20~25 分钟。

## 使用

### 图形界面(双击 `dist/GPU压图/GPU压图.exe`)

1. 源文件夹(打开后自动记住上次目录),输出默认 `源文件夹\webp`(原图不动)。
2. 拖动质量滑条(默认 90),选择引擎(GPU/CPU)、GPU 卡、线程数。
3. 点"开始压缩"。实时显示进度、速度、剩余时间、已节省百分比。
4. "完成后删除原PNG"可选,勾选后结束时还有二次确认。
5. "打开输出目录"可直接查看结果;日志区滚动显示每张图的处理情况。
6. **转换失败或无法识别的文件**会原样复制到输出目录下的 `未转换/`
   子文件夹(保留原始字节),保证输出目录永远是一份完整的集合。

> 分发到别的电脑:拷贝**整个 `GPU压图` 文件夹**(不能只拷 exe)。
> 要求 Windows 10/11 64 位;有 NVIDIA 显卡(驱动 2023 年以后)走 GPU,
> 没有则自动切换 CPU 引擎,功能完全一致。首次运行会现场编译 CUDA/机器码
> 内核(约 1 分钟)并缓存到文件夹内,之后启动即用;放在中文路径下也能正常工作。

### 命令行

```
GPU压图.exe --src D:\我的图片 --quality 90
可选项: --dst 输出目录   --cpu 纯CPU   --device 0 选卡
        --recursive 含子目录   --no-verify 关闭逐张校验   --workers 3
GPU压图.exe --diag    环境自检(检查GPU/依赖)
```

### 从 WebP 恢复原 PNG 元数据

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...] 原始字节
```

## 实测数据(V100,300 张 832×1216 真实图片,批量高速模式)

| 指标 | 数值 |
|---|---|
| 吞吐 | **~27-34 张/秒**(取决于后台负载;含逐张校验;旧单张模式 ~5.5 张/秒) |
| GPU 占用 | 均值 ~26-32%,内核执行期峰值 65-98%(瓶颈在 CPU 段:PNG 解码/校验) |
| 体积 | 压到原 PNG 的 8~18% |
| PSNR(q90) | 36~44 dB(与 Pillow/libwebp 同档) |
| alpha | 逐位一致 |
| 元数据 | 100% 字节级一致(回读比对) |
| 确定性 | 同一文件多次编码/不同批量大小,输出逐字节一致 |
| 失败率 | 0(奇数尺寸自动 CPU 兜底;显存不足自动拆批) |
| exe 体积 | 1.6GB(onedir 文件夹,含 CUDA/NVRTC 运行库,免装任何依赖) |

批量高速模式全并行:解码线程池(**imagecodecs** 的 libdeflate 后端,
单线程 2 倍、多线程 3.3 倍于 Pillow,与 Pillow 解码结果逐字节一致,
异常格式自动回退 Pillow)→ **融合模式搜索内核**(一条 CUDA
内核算完整个批次的 i16/uv/i4 全部模式决策,替代 ~50 次 Python 派发)→
**GPU 波前闭环量化内核**(每宏块行一个线程,行间标志位同步,与 CPU 版
逐位一致)→ 收尾线程池(Numba 无 GIL 熵编码)→ **子进程校验池**(把
每张输出重新解码比对 PSNR/alpha/元数据的工作放到独立进程,绕开
Pillow/numpy 持有 GIL 导致的线程扩展性上限)。16 整除尺寸走零拷贝
快速路径,任意其它分辨率自动回退补边路径。
实测说明:多进程方案能把 GPU 占用拉到 90%+ 但 Windows 多 CUDA 上下文
切换让总吞吐反而下降 10 倍,故未采用——本工具以总吞吐最高为准。

## 安装程序与源码保护

正式分发用 **`installer/GPU压图-安装程序.exe`**(约 620MB,Inno Setup 制作):
中文安装向导、可选桌面快捷方式、可选 PNG 右键菜单、自带卸载程序。
静默安装:`GPU压图-安装程序.exe /VERYSILENT /DIR=路径`。

**算法源码保护**(两层):
- 纯 Python 模块(容器/YUV/引擎/界面等)经 **Nuitka 编译为机器码**(.pyd,无字节码)
- 核心算法模块(闭环量化/熵编码/模式选择/VP8L/快速PNG解码)以**加密字节码**
  (zlib+XOR+base85)嵌入编译后的加载器,运行时解密到内存执行;CUDA 内核
  源码同样加密存储。发行包内没有任何 .py 源文件或可反编译的 .pyc。

构建流程:
```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
python tools/protect_build.py            # 生成加密包并 Nuitka 编译
cd build_pkg && python -m nuitka --module gpuwebp --include-package=gpuwebp
python -m PyInstaller --noconfirm GPU压图.spec
C:\InnoSetup\ISCC.exe installer.iss      # 生成安装程序
```

打包要点(都已在 spec / 代码里处理):排除无关大包;补齐 cupy_backends、
cuda.pathfinder、graphlib 等隐藏依赖;预建 bin 目录绕过 cupy 的
add_dll_directory 检查;代理 `python -m` 子进程;中文路径下把 CUDA 头文件
镜像到 ASCII 目录并注入 NVRTC -I。

## 目录结构

```
gpuwebp/            编码器源码
  vp8_encode.py     VP8 帧头/模式/系数布尔编码(Numba)
  closed_loop_jit.py 闭环重建+量化(与解码器逐位一致,Numba)
  gpu_engine.py     CuPy 批量内核:预测/模式搜索/变换
  alpha_enc.py      ALPH 通道 VP8L 最小编码器
  png_meta.py       PNG 元数据解析/XMP 打包/回读校验
  webp_container.py RIFF/VP8X 容器
  encoder.py        高层 API(GPU/CPU 引擎)
  pipeline.py       批量流水线+校验+兜底
  app.py            中文界面 + CLI
vendor/libwebp-1.5.0  参照源码(BSD,常量与语义对照)
tests/              各组件与 C 参照的逐字节验证工具
main.py             exe 入口
```

## 技术说明(诚实预期)

- 编码器为自研(非 libwebp),压缩效率与官方相差 ±10% 以内:同 PSNR 下文件
  略大或略小都有可能,上面 50 张实测比 Pillow 略小。
- VP8 有损编码按 4:2:0 处理色彩,极高饱和度细线条的图片在 90 档可能出现
  轻微色彩柔化;把质量滑条调到 95 即可基本消除。
- 需要 NVIDIA GPU(CUDA 12.x);无 GPU 时自动全程 CPU(Pillow)。
