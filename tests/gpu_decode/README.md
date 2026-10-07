# GPU 解码/编码位准与性能门测试

从仓库根目录运行(脚本内的 DLL/语料路径均相对根目录):

- `pngproto.py` — PNG 块扫描/解码参考实现(IDAT 提取, 其他脚本的公共依赖)
- `mkidat.py N` — 从 `D:/gpuimgtest3` 生成分层 IDAT 测试素材到 `_infltest/`
- `s3test.py` — 176 样本 A/B 基准
- `s4test.py [MODE] [DST] [N]` — 全量/子集流水线跑 + 输出摘要对比
- `s5png.py [N]` — GPU PNG 解码位准门(全语料 RGBA==libpng, 编码输入==zc 路径)
- `s6var.py` — 穷举变体位准门(1512 组合: 位深×颜色类型×隔行×tRNS×滤波×尺寸×zlib)
- `s7pad.py` — padded 混合尺寸批位准门

配套独立内核基准: `cpp/test_inflate.cu`, `cpp/test_defilter.cu`
(build_test.bat / build_deftest.bat)。
