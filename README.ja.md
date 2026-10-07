# GPU 画像圧縮 — 自作 GPU WebP エンコーダー + バッチツール

[简体中文](README.md) | [English](README.en.md) | **[日本語](README.ja.md)** | [한국어](README.ko.md) | [Русский](README.ru.md) | [Español](README.es.md)

大量の画像を WebP に圧縮します:**画質・解像度はほぼ変わらず、容量は
約 10〜15% に、メタデータはバイト単位でそのまま保存されます。**

## 主な特徴

- **本物の GPU エンコード**:ビデオコーデック級の VP8 キーフレーム
  エンコーダが、マクロブロック予測モード探索とウェーブフロント
  クローズドループ量子化を NVIDIA GPU(CUDA / CuPy / NVRTC)で実行
  ——パイプライン中最も重い処理です。エントロピー符号化は Numba で
  機械語化され CPU で実行されます。
- **40 以上の入力フォーマット**:PNG / JPEG / WebP(再圧縮)/ TIFF /
  GIF / JP2 / JXL / AVIF / HEIC / QOI / DDS / BMP など。imagecodecs の
  ネイティブ判定 + Pillow フルプラグインフォールバックで、特殊な
  ファイルも変換できます。
- **libwebp と同等の画質**:同品質設定で PSNR 誤差 ±0.3dB 以内
  (q90 で実測 36〜44dB、内容により変動)。
- **メタデータ完全保存**:PNG の `tEXt` / `pHYs` / `eXIf` / `iCCP`、
  JPEG の EXIF / XMP を WebP の XMP/EXIF/ICCP チャンクへ格納。
  出力は毎回**バイト単位で読み戻し比較**し、不一致なら CPU エンジンで
  自動再圧縮します。
- **アルファチャンネルはビット完全**:自作の最小 VP8L 可逆エンコーダ
  (LZ77 + Huffman)により 1 ビットも狂いません。
- **圧縮率の継続改善**:フレーム単位の係数確率適応 + GPU トレリス
  レート歪み量子化により、同じ画質のままさらに 9.0% 削減(6722 枚の
  実測で 1073→977MB)。混合サイズはパディング統合バッチ処理。
- **1 枚ずつ検証 + CPU フォールバック**:全出力を再デコードして検証
  (サイズ/アルファ/メタデータ/PSNR≥34dB)。不合格なら Pillow(libwebp)
  で自動再圧縮します。
- **リソース制御**:UI で **CPU コア数上限**を設定可能(`--cores N`、
  0=無制限)。メモリガバナーが低メモリ時に自動スロットルしフリーズを防ぎます。
- **マルチ GPU**:UI / `--device` でカード選択。GTX 10 系〜RTX 40 系と
  Tesla V100 に対応し、カードをまたいで出力はビット単位で一致。
  V100 でフル検証あり **約 150 枚/秒**(6722 枚を 45 秒)、検証なしで
  250+ 枚/秒。
- **実験的:GPU PNG デコード**(`PNG_GPU=1`):inflate + 逆フィルタの
  全バリアント(ビット深度 1〜16、グレー/RGB/パレット/グレー+A/RGBA、
  Adam7 インターレース、tRNS)を GPU で処理。zlib adler32 + チャンク毎
  CRC32 の二重検証で libwebp とバイト単位で一致(1512 組合せの網羅検証
  済み)。スループット調整中のため既定はオフです。

## 使い方

### GUI(`dist/GPU压图/GPU压图.exe` をダブルクリック)

1. ソースフォルダを選択(前回の位置を記憶)。出力は既定で
   `ソース\webp` に保存され、元画像は変更されません。
2. 品質スライダー(既定 90)、エンジン(GPU/CPU)、GPU カード、
   CPU コア上限を選択。
3. 「開始圧縮」をクリック。進捗・速度・残り時間・削減率をリアルタイム表示。
4. 完了後の元 PNG 削除はオプション(確認ダイアログ付き)。
5. 変換失敗したファイルはそのまま `未转换/` サブフォルダへコピーされ、
   出力フォルダは常に完全なセットになります。

> 他の PC へ配布する場合は `GPU压图` フォルダ全体をコピーしてください
> (exe 単体では不可)。Windows 10/11 64bit。NVIDIA GPU(2023 年以降の
> ドライバ)があれば GPU エンジン、なければ同一機能の CPU エンジンで
> 動作します。

### コマンドライン

```
GPU压图.exe --src D:\MyPictures --quality 90
オプション: --dst 出力先   --cpu CPUのみ   --device 0 (nvidia-smiの番号)
           --cores N CPUコア上限(0=無制限)   --recursive サブフォルダ含む
           --no-verify 逐次検証を無効化
GPU压图.exe --diag    環境診断(GPU/依存関係)
```

### WebP から元の PNG メタデータを復元

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...]
```

## 実測データ(V100、6722 枚の実コーパス)

| 指標 | 値 |
|---|---|
| スループット | **約 150 枚/秒**(フル検証、16 コア)/ 検証なし 250+ 枚/秒 |
| 容量 | 元 PNG の 8〜18% |
| PSNR(q90) | 36〜44 dB(Pillow/libwebp と同水準) |
| アルファ | ビット完全一致 |
| メタデータ | 100% バイト単位一致 |
| 決定性 | 実行回数・バッチサイズ・GPU を問わずバイト単位で一致 |
| 失敗 | 0(奇数サイズは CPU へフォールバック、OOM は自動分割) |
| 容量 | ~410MB フォルダ / 141MB インストーラ(依存関係なし) |

## ソースからのビルド

```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
cpp\build.bat          # CUDA パイプライン DLL(CUDA 12.x + MSVC 2022)
cpp\build_entropy.bat  # エントロピー DLL
python tools/protect_build.py   # 任意: 暗号化リリースビルド
python -m PyInstaller --noconfirm GPU压图.spec
```

## ディレクトリ構成

```
gpuwebp/            エンコーダ Python ソース
cpp/                CUDA/C++ パイプライン(gpu_pipeline.cu, kernels.cuh,
                    gpu_inflate.cuh, entropy.cpp, pngdec.cpp, build*.bat)
vendor/libwebp-1.5.0  参照用ソース(BSD)
tests/              C 参照とのバイト検証ツール
tests/gpu_decode/   GPU デコード/エンコードの検証とベンチマーク
main.py             exe エントリポイント
```

## 正直な注意点

- エンコーダは自作です(libwebp ではない)。確率適応とトレリス導入後、
  同 PSNR での容量差は libwebp method=6 比で約 35%(5〜8 倍高速)。
  ピクセル完全な決定性は GPU アーキテクチャを跨いで保たれます。
- VP8 非可逆符号化は 4:2:0。高彩度の細い線は q90 でわずかな色にじみが
  出る可能性があります。品質 95 でほぼ解消します。
- NVIDIA GPU(CUDA 12.x)が必要。なければ自動で全処理を CPU(Pillow)で実行。
- GPU PNG デコードは実験的(既定オフ):正確性は網羅検証済み、
  スループット調整中です。
