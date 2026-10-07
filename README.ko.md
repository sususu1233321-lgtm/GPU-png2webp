# GPU 이미지 압축 — 자체 개발 GPU WebP 인코더 + 배치 도구

[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | **[한국어](README.ko.md)** | [Русский](README.ru.md) | [Español](README.es.md)

대량의 이미지를 WebP로 압축합니다: **품질과 해상도는 거의 그대로,
용량은 약 10~15%로 줄어들며, 메타데이터가 바이트 단위로 그대로
보존됩니다.**

## 주요 특징

- **진짜 GPU 인코딩**: 비디오 코덱 수준의 VP8 키프레임 인코더가 매크로
  블록 인트라 예측 모드 탐색과 웨이브프론트 폐루프 양자화를 NVIDIA
  GPU(CUDA / CuPy / NVRTC)에서 수행합니다 — 파이프라인에서 가장
  무거운 부분입니다. 엔트로피 부호화는 Numba로 기계어 컴파일되어
  CPU에서 실행됩니다.
- **40개 이상의 입력 형식**: PNG / JPEG / WebP(재압축) / TIFF / GIF /
  JP2 / JXL / AVIF / HEIC / QOI / DDS / BMP 등. imagecodecs 네이티브
  판별 + Pillow 전체 플러그인 폴백으로 특이한 파일도 변환됩니다.
- **libwebp와 동급 품질**: 동일 품질 설정에서 PSNR 오차 ±0.3dB 이내
  (q90 실측 36~44dB, 내용에 따라 다름).
- **메타데이터 완전 보존**: PNG의 `tEXt` / `pHYs` / `eXIf` / `iCCP`,
  JPEG의 EXIF / XMP를 WebP의 XMP/EXIF/ICCP 청크에 패키징합니다.
  모든 출력은 **바이트 단위로 다시 읽어 비교**하며, 불일치 시 CPU
  엔진으로 자동 재압축합니다.
- **알파 채널 비트 정확**: 자체 제작한 최소 VP8L 무손실 인코더
  (LZ77 + Huffman)로 한 비트도 틀리지 않습니다.
- **지속적인 압축률 개선**: 프레임별 계수 확률 적응 + GPU 트렐리스
  레이트-디스토션 양자화로 동일 화질에서 추가 9.0% 감소(6722장 실측
  1073→977MB). 혼합 크기는 패딩 통합 배치로 처리됩니다.
- **장별 검증 + CPU 폴백**: 모든 출력을 다시 디코딩해 검증(크기/알파/
  메타데이터/PSNR≥34dB)하며, 실패 시 Pillow(libwebp)로 자동 재압축합니다.
- **리소스 제어**: UI에서 **CPU 코어 상한** 설정 가능(`--cores N`,
  0=무제한). 메모리 가버너가 저메모리 시 자동 스로틀해 멈춤을 방지합니다.
- **멀티 GPU**: UI / `--device`로 카드 선택. GTX 10세대~RTX 40세대와
  Tesla V100을 지원하며 카드 간 출력이 비트 단위로 동일합니다.
  V100 기준 전체 검증 포함 **약 150장/초**(6722장 45초), 검증 없이
  250+장/초.
- **실험적: GPU PNG 디코딩**(`PNG_GPU=1`): 모든 PNG 변형(비트 심도
  1~16, 그레이/RGB/팔레트/그레이+A/RGBA, Adam7 인터레이스, tRNS)의
  inflate + 역필터를 GPU에서 처리합니다. zlib adler32 + 청크별 CRC32
  이중 검증으로 libpng와 바이트 단위로 일치(1512 조합 전수 검증).
  처리량 조율 중이라 기본값은 꺼져 있습니다.

## 사용법

### GUI(`dist/GPU压图/GPU压图.exe` 더블 클릭)

1. 소스 폴더 선택(이전 위치 기억). 출력은 기본적으로 `소스\webp`에
   저장되며 원본은 변경되지 않습니다.
2. 품질 슬라이더(기본 90), 엔진(GPU/CPU), GPU 카드, CPU 코어 상한 선택.
3. "开始压缩" 클릭. 진행률·속도·남은 시간·절감률이 실시간 표시됩니다.
4. 완료 후 원본 PNG 삭제는 선택 사항(확인 대화상자 포함).
5. 변환 실패한 파일은 `未转换/` 하위 폴더에 원본 그대로 복사되어,
   출력 폴더는 항상 완전한 집합이 됩니다.

> 다른 PC에 배포할 때는 `GPU压图` 폴더 전체를 복사하세요(exe만으로는
> 불가). Windows 10/11 64비트. NVIDIA GPU(2023년 이후 드라이버)가 있으면
> GPU 엔진, 없으면 동일 기능의 CPU 엔진으로 자동 전환됩니다.

### 명령줄

```
GPU压图.exe --src D:\MyPictures --quality 90
옵션: --dst 출력 디렉터리   --cpu CPU 전용   --device 0 (nvidia-smi 번호)
     --cores N CPU 코어 상한(0=무제한)   --recursive 하위 폴더 포함
     --no-verify 장별 검증 비활성화
GPU压图.exe --diag    환경 자가 진단(GPU/의존성)
```

### WebP에서 원본 PNG 메타데이터 복원

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...]
```

## 실측 데이터(V100, 6722장 실제 코퍼스)

| 지표 | 값 |
|---|---|
| 처리량 | **약 150장/초**(전체 검증, 16코어) / 검증 없이 250+장/초 |
| 용량 | 원본 PNG의 8~18% |
| PSNR(q90) | 36~44 dB(Pillow/libwebp와 동급) |
| 알파 | 비트 단위 일치 |
| 메타데이터 | 100% 바이트 단위 일치 |
| 결정성 | 실행 횟수·배치 크기·GPU와 무관하게 바이트 단위 일치 |
| 실패 | 0(홀수 크기는 CPU 폴백, OOM 시 자동 분할) |
| 용량 | ~410MB 폴더 / 141MB 설치 프로그램(의존성 없음) |

## 소스에서 빌드

```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
cpp\build.bat          # CUDA 파이프라인 DLL(CUDA 12.x + MSVC 2022)
cpp\build_entropy.bat  # 엔트로피 DLL
python tools/protect_build.py   # 선택: 암호화 릴리스 빌드
python -m PyInstaller --noconfirm GPU压图.spec
```

## 디렉터리 구조

```
gpuwebp/            인코더 Python 소스
cpp/                CUDA/C++ 파이프라인(gpu_pipeline.cu, kernels.cuh,
                    gpu_inflate.cuh, entropy.cpp, pngdec.cpp, build*.bat)
vendor/libwebp-1.5.0  참조용 소스(BSD)
tests/              C 참조와의 바이트 검증 도구
tests/gpu_decode/   GPU 디코딩/인코딩 검증 및 벤치마크
main.py             exe 진입점
```

## 솔직한 주의사항

- 인코더는 자체 개발입니다(libwebp 아님). 확률 적응과 트렐리스 도입 후
  동일 PSNR에서 libwebp method=6 대비 약 35% 크지만 5~8배 빠릅니다.
  픽셀 완전 결정성은 GPU 아키텍처와 무관하게 유지됩니다.
- VP8 손실 부호화는 4:2:0을 사용합니다. 고채도 세선은 q90에서 약간의
  색 번짐이 있을 수 있으며 품질 95에서 거의 사라집니다.
- NVIDIA GPU(CUDA 12.x)가 필요하며, 없으면 자동으로 전체가
  CPU(Pillow)에서 실행됩니다.
- GPU PNG 디코딩은 실험적(기본 꺼짐): 정확성은 전수 검증되었으며
  처리량 조율 중입니다.
