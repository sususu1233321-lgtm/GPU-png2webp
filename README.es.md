# Compresor de imágenes por GPU — codificador WebP en GPU propio + herramienta por lotes

[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Русский](README.ru.md) | **[Español](README.es.md)**

Comprime grandes colecciones de imágenes a WebP: **la calidad y la
resolución apenas cambian, el tamaño queda en torno al 10–15% del original
y los metadatos se conservan byte a byte dentro de los archivos.**

## Características principales

- **Codificación real en GPU**: un codificador VP8 de fotogramas clave de
  nivel códec de vídeo ejecuta la búsqueda de modos de predicción
  intramacrobloque y el bucle cerrado de cuantización en forma de frente de
 onda en tu GPU NVIDIA (CUDA / CuPy / NVRTC) — la parte más costosa del
  proceso; la codificación entrópica se ejecuta en CPU como código máquina
  compilado con Numba.
- **Más de 40 formatos de entrada**: PNG / JPEG / WebP (recompresión) /
  TIFF / GIF / JP2 / JXL / AVIF / HEIC / QOI / DDS / BMP, etc.;
  despacho nativo de imagecodecs con respaldo completo de complementos de
  Pillow, de modo que incluso los formatos exóticos se convierten.
- **Calidad a la par de libwebp**: dentro de ±0.3 dB de PSNR con el mismo
  nivel de calidad (36–44 dB medidos en q90, según el contenido).
- **Metadatos intactos**: los `tEXt` / `pHYs` / `eXIf` / `iCCP` del PNG y
  los EXIF / XMP del JPEG se reempaquetan en los bloques XMP/EXIF/ICCP del
  WebP; cada salida se **relee y compara byte a byte**, y ante cualquier
  discrepancia se recomprime automáticamente con el motor de CPU.
- **Canal alfa exacto al bit**: un codificador VP8L sin pérdidas propio y
  minimalista (LZ77 + Huffman) — ni un solo bit de alfa cambia.
- **Compresión en mejora continua**: la adaptación de probabilidades por
  fotograma más la cuantización trellis de tasa-distorsión en GPU recortan
  otro 9.0% con píxeles idénticos (corpus de 6722 imágenes:
  1073→977 MB); los tamaños mixtos se fusionan en lotes con relleno.
- **Verificación por imagen + respaldo en CPU**: cada salida se vuelve a
  decodificar y comprobar (dimensiones / alfa / metadatos / PSNR ≥ 34 dB);
  cualquier fallo se recomprime automáticamente con Pillow (libwebp).
- **Control de recursos**: límite de núcleos de CPU desde la interfaz
  (`--cores N`, 0 = sin límite); un gobernador de memoria reduce la
  velocidad si la RAM escasea, evitando bloqueos del sistema.
- **Varias GPU**: elige la tarjeta en la interfaz o con `--device`; funciona
  desde GTX serie 10 hasta RTX serie 40 y Tesla V100, con salidas idénticas
  al bit entre tarjetas. Rendimiento: ~**150 imágenes/s** con verificación
  completa en una V100 (el corpus de 6722 imágenes tarda 45 s); 250+
  imágenes/s sin verificar.
- **Experimental: decodificación PNG en GPU** (`PNG_GPU=1`): inflate +
  desfiltrado de todas las variantes PNG (profundidades 1–16,
  gris/RGB/paleta/gris+A/RGBA, entrelazado Adam7, tRNS) íntegramente en
  GPU, con doble comprobación del adler32 de zlib y del CRC32 por bloque,
  idéntica byte a byte a libpng (verificada exhaustivamente en 1512
  combinaciones). Desactivada por defecto mientras se ajusta el rendimiento.

## Uso

### Interfaz gráfica (doble clic en `dist/GPU压图/GPU压图.exe`)

1. Elige la carpeta de origen (se recuerda entre sesiones); la salida por
   defecto es `origen\webp` y los originales no se tocan.
2. Mueve el deslizador de calidad (90 por defecto), elige motor
   (GPU/CPU), tarjeta GPU y el límite de núcleos de CPU.
3. Pulsa «开始压缩»: el progreso, la velocidad, el tiempo restante y el
   porcentaje ahorrado se actualizan en vivo.
4. El borrado de los PNG originales al terminar es opcional (con diálogo de
   confirmación).
5. Los archivos que no se puedan convertir se copian tal cual a la
   subcarpeta `未转换/`, de modo que la carpeta de salida siempre es un
   conjunto completo.

> Para desplegar en otro equipo, copia la **carpeta `GPU压图` completa** (no
> solo el exe). Windows 10/11 de 64 bits; con una GPU NVIDIA (controlador
> de 2023 en adelante) se usa el motor GPU, y si no, el motor de CPU
> idéntico.

### Línea de comandos

```
GPU压图.exe --src D:\MisImagenes --quality 90
Opciones: --dst CARPETA_SALIDA   --cpu solo CPU   --device 0 (índice nvidia-smi)
          --cores N límite de núcleos (0 = ilimitado)   --recursive con subcarpetas
          --no-verify desactivar la verificación por imagen
GPU压图.exe --diag    autodiagnóstico del entorno (GPU/dependencias)
```

### Restaurar los metadatos PNG originales desde un WebP

```python
from gpuwebp.png_meta import extract_from_webp, restore_png_text_chunks
meta = extract_from_webp(open("out.webp", "rb").read())
chunks = restore_png_text_chunks(meta)   # [(type, raw_payload), ...]
```

## Resultados medidos (V100, corpus real de 6722 imágenes)

| Métrica | Valor |
|---|---|
| Rendimiento | **~150 imágenes/s** (verificación completa, 16 núcleos); 250+ sin verificar |
| Tamaño | 8–18% del PNG original |
| PSNR (q90) | 36–44 dB (mismo nivel que Pillow/libwebp) |
| Alfa | idéntica al bit |
| Metadatos | 100% idénticos a nivel de bytes |
| Determinismo | salida idéntica byte a byte entre ejecuciones, tamaños de lote y GPU |
| Fallos | 0 (tamaños impares pasan a CPU; la falta de VRAM divide el lote) |
| Paquete | ~410 MB de carpeta / 141 MB de instalador, sin dependencias |

## Compilación desde el código fuente

```
pip install cupy-cuda12x numba numpy pillow imagecodecs pyinstaller nuitka
cpp\build.bat          # DLL del pipeline CUDA (CUDA 12.x + MSVC 2022)
cpp\build_entropy.bat  # DLL de codificación entrópica
python tools/protect_build.py   # opcional: compilación cifrada
python -m PyInstaller --noconfirm GPU压图.spec
```

## Estructura del repositorio

```
gpuwebp/            fuentes Python del codificador
cpp/                fuentes del pipeline CUDA/C++ (gpu_pipeline.cu,
                    kernels.cuh, gpu_inflate.cuh, entropy.cpp,
                    pngdec.cpp, build*.bat)
vendor/libwebp-1.5.0  fuentes de referencia (BSD)
tests/              herramientas de verificación byte a byte contra referencias C
tests/gpu_decode/   pruebas de exactitud y rendimiento del decodificado en GPU
main.py             punto de entrada del exe
```

## Notas honestas

- El codificador es propio (no es libwebp); tras la adaptación de
  probabilidades y el trellis, con el mismo PSNR los archivos son ~35% más
  grandes que libwebp method=6, pero 5–8 veces más rápidos. El determinismo
  exacto por píxel se mantiene entre arquitecturas de GPU.
- La codificación VP8 con pérdida usa 4:2:0; las líneas finas muy saturadas
  pueden mostrar un leve suavizado de color en q90 — súbelo a 95 para
  eliminarlo.
- Requiere una GPU NVIDIA (CUDA 12.x); sin ella todo se ejecuta en CPU
  (Pillow).
- La decodificación PNG en GPU es experimental (desactivada por defecto):
  la corrección está verificada exhaustivamente; el ajuste de rendimiento
  continúa.
