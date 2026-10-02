@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set SRC=..\..\vendor\libwebp-1.5.0
set DSPSSE=%SRC%\src\dsp
set DEC=%SRC%\src\dec
set UTL=%SRC%\src\utils
cl /O2 /nologo /DDUMP_MB /I"%SRC%\src" /I"%SRC%" ^
  "%DEC%\alpha_dec.c" "%DEC%\buffer_dec.c" "%DEC%\frame_dec.c" ^
  "%DEC%\idec_dec.c" "%DEC%\io_dec.c" "%DEC%\quant_dec.c" ^
  "%DEC%\tree_dec.c" "%DEC%\vp8_dec.c" "%DEC%\vp8l_dec.c" ^
  "%DEC%\webp_dec.c" ^
  "%DSPSSE%\alpha_processing.c" "%DSPSSE%\cpu.c" "%DSPSSE%\dec.c" ^
  "%DSPSSE%\dec_clip_tables.c" "%DSPSSE%\filters.c" "%DSPSSE%\lossless.c" ^
  "%DSPSSE%\rescaler.c" "%DSPSSE%\upsampling.c" "%DSPSSE%\yuv.c" ^

  ^

  ^

  "%UTL%\bit_reader_utils.c" "%UTL%\color_cache_utils.c" ^
  "%UTL%\filters_utils.c" "%UTL%\huffman_utils.c" ^
  "%UTL%\quant_levels_dec_utils.c" "%UTL%\rescaler_utils.c" ^
  "%UTL%\random_utils.c" "%UTL%\thread_utils.c" ^
  "%UTL%\utils.c" "%UTL%\palette.c" ^
  dwebp_main.c /Fedwebp.exe /link /SUBSYSTEM:CONSOLE
