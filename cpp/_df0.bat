@echo off
setlocal
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"
set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
"%NVCC%" %OPT% -gencode arch=compute_70,code=sm_70 -allow-unsupported-compiler -o test_defilter.exe test_defilter.cu
if %ERRORLEVEL% == 0 (echo BUILD OK) else (echo BUILD FAILED)
