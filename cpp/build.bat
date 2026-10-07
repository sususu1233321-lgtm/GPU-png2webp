@echo off
setlocal
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;%PATH%"
set "INCLUDE=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared"
set "LIB=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe"

set CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
set "GEN=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89"
if not "%GPUARCH%"=="" set "GEN=-gencode arch=compute_%GPUARCH%,code=sm_%GPUARCH%"

"%NVCC%" -O2 --pre-include _prelude.h -shared -D_ALLOW_COMPILER_AND_STL_VERSION_CHECK %GEN% -allow-unsupported-compiler -Xcompiler "/LD /MD /Zc:preprocessor" -o gpu_pipeline.dll gpu_pipeline.cu

if %ERRORLEVEL% == 0 (
    echo BUILD OK
) else (
    echo BUILD FAILED
)
