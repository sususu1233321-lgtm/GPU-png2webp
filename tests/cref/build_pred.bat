@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /nologo pred_ref.c /Fe:pred_ref.exe 2>&1 | findstr /i error
