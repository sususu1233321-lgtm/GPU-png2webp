@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /nologo fdct_ref.c /Fe:fdct_ref.exe
