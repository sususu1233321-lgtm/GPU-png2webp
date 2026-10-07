@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /nologo iwht_ref.c /Fe:iwht_ref.exe
