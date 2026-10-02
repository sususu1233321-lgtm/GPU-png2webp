@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /O2 /nologo "%~dp0bool_writer_ref.c" /Fe:"%~dp0bool_writer_ref.exe"
cl /O2 /nologo "%~dp0bool_reader_ref.c" /Fe:"%~dp0bool_reader_ref.exe"
