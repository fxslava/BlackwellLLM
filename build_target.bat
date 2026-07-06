@echo off
rem Transitional wrapper over CMakePresets.json: usage  build_target.bat <target>
rem (vcvars64 puts cl.exe on PATH; nvcc additionally needs the CUDA headers on INCLUDE)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "INCLUDE=%INCLUDE%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"
cmake --build --preset x64-debug --target %1
