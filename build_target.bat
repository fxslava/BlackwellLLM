@echo off
rem Builds one CMake target from the CLI: usage  build_target.bat <target>
rem MSVC env comes from vcvars64; nvcc additionally needs the CUDA headers
rem on INCLUDE because the CMake cache was generated from the VS IDE.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "INCLUDE=%INCLUDE%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include"
cmake --build out/build/x64-Debug --target %1
