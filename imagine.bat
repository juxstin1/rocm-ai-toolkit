@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%pytorch-rocm-venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if defined AIKIT_PYTHON set "PYTHON_EXE=%AIKIT_PYTHON%"
if defined AIKIT_GPU_INDEX set "HIP_VISIBLE_DEVICES=%AIKIT_GPU_INDEX%"
"%PYTHON_EXE%" -m aikit imagine %*
endlocal
