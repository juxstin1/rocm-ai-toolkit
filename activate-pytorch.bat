@echo off
title PyTorch ROCm Environment
echo.
echo  ========================================
echo   PyTorch ROCm Environment
echo  ========================================
echo.
set "SCRIPT_DIR=%~dp0"
set "VENV_ACTIVATE=%SCRIPT_DIR%pytorch-rocm-venv\Scripts\activate.bat"

if defined AIKIT_GPU_INDEX set "HIP_VISIBLE_DEVICES=%AIKIT_GPU_INDEX%"

REM Activate the virtual environment
if exist "%VENV_ACTIVATE%" (
  call "%VENV_ACTIVATE%"
) else (
  echo  Virtual environment not found: %VENV_ACTIVATE%
  echo  Create it with: python -m venv pytorch-rocm-venv
)

echo  Virtual environment activated.
if defined HIP_VISIBLE_DEVICES (
  echo  HIP_VISIBLE_DEVICES=%HIP_VISIBLE_DEVICES%
)
echo.
echo  Quick test: python -c "import torch; print(torch.cuda.get_device_name(0))"
echo.
