@echo off
setlocal EnableDelayedExpansion

echo ==================================================
echo      ROCm Diagnostic Tool - Clock Probe
echo ==================================================

REM 1. Check if HIPCC is already set
if not defined HIPCC (
    REM Try to find it in default 6.x/5.x locations
    if exist "C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe" (
        set "HIPCC=C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe"
    ) else if exist "C:\Program Files\AMD\ROCm\5.7\bin\hipcc.exe" (
        set "HIPCC=C:\Program Files\AMD\ROCm\5.7\bin\hipcc.exe"
    ) else (
        echo [ERROR] HIPCC path not found.
        echo Please edit this file or set HIPCC environment variable.
        pause
        exit /b 1
    )
)

echo [INFO] Using HIPCC: "!HIPCC!"

REM 2. Build the Kernel
echo [INFO] Building HIP Kernel...
call build.bat
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Build failed. Check your AMD HIP SDK installation.
    pause
    exit /b 1
)

REM 3. Run the Python Probe
echo [INFO] Running Clock Probe...
python clock_probe.py

echo.
echo ==================================================
echo [DONE] Diagnostic complete.
pause
