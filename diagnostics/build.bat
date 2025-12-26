@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

if defined HIPCC (
  set "HIPCC_BIN=%HIPCC%"
) else (
  set "HIPCC_BIN=hipcc"
)

echo Compiling HIP kernel...
"%HIPCC_BIN%" -O3 --shared -o clock_probe.dll clock_probe.cpp

if %ERRORLEVEL% EQU 0 (
  echo [SUCCESS] clock_probe.dll created.
) else (
  echo [FAIL] Compilation failed. Ensure HIPCC is on PATH or set HIPCC.
)

popd
endlocal
