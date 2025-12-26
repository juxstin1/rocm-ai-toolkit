@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%pytorch-rocm-venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
if defined AIKIT_PYTHON set "PYTHON_EXE=%AIKIT_PYTHON%"
if defined AIKIT_GPU_INDEX set "HIP_VISIBLE_DEVICES=%AIKIT_GPU_INDEX%"

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help

if "%1"=="img" goto imagine
if "%1"=="image" goto imagine
if "%1"=="gen" goto imagine
if "%1"=="draw" goto imagine

if "%1"=="txt" goto transcribe
if "%1"=="text" goto transcribe
if "%1"=="hear" goto transcribe

if "%1"=="find" goto search
if "%1"=="search" goto search

if "%1"=="index" goto index

if "%1"=="seg" goto segment
if "%1"=="cut" goto segment

if "%1"=="bg" goto rmbg
if "%1"=="nobg" goto rmbg

echo Unknown command: %1
goto help

:imagine
shift
"%PYTHON_EXE%" -m aikit imagine %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:transcribe
shift
"%PYTHON_EXE%" -m aikit transcribe %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:search
shift
"%PYTHON_EXE%" -m aikit embed search %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:index
shift
"%PYTHON_EXE%" -m aikit embed index %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:segment
shift
"%PYTHON_EXE%" -m aikit segment %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:rmbg
shift
"%PYTHON_EXE%" -m aikit rmbg %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end

:help
echo.
echo   ai - Your local AI toolkit
echo.
echo   GENERATE IMAGES
echo     ai img "a cyberpunk city"
echo     ai img "warrior" --seed 42
echo     ai img "cat" --lora style.safetensors
echo.
echo   TRANSCRIBE AUDIO/VIDEO
echo     ai txt meeting.mp3
echo     ai txt video.mp4 -o transcript.txt
echo     ai txt podcast.wav --timestamps
echo.
echo   SEARCH YOUR FILES
echo     ai index ./docs              (first, index a folder)
echo     ai find "how does auth work"
echo.
echo   SEGMENT IMAGES
echo     ai seg photo.jpg
echo     ai seg photo.jpg --point 100 200
echo.
echo   REMOVE BACKGROUNDS
echo     ai bg photo.jpg
echo     ai bg folder/ --batch
echo.

:end
endlocal
