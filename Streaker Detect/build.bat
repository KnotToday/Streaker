@echo off
REM Build script for StreakerDetect.exe — works on any Windows machine.
REM FFmpeg is auto-downloaded into bin\ if not already present.

setlocal

set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
set FFMPEG_ZIP=%TEMP%\ffmpeg_build.zip
set FFMPEG_BIN=bin\ffmpeg.exe

REM --- Download ffmpeg if missing ---
if not exist "%FFMPEG_BIN%" (
    echo [1/4] Downloading ffmpeg...
    if not exist bin mkdir bin
    powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing"
    if errorlevel 1 ( echo ERROR: Download failed. & exit /b 1 )

    echo [2/4] Extracting ffmpeg...
    powershell -Command "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath '%TEMP%\ffmpeg_extract' -Force"
    powershell -Command "Get-ChildItem '%TEMP%\ffmpeg_extract' -Recurse -Filter ffmpeg.exe  | Copy-Item -Destination 'bin\ffmpeg.exe'"
    powershell -Command "Get-ChildItem '%TEMP%\ffmpeg_extract' -Recurse -Filter ffprobe.exe | Copy-Item -Destination 'bin\ffprobe.exe'"
    del "%FFMPEG_ZIP%"
    powershell -Command "Remove-Item '%TEMP%\ffmpeg_extract' -Recurse -Force"
    echo     ffmpeg ready.
) else (
    echo [1/4] ffmpeg already present in bin\, skipping download.
)

REM --- Install Python dependencies ---
echo [2/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: pip install failed. & exit /b 1 )

REM --- Build StreakerDetect.exe ---
echo [3/4] Building StreakerDetect.exe...
pyinstaller StreakerDetect.spec --noconfirm
if errorlevel 1 ( echo ERROR: PyInstaller failed. & exit /b 1 )

REM --- Build StreakerPlayer.exe ---
echo [4/4] Building StreakerPlayer.exe...
pyinstaller --onefile --windowed ^
  --add-data "platform_utils.py:." ^
  --add-binary "bin\ffmpeg.exe:." ^
  --add-binary "bin\ffprobe.exe:." ^
  --collect-all cv2 ^
  --name StreakerPlayer ^
  StreakerPlayer.py
if errorlevel 1 ( echo ERROR: StreakerPlayer build failed. & exit /b 1 )

echo.
echo Build complete. Executables are in the dist\ folder.
pause
