@echo off
REM Build script for Windows: creates StreakerDetect.exe and StreakerPlayer.exe

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Building StreakerDetect.exe...
pyinstaller --onefile --windowed ^
  --add-data "platform_utils.py:." ^
  --add-data "streaker_config.json:." ^
  --collect-all cv2 ^
  --icon=streaker.ico ^
  --name StreakerDetect ^
  StreakerDetect.py

echo.
echo Building StreakerPlayer.exe...
pyinstaller --onefile --windowed ^
  --add-data "platform_utils.py:." ^
  --collect-all cv2 ^
  --icon=streaker.ico ^
  --name StreakerPlayer ^
  StreakerPlayer.py

echo.
echo Build complete. Executables in dist/ folder.
pause
