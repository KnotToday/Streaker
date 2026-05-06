#!/bin/bash
# Build script for Linux: creates StreakerDetect and StreakerPlayer executables

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Building StreakerDetect..."
pyinstaller --onefile --windowed \
  --add-data "platform_utils.py:." \
  --add-data "streaker_config.json:." \
  --collect-all cv2 \
  --name StreakerDetect \
  StreakerDetect.py

echo ""
echo "Building StreakerPlayer..."
pyinstaller --onefile --windowed \
  --add-data "platform_utils.py:." \
  --collect-all cv2 \
  --name StreakerPlayer \
  StreakerPlayer.py

echo ""
echo "Build complete. Executables in dist/ folder."
