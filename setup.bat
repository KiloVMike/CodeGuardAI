@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher was not found. Install Python 3.12 from python.org.
  exit /b 1
)

py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 is required. Install it from python.org and run this file again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the local Python environment...
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -e ".[dev]"
if errorlevel 1 exit /b 1

echo.
echo CodeGuard AI is installed locally.
codeguard doctor
echo.
echo Next: run prepare.bat to download the dataset and CodeBERT.
