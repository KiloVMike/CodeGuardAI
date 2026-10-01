@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\codeguard.exe" (
  echo Run setup.bat first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
codeguard prepare
