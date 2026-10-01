@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\codeguard.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if not exist "artifacts\cpu_v3\final_config.json" (
  echo No finalized model exists yet.
  echo Run train_local.bat first. For a smoke UI use: codeguard demo --quick --open
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Starting CodeGuard AI demo - model loads first, then the browser opens.
echo Close this window to stop the demo.
echo.
codeguard demo --open
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo The demo exited with code %RC%.
  pause
)
exit /b %RC%
