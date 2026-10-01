@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\codeguard.exe" (
  echo Run setup.bat first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Evaluating the frozen baseline and selected model. This closes study development.
codeguard evaluate
if errorlevel 1 exit /b 1
codeguard notebook
if errorlevel 1 exit /b 1
codeguard interview
echo Interview notebook: artifacts\cpu_v3\CodeGuardAI_Interview_Presentation.ipynb
exit /b %errorlevel%
