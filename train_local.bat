@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\codeguard.exe" (
  echo Run setup.bat first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Full official dataset. Feature extraction may take around an hour or more on this CPU.
echo Re-running this script resumes feature extraction and committed training epochs.
codeguard extract
if errorlevel 1 exit /b 1
for %%E in (linear mlp weighted) do (
  call :train %%E
  if errorlevel 1 exit /b 1
)
codeguard finalize
if errorlevel 1 exit /b 1
codeguard notebook
if errorlevel 1 exit /b 1
codeguard interview
exit /b %errorlevel%

:train
set "RUN="
if "%~1"=="linear" set "RUN=C0_LINEAR_CE"
if "%~1"=="mlp" set "RUN=C1_MLP_CE"
if "%~1"=="weighted" set "RUN=C2_MLP_WEIGHTED"
if exist "artifacts\cpu_v3\checkpoints\%RUN%\latest.pt" (
  codeguard train --experiment %~1 --resume
) else (
  codeguard train --experiment %~1
)
exit /b %errorlevel%
