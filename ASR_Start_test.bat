@echo off
cd /d %~dp0

REM Set the environment name as a variable
set VENV_DIR=realtimestt_gpu_env
set PYTHON_DIR=tests

REM Check if the venv directory exists
if not exist %VENV_DIR%\Scripts\python.exe (
    echo Creating VENV
    python -m venv %VENV_DIR%
) else (
    echo VENV already exists
)

echo Activating VENV
call %VENV_DIR%\Scripts\activate.bat && python %PYTHON_DIR%\Thomas_audio_control_src.py

pause