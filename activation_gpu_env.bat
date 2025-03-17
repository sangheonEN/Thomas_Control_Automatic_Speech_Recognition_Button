@echo off
cd /d %~dp0

REM Check if the venv directory exists
if not exist realtimestt_gpu_env\Scripts\python.exe (
    echo Creating VENV
    python -m venv realtimestt_gpu_env
) else (
    echo VENV already exists
)

echo Activating VENV
start cmd /k "call realtimestt_gpu_env\Scripts\activate.bat && stt_python_lib_install_with_gpu_support.bat"