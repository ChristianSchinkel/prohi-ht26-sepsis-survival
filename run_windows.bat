@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The project virtual environment was not found.
    echo Create it at .venv and install the project dependencies first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "src\main.py"
pause
