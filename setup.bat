@echo off
title DMESR Emotion-AI Setup

echo ========================================================
echo DMESR Emotion-AI Project Setup
echo ========================================================
echo.

:: 1. Backend Setup
echo [1/2] Setting up Python virtual environment for Backend...
cd /d "%~dp0backend"
if not exist "venv" (
    echo Creating virtual environment in backend\venv...
    python -m venv venv
) else (
    echo Virtual environment already exists in backend\venv.
)

echo Activating virtual environment and installing backend requirements...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies.
    pause
    exit /b %errorlevel%
)
echo Backend setup complete.
echo.

:: 2. Frontend Setup
echo [2/2] Installing Node.js packages for Frontend...
cd /d "%~dp0frontend"
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node dependencies.
    pause
    exit /b %errorlevel%
)
echo Frontend setup complete.
echo.

cd /d "%~dp0"
echo ========================================================
echo Setup finished successfully!
echo You can now run start.bat to launch the application.
echo ========================================================
pause
