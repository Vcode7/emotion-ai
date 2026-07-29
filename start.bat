@echo off
title DMESR Emotion-AI Launcher

echo ========================================================
echo Starting DMESR Backend and Frontend...
echo ========================================================

:: Start Backend in a new window (activating venv if present)
if exist "%~dp0backend\venv\Scripts\activate.bat" (
    start "DMESR Backend (FastAPI)" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
) else (
    start "DMESR Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
)

:: Start Frontend in a new window
start "DMESR Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are starting up in separate terminal windows:
echo   - Backend:  http://localhost:8000
echo   - Frontend: http://localhost:5173
echo ========================================================
