@echo off
title TremorSense — Server Launcher
color 0A

echo ╔══════════════════════════════════════════════════╗
echo ║       TremorSense AI — Starting Servers          ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: ── Paths ─────────────────────────────────────────────
set "PROJECT_DIR=%~dp0"
set "BACKEND_DIR=%PROJECT_DIR%ai_dashboard\backend"
set "FRONTEND_DIR=%PROJECT_DIR%ai_dashboard"

:: ── 1. Kill any leftover processes on ports 5173 & 8000
echo [1/3] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5173" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)
echo       Done.
echo.

:: ── 2. Start FastAPI Backend (uvicorn) ────────────────
echo [2/3] Starting FastAPI backend (port 8000)...
start "TremorSense Backend" cmd /k "cd /d %BACKEND_DIR% && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 2 /nobreak >nul

:: ── 3. Start Vite Frontend (npm run dev) ──────────────
echo [3/3] Starting Vite frontend (port 5173)...
start "TremorSense Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm run dev"
timeout /t 2 /nobreak >nul

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║  All servers started!                            ║
echo ║                                                  ║
echo ║  Frontend:  http://localhost:5173                 ║
echo ║  Backend:   http://localhost:8000                 ║
echo ║  Health:    http://localhost:8000/health           ║
echo ║                                                  ║
echo ║  Close the terminal windows to stop servers.     ║
echo ╚══════════════════════════════════════════════════╝
echo.
pause
