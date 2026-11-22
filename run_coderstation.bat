@echo off
title Coderstation Voice Assistant & Web GUI
cd /d "%~dp0"

:: Optimize Ollama VRAM for RTX 4060 (8GB)
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0

if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    echo [SETUP] Installing dependencies...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\pip.exe install -r requirements.txt
)

echo [START] Restarting any existing Coderstation instance on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo [START] Launching Coderstation Web GUI (http://127.0.0.1:8000)...
.venv\Scripts\python.exe coderstation.py %*
pause
