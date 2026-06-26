@echo off
chcp 65001 >nul
title ML Diagnostic Tool

echo ============================================
echo  Materials ML Diagnostic Tool
echo ============================================
echo.

:: Kill any leftover Streamlit from previous session
taskkill /F /IM streamlit.exe >nul 2>&1
ping -n 2 127.0.0.1 >nul

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.9+
    echo Download: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] Python detected

:: Switch to script directory
cd /d "%~dp0"

:: First-run setup
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ============ First-time setup ============
    echo.
    echo [1/2] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 ( echo [ERROR] Failed; pause; exit /b 1 )
    echo [OK] Virtual environment created
    echo.
    echo [2/2] Installing dependencies...
    echo.
    call .venv\Scripts\pip.exe install -r requirements.txt --no-index --find-links=packages
    if %errorlevel% neq 0 (
        echo [..] Trying network...
        call .venv\Scripts\pip.exe install -r requirements.txt
        if %errorlevel% neq 0 ( echo [ERROR] Failed; pause; exit /b 1 )
    )
    echo [OK] Dependencies installed
    echo ==========================================
)

echo.
echo [OK] Starting Streamlit server...
echo.

:: Start Streamlit in background
start /B "" ".venv\Scripts\streamlit.exe" run app.py --server.headless true > streamlit_log.txt 2>&1

:: Wait for server (check every 2 seconds, up to 30 seconds)
set WAIT_COUNT=0
:SERVER_CHECK
set /a WAIT_COUNT+=1
python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501')" >nul 2>&1 && goto SERVER_READY
echo  Starting... (%WAIT_COUNT%/15)
ping -n 2 127.0.0.1 >nul
if %WAIT_COUNT% lss 15 goto SERVER_CHECK

:SERVER_READY
echo.
echo [OK] Server is ready!
echo.
echo Opening browser...
start http://localhost:8501

echo.
echo ============================================
echo  If browser doesn't open, visit:
echo  http://localhost:8501
echo  Close this window to exit.
echo ============================================
echo.
pause >nul
