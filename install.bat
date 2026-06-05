@echo off
setlocal enabledelayedexpansion
title SuperCool Environment Installer

echo ====================================================
echo Starting Automated Pipeline Setup
echo ====================================================

:: Step 1: Detect Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on this system path.
    echo Please install Python 3.10 or 3.11 and ensure 'Add to PATH' is checked.
    pause
    exit /b 1
)

:: Step 2: Establish Virtual Environment
if not exist "venv\" (
    echo [SYSTEM] Creating isolated deployment virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to construct local virtual environment layout.
        pause
        exit /b 1
    )
)

:: Step 3: Upgrade pip and dependencies inside environment context
echo [SYSTEM] Activating virtual architecture pipeline context...
call venv\Scripts\activate.bat

echo [SYSTEM] Upgrading base pipeline pip installer module...
python -m pip install --upgrade pip

echo [SYSTEM] Resolving requirements manifest dependencies...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Dependency collection validation failed.
    pause
    exit /b 1
)

:: Step 4: Run Asset Auto-Downloader
python download_checkpoints.py

:: Step 5: Start Software Layer Execution Environment
echo ====================================================
echo Launching Application Engine...
echo ====================================================
python ui_select.py

pause