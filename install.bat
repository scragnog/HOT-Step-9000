@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title ACE-Step Installer

:: ====================================================================
::  ACE-Step 1.5 ? First-Time Installer (Windows)
::  Sets up: UV, Python venv, dependencies, models, UI
:: ====================================================================

echo.
echo  ============================================================
echo    ACE-Step 1.5 ? Installer
echo  ============================================================
echo.

:: -------------------------------------------------------------------
:: 0. Ensure we're in the right directory
:: -------------------------------------------------------------------
cd /d "%~dp0"
echo  [*] Working directory: %CD%
echo.

:: -------------------------------------------------------------------
:: 1. Check Python is available
:: -------------------------------------------------------------------
echo  [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed or not on PATH.
    echo  Please install Python 3.11 or 3.12 from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Found: %%v
echo.

:: -------------------------------------------------------------------
:: 2. Install UV (Python package manager)
:: -------------------------------------------------------------------
echo  [2/6] Checking UV package manager...
where uv >nul 2>&1
if errorlevel 1 (
    echo  UV not found. Installing...
    pip install uv
    if errorlevel 1 (
        echo.
        echo  ERROR: Failed to install UV. Try manually: pip install uv
        pause
        exit /b 1
    )
    echo  UV installed successfully.
) else (
    for /f "tokens=*" %%v in ('uv --version 2^>^&1') do echo  Found: %%v
)
echo.

:: -------------------------------------------------------------------
:: 3. Create virtual environment
:: -------------------------------------------------------------------
echo  [3/6] Setting up Python virtual environment...
if exist ".venv\Scripts\activate.bat" (
    echo  Virtual environment already exists.
) else (
    echo  Creating .venv...
    uv venv .venv
    if errorlevel 1 (
        echo.
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created.
)

:: Activate the venv for the rest of the script
call .venv\Scripts\activate.bat
echo  Activated: .venv
echo.

:: -------------------------------------------------------------------
:: 4. Install Python dependencies
:: -------------------------------------------------------------------
echo  [4/6] Installing Python dependencies...
echo  This may take several minutes on first run.
echo.
uv pip install -e .
if errorlevel 1 (
    echo.
    echo  WARNING: uv pip install -e . failed, trying with requirements.txt...
    uv pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  ERROR: Failed to install Python dependencies.
        pause
        exit /b 1
    )
)
echo.
echo  Python dependencies installed.
echo.

:: -------------------------------------------------------------------
:: 5. Download AI Models
:: -------------------------------------------------------------------
echo  [5/6] AI Model Download
echo.
echo  ============================================================
echo    Model Download Options
echo  ============================================================
echo.
echo    1) Default ? Download main model
echo       (turbo DiT + 1.7B LM ? recommended for most users)
echo.
echo    2) All ? Download all available models
echo       (all DiT variants + all LM sizes ? large download)
echo.
echo    3) List ? Show all available models, then choose
echo.
echo    4) Skip ? Don't download models now
echo       (you can run:  python -m acestep.model_downloader  later)
echo.
echo  ============================================================
echo.

set /p MODEL_CHOICE="  Your choice [1/2/3/4] (default=1): "
if "%MODEL_CHOICE%"=="" set MODEL_CHOICE=1

if "%MODEL_CHOICE%"=="1" (
    echo.
    echo  Downloading main model...
    python -m acestep.model_downloader
    if errorlevel 1 (
        echo.
        echo  WARNING: Model download had errors. You can retry later with:
        echo    python -m acestep.model_downloader
    )
) else if "%MODEL_CHOICE%"=="2" (
    echo.
    echo  Downloading all models...
    python -m acestep.model_downloader --all
    if errorlevel 1 (
        echo.
        echo  WARNING: Some downloads had errors. You can retry later with:
        echo    python -m acestep.model_downloader --all
    )
) else if "%MODEL_CHOICE%"=="3" (
    echo.
    python -m acestep.model_downloader --list
    echo.
    echo  ============================================================
    echo   Enter a model name from the list above, or press Enter
    echo   to download the default main model.
    echo  ============================================================
    echo.
    set /p SPECIFIC_MODEL="  Model name (or Enter for default): "
    if "!SPECIFIC_MODEL!"=="" (
        echo.
        echo  Downloading main model...
        python -m acestep.model_downloader
    ) else (
        echo.
        echo  Downloading !SPECIFIC_MODEL!...
        python -m acestep.model_downloader --model !SPECIFIC_MODEL!
    )
    if errorlevel 1 (
        echo.
        echo  WARNING: Download had errors. You can retry later with:
        echo    python -m acestep.model_downloader --model [name]
    )
) else if "%MODEL_CHOICE%"=="4" (
    echo.
    echo  Skipping model download. You can download later with:
    echo    python -m acestep.model_downloader
) else (
    echo.
    echo  Invalid choice. Skipping model download.
    echo  You can download later with: python -m acestep.model_downloader
)
echo.

:: -------------------------------------------------------------------
:: 5b. Optional: Download Vocoder Enhancement Model (HiFi-GAN)
:: -------------------------------------------------------------------
echo  ============================================================
echo    Optional: Vocoder Enhancement Model (HiFi-GAN)
echo  ============================================================
echo.
echo    The ADaMoSHiFiGAN vocoder model improves audio quality by
echo    re-decoding generated audio through a dedicated HiFi-GAN.
echo    Size: ~206 MB
echo.

if exist "checkpoints\music_vocoder\diffusion_pytorch_model.safetensors" (
    echo  Vocoder model already installed. Skipping.
) else (
    set /p VOCODER_CHOICE="  Download vocoder model? [Y/n] (default=Y): "
    if "!VOCODER_CHOICE!"=="" set VOCODER_CHOICE=Y
    if /i "!VOCODER_CHOICE!"=="Y" (
        echo.
        echo  Downloading vocoder model from HuggingFace...
        if not exist "checkpoints\music_vocoder" mkdir "checkpoints\music_vocoder"
        curl -L -o "checkpoints\music_vocoder\config.json" "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/config.json"
        if errorlevel 1 (
            echo  WARNING: Failed to download config.json
        )
        curl -L -o "checkpoints\music_vocoder\diffusion_pytorch_model.safetensors" "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors"
        if errorlevel 1 (
            echo  WARNING: Failed to download vocoder model weights
        ) else (
            echo  Vocoder model downloaded successfully.
        )
    ) else (
        echo  Skipping vocoder download. You can manually download later from:
        echo    https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/tree/main/music_vocoder
        echo  Place files in: checkpoints\music_vocoder\
    )
)
echo.

:: -------------------------------------------------------------------
:: 6. Install UI dependencies (Node.js / npm)
:: -------------------------------------------------------------------
echo  [6/6] Setting up React UI...

:: Check if Node.js is available
where node >nul 2>&1
if errorlevel 1 (
    echo.
    echo  WARNING: Node.js is not installed or not on PATH.
    echo  The React UI requires Node.js 18+.
    echo  Download from: https://nodejs.org/
    echo  After installing Node.js, re-run this installer or run:
    echo    cd ace-step-ui ^&^& npm install ^&^& cd server ^&^& npm install ^&^& npm run build
    echo.
    goto :done
)
for /f "tokens=*" %%v in ('node --version 2^>^&1') do echo  Found Node.js: %%v

echo  Installing UI frontend dependencies...
cd /d "%~dp0\ace-step-ui"
call npm install
if errorlevel 1 (
    echo  WARNING: npm install failed for frontend.
)

echo  Installing UI backend dependencies...
cd /d "%~dp0\ace-step-ui\server"
call npm install
if errorlevel 1 (
    echo  WARNING: npm install failed for backend.
)

echo  Building UI backend...
call npm run build
if errorlevel 1 (
    echo  WARNING: npm run build failed for backend.
)

cd /d "%~dp0"
echo.
echo  UI setup complete.

:: -------------------------------------------------------------------
:: Done!
:: -------------------------------------------------------------------
:done
echo.
echo  ============================================================
echo    Installation Complete!
echo  ============================================================
echo.
echo    To start ACE-Step, run:   LAUNCH.bat
echo.
echo    To download more models:  python -m acestep.model_downloader --list
echo.
echo  ============================================================
echo.
pause
