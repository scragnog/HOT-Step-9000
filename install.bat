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
:: 5c. Optional: Download SDXL Turbo (AI Cover Art)
:: -------------------------------------------------------------------
echo  ============================================================
echo    Optional: AI Cover Art Model (SDXL Turbo)
echo  ============================================================
echo.
echo    SDXL Turbo generates relevant album cover art after each
echo    song generation. You can enable this in Settings.
echo    Size: ~3.5 GB
echo.

set /p COVERART_CHOICE="  Download AI cover art model? [Y/n] (default=n): "
if "%COVERART_CHOICE%"=="" set COVERART_CHOICE=n
if /i "!COVERART_CHOICE!"=="Y" (
    echo.
    echo  Downloading SDXL Turbo from HuggingFace...
    python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sdxl-turbo', variant='fp16')"
    if errorlevel 1 (
        echo  WARNING: Failed to download SDXL Turbo model.
        echo  It will be downloaded automatically on first use.
    ) else (
        echo  SDXL Turbo model downloaded successfully.
    )
) else (
    echo  Skipping SDXL Turbo download. It will auto-download on first use
    echo  if you enable AI Cover Art in Settings.
)
echo.

:: -------------------------------------------------------------------
:: 5d. Optional: Download XL (4B DiT) Models
:: -------------------------------------------------------------------
echo.
echo  ============================================================
echo    Optional: XL (4B DiT) Models
echo  ============================================================
echo.
echo    The XL models are larger, higher-quality 4B-parameter DiT
echo    variants of ACE-Step 1.5. They produce richer audio but
echo    require more VRAM (~12 GB minimum).
echo    Size: ~10 GB each
echo.
echo    1) XL Turbo only  (fastest XL variant, 8 steps)
echo    2) All three XL models  (base + sft + turbo)
echo    3) Skip
echo.

set /p XL_CHOICE="  Your choice [1/2/3] (default=3): "
if "%XL_CHOICE%"=="" set XL_CHOICE=3

if "%XL_CHOICE%"=="1" (
    echo.
    echo  Downloading XL Turbo model...
    python -m acestep.model_downloader --model acestep-v15-xl-turbo --skip-main
    if errorlevel 1 (
        echo  WARNING: XL Turbo download had errors. You can retry later with:
        echo    python -m acestep.model_downloader --model acestep-v15-xl-turbo --skip-main
    ) else (
        echo  XL Turbo model downloaded successfully.
    )
) else if "%XL_CHOICE%"=="2" (
    echo.
    echo  Downloading all XL models...
    echo.
    echo  [1/3] XL Base...
    python -m acestep.model_downloader --model acestep-v15-xl-base --skip-main
    if errorlevel 1 (
        echo  WARNING: XL Base download had errors.
    )
    echo.
    echo  [2/3] XL SFT...
    python -m acestep.model_downloader --model acestep-v15-xl-sft --skip-main
    if errorlevel 1 (
        echo  WARNING: XL SFT download had errors.
    )
    echo.
    echo  [3/3] XL Turbo...
    python -m acestep.model_downloader --model acestep-v15-xl-turbo --skip-main
    if errorlevel 1 (
        echo  WARNING: XL Turbo download had errors.
    )
    echo.
    echo  XL model downloads complete.
) else (
    echo  Skipping XL models. They will auto-download on first use if selected,
    echo  or you can download later with:
    echo    python -m acestep.model_downloader --model acestep-v15-xl-turbo --skip-main
)
echo.

:: -------------------------------------------------------------------
:: 5e. Optional: Download Redmond Mode (DPO Quality Refinement)
:: -------------------------------------------------------------------
echo.
echo  ============================================================
echo    Optional: Redmond Mode (DPO Quality Refinement)
echo  ============================================================
echo.
echo    Merges a quality-improvement adapter into the DiT at startup.
echo    Improves musicality, arrangement coherence, and vocals.
echo    Size: ~750 MB
echo.

set /p REDMOND_CHOICE="  Download Redmond Refine adapter? [Y/n] (default=n): "
if "%REDMOND_CHOICE%"=="" set REDMOND_CHOICE=n
if /i "!REDMOND_CHOICE!"=="Y" (
    echo.
    echo  Downloading AceStep_Refine_Redmond from HuggingFace...
    if not exist "checkpoints\redmond-refine" mkdir "checkpoints\redmond-refine"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('artificialguybr/AceStep_Refine_Redmond', allow_patterns='standard/*', local_dir='checkpoints/redmond-refine')"
    if errorlevel 1 (
        echo  WARNING: Failed to download Redmond adapter.
        echo  You can download manually from: https://huggingface.co/artificialguybr/AceStep_Refine_Redmond
    ) else (
        echo  Redmond Refine adapter downloaded successfully.
        echo  Enable it in the loading screen or Settings.
    )
) else (
    echo  Skipping Redmond download. You can download later by re-running install.bat
    echo  or from: https://huggingface.co/artificialguybr/AceStep_Refine_Redmond
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
