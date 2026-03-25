@echo off
REM ============================================================
REM  ACE-Step ??? Single-Click Launcher
REM  Opens loading screen, then starts all services directly.
REM  No PowerShell scripts needed.
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

set "IS_RESTART=0"

:start_system
if exist update.lock del update.lock
if exist boot.lock del boot.lock


REM Tell ace-step-ui start.bat not to open a browser - our loading page handles that
set "ACESTEP_NO_BROWSER=1"

REM Generate timestamp for logging session (PowerShell — wmic is deprecated on Win 11)
for /f "usebackq" %%I in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'"`) do set "LOG_TIMESTAMP=%%I"
set "ACESTEP_LOG_DIR=%~dp0logs\%LOG_TIMESTAMP%"
if not exist "%ACESTEP_LOG_DIR%" mkdir "%ACESTEP_LOG_DIR%"
if not exist "%ACESTEP_LOG_DIR%\generations" mkdir "%ACESTEP_LOG_DIR%\generations"
echo [Logging] Session logs will be written to: logs\%LOG_TIMESTAMP%
echo.
REM Read frontend port from .env
set "VITE_PORT=3000"
if exist ".env" (
    for /f "tokens=2 delims==" %%a in ('findstr /b "VITE_PORT" ".env"') do set "VITE_PORT=%%a"
)

REM Read current model selections from .env
set "CURRENT_MODEL=acestep-v15-base"
set "CURRENT_LM_MODEL=acestep-5Hz-lm-0.6B"
set "CURRENT_LM_BACKEND=vllm"
set "CURRENT_NO_INIT=false"
if exist ".env" (
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_CONFIG_PATH=" ".env"') do set "CURRENT_MODEL=%%b"
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_LM_MODEL_PATH=" ".env"') do set "CURRENT_LM_MODEL=%%b"
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_LM_BACKEND=" ".env"') do set "CURRENT_LM_BACKEND=%%b"
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_NO_INIT=" ".env"') do set "CURRENT_NO_INIT=%%b"
)

REM Read Redmond Mode settings from .env
set "CURRENT_REDMOND_MODE=false"
set "CURRENT_REDMOND_SCALE=0.7"
if exist ".env" (
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_REDMOND_MODE=" ".env"') do set "CURRENT_REDMOND_MODE=%%b"
    for /f "tokens=1,* delims==" %%a in ('findstr /b "ACESTEP_REDMOND_SCALE=" ".env"') do set "CURRENT_REDMOND_SCALE=%%b"
)

REM Check if Redmond adapter is available on disk
set "REDMOND_AVAILABLE=false"
if exist "checkpoints\redmond-refine\standard\adapter_config.json" set "REDMOND_AVAILABLE=true"

REM Scan checkpoints/ for available ACE-Step models (acestep-v15-*)
set "MODEL_LIST="
for /d %%d in (checkpoints\acestep-v15-*) do (
    set "DIRNAME=%%~nxd"
    if defined MODEL_LIST (
        set "MODEL_LIST=!MODEL_LIST!,'!DIRNAME!'"
    ) else (
        set "MODEL_LIST='!DIRNAME!'"
    )
)
if not defined MODEL_LIST set "MODEL_LIST='acestep-v15-base'"

REM Scan checkpoints/ for available LM models (acestep-5Hz-lm-*)
set "LM_MODEL_LIST="
for /d %%d in (checkpoints\acestep-5Hz-lm-*) do (
    set "DIRNAME=%%~nxd"
    if defined LM_MODEL_LIST (
        set "LM_MODEL_LIST=!LM_MODEL_LIST!,'!DIRNAME!'"
    ) else (
        set "LM_MODEL_LIST='!DIRNAME!'"
    )
)
if not defined LM_MODEL_LIST set "LM_MODEL_LIST='acestep-5Hz-lm-0.6B'"

echo =============================================
echo   ACE-Step One-Click Launcher
echo =============================================
echo.

REM ---- Step 1: Write config and open loading page ----
echo [1/5] Writing config...
(
echo var VITE_PORT = '%VITE_PORT%';
echo var AVAILABLE_MODELS = [%MODEL_LIST%];
echo var AVAILABLE_LM_MODELS = [%LM_MODEL_LIST%];
echo var CURRENT_MODEL = '%CURRENT_MODEL%';
echo var CURRENT_LM_MODEL = '%CURRENT_LM_MODEL%';
echo var CURRENT_LM_BACKEND = '%CURRENT_LM_BACKEND%';
echo var CURRENT_REDMOND_MODE = '%CURRENT_REDMOND_MODE%';
echo var CURRENT_REDMOND_SCALE = '%CURRENT_REDMOND_SCALE%';
echo var REDMOND_AVAILABLE = '%REDMOND_AVAILABLE%';
echo var CURRENT_NO_INIT = '%CURRENT_NO_INIT%';
) > "%~dp0loading-config.js"

if "%IS_RESTART%"=="0" (
    echo [1.5] Opening loading screen...
    start "" "%~dp0loading.html"
)
echo   Done.
echo.

REM ---- Step 2: Check UI dependencies ----
echo [2/5] Checking UI dependencies...
if not exist "ace-step-ui\node_modules" (
    echo   Installing frontend dependencies...
    cd ace-step-ui
    call npm install
    cd ..
)
if not exist "ace-step-ui\server\node_modules" (
    echo   Installing server dependencies...
    cd ace-step-ui\server
    call npm install
    cd ..\..
)
echo   Done.
echo.

REM ---- Step 2b: Rebuild server TypeScript ----
echo [2b/5] Building server...
cd ace-step-ui\server
call npx tsc
if errorlevel 1 (
    echo   [!] TypeScript build had errors ??? server will use source via tsx.
)
cd ..\..\
echo   Done.
echo.

REM ---- Step 2c: Clear Python bytecode cache for fresh code ----
echo [2c/5] Clearing Python bytecode cache...
for /d /r "acestep" %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d"
)
echo   Done.
echo.

REM ---- Step 2d: Apply Triton DLL patch (Windows PyTorch 2.9.x fix) ----
echo [2d/5] Checking Triton compatibility patch...
.venv\Scripts\python.exe scripts\patch_torch_triton.py --quiet
echo   Done.
echo.

REM ---- Step 3: Start UI servers (Express backend + Vite frontend) ----
echo [3/5] Starting UI servers...

REM Set ACE-Step paths for the Express backend
set "ACESTEP_PATH=%~dp0"
set "PYTHON_PATH=%~dp0.venv\Scripts\python.exe"

REM Start Express backend
start /min "ACE-Step UI Backend" cmd /k "cd /d "%~dp0ace-step-ui\server" && npm run dev"
timeout /t 3 /nobreak >nul

REM Start Vite frontend
start /min "HOT-Step 9000 UI Frontend" cmd /k "set VITE_PORT=%VITE_PORT%&& cd /d "%~dp0ace-step-ui" && npm run dev"
echo   Started (minimized windows).
echo.

REM ---- Step 4: Wait for UI selection ----
echo [4/5] Waiting for user selection in loading screen...
:wait_for_user
if exist "update.lock" (
    del "update.lock"
    echo.
    echo =============================================
    echo   [UI Action] Update requested.
    echo   Updating from GitHub...
    echo =============================================
    git pull origin main
    echo.
    echo Update complete. Restarting UI services...
    set "IS_RESTART=1"
    goto start_system
)
if exist "boot.lock" (
    del "boot.lock"
    echo.
    echo =============================================
    echo   [UI Action] Boot sequence initiated...
    echo =============================================
    goto start_python
)
timeout /t 1 /nobreak >nul
goto wait_for_user

:start_python
REM ---- Step 5: Set environment and start Python API server ----
echo [5/5] Starting Python API server...
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "HF_HOME=huggingface"
set "XFORMERS_FORCE_DISABLE_TRITON=1"
set "PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG=1"
set "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130"
set "UV_CACHE_DIR=%LOCALAPPDATA%\uv\cache"
set "UV_NO_BUILD_ISOLATION=1"
set "UV_NO_CACHE=0"
set "UV_LINK_MODE=symlink"
set "UV_INDEX_STRATEGY=unsafe-best-match"

REM Pre-install build deps (required when UV_NO_BUILD_ISOLATION=1)
.venv\Scripts\python.exe -m pip install hatchling editables >nul 2>&1

start /min "ACE-Step Python API" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && set "PYTHONPATH=%~dp0" && set "HF_HOME=huggingface" && set "XFORMERS_FORCE_DISABLE_TRITON=1" && set "UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130" && set "UV_CACHE_DIR=%LOCALAPPDATA%\uv\cache" && set "UV_NO_BUILD_ISOLATION=1" && set "UV_LINK_MODE=symlink" && set "UV_INDEX_STRATEGY=unsafe-best-match" && set "PYTHONUNBUFFERED=1" && python acestep/api_server.py --port 8001 --host 127.0.0.1"
echo   Started (minimized window).
echo.


REM ---- Step 5: Done ----
echo =============================================
echo   All services starting up!
echo =============================================
echo.
echo   The loading screen is open in your browser.
echo   It will auto-redirect to ACE-Step once
echo   all services are ready.
echo.
echo   Python API:  http://localhost:8001
echo   Backend:     http://localhost:3001
echo   Frontend:    http://localhost:%VITE_PORT%
echo.
echo   Three minimized windows are running:
echo     - Python API server
echo     - UI Backend (Express)
echo     - UI Frontend (Vite)
echo.
echo   Close those windows to stop the services.
echo =============================================
echo.
echo   This window will close automatically.
echo   (Services will keep running in the background)
timeout /t 5 /nobreak >nul
