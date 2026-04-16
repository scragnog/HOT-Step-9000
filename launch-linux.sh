#!/usr/bin/env bash
# ============================================================
#  ACE-Step 🎵 Single-Click Launcher (Linux / NVIDIA CUDA)
#  Opens loading screen, then starts all services.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_RESTART=0

# Cleanup function for graceful shutdown
cleanup() {
    echo ""
    echo "  Shutting down services..."
    [[ -n "${PID_EXPRESS:-}" ]] && kill "$PID_EXPRESS" 2>/dev/null && echo "  Stopped Express backend."
    [[ -n "${PID_VITE:-}" ]] && kill "$PID_VITE" 2>/dev/null && echo "  Stopped Vite frontend."
    [[ -n "${PID_PYTHON:-}" ]] && kill "$PID_PYTHON" 2>/dev/null && echo "  Stopped Python API."
    rm -f boot.lock update.lock 2>/dev/null
    echo "  All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Open a URL in the default browser (cross-distro)
open_browser() {
    local url="$1"
    if command -v xdg-open &>/dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif command -v sensible-browser &>/dev/null; then
        sensible-browser "$url" 2>/dev/null &
    elif command -v firefox &>/dev/null; then
        firefox "$url" 2>/dev/null &
    elif command -v google-chrome &>/dev/null; then
        google-chrome "$url" 2>/dev/null &
    else
        echo "  ⚠ Could not detect a browser. Please open manually:"
        echo "    $url"
    fi
}

start_system() {
    rm -f update.lock boot.lock 2>/dev/null

    # Tell ace-step-ui not to open a browser — our loading page handles that
    export ACESTEP_NO_BROWSER=1

    # Generate timestamp for logging session
    LOG_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    export ACESTEP_LOG_DIR="$SCRIPT_DIR/logs/$LOG_TIMESTAMP"
    mkdir -p "$ACESTEP_LOG_DIR/generations"
    echo "[Logging] Session logs: logs/$LOG_TIMESTAMP"
    echo ""

    # Read frontend port from .env
    VITE_PORT=3000
    if [[ -f ".env" ]]; then
        VITE_PORT=$(grep '^VITE_PORT' .env 2>/dev/null | cut -d= -f2 || echo "3000")
        VITE_PORT=${VITE_PORT:-3000}
    fi

    # Read current model selections from .env
    CURRENT_MODEL="acestep-v15-base"
    CURRENT_LM_MODEL="acestep-5Hz-lm-0.6B"
    CURRENT_LM_BACKEND="vllm"
    CURRENT_NO_INIT="false"
    CURRENT_QUANTIZATION="auto"
    CURRENT_LM_GPU_LAYERS="-1"
    CURRENT_GGUF_QUANT="auto"
    CURRENT_REDMOND_MODE="false"
    CURRENT_REDMOND_SCALE="0.7"
    CURRENT_ADAPTER_MERGE_MODE="false"
    CURRENT_VAE_MODEL="stock"

    if [[ -f ".env" ]]; then
        _read_env() { grep "^$1=" .env 2>/dev/null | cut -d= -f2- || echo "$2"; }
        CURRENT_MODEL=$(_read_env ACESTEP_CONFIG_PATH "$CURRENT_MODEL")
        CURRENT_LM_MODEL=$(_read_env ACESTEP_LM_MODEL_PATH "$CURRENT_LM_MODEL")
        CURRENT_LM_BACKEND=$(_read_env ACESTEP_LM_BACKEND "$CURRENT_LM_BACKEND")
        CURRENT_NO_INIT=$(_read_env ACESTEP_NO_INIT "$CURRENT_NO_INIT")
        CURRENT_QUANTIZATION=$(_read_env ACESTEP_QUANTIZATION "$CURRENT_QUANTIZATION")
        CURRENT_LM_GPU_LAYERS=$(_read_env ACESTEP_LM_GPU_LAYERS "$CURRENT_LM_GPU_LAYERS")
        CURRENT_GGUF_QUANT=$(_read_env ACESTEP_GGUF_QUANT "$CURRENT_GGUF_QUANT")
        CURRENT_REDMOND_MODE=$(_read_env ACESTEP_REDMOND_MODE "$CURRENT_REDMOND_MODE")
        CURRENT_REDMOND_SCALE=$(_read_env ACESTEP_REDMOND_SCALE "$CURRENT_REDMOND_SCALE")
        CURRENT_ADAPTER_MERGE_MODE=$(_read_env ACESTEP_ADAPTER_MERGE_MODE "$CURRENT_ADAPTER_MERGE_MODE")
        CURRENT_VAE_MODEL=$(_read_env ACESTEP_VAE_MODEL "$CURRENT_VAE_MODEL")
    fi

    # Check if Redmond adapter is available on disk
    REDMOND_AVAILABLE="false"
    [[ -f "checkpoints/redmond-refine/standard/adapter_config.json" ]] && REDMOND_AVAILABLE="true"

    # Check if ScragVAE is available
    SCRAGVAE_AVAILABLE="false"
    [[ -f "checkpoints/scragvae/diffusion_pytorch_model.safetensors" ]] && SCRAGVAE_AVAILABLE="true"

    # Scan checkpoints/ for available models
    MODEL_LIST=""
    for d in checkpoints/acestep-v15-*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        [[ -n "$MODEL_LIST" ]] && MODEL_LIST="$MODEL_LIST,'$name'" || MODEL_LIST="'$name'"
    done
    [[ -z "$MODEL_LIST" ]] && MODEL_LIST="'acestep-v15-base'"

    LM_MODEL_LIST=""
    for d in checkpoints/acestep-5Hz-lm-*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        [[ -n "$LM_MODEL_LIST" ]] && LM_MODEL_LIST="$LM_MODEL_LIST,'$name'" || LM_MODEL_LIST="'$name'"
    done
    [[ -z "$LM_MODEL_LIST" ]] && LM_MODEL_LIST="'acestep-5Hz-lm-0.6B'"

    echo "============================================="
    echo "  ACE-Step One-Click Launcher (Linux)"
    echo "============================================="
    echo ""

    # ---- Step 1: Write config and open loading page ----
    echo "[1/5] Writing config..."
    cat > "$SCRIPT_DIR/loading-config.js" << CONFIGEOF
var VITE_PORT = '${VITE_PORT}';
var AVAILABLE_MODELS = [${MODEL_LIST}];
var AVAILABLE_LM_MODELS = [${LM_MODEL_LIST}];
var CURRENT_MODEL = '${CURRENT_MODEL}';
var CURRENT_LM_MODEL = '${CURRENT_LM_MODEL}';
var CURRENT_LM_BACKEND = '${CURRENT_LM_BACKEND}';
var CURRENT_REDMOND_MODE = '${CURRENT_REDMOND_MODE}';
var CURRENT_REDMOND_SCALE = '${CURRENT_REDMOND_SCALE}';
var REDMOND_AVAILABLE = '${REDMOND_AVAILABLE}';
var CURRENT_NO_INIT = '${CURRENT_NO_INIT}';
var CURRENT_QUANTIZATION = '${CURRENT_QUANTIZATION}';
var CURRENT_LM_GPU_LAYERS = '${CURRENT_LM_GPU_LAYERS}';
var CURRENT_GGUF_QUANT = '${CURRENT_GGUF_QUANT}';
var CURRENT_ADAPTER_MERGE_MODE = '${CURRENT_ADAPTER_MERGE_MODE}';
var CURRENT_VAE_MODEL = '${CURRENT_VAE_MODEL}';
var SCRAGVAE_AVAILABLE = '${SCRAGVAE_AVAILABLE}';
CONFIGEOF

    if [[ "$IS_RESTART" -eq 0 ]]; then
        echo "[1.5] Opening loading screen..."
        open_browser "file://$SCRIPT_DIR/loading.html"
    fi
    echo "  Done."
    echo ""

    # ---- Step 2: Check UI dependencies ----
    echo "[2/5] Checking UI dependencies..."
    [[ ! -d "ace-step-ui/node_modules" ]] && (cd ace-step-ui && npm install)
    [[ ! -d "ace-step-ui/server/node_modules" ]] && (cd ace-step-ui/server && npm install)
    echo "  Done."
    echo ""

    # ---- Step 2b: Rebuild server TypeScript ----
    echo "[2b/5] Building server..."
    (cd ace-step-ui/server && npx tsc 2>/dev/null) || echo "  [!] TypeScript build had errors."
    echo "  Done."
    echo ""

    # ---- Step 2c: Clear Python bytecode cache ----
    echo "[2c/5] Clearing Python bytecode cache..."
    find acestep -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    echo "  Done."
    echo ""

    # ---- Step 3: Start UI servers ----
    echo "[3/5] Starting UI servers..."
    export ACESTEP_PATH="$SCRIPT_DIR"
    export PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python3"

    # Start Express backend
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm run dev) &>/dev/null &
    PID_EXPRESS=$!
    sleep 2

    # Start Vite frontend
    (cd "$SCRIPT_DIR/ace-step-ui" && VITE_PORT="$VITE_PORT" npm run dev) &>/dev/null &
    PID_VITE=$!
    echo "  Started UI servers (PIDs: Express=$PID_EXPRESS, Vite=$PID_VITE)."
    echo ""

    # ---- Step 4: Wait for user selection in loading screen ----
    echo "[4/5] Waiting for user selection in loading screen..."
    while true; do
        if [[ -f "update.lock" ]]; then
            rm -f update.lock
            echo ""
            echo "============================================="
            echo "  [UI Action] Update requested."
            echo "  Updating from GitHub..."
            echo "============================================="
            git pull origin main
            echo ""
            echo "Update complete. Restarting UI services..."
            # Kill existing UI services
            kill "$PID_EXPRESS" 2>/dev/null || true
            kill "$PID_VITE" 2>/dev/null || true
            IS_RESTART=1
            start_system
            return
        fi
        if [[ -f "boot.lock" ]]; then
            rm -f boot.lock
            echo ""
            echo "============================================="
            echo "  [UI Action] Boot sequence initiated..."
            echo "============================================="
            break
        fi
        sleep 1
    done

    # ---- Step 5: Start Python API server ----
    echo "[5/5] Starting Python API server..."
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
    export HF_HOME="huggingface"
    export XFORMERS_FORCE_DISABLE_TRITON=1
    export PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG=1
    export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"
    export PYTHONUNBUFFERED=1

    source "$SCRIPT_DIR/.venv/bin/activate"
    python3 "$SCRIPT_DIR/acestep/api_server.py" --port 8001 --host 127.0.0.1 &
    PID_PYTHON=$!
    echo "  Started Python API (PID: $PID_PYTHON)."
    echo ""

    echo "============================================="
    echo "  All services starting up!"
    echo "============================================="
    echo ""
    echo "  The loading screen is open in your browser."
    echo "  It will auto-redirect to ACE-Step once"
    echo "  all services are ready."
    echo ""
    echo "  Python API:  http://localhost:8001"
    echo "  Backend:     http://localhost:3001"
    echo "  Frontend:    http://localhost:$VITE_PORT"
    echo ""
    echo "  Press Ctrl+C to stop all services."
    echo "============================================="
    echo ""

    # Wait for any child process to exit
    wait
}

start_system
