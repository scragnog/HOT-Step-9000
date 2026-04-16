#!/usr/bin/env bash
# ============================================================
#  ACE-Step 🎵 Single-Click Launcher (macOS / Apple Silicon)
#  Opens loading screen, then starts all services.
#
#  ⚠️  EXPERIMENTAL — Community-tested, not officially supported.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IS_RESTART=0

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

start_system() {
    rm -f update.lock boot.lock 2>/dev/null

    export ACESTEP_NO_BROWSER=1

    # Logging
    LOG_TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    export ACESTEP_LOG_DIR="$SCRIPT_DIR/logs/$LOG_TIMESTAMP"
    mkdir -p "$ACESTEP_LOG_DIR/generations"
    echo "[Logging] Session logs: logs/$LOG_TIMESTAMP"
    echo ""

    # Read .env
    VITE_PORT=3000
    CURRENT_MODEL="acestep-v15-base"
    CURRENT_LM_MODEL="acestep-5Hz-lm-0.6B"
    CURRENT_LM_BACKEND="mlx"  # Default to MLX on macOS
    CURRENT_NO_INIT="false"
    CURRENT_QUANTIZATION="none"  # No quantization on macOS
    CURRENT_LM_GPU_LAYERS="-1"
    CURRENT_GGUF_QUANT="auto"
    CURRENT_REDMOND_MODE="false"
    CURRENT_REDMOND_SCALE="0.7"
    CURRENT_ADAPTER_MERGE_MODE="false"
    CURRENT_VAE_MODEL="stock"

    if [[ -f ".env" ]]; then
        _read_env() { grep "^$1=" .env 2>/dev/null | cut -d= -f2- || echo "$2"; }
        VITE_PORT=$(_read_env VITE_PORT "$VITE_PORT")
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

    # Check Redmond/ScragVAE availability
    REDMOND_AVAILABLE="false"
    [[ -f "checkpoints/redmond-refine/standard/adapter_config.json" ]] && REDMOND_AVAILABLE="true"
    SCRAGVAE_AVAILABLE="false"
    [[ -f "checkpoints/scragvae/diffusion_pytorch_model.safetensors" ]] && SCRAGVAE_AVAILABLE="true"

    # Scan models
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
    echo "  ACE-Step One-Click Launcher (macOS)"
    echo "  ⚠️  EXPERIMENTAL — MPS/MLX backend"
    echo "============================================="
    echo ""

    # Write config
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
        open "$SCRIPT_DIR/loading.html" 2>/dev/null || echo "  ⚠ Could not open browser. Open loading.html manually."
    fi
    echo "  Done."
    echo ""

    # Check UI deps
    echo "[2/5] Checking UI dependencies..."
    [[ ! -d "ace-step-ui/node_modules" ]] && (cd ace-step-ui && npm install)
    [[ ! -d "ace-step-ui/server/node_modules" ]] && (cd ace-step-ui/server && npm install)
    echo "  Done."
    echo ""

    echo "[2b/5] Building server..."
    (cd ace-step-ui/server && npx tsc 2>/dev/null) || echo "  [!] TypeScript build had errors."
    echo "  Done."
    echo ""

    echo "[2c/5] Clearing Python bytecode cache..."
    find acestep -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    echo "  Done."
    echo ""

    # Start UI servers
    echo "[3/5] Starting UI servers..."
    export ACESTEP_PATH="$SCRIPT_DIR"
    export PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python3"

    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm run dev) &>/dev/null &
    PID_EXPRESS=$!
    sleep 2

    (cd "$SCRIPT_DIR/ace-step-ui" && VITE_PORT="$VITE_PORT" npm run dev) &>/dev/null &
    PID_VITE=$!
    echo "  Started UI servers (PIDs: Express=$PID_EXPRESS, Vite=$PID_VITE)."
    echo ""

    # Wait for boot.lock
    echo "[4/5] Waiting for user selection in loading screen..."
    while true; do
        if [[ -f "update.lock" ]]; then
            rm -f update.lock
            echo ""
            echo "  [UI Action] Update requested. Pulling from GitHub..."
            git pull origin main
            kill "$PID_EXPRESS" 2>/dev/null || true
            kill "$PID_VITE" 2>/dev/null || true
            IS_RESTART=1
            start_system
            return
        fi
        if [[ -f "boot.lock" ]]; then
            rm -f boot.lock
            echo ""
            echo "  [UI Action] Boot sequence initiated..."
            break
        fi
        sleep 1
    done

    # Start Python API — no CUDA env vars on macOS
    echo "[5/5] Starting Python API server..."
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
    export HF_HOME="huggingface"
    export PYTHONUNBUFFERED=1
    # macOS-specific: no XFORMERS, no Triton, no CUDA index
    # MPS and MLX are auto-detected by the Python backend

    source "$SCRIPT_DIR/.venv/bin/activate"
    python3 "$SCRIPT_DIR/acestep/api_server.py" --port 8001 --host 127.0.0.1 &
    PID_PYTHON=$!
    echo "  Started Python API (PID: $PID_PYTHON)."
    echo ""

    echo "============================================="
    echo "  All services starting up!"
    echo "============================================="
    echo ""
    echo "  Python API:  http://localhost:8001"
    echo "  Backend:     http://localhost:3001"
    echo "  Frontend:    http://localhost:$VITE_PORT"
    echo ""
    echo "  Press Ctrl+C to stop all services."
    echo "============================================="
    echo ""

    wait
}

start_system
