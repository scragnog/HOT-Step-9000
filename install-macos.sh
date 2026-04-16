#!/usr/bin/env bash
# ====================================================================
#  ACE-Step 1.5 — First-Time Installer (macOS / Apple Silicon)
#  Sets up: UV, Python venv, dependencies, models, UI
#
#  ⚠️  EXPERIMENTAL — Community-tested, not officially supported.
#  Requires: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.11+
# ====================================================================
set -euo pipefail

echo ""
echo "  ============================================================"
echo "    ACE-Step 1.5 — macOS Apple Silicon Installer"
echo "    ⚠️  EXPERIMENTAL — MPS/MLX backend"
echo "  ============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "  [*] Working directory: $(pwd)"
echo ""

# Check we're actually on macOS ARM64
if [[ "$(uname)" != "Darwin" ]]; then
    echo "  ERROR: This installer is for macOS only."
    echo "  For Linux, use: ./install-linux.sh"
    exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "  ERROR: This installer requires Apple Silicon (M1/M2/M3/M4)."
    echo "  Intel Macs are not supported (no MPS backend)."
    exit 1
fi
echo "  ✓ macOS Apple Silicon detected."
echo ""

# -------------------------------------------------------------------
# 1. Check Python is available
# -------------------------------------------------------------------
echo "  [1/6] Checking Python..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo ""
    echo "  ERROR: Python 3 is not installed."
    echo "  Install via Homebrew:"
    echo "    brew install python@3.12"
    echo "  Or download from: https://python.org"
    echo ""
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  Found: $PY_VERSION"

PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")

if [[ "$PY_MAJOR" != "3" ]]; then
    echo "  ERROR: Python 3 is required. Detected: $PY_VERSION"
    exit 1
fi
if (( PY_MINOR < 11 )); then
    echo "  ERROR: Python 3.11 or 3.12 is required. Detected: $PY_VERSION"
    echo "  Install via: brew install python@3.12"
    exit 1
fi
if (( PY_MINOR >= 13 )); then
    echo "  WARNING: $PY_VERSION detected. Tested with 3.11/3.12. Proceed with caution."
    sleep 3
fi
echo ""

# -------------------------------------------------------------------
# 2. Install UV (Python package manager)
# -------------------------------------------------------------------
echo "  [2/6] Checking UV package manager..."
if ! command -v uv &>/dev/null; then
    echo "  UV not found. Installing..."
    pip3 install uv || { echo "  ERROR: Failed to install UV."; exit 1; }
    echo "  UV installed successfully."
else
    echo "  Found: $(uv --version 2>&1)"
fi
echo ""

# -------------------------------------------------------------------
# 3. Create virtual environment
# -------------------------------------------------------------------
echo "  [3/6] Setting up Python virtual environment..."
if [[ -f ".venv/bin/activate" ]]; then
    echo "  Virtual environment already exists."
else
    echo "  Creating .venv..."
    uv venv .venv || { echo "  ERROR: Failed to create virtual environment."; exit 1; }
    echo "  Virtual environment created."
fi

source .venv/bin/activate
echo "  Activated: .venv"
echo ""

# -------------------------------------------------------------------
# 4. Install Python dependencies
# -------------------------------------------------------------------
echo "  [4/6] Installing Python dependencies..."
echo "  This may take several minutes on first run."
echo ""
echo "  NOTE: macOS uses default PyPI torch (with MPS support)."
echo "  MLX dependencies will be installed automatically."
echo ""

# Do NOT set UV_EXTRA_INDEX_URL — macOS torch comes from default PyPI
export UV_INDEX_STRATEGY="unsafe-best-match"

uv pip install -e . || {
    echo ""
    echo "  WARNING: uv pip install -e . failed, trying with requirements.txt..."
    uv pip install -r requirements.txt || {
        echo "  ERROR: Failed to install Python dependencies."
        exit 1
    }
}

# Verify MPS is available
echo "  Verifying MPS (Apple Silicon GPU) support..."
if python3 -c "import torch; exit(0 if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "  ✓ MPS backend available."
else
    echo "  ⚠ MPS backend not detected. Generation will fall back to CPU (much slower)."
fi

# Verify MLX is available
echo "  Verifying MLX (Apple native acceleration)..."
if python3 -c "import mlx; print(f'MLX {mlx.__version__}')" 2>/dev/null; then
    echo "  ✓ MLX available — native Apple Silicon acceleration for LM."
else
    echo "  ⚠ MLX not available. LM backend will use PyTorch (pt) instead."
fi
echo ""

echo "  ============================================================"
echo "    macOS Platform Notes"
echo "  ============================================================"
echo ""
echo "    ✓ MPS GPU acceleration for DiT and VAE"
echo "    ✓ MLX native acceleration for 5Hz Language Model"
echo "    ✗ No torch.compile (uses mx.compile for MLX instead)"
echo "    ✗ No quantization (torchao requires CUDA)"
echo "    ✗ No vllm LM backend (use 'mlx' or 'pt' instead)"
echo "    ✗ CPU offload disabled (unified memory — no benefit)"
echo ""

# -------------------------------------------------------------------
# 5. Download AI Models
# -------------------------------------------------------------------
echo "  [5/6] AI Model Download"
echo ""
echo "  ============================================================"
echo "    Model Download Options"
echo "  ============================================================"
echo ""
echo "    1) Default — Download main model"
echo "       (turbo DiT + 1.7B LM — recommended)"
echo ""
echo "    2) All — Download all available models"
echo ""
echo "    3) Skip — Don't download models now"
echo ""
echo "  ============================================================"
echo ""
read -rp "  Your choice [1/2/3] (default=1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case "$MODEL_CHOICE" in
    1)
        echo "  Downloading main model..."
        python3 -m acestep.model_downloader || echo "  WARNING: Model download had errors."
        ;;
    2)
        echo "  Downloading all models..."
        python3 -m acestep.model_downloader --all || echo "  WARNING: Some downloads had errors."
        ;;
    *)
        echo "  Skipping. Run later: python3 -m acestep.model_downloader"
        ;;
esac
echo ""

# 5b. Optional: Vocoder
if [[ ! -f "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" ]]; then
    read -rp "  Download vocoder model (~206 MB)? [Y/n] (default=Y): " VOCODER_CHOICE
    VOCODER_CHOICE=${VOCODER_CHOICE:-Y}
    if [[ "${VOCODER_CHOICE,,}" == "y" ]]; then
        mkdir -p checkpoints/music_vocoder
        curl -L -o "checkpoints/music_vocoder/config.json" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/config.json" || true
        curl -L -o "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors" || true
    fi
fi
echo ""

# -------------------------------------------------------------------
# 6. Install UI dependencies
# -------------------------------------------------------------------
echo "  [6/6] Setting up React UI..."
if ! command -v node &>/dev/null; then
    echo "  WARNING: Node.js not found. Install via:"
    echo "    brew install node"
else
    echo "  Found Node.js: $(node --version)"
    (cd "$SCRIPT_DIR/ace-step-ui" && npm install) || echo "  WARNING: npm install failed for frontend."
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm install) || echo "  WARNING: npm install failed for backend."
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm run build) || echo "  WARNING: npm run build failed for backend."
fi
echo ""

chmod +x "$SCRIPT_DIR/launch-macos.sh" 2>/dev/null || true

echo "  ============================================================"
echo "    Installation Complete!"
echo "  ============================================================"
echo ""
echo "    To start ACE-Step, run:   ./launch-macos.sh"
echo "    To download more models:  python3 -m acestep.model_downloader --list"
echo ""
echo "    ⚠️  This is an EXPERIMENTAL macOS port."
echo "    Please report issues at: github.com/scragnog/HOT-Step-9000"
echo ""
echo "  ============================================================"
