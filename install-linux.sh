#!/usr/bin/env bash
# ====================================================================
#  ACE-Step 1.5 — First-Time Installer (Linux / NVIDIA CUDA)
#  Sets up: UV, Python venv, dependencies, models, UI
# ====================================================================
set -euo pipefail

echo ""
echo "  ============================================================"
echo "    ACE-Step 1.5 — Linux CUDA Installer"
echo "  ============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "  [*] Working directory: $(pwd)"
echo ""

# -------------------------------------------------------------------
# 1. Check Python is available
# -------------------------------------------------------------------
echo "  [1/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo ""
    echo "  ERROR: Python 3 is not installed or not on PATH."
    echo "  Please install Python 3.11 or 3.12."
    echo "    Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "    Fedora:        sudo dnf install python3.11"
    echo ""
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1)
echo "  Found: $PY_VERSION"

PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")

if [[ "$PY_MAJOR" != "3" ]]; then
    echo "  ERROR: Python 3 is required. Detected: $PY_VERSION"
    exit 1
fi
if (( PY_MINOR < 11 )); then
    echo "  ERROR: Python 3.11 or 3.12 is required. Detected: $PY_VERSION"
    echo "  Python 3.10 and older are not supported."
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

# Activate the venv for the rest of the script
source .venv/bin/activate
echo "  Activated: .venv"
echo ""

# -------------------------------------------------------------------
# 4. Install Python dependencies
# -------------------------------------------------------------------
echo "  [4/6] Installing Python dependencies..."
echo "  This may take several minutes on first run."
echo ""

# Ensure UV resolves PyTorch from the CUDA wheel index
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"
export UV_INDEX_STRATEGY="unsafe-best-match"

uv pip install -e . || {
    echo ""
    echo "  WARNING: uv pip install -e . failed, trying with requirements.txt..."
    uv pip install -r requirements.txt || {
        echo "  ERROR: Failed to install Python dependencies."
        exit 1
    }
}

# Install nano-vllm from local source
if [[ -d "acestep/third_parts/nano-vllm" ]]; then
    echo "  Installing nano-vllm from local source..."
    uv pip install -e acestep/third_parts/nano-vllm 2>/dev/null || true
fi

# Verify PyTorch has CUDA support
echo "  Verifying PyTorch CUDA support..."
if python3 -c "import torch; exit(0 if torch.cuda.is_available() or hasattr(torch.version,'hip') else 1)" 2>/dev/null; then
    echo "  PyTorch CUDA support confirmed."
else
    echo "  WARNING: CPU-only PyTorch detected. Reinstalling with CUDA support..."
    uv pip install --force-reinstall --no-cache torch torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/cu130 || {
        echo "  ERROR: Failed to install CUDA PyTorch."
        echo "  You can try manually:"
        echo "    source .venv/bin/activate"
        echo "    uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130"
        exit 1
    }
    echo "  CUDA PyTorch installed successfully."
fi
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
echo "       (turbo DiT + 1.7B LM — recommended for most users)"
echo ""
echo "    2) All — Download all available models"
echo "       (all DiT variants + all LM sizes — large download)"
echo ""
echo "    3) Skip — Don't download models now"
echo "       (you can run:  python3 -m acestep.model_downloader  later)"
echo ""
echo "  ============================================================"
echo ""
read -rp "  Your choice [1/2/3] (default=1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case "$MODEL_CHOICE" in
    1)
        echo ""
        echo "  Downloading main model..."
        python3 -m acestep.model_downloader || echo "  WARNING: Model download had errors."
        ;;
    2)
        echo ""
        echo "  Downloading all models..."
        python3 -m acestep.model_downloader --all || echo "  WARNING: Some downloads had errors."
        ;;
    *)
        echo "  Skipping model download. You can download later with:"
        echo "    python3 -m acestep.model_downloader"
        ;;
esac
echo ""

# 5b. Optional: Vocoder
echo "  ============================================================"
echo "    Optional: Vocoder Enhancement Model (HiFi-GAN, ~206 MB)"
echo "  ============================================================"
if [[ -f "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" ]]; then
    echo "  Vocoder model already installed. Skipping."
else
    read -rp "  Download vocoder model? [Y/n] (default=Y): " VOCODER_CHOICE
    VOCODER_CHOICE=${VOCODER_CHOICE:-Y}
    if [[ "${VOCODER_CHOICE,,}" == "y" ]]; then
        mkdir -p checkpoints/music_vocoder
        curl -L -o "checkpoints/music_vocoder/config.json" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/config.json" || true
        curl -L -o "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors" || true
        echo "  Vocoder download complete."
    else
        echo "  Skipping vocoder download."
    fi
fi
echo ""

# -------------------------------------------------------------------
# 6. Install UI dependencies (Node.js / npm)
# -------------------------------------------------------------------
echo "  [6/6] Setting up React UI..."
if ! command -v node &>/dev/null; then
    echo ""
    echo "  WARNING: Node.js is not installed or not on PATH."
    echo "  The React UI requires Node.js 18+."
    echo "  Install via: https://nodejs.org/ or your package manager"
    echo "    Ubuntu/Debian: sudo apt install nodejs npm"
    echo "    Or use nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash"
else
    echo "  Found Node.js: $(node --version)"

    echo "  Installing UI frontend dependencies..."
    (cd "$SCRIPT_DIR/ace-step-ui" && npm install) || echo "  WARNING: npm install failed for frontend."

    echo "  Installing UI backend dependencies..."
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm install) || echo "  WARNING: npm install failed for backend."

    echo "  Building UI backend..."
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm run build) || echo "  WARNING: npm run build failed for backend."
fi
echo ""

# Make launch script executable
chmod +x "$SCRIPT_DIR/launch-linux.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/launch-rocm.sh" 2>/dev/null || true

echo "  ============================================================"
echo "    Installation Complete!"
echo "  ============================================================"
echo ""
echo "    To start ACE-Step, run:   ./launch-linux.sh"
echo "    To download more models:  python3 -m acestep.model_downloader --list"
echo ""
echo "  ============================================================"
