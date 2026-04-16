#!/usr/bin/env bash
# ====================================================================
#  ACE-Step 1.5 — First-Time Installer (Linux / AMD ROCm)
#  Sets up: UV, Python venv, dependencies (ROCm wheels), models, UI
#
#  ⚠️  EXPERIMENTAL — Community-tested, not officially supported.
#  Requires: Linux, AMD GPU with ROCm 6.x drivers installed.
# ====================================================================
set -euo pipefail

echo ""
echo "  ============================================================"
echo "    ACE-Step 1.5 — Linux AMD ROCm Installer"
echo "    ⚠️  EXPERIMENTAL — ROCm backend"
echo "  ============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "  [*] Working directory: $(pwd)"
echo ""

# -------------------------------------------------------------------
# 0. Check ROCm environment
# -------------------------------------------------------------------
echo "  [0/6] Checking ROCm environment..."
ROCM_DETECTED=false

if command -v rocm-smi &>/dev/null; then
    echo "  ✓ rocm-smi found."
    rocm-smi --showid 2>/dev/null | head -5 || true
    ROCM_DETECTED=true
elif command -v rocminfo &>/dev/null; then
    echo "  ✓ rocminfo found."
    ROCM_DETECTED=true
else
    echo "  ⚠ Neither rocm-smi nor rocminfo found."
    echo "  ROCm drivers may not be installed."
    echo ""
    echo "  Install ROCm:"
    echo "    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo ""
    read -rp "  Continue anyway? [y/N]: " ROCM_CONTINUE
    if [[ "$(echo "$ROCM_CONTINUE" | tr '[:upper:]' '[:lower:]')" != "y" ]]; then
        echo "  Aborting. Install ROCm first."
        exit 1
    fi
fi

# HSA_OVERRIDE_GFX_VERSION setup
echo ""
echo "  ============================================================"
echo "    HSA_OVERRIDE_GFX_VERSION Setup"
echo "  ============================================================"
echo ""
echo "  Many AMD GPUs need HSA_OVERRIDE_GFX_VERSION to work with"
echo "  PyTorch ROCm. Common values:"
echo ""
echo "    RX 7900 XT/XTX, RX 9070 XT:  11.0.0"
echo "    RX 7800 XT, RX 7700 XT:      11.0.1"
echo "    RX 7600:                       11.0.2"
echo "    RX 6900 XT, RX 6800 XT:      10.3.0"
echo "    RX 6700 XT:                   10.3.1"
echo ""
echo "  If unsure, check: https://rocm.docs.amd.com/en/latest/"
echo ""

CURRENT_HSA="${HSA_OVERRIDE_GFX_VERSION:-}"
if [[ -n "$CURRENT_HSA" ]]; then
    echo "  Current HSA_OVERRIDE_GFX_VERSION: $CURRENT_HSA"
    read -rp "  Keep this value? [Y/n]: " KEEP_HSA
    if [[ "$(echo "$KEEP_HSA" | tr '[:upper:]' '[:lower:]')" == "n" ]]; then
        CURRENT_HSA=""
    fi
fi

if [[ -z "$CURRENT_HSA" ]]; then
    read -rp "  Enter HSA_OVERRIDE_GFX_VERSION (or press Enter to skip): " HSA_INPUT
    if [[ -n "$HSA_INPUT" ]]; then
        export HSA_OVERRIDE_GFX_VERSION="$HSA_INPUT"
        echo "  Set HSA_OVERRIDE_GFX_VERSION=$HSA_INPUT"
        # Save to .env for launch script
        if [[ -f ".env" ]]; then
            grep -v '^HSA_OVERRIDE_GFX_VERSION=' .env > .env.tmp || true
            echo "HSA_OVERRIDE_GFX_VERSION=$HSA_INPUT" >> .env.tmp
            mv .env.tmp .env
        else
            echo "HSA_OVERRIDE_GFX_VERSION=$HSA_INPUT" > .env
        fi
    fi
fi
echo ""

# -------------------------------------------------------------------
# 1. Check Python
# -------------------------------------------------------------------
echo "  [1/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "  ERROR: Python 3 is not installed."
    echo "    Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1)
echo "  Found: $PY_VERSION"

PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if (( PY_MINOR < 11 )); then
    echo "  ERROR: Python 3.11 or 3.12 required."
    exit 1
fi
echo ""

# -------------------------------------------------------------------
# 2. Install UV
# -------------------------------------------------------------------
echo "  [2/6] Checking UV package manager..."
if ! command -v uv &>/dev/null; then
    pip3 install uv || { echo "  ERROR: Failed to install UV."; exit 1; }
    echo "  UV installed."
else
    echo "  Found: $(uv --version 2>&1)"
fi
echo ""

# -------------------------------------------------------------------
# 3. Create venv
# -------------------------------------------------------------------
echo "  [3/6] Setting up Python virtual environment..."
if [[ -f ".venv/bin/activate" ]]; then
    echo "  Virtual environment already exists."
else
    uv venv .venv || { echo "  ERROR: Failed to create venv."; exit 1; }
fi
source .venv/bin/activate
echo "  Activated: .venv"
echo ""

# -------------------------------------------------------------------
# 4. Install Python dependencies (ROCm wheels)
# -------------------------------------------------------------------
echo "  [4/6] Installing Python dependencies (ROCm)..."
echo "  This may take several minutes on first run."
echo ""

# Point to ROCm wheel index instead of CUDA
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/rocm6.2"
export UV_INDEX_STRATEGY="unsafe-best-match"

# Install main project (will pull ROCm torch via the index)
uv pip install -e . || {
    echo "  WARNING: uv pip install -e . failed. Trying requirements.txt..."
    # Install base requirements but override torch for ROCm
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    uv pip install -r requirements.txt || {
        echo "  ERROR: Failed to install dependencies."
        exit 1
    }
}

# Install nano-vllm from local source
if [[ -d "acestep/third_parts/nano-vllm" ]]; then
    echo "  Installing nano-vllm from local source..."
    uv pip install -e acestep/third_parts/nano-vllm 2>/dev/null || true
fi

# Verify ROCm/HIP PyTorch
echo "  Verifying PyTorch ROCm support..."
if python3 -c "import torch; assert hasattr(torch.version,'hip') and torch.version.hip is not None; print(f'HIP: {torch.version.hip}')" 2>/dev/null; then
    echo "  ✓ PyTorch ROCm/HIP build confirmed."
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  ⚠ PyTorch CUDA available but not ROCm build. Generation may still work."
else
    echo "  ❌ No GPU support detected in PyTorch."
    echo "  Try reinstalling:"
    echo "    uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
fi
echo ""

echo "  ============================================================"
echo "    ROCm Platform Notes"
echo "  ============================================================"
echo ""
echo "    ✓ ROCm GPU acceleration via PyTorch HIP backend"
echo "    ✓ SDPA attention (flash-attn not available on ROCm)"
echo "    ✗ No flash-attn (SDPA fallback used — same quality, slightly slower)"
echo "    ✗ No torch.compile (disabled by default on ROCm)"
echo "    ✗ No torchao quantization (disabled — higher VRAM usage)"
echo "    ✗ Default dtype is float32 (set ACESTEP_ROCM_DTYPE=bfloat16 to override)"
echo ""
echo "    Set ACESTEP_ROCM_DTYPE=bfloat16 in .env if your GPU supports bf16"
echo "    (most RDNA3/RDNA4 GPUs do, but some iGPUs may segfault)"
echo ""

# -------------------------------------------------------------------
# 5. Download models (same as Linux CUDA)
# -------------------------------------------------------------------
echo "  [5/6] AI Model Download"
echo ""
echo "    1) Default — turbo DiT + 1.7B LM"
echo "    2) All models"
echo "    3) Skip"
echo ""
read -rp "  Your choice [1/2/3] (default=1): " MODEL_CHOICE
MODEL_CHOICE=${MODEL_CHOICE:-1}

case "$MODEL_CHOICE" in
    1) python3 -m acestep.model_downloader || echo "  WARNING: Download had errors." ;;
    2) python3 -m acestep.model_downloader --all || echo "  WARNING: Some downloads had errors." ;;
    *) echo "  Skipping. Run later: python3 -m acestep.model_downloader" ;;
esac

# Vocoder
if [[ ! -f "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" ]]; then
    read -rp "  Download vocoder (~206 MB)? [Y/n] (default=Y): " VOCODER_CHOICE
    VOCODER_CHOICE=${VOCODER_CHOICE:-Y}
    if [[ "$(echo "$VOCODER_CHOICE" | tr '[:upper:]' '[:lower:]')" == "y" ]]; then
        mkdir -p checkpoints/music_vocoder
        curl -L -o "checkpoints/music_vocoder/config.json" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/config.json" || true
        curl -L -o "checkpoints/music_vocoder/diffusion_pytorch_model.safetensors" \
            "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors" || true
    fi
fi
echo ""

# -------------------------------------------------------------------
# 6. UI dependencies
# -------------------------------------------------------------------
echo "  [6/6] Setting up React UI..."
if command -v node &>/dev/null; then
    echo "  Found Node.js: $(node --version)"
    (cd "$SCRIPT_DIR/ace-step-ui" && npm install) || true
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm install) || true
    (cd "$SCRIPT_DIR/ace-step-ui/server" && npm run build) || true
else
    echo "  WARNING: Node.js not found. Install: sudo apt install nodejs npm"
fi
echo ""

chmod +x "$SCRIPT_DIR/launch-rocm.sh" 2>/dev/null || true

echo "  ============================================================"
echo "    Installation Complete!"
echo "  ============================================================"
echo ""
echo "    To start ACE-Step, run:   ./launch-rocm.sh"
echo ""
echo "    ⚠️  This is an EXPERIMENTAL ROCm port."
echo "    Please report issues at: github.com/scragnog/HOT-Step-9000"
echo "  ============================================================"
