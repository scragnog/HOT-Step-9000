# ============================================================
#  ACE-Step 1.5 — Docker Build (Linux / NVIDIA CUDA)
#  Multi-stage build for containerized deployment.
#
#  Usage:
#    docker compose up --build
#    OR:
#    docker build -t ace-step .
#    docker run --gpus all -p 3000:3000 -p 3001:3001 -p 8001:8001 \
#      -v ./checkpoints:/app/checkpoints \
#      -v ./data:/app/data \
#      ace-step
# ============================================================

# --- Stage 1: Build ---
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install Node.js 20 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip3 install uv

WORKDIR /app

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml requirements.txt ./
COPY acestep/third_parts/nano-vllm/ ./acestep/third_parts/nano-vllm/

# Create venv and install Python dependencies
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"
ENV UV_INDEX_STRATEGY="unsafe-best-match"

RUN uv pip install -e . || uv pip install -r requirements.txt
RUN uv pip install -e acestep/third_parts/nano-vllm 2>/dev/null || true

# Copy UI dependency files and install
COPY ace-step-ui/package.json ace-step-ui/package-lock.json* ./ace-step-ui/
COPY ace-step-ui/server/package.json ace-step-ui/server/package-lock.json* ./ace-step-ui/server/

RUN cd ace-step-ui && npm ci --omit=dev 2>/dev/null || npm install
RUN cd ace-step-ui/server && npm ci --omit=dev 2>/dev/null || npm install

# --- Stage 2: Runtime ---
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    nodejs \
    npm \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /app

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy node_modules from builder
COPY --from=builder /app/ace-step-ui/node_modules ./ace-step-ui/node_modules
COPY --from=builder /app/ace-step-ui/server/node_modules ./ace-step-ui/server/node_modules

# Copy application code
COPY . .

# Build server TypeScript
RUN cd ace-step-ui/server && npx tsc 2>/dev/null || true

# Environment
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME="/app/huggingface"
ENV XFORMERS_FORCE_DISABLE_TRITON=1
ENV PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG=1
ENV ACESTEP_NO_BROWSER=1
ENV ACESTEP_PATH="/app"

# Expose ports: Vite (3000), Express (3001), Python API (8001)
EXPOSE 3000 3001 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start all services
# Docker mode: skip loading.html, boot Python API directly
CMD ["bash", "-c", "\
    echo '=== ACE-Step Docker Mode ===' && \
    cd /app/ace-step-ui/server && npm run dev &>/dev/null & \
    sleep 2 && \
    cd /app/ace-step-ui && npm run dev &>/dev/null & \
    sleep 2 && \
    touch /app/boot.lock && \
    cd /app && python3 acestep/api_server.py --port 8001 --host 0.0.0.0 \
"]
