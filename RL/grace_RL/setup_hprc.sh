#!/usr/bin/env bash
# setup_hprc.sh — one-time environment setup for TAMU HPRC (Grace)
# Run once from your scratch directory before submitting jobs:
#   bash setup_hprc.sh
set -e

ENV_NAME="rl_env"
PYTHON_VERSION="3.9"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.1.1

# ── Create conda env ──────────────────────────────────────────────────────────
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
source activate "$ENV_NAME"

# ── PyTorch (CUDA 12.1) ───────────────────────────────────────────────────────
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# ── Project dependencies ──────────────────────────────────────────────────────
pip install -r requirements.txt

# ── Verify GPU is visible ─────────────────────────────────────────────────────
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "Environment '$ENV_NAME' is ready."
echo "Activate with:  source activate $ENV_NAME"
