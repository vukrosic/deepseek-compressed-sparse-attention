#!/usr/bin/env bash
set -euo pipefail

GPU_HOST="${CSA_GPU_HOST:-root@142.127.68.223}"
GPU_PORT="${CSA_GPU_PORT:-11578}"
REPO_URL="${CSA_REPO_URL:-https://github.com/vukrosic/deepseek-compressed-sparse-attention.git}"
REMOTE_DIR="${CSA_REMOTE_DIR:-/root/deepseek-compressed-sparse-attention}"
TORCH_INDEX_URL="${CSA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

ssh -p "$GPU_PORT" -o StrictHostKeyChecking=accept-new "$GPU_HOST" bash -s <<REMOTE
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if [ ! -d "$REMOTE_DIR/.git" ]; then
  git clone "$REPO_URL" "$REMOTE_DIR"
fi

cd "$REMOTE_DIR"
git fetch origin main
git checkout main
git pull --ff-only origin main

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
REMOTE
