#!/usr/bin/env bash
set -euo pipefail

GPU_HOST="${CSA_GPU_HOST:-root@142.127.68.223}"
GPU_PORT="${CSA_GPU_PORT:-11578}"
REMOTE_DIR="${CSA_REMOTE_DIR:-/root/deepseek-compressed-sparse-attention}"
DATASET_DIR="${CSA_DATASET_DIR:-processed_data/speedrun_40M}"

ssh -p "$GPU_PORT" -o StrictHostKeyChecking=accept-new "$GPU_HOST" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"
. .venv/bin/activate

python - <<'PY'
from datasets import load_dataset
import os

dataset_dir = "$DATASET_DIR"
print(f"Downloading 40M-token speedrun subset to {dataset_dir}...")
ds = load_dataset("vukrosic/blueberry-1B-pretrain", split="train[:20000]")
os.makedirs(dataset_dir, exist_ok=True)
ds.save_to_disk(dataset_dir)
print("Data ready.")
PY
REMOTE
