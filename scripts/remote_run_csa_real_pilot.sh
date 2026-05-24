#!/usr/bin/env bash
set -euo pipefail

GPU_HOST="${CSA_GPU_HOST:-root@142.127.68.223}"
GPU_PORT="${CSA_GPU_PORT:-11578}"
REMOTE_DIR="${CSA_REMOTE_DIR:-/root/deepseek-compressed-sparse-attention}"
DATASET_DIR="${CSA_DATASET_DIR:-processed_data/speedrun_40M}"
TRAIN_TOKENS="${CSA_TRAIN_TOKENS:-200000}"
MAX_SECONDS="${CSA_MAX_SECONDS:-600}"
BATCH_SIZE="${CSA_BATCH_SIZE:-4}"

ssh -p "$GPU_PORT" -o StrictHostKeyChecking=accept-new "$GPU_HOST" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"
. .venv/bin/activate
git pull --ff-only origin main

python -m experiments.csa_top_k_sweep \
  --mode pilot \
  --config_class configs.research_configs.CSAMinimumPaperConfig \
  --dataset_path "$DATASET_DIR" \
  --train_tokens "$TRAIN_TOKENS" \
  --max_train_seconds "$MAX_SECONDS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers 2 \
  --compile false \
  --warmup false
REMOTE
