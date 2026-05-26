#!/usr/bin/env bash
set -euo pipefail

python experiments/forgetting_scaling_sweep.py \
  --contexts 2048 4096 8192 \
  --max_train_seconds 300 \
  --train_tokens 200000000 \
  --batch_size 4 \
  --eval_every 200 \
  --run_root runs/forgetting_scaling

