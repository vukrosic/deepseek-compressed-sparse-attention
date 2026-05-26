#!/usr/bin/env bash
set -euo pipefail

python experiments/forgetting_scaling_sweep.py \
  --contexts 512 \
  --archs dense local compressed_memory age_forgetting surprise_retention \
  --max_train_seconds 30 \
  --train_tokens 1000000 \
  --batch_size 2 \
  --eval_every 25 \
  --run_root runs/small_forgetting_sweep

