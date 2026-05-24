#!/usr/bin/env bash
set -euo pipefail

GPU_HOST="${CSA_GPU_HOST:-root@142.127.68.223}"
GPU_PORT="${CSA_GPU_PORT:-11578}"
REMOTE_DIR="${CSA_REMOTE_DIR:-/root/deepseek-compressed-sparse-attention}"

ssh -p "$GPU_PORT" -o StrictHostKeyChecking=accept-new "$GPU_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd "${CSA_REMOTE_DIR:-/root/deepseek-compressed-sparse-attention}"
latest="$(find runs/csa_top_k -name stdout.log -type f -print 2>/dev/null | sort | tail -n 1)"
if [ -z "$latest" ]; then
  echo "No stdout.log found under runs/csa_top_k yet."
  exit 0
fi
echo "== $latest =="
tail -n 80 "$latest"
REMOTE
