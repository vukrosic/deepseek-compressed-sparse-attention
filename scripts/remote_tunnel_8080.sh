#!/usr/bin/env bash
set -euo pipefail

GPU_HOST="${CSA_GPU_HOST:-root@142.127.68.223}"
GPU_PORT="${CSA_GPU_PORT:-11578}"

ssh -p "$GPU_PORT" -L 8080:localhost:8080 "$GPU_HOST"
