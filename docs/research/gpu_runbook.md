# CSA GPU Runbook

This is the practical path for the rented GPU box.

The current box is reachable with:

```bash
ssh -p 11578 root@142.127.68.223 -L 8080:localhost:8080
```

The scripts use these defaults, but every value can be overridden:

```bash
export CSA_GPU_HOST=root@142.127.68.223
export CSA_GPU_PORT=11578
export CSA_REMOTE_DIR=/root/deepseek-compressed-sparse-attention
```

## Step 1: Set Up The GPU Box

Run from your Mac:

```bash
./scripts/remote_setup_gpu.sh
```

This clones the repo, creates `.venv`, installs CUDA PyTorch, installs the repo
requirements, and verifies that CUDA is visible.

## Step 2: Run A GPU Smoke Test

```bash
./scripts/remote_run_csa_smoke.sh
```

This uses synthetic data and the tiny config. It is only a machinery check:

```text
dense -> csa-k1 -> csa-k8
```

Good result:

```text
all three runs exit 0
each run writes metrics.json
CUDA is available in the metrics
```

## Step 3: Download The Small Real Dataset

```bash
./scripts/remote_download_speedrun_data.sh
```

Default output:

```text
processed_data/speedrun_40M
```

## Step 4: Run The Real Pilot

```bash
./scripts/remote_run_csa_real_pilot.sh
```

Default pilot:

```text
model: CSAMinimumPaperConfig
runs: dense, csa-k1, csa-k8
train tokens: 200000
max seconds per run: 600
```

Override the budget like this:

```bash
CSA_TRAIN_TOKENS=500000 CSA_MAX_SECONDS=900 ./scripts/remote_run_csa_real_pilot.sh
```

## Step 5: Watch Logs

```bash
./scripts/remote_tail_latest.sh
```

If you start a notebook, TensorBoard, or small dashboard on the remote at port
8080, keep the tunnel open with:

```bash
./scripts/remote_tunnel_8080.sh
```

Then open:

```text
http://localhost:8080
```
