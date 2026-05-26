# GPU Runbook

This is the shortest path to reproduce the paper sweep on a CUDA machine.

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch CUDA build that matches your machine if `requirements.txt` does not already match your environment.

Check CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 2. Prepare Data

```bash
python3 - <<'PY'
from datasets import load_dataset
from pathlib import Path

out = Path("processed_data/speedrun_40M")
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("vukrosic/blueberry-1B-pretrain", split="train[:20000]")
ds.save_to_disk(str(out))
print(f"saved {out}")
PY
```

## 3. Smoke Test

```bash
bash scripts/run_tests.sh
bash scripts/run_small_sweep.sh
```

## 4. Full Sweep

Use `tmux` so the job survives disconnects:

```bash
tmux new -s forgetting_sweep
bash scripts/run_gpu_sweep.sh
```

Detach with `Ctrl-b d`.

Progress is written to:

```text
runs/forgetting_scaling/<timestamp>/summary.json
```

## 5. Regenerate Figures

```bash
python experiments/forgetting_scaling_plot.py \
  runs/forgetting_scaling/<timestamp> \
  --out docs/research/results/forgetting_scaling_latest
```

## 6. Rebuild Paper

```bash
cd docs/research/reports
pdflatex -interaction=nonstopmode -halt-on-error arch_compare_20260526.tex
pdflatex -interaction=nonstopmode -halt-on-error arch_compare_20260526.tex
```

