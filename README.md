# Forgetting Attention

Plain PyTorch reference implementations of ten forgetting and memory attention policies, with correctness tests, a reproducible one-GPU sweep, and a short paper.

## Onboard with an AI agent

Paste this into Claude Code, Cursor, or any agent that can run shell commands. It will set up the environment, fetch the data, and brief you on the project.

```text
Clone https://github.com/vukrosic/forgetting-attention into the current directory.
Create a Python venv, install requirements.txt, and download the dataset
(see the "Data" section of the README). Then read
docs/research/reports/arch_compare_20260526.pdf along with models/, experiments/,
and tests/, and tell me:
  1. what the project is about
  2. what each of the ten forgetting mechanisms does, in one sentence each
  3. what the preliminary results say and what they do not say
  4. the main limitations of the current setup
  5. the most promising next experiments
Then ask me which experiment I want to run first.
```

## Paper

- PDF: [docs/research/reports/arch_compare_20260526.pdf](docs/research/reports/arch_compare_20260526.pdf)
- Final plots and results table: [docs/research/results/forgetting_scaling_latest](docs/research/results/forgetting_scaling_latest)

## Policies

| Policy | Idea |
|---|---|
| `dense` | Full causal SDPA (baseline). |
| `local` | Sliding window, no memory. |
| `csa` | DeepSeek-style indexer picks top-k compressed blocks. |
| `compressed_memory` | Block-mean summaries of older tokens. |
| `age_forgetting` | Linear age-decay gate on compressed blocks. |
| `hierarchical` | Blocks plus summaries of summaries. |
| `predictive` | Learned router picks blocks per query. |
| `surprise_retention` | Keep blocks hardest to predict from their own keys. |
| `frequency_lfu` | Keep blocks that received past attention mass. |
| `token_merge` | Merge similar adjacent memory blocks. |
| `recurrent_state` | Fixed-size linear-attention accumulator. |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA runs, install a PyTorch build matching your GPU before running the sweep.

## Data

```bash
python3 - <<'PY'
from datasets import load_dataset
from pathlib import Path
out = Path("processed_data/speedrun_40M")
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("vukrosic/blueberry-1B-pretrain", split="train[:20000]")
ds.save_to_disk(str(out))
PY
```

## Tests

```bash
python -m tests.test_forgetting_mechanisms
```

Checks shape/dtype, causality, mask correctness, gradient flow, and peak memory across all eleven policies.

## Reproduce the sweep

```bash
bash scripts/run_gpu_sweep.sh
```

Runs 11 policies × 3 context lengths (2K/4K/8K) × 300 seconds each on one CUDA GPU. Outputs go to `runs/forgetting_scaling/<timestamp>/`. Rebuild plots with:

```bash
python experiments/forgetting_scaling_plot.py \
  runs/forgetting_scaling/<timestamp> \
  --out docs/research/results/forgetting_scaling_latest
```

## Known limits

One seed, one dataset, 18M-class model, 300-second runs, unfused PyTorch, ordinary next-token prediction. The next useful experiment is a synthetic long-range retrieval benchmark where a query must recover values from thousands of tokens earlier.
