# Forgetting Attention

**Language**: English | [中文](README_zh.md)

Plain PyTorch reference implementations of ten forgetting and memory attention policies, with correctness tests, a reproducible one-GPU sweep, and a short paper.

![Mechanism diagrams](docs/research/results/arch_compare_20260526/mechanism_diagrams.png)

Black regions are visible to the query, gray regions are retained through memory or gating, white regions are forgotten.

## Paper

- **PDF (English)**: [paper.pdf](paper.pdf)
- **PDF (中文)**: [paper_zh.pdf](paper_zh.pdf)
- **Source (English)**: [docs/research/reports/arch_compare_20260526.tex](docs/research/reports/arch_compare_20260526.tex)
- **Source (中文)**: [docs/research/reports/arch_compare_20260526_zh.tex](docs/research/reports/arch_compare_20260526_zh.tex)
- **Final plots and results table**: [docs/research/results/forgetting_scaling_latest](docs/research/results/forgetting_scaling_latest)

## Onboard with an AI agent

Paste this into Claude Code, Cursor, or any agent that can run shell commands. It will set up the environment, fetch the data, and brief you on the project.

```text
Clone https://github.com/vukrosic/forgetting-attention into the current directory.
Create a Python venv, install requirements.txt, and download the dataset
(see the "Data" section of the README). Then read the paper source at
docs/research/reports/arch_compare_20260526.tex along with the implementations
under models/ (especially models/layers.py, models/memory_policies.py,
models/new_forgetting.py, models/compressed_sparse_attention.py), the sweep at
experiments/forgetting_scaling_sweep.py, and the correctness suite at
tests/test_forgetting_mechanisms.py. Ignore anything listed in docs/STALE.md.
Then tell me:
  1. what the project is about
  2. what each of the ten forgetting mechanisms does, in one sentence each,
     with the file:line where it is defined
  3. what the preliminary results say and what they do not say
  4. the main limitations of the current setup
  5. the most promising next experiments
Then ask me which experiment I want to run first.
```

## Policies

Click any class name to jump straight to its implementation.

| Policy | Idea | Implementation |
|---|---|---|
| `dense` | Full causal SDPA (baseline). | [`MultiHeadAttention`](models/layers.py#L45) |
| `local` | Sliding window, no memory. | [`LocalSlidingWindowAttention`](models/layers.py#L131) |
| `csa` | DeepSeek-style indexer picks top-k compressed blocks. | [`CompressedSparseAttention`](models/compressed_sparse_attention.py#L193) |
| `compressed_memory` | Block-mean summaries of older tokens. | [`CompressedMemoryNoGateAttention`](models/memory_policies.py#L607) |
| `age_forgetting` | Linear age-decay gate on compressed blocks. | [`AgeForgettingAttention`](models/memory_policies.py#L332) |
| `hierarchical` | Blocks plus summaries of summaries. | [`HierarchicalSummarizationAttention`](models/memory_policies.py#L801) |
| `predictive` | Learned router picks blocks per query. | [`PredictiveImportanceAttention`](models/memory_policies.py#L919) |
| `surprise_retention` | Keep blocks hardest to predict from their own keys. | [`SurpriseRetentionAttention`](models/new_forgetting.py#L41) |
| `frequency_lfu` | Keep blocks that received past attention mass. | [`FrequencyLFUAttention`](models/new_forgetting.py#L158) |
| `token_merge` | Merge similar adjacent memory blocks. | [`TokenMergeAttention`](models/new_forgetting.py#L273) |
| `recurrent_state` | Fixed-size linear-attention accumulator. | [`RecurrentStateAttention`](models/new_forgetting.py#L437) |

Other modules in `models/` that are not exercised by the paper (legacy variants, exploratory attentions) are listed in [docs/STALE.md](docs/STALE.md).

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

Runs 11 policies × 3 context lengths (2K/4K/8K) × 300 seconds each on one CUDA GPU. Outputs go to `runs/forgetting_scaling/<timestamp>/`. Rebuild plots and the top-level `paper.pdf` with:

```bash
python experiments/forgetting_scaling_plot.py \
  runs/forgetting_scaling/<timestamp> \
  --out docs/research/results/forgetting_scaling_latest

cd docs/research/reports
pdflatex -interaction=nonstopmode arch_compare_20260526.tex
pdflatex -interaction=nonstopmode arch_compare_20260526.tex
cp arch_compare_20260526.pdf ../../../paper.pdf
```

## Known limits

One seed, one dataset, 18M-class model, 300-second runs, unfused PyTorch, ordinary next-token prediction. The next useful experiment is a synthetic long-range retrieval benchmark where a query must recover values from thousands of tokens earlier.
