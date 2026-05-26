# Attention Forgetting Lab

Plain PyTorch reference implementations for forgetting and memory attention policies.

This repository is a small research lab for asking:

```text
If a transformer cannot attend to everything forever, what should it keep?
```

It includes ten non-dense attention policies, a dense baseline, correctness tests, a reproducible GPU sweep, and a short paper with the first results.

## Paper

- PDF: [docs/research/reports/arch_compare_20260526.pdf](docs/research/reports/arch_compare_20260526.pdf)
- LaTeX: [docs/research/reports/arch_compare_20260526.tex](docs/research/reports/arch_compare_20260526.tex)
- Final plots/table: [docs/research/results/forgetting_scaling_latest](docs/research/results/forgetting_scaling_latest)

Main result from the one-GPU reference sweep:

```text
dense and local attention dominate in unfused PyTorch.
simple compressed-memory rules are the strongest memory variants.
the current value is reference correctness, not a speed/quality win.
```

That is still useful: these modules are readable baselines for researchers who want to write Triton/CUDA kernels or scale the comparison on larger hardware.

## Attention Policies

Dense attention is the ceiling baseline. The ten non-dense policies are:

| Policy | Idea |
|---|---|
| `local` | Hard recency forgetting: keep only the recent sliding window. |
| `csa` | DeepSeek-style compressed sparse attention with a lightweight indexer. |
| `compressed_memory` | Keep recent tokens plus block-mean summaries of older tokens. |
| `age_forgetting` | Weaken older compressed blocks with an age-decay gate. |
| `hierarchical` | Store blocks plus summaries of summaries. |
| `predictive` | Use a small learned router to predict which blocks matter. |
| `surprise_retention` | Keep blocks that are hard to predict from their own keys. |
| `frequency_lfu` | Keep blocks that received more past attention mass. |
| `token_merge` | Merge similar adjacent memory blocks. |
| `recurrent_state` | Replace block memory with a fixed-size recurrent state. |

## Repository Map

```text
models/
  layers.py                    attention dispatch
  memory_policies.py           compressed-memory policies
  new_forgetting.py            surprise, LFU, token merge, recurrent state

configs/
  research_configs.py          small paper model config
  memory_policy_config.py      memory-policy knobs

experiments/
  forgetting_scaling_sweep.py  resilient 11-policy x context sweep
  forgetting_scaling_plot.py   rebuilds paper plots and results table

tests/
  test_forgetting_mechanisms.py  shape, causality, mask, gradients, memory smoke tests

docs/research/
  reports/                     paper PDF and LaTeX
  results/forgetting_scaling_latest/
                               final paper figures and table
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA runs, install a PyTorch build matching your GPU environment before running the sweep.

## Data

The paper sweep used a 40M-token speedrun slice:

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

`processed_data/` is ignored by Git.

## Run Tests

```bash
bash scripts/run_tests.sh
```

The main correctness grid is:

```bash
python -m tests.test_forgetting_mechanisms
```

It checks shape/dtype, causality, mask/finite output, gradient flow, and memory-budget smoke behavior.

## Run A Tiny Sweep

Use this to catch bugs before renting a GPU:

```bash
bash scripts/run_small_sweep.sh
```

It runs a short context-512 sweep over a few policies and writes under `runs/`.

## Reproduce The GPU Sweep

On a CUDA machine:

```bash
bash scripts/run_gpu_sweep.sh
```

This launches:

```text
11 policies x 3 context lengths x 300 seconds
```

Outputs go to:

```text
runs/forgetting_scaling/<timestamp>/
```

To regenerate paper plots from a completed run:

```bash
python experiments/forgetting_scaling_plot.py \
  runs/forgetting_scaling/<timestamp> \
  --out docs/research/results/forgetting_scaling_latest
```

Then rebuild the paper:

```bash
cd docs/research/reports
pdflatex -interaction=nonstopmode -halt-on-error arch_compare_20260526.tex
pdflatex -interaction=nonstopmode -halt-on-error arch_compare_20260526.tex
```

## Known Limits

- one seed
- one dataset
- small 18M-class model
- short 300-second runs
- unfused PyTorch kernels
- ordinary next-token prediction, not an explicit long-memory retrieval task

The next useful experiment is a synthetic key-value retrieval benchmark where a query must recover values from thousands of tokens earlier.

## DeepSeek V4 Tutorial Material

This repo started as a DeepSeek compressed sparse attention tutorial. Those materials are still here:

- Tutorial: [docs/tutorial.md](docs/tutorial.md)
- Paper PDF: [papers/DeepSeek_V4.pdf](papers/DeepSeek_V4.pdf)

