# Does Compressed Memory Help A Tiny Transformer?

This is the simple version of the research project.

We are not trying to reproduce DeepSeek. We are teaching the smallest honest
research loop:

```text
Question -> experiment -> result -> interpretation -> next question
```

## Question

If a tiny Transformer cannot attend to the whole past, does compressed memory
help?

We compare three ways for a token to read previous tokens:

```text
dense      = read the whole causal past
local      = read only nearby previous tokens
csa        = read nearby previous tokens + compressed summaries of older tokens
```

## The One Experiment

Run these three settings under the same budget:

```text
dense
local
csa
```

Measure:

```text
validation loss
tokens processed
tokens/sec
active training time
peak VRAM
attention budget
raw-token coverage
```

## Fairness Rule

We do not let one method spend extra VRAM to look better.

This is a plain PyTorch research harness, not an optimized sparse-kernel speed
benchmark. If local or CSA is slower, that may be partly implementation
overhead.

Fixed for every run:

```text
model size
dataset
tokenizer
batch size
sequence length
optimizer
learning-rate schedule
training-time budget
seed list
hardware
code commit
```

VRAM is reported as a result, not reused to increase batch size.

Set `train_tokens` high enough that every setting stops because of
`max_train_seconds`, not because it ran out of token budget.

That keeps the claim simple:

```text
With the same training setup and same GPU time, which attention style learns
more?
```

## Mac Smoke Run

Use this only to prove the code works.

```bash
python -m experiments.attention_memory_experiment \
  --config_class configs.research_configs.CSAMacSmokeConfig \
  --synthetic_data true \
  --synthetic_pattern copy_lag \
  --synthetic_lag 32 \
  --train_tokens 1000000 \
  --max_train_seconds 20 \
  --batch_size 4 \
  --num_workers 0 \
  --seeds 42
```

Do not treat this as evidence. It is a debug run.

## NVIDIA Run

For a video-quality result, start with three seeds and 10 minutes per run:

```bash
python -m experiments.attention_memory_experiment \
  --config_class configs.research_configs.CSAMinimumPaperConfig \
  --dataset_path processed_data/speedrun_40M \
  --train_tokens 100000000 \
  --max_train_seconds 600 \
  --batch_size 4 \
  --num_workers 2 \
  --seeds 42,43,44 \
  --local_window_size 64 \
  --csa_top_k 4 \
  --csa_compression_block_size 16
```

This is:

```text
3 attention settings x 3 seeds x 10 minutes = about 90 GPU-minutes
```

If the result is very close, run five seeds:

```bash
--seeds 42,43,44,45,46
```

## How Much Training Is Enough?

For teaching, the result is usable when the gap is bigger than run-to-run noise.

A practical rule:

```text
big gap:    >= 0.10 validation loss -> probably visible with 3 seeds
small gap:  0.03 to 0.10            -> use 5 seeds or train longer
tiny gap:   < 0.03                  -> do not claim a winner
```

Mac runs are for debugging only. The real claim should come from NVIDIA runs.

## Paper Claim Template

If CSA beats local:

```text
Compressed memory helped this tiny model use older context better than a local
window alone under the same GPU-time budget.
```

If CSA ties local:

```text
In this small setup, compressed memory did not clearly help beyond local
attention. The next question is whether the compressor, selector, context
length, or training scale is the bottleneck.
```

If dense wins:

```text
Dense attention remained the strongest baseline under equal GPU time. This does
not disprove CSA; it says this plain PyTorch CSA implementation has not yet
converted compression into a quality or speed win.
```

## Output Files

Each run writes:

```text
runs/attention_memory/<timestamp>/manifest.jsonl
runs/attention_memory/<timestamp>/summary.csv
runs/attention_memory/<timestamp>/summary.json
runs/attention_memory/<timestamp>/<condition>-seed<seed>/metrics.json
runs/attention_memory/<timestamp>/<condition>-seed<seed>/stdout.log
```

Use `summary.csv` for the paper table.
