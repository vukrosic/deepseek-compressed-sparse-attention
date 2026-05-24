# CSA top_k Experiment Implementation Plan

This plan defines how we will make the seven-day CSA experiment fair enough to
write about.

The key correction:

```text
Same training tokens is data-normalized.
Same GPU-hours is compute-normalized.
```

We should report both, but the paper's main fairness claim should use the
compute-normalized view.

## 1. Fair Comparison Design

Run three views of the same experiment.

### View A: Fixed Tokens

Every run trains for the same number of tokens.

Purpose:

```text
Measure learning from the same data exposure.
```

This is useful, but not compute-fair, because higher top_k can spend more work
per token.

### View B: Fixed GPU-Hours

Every run gets the same active training time.

Purpose:

```text
Measure what result you get for the same rented-GPU budget.
```

This should be the main fairness view.

### View C: Analytical Attention Budget

Every run logs the attention pattern budget.

Purpose:

```text
Explain why a setting costs more or less.
```

For this implementation:

```text
CSA attention-vector budget = sliding_window_size + top_k
CSA coverage budget         = sliding_window_size + top_k * compression_block_size
Dense average causal budget = seq_len / 2
Dense max causal budget     = seq_len
```

Important:

```text
coverage budget is not FLOPs.
```

Coverage says how much old text the summaries cover. Attention-vector budget is
closer to attention compute.

## 2. Metrics We Must Log

Every run should produce one JSON file with:

```text
run_id
git_commit
seed
attention_impl
csa_top_k
csa_compression_block_size
csa_sliding_window_size
train_tokens_target
active_training_time_budget_seconds
tokens_seen
actual_steps
final_val_loss
final_val_perplexity
tokens_per_second
active_training_time_seconds
total_wall_time_seconds
peak_cuda_memory_allocated_bytes
peak_cuda_memory_reserved_bytes
attention_vector_budget
raw_token_equivalent_coverage
dense_avg_causal_budget
hardware_name
torch_version
cuda_version
```

Why both memory metrics:

- `allocated` is memory actively used by tensors.
- `reserved` includes PyTorch's caching allocator.

For the paper, report both if possible and use `peak allocated` as the cleaner
model-memory number.

## 3. Where To Code It

Add a small experiment harness instead of hiding logic inside `train_llm.py`.

Proposed files:

```text
experiments/csa_top_k_sweep.py
experiments/csa_metrics.py
docs/research/csa_top_k_tradeoff_mini_paper.md
docs/research/experiment_implementation_plan.md
```

Responsibilities:

```text
experiments/csa_metrics.py
```

- compute attention-vector budget
- compute raw-token-equivalent coverage
- read GPU name and CUDA info
- read/reset peak CUDA memory
- build run metadata

```text
experiments/csa_top_k_sweep.py
```

- define the run matrix
- launch dense and CSA runs
- pass exact CLI flags
- write one manifest JSONL file
- optionally stop runs by GPU-hour budget

```text
training/trainer.py
```

- add peak CUDA memory to saved metrics
- add tokens/sec to saved metrics
- optionally support a max active training time limit

```text
train_llm.py
```

- expose `--max_train_seconds`
- pass run metadata into the metrics JSON

## 4. First Code Change

The first implementation change should be instrumentation, not more modeling.

Add:

```text
--max_train_seconds
```

Behavior:

```text
If max_train_seconds is set, stop after the current optimization step once
active training time exceeds that budget.
```

This lets us run fixed GPU-hour comparisons.

Also add:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Use them only when CUDA is available.

## 5. Pilot Run

Before the NVIDIA sweep, run a Mac smoke pilot using synthetic data:

```bash
python -m experiments.csa_top_k_sweep \
  --mode pilot \
  --config_class configs.research_configs.CSAMacSmokeConfig \
  --synthetic_data true \
  --train_tokens 8192 \
  --batch_size 4 \
  --num_workers 0 \
  --compile false \
  --warmup false
```

This proves the code path works. It is not a paper result.

Then run a short real-data NVIDIA pilot:

```text
dense, 200k tokens
csa-k1, 200k tokens
csa-k8, 200k tokens
```

Pilot goals:

- confirm no crashes
- estimate tokens/sec
- estimate VRAM
- choose batch size
- choose the fixed GPU-hour budget

Do not choose the full seven-day budget before seeing pilot throughput.

## 6. Main Sweep

After the pilot, run:

```text
dense
csa-k1
csa-k2
csa-k4
csa-k8
csa-k16
```

For each run, save:

```text
metrics JSON
stdout log
exact command
git commit
hardware
```

Run order should alternate cheap and expensive settings to catch failures early:

```text
dense
csa-k1
csa-k16
csa-k4
csa-k2
csa-k8
```

## 7. Paper Claim Rules

Allowed claim:

```text
Under a fixed GPU-hour budget, top_k=X gave the best validation-loss tradeoff in
this small 88M CSA implementation.
```

Not allowed:

```text
CSA is faster than dense attention.
```

Not unless the measured wall-clock data says so.

Allowed negative result:

```text
Dense attention remained stronger at seq_len=2048, but CSA showed a monotonic
quality recovery as top_k increased. This suggests the selector/compressor path
is functioning, while speed claims require longer contexts or custom kernels.
```
