# A Small-Scale Study Of top_k In Compressed Sparse Attention

Status: living mini-paper. Results are early and should be treated as pilot
evidence, not a final claim.

<!-- Note: [2026-05-24 14:17] test note at line 5 -->

Companion critique: [Critique Of The CSA top_k Mini Paper](csa_top_k_tradeoff_critique.md).

Archived results bundle: [CSA top_k Results Bundle](results/csa_top_k_20260524/README.md).

## Abstract

Compressed Sparse Attention (CSA) lets each token attend to a local window plus
a small number of compressed summaries of older tokens. This study asks one
narrow question:

```text
As top_k increases, how much quality do we recover, and what do we pay in
throughput and memory?
```

DeepSeek-V4 reports large-model CSA settings, but it does not provide the small
plain-PyTorch top_k sweep we need for a seven-day tutorial experiment. We run
that sweep on one RTX 3090 using a small Transformer and report both
data-normalized and compute-normalized views.

## Question

We vary only `top_k`, the number of compressed history blocks selected per
query.

The runs are:

```text
dense baseline
CSA top_k = 1, 2, 4, 8, 16
```

The main fairness rule is:

```text
fixed GPU time is the headline comparison
fixed training tokens is a secondary diagnostic
```

Why? Higher `top_k` changes how much attention work happens per token. Same
training tokens is useful, but it is not compute-fair.

## Method

Hardware:

```text
1x NVIDIA RTX 3090, 24GB VRAM
```

Model:

```text
config: configs.research_configs.CSAMinimumPaperConfig
d_model = 256
n_heads = 8
n_layers = 8
d_ff = 1024
max_seq_len = 2048
parameters ~= 18.4M in the current tokenizer setup
```

Dataset:

```text
processed_data/speedrun_40M
```

Implementation:

```text
plain PyTorch CSA
no custom sparse CUDA kernels
```

This means timing is not a production-kernel speed claim. It is a small
architecture and research-loop measurement.

## Budget Accounting

CSA has two useful budget numbers.

Actual attention-vector budget:

```text
attention_vectors = sliding_window_size + top_k
```

Raw-token-equivalent coverage:

```text
coverage = sliding_window_size + top_k * compression_block_size
```

For this run:

```text
sliding_window_size = 64
compression_block_size = 16
```

So:

| Run | Attention vectors | Raw-token coverage |
| --- | ---: | ---: |
| dense | 1024 avg | 2048 |
| csa-k1 | 65 | 80 |
| csa-k2 | 66 | 96 |
| csa-k4 | 68 | 128 |
| csa-k8 | 72 | 192 |
| csa-k16 | 80 | 320 |

Coverage is not FLOPs. It tells us how much old text the compressed summaries
represent.

## Result A: Same Training Tokens

All runs below saw `204,800` training tokens.

This view asks:

```text
If every run reads the same text budget, what happens?
```

It does not ask:

```text
Which run used the same compute?
```

Time columns:

```text
active sec = measured trainer time
total wall sec = setup + training + final evaluation
harness sec = subprocess elapsed time from the sweep launcher
```

| Run | top_k | Val loss | PPL | Tokens seen | Tok/s | Active sec | Total wall sec | Harness sec | Peak alloc GiB | Peak reserved GiB | Attention vectors | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 7.0048 | 1101.90 | 204,800 | 21032.8 | 9.7 | 10.5 | 39.9 | 6.18 | 6.94 | 1024 | 2048 |
| csa-k1 | 1 | 6.9797 | 1074.56 | 204,800 | 12794.4 | 16.0 | 16.7 | 25.8 | 6.83 | 7.64 | 65 | 80 |
| csa-k2 | 2 | 6.9804 | 1075.37 | 204,800 | 12908.6 | 15.9 | 16.7 | 26.0 | 6.85 | 7.64 | 66 | 96 |
| csa-k4 | 4 | 6.9850 | 1080.32 | 204,800 | 12814.5 | 16.0 | 16.8 | 26.0 | 6.87 | 7.66 | 68 | 128 |
| csa-k8 | 8 | 6.9828 | 1077.96 | 204,800 | 12832.9 | 16.0 | 16.7 | 26.1 | 6.91 | 7.71 | 72 | 192 |
| csa-k16 | 16 | 6.9688 | 1062.93 | 204,800 | 13146.2 | 15.6 | 16.3 | 26.1 | 7.00 | 7.80 | 80 | 320 |

Early read:

```text
At the same token budget, CSA is competitive with dense in this tiny pilot.
csa-k16 is best among these six rows.
```

But this is not the headline result. CSA and dense spent different amounts of
wall-clock time.

## Result B: Same GPU Time

This is the fairer comparison.

Each run gets about `300` active training seconds.

Completed rows:

| Run | top_k | Val loss | PPL | Tokens seen | Tok/s | Active sec | Total wall sec | Harness sec | Peak alloc GiB | Peak reserved GiB | Attention vectors | Coverage | Stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | - | 4.3713 | 79.15 | 33,275,904 | 109930.2 | 302.7 | 303.4 | 312.8 | 6.18 | 6.94 | 1024 | 2048 | max_train_seconds |
| csa-k1 | 1 | 5.4255 | 227.12 | 13,500,416 | 44373.0 | 304.2 | 305.1 | 314.9 | 6.83 | 7.64 | 65 | 80 | max_train_seconds |
| csa-k2 | 2 | 5.4255 | 227.13 | 13,524,992 | 44441.9 | 304.3 | 305.2 | 314.5 | 6.85 | 7.64 | 66 | 96 | max_train_seconds |
| csa-k4 | 4 | 5.4414 | 230.77 | 13,344,768 | 43847.6 | 304.3 | 305.2 | 314.6 | 6.87 | 7.66 | 68 | 128 | max_train_seconds |
| csa-k8 | 8 | 5.4261 | 227.25 | 14,057,472 | 46198.1 | 304.3 | 305.1 | 314.0 | 6.91 | 7.71 | 72 | 192 | max_train_seconds |
| csa-k16 | 16 | 5.4287 | 227.85 | 13,574,144 | 44626.6 | 304.2 | 305.1 | 313.8 | 7.00 | 7.80 | 80 | 320 | max_train_seconds |

Early read:

```text
At equal GPU time, dense is much better than the completed CSA rows.
```

That is expected for `csa-k1`: it has only `80` raw-token-equivalent coverage,
while dense can use the full causal context.

The surprising early signal is that more `top_k` does not yet improve quality.
`csa-k16` increases coverage from `80` to `320`, but all CSA validation losses
stay clustered around `5.43`.

The important next question is:

```text
Why does extra compressed-history coverage not improve validation loss in this
budget?
```

## Current Interpretation

The first result says:

```text
CSA can look strong when every setting sees the same number of tokens.
```

The second result says:

```text
That is not enough. Under equal GPU time, CSA trains on fewer tokens and loses
to dense in this setup.
```

The completed CSA sweep adds one sharper concern:

```text
More coverage alone did not improve quality in the fixed-time CSA rows.
```

So the paper's real claim should not be:

```text
CSA beats dense.
```

The hypothesis we tested is:

```text
top_k is a controllable budget knob. Higher top_k should recover quality by
increasing compressed history coverage, but it may cost memory and throughput.
```

This pilot does not support that hypothesis yet.

## Limitations

- One seed.
- Small model.
- Short training.
- Plain PyTorch, not optimized sparse kernels.
- Sequence length is `2048`, not million-token context.
- The compute-normalized sweep is complete, but still only one pilot sweep.

## Next Checks

1. Plot validation loss vs `top_k` for fixed GPU time.
2. Inspect the selector and compressor, because `top_k` did not help here.
3. Repeat only the most informative rows: dense, csa-k1, csa-k8, csa-k16.
4. Try a longer CSA-only run to see whether compressed summaries need more time.
5. Do not change multiple CSA knobs until this top_k result is understood.

## Provisional Claim

The safe claim today is:

```text
In a small plain-PyTorch CSA implementation, fixed-token results are not enough.
Under equal GPU time, increasing top_k from 1 to 16 did not recover quality in
this pilot.
```

The next research step is diagnostic, not bigger hype: find out whether the
selector/compressor is learning useful compressed summaries.
