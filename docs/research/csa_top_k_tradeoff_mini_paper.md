# A Small-Scale Study Of top_k In Compressed Sparse Attention

Status: living mini-paper. Results are early and should be treated as pilot
evidence, not a final claim.

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

| Run | top_k | Val loss | PPL | Tok/s | Peak alloc GiB | Attention vectors | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 7.0048 | 1101.90 | 21032.8 | 6.18 | 1024 | 2048 |
| csa-k1 | 1 | 6.9797 | 1074.56 | 12794.4 | 6.83 | 65 | 80 |
| csa-k2 | 2 | 6.9804 | 1075.37 | 12908.6 | 6.85 | 66 | 96 |
| csa-k4 | 4 | 6.9850 | 1080.32 | 12814.5 | 6.87 | 68 | 128 |
| csa-k8 | 8 | 6.9828 | 1077.96 | 12832.9 | 6.91 | 72 | 192 |
| csa-k16 | 16 | 6.9688 | 1062.93 | 13146.2 | 7.00 | 80 | 320 |

Early read:

```text
At the same token budget, CSA is competitive with dense in this tiny pilot.
csa-k16 is best among these six rows.
```

But this is not the headline result. CSA and dense spent different amounts of
wall-clock time.

## Result B: Same GPU Time

This is the fairer comparison.

Each run gets about `300` active training seconds. This sweep is still running.

Completed rows so far:

| Run | top_k | Val loss | PPL | Tokens seen | Tok/s | Active sec | Peak alloc GiB | Attention vectors | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 4.3713 | 79.15 | 33,275,904 | 109930.2 | 302.7 | 6.18 | 1024 | 2048 |
| csa-k1 | 1 | 5.4255 | 227.12 | 13,500,416 | 44373.0 | 304.2 | 6.83 | 65 | 80 |
| csa-k16 | 16 | 5.4287 | 227.85 | 13,574,144 | 44626.6 | 304.2 | 7.00 | 80 | 320 |

Early read:

```text
At equal GPU time, dense is much better than csa-k1 and csa-k16.
```

That is expected for `csa-k1`: it has only `80` raw-token-equivalent coverage,
while dense can use the full causal context.

The surprising early signal is that `csa-k16` does not yet improve over
`csa-k1`, even though it increases coverage from `80` to `320`.

The important next question is:

```text
Do the middle values k=2, k=4, and k=8 show any recovery, or is this CSA setup
not learning useful compressed summaries at this budget?
```

## Current Interpretation

The first result says:

```text
CSA can look strong when every setting sees the same number of tokens.
```

The second result says:

```text
That is not enough. Under equal GPU time, the strongest compression setting
trains on fewer tokens and currently loses to dense.
```

The early `k=16` result adds one sharper concern:

```text
More coverage alone did not improve quality in the first completed fixed-time
CSA rows.
```

So the paper's real claim should not be:

```text
CSA beats dense.
```

The claim we are testing is:

```text
top_k is a controllable budget knob. Higher top_k should recover quality by
increasing compressed history coverage, but it may cost memory and throughput.
```

## Limitations

- One seed.
- Small model.
- Short training.
- Plain PyTorch, not optimized sparse kernels.
- Sequence length is `2048`, not million-token context.
- The compute-normalized sweep is still incomplete.

## Next Checks

1. Finish the 300-second fixed-time sweep.
2. Plot validation loss vs `top_k` for fixed GPU time.
3. Repeat only the most informative rows: dense, csa-k1, best CSA.
4. If CSA does not improve with `top_k`, inspect the selector and compressor.
5. If CSA improves smoothly, increase training budget before changing any other knob.

## Provisional Claim

The safe claim today is:

```text
In a small plain-PyTorch CSA implementation, fixed-token results are not enough.
The meaningful question is whether larger top_k recovers quality under equal
GPU time.
```

That is exactly what the current 300-second sweep is measuring.
