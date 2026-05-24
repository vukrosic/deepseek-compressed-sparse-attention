# CSA Pilot Report: Can Compressed Attention Compete Under Equal GPU Time?

Date: 2026-05-24

Repository: `vukrosic/deepseek-compressed-sparse-attention`

## Executive Summary

This report summarizes a small pilot study of the repo's plain-PyTorch
Compressed Sparse Attention implementation.

The headline result is simple:

```text
Under equal GPU time, dense attention clearly won in this pilot.
```

CSA looked competitive when every run saw the same number of training tokens,
but that comparison was not compute-fair. When every run received about 300
active training seconds on the same RTX 3090, dense attention processed far more
tokens and achieved much lower validation loss.

This does not disprove CSA. It says this first research implementation has not
yet turned compressed history into a quality or speed win.

## Experiment

Question:

```text
If CSA can read compressed summaries of old tokens, does increasing top_k
recover quality under a fixed GPU-time budget?
```

Settings:

```text
dense
csa-k1
csa-k2
csa-k4
csa-k8
csa-k16
```

Hardware:

```text
1x NVIDIA RTX 3090, 24GB VRAM
```

Fairness rule:

```text
Same model, data, batch size, sequence length, optimizer, seed, hardware,
and active training-time budget.
```

VRAM was measured as an outcome. It was not reused to give any method a larger
batch size.

## Fixed-Time Result

All rows used about 300 active training seconds.

| Run | Val loss | Tokens seen | Tokens/sec | Peak alloc GiB | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 4.3713 | 33,275,904 | 109,930 | 6.18 | 2048 |
| csa-k1 | 5.4255 | 13,500,416 | 44,373 | 6.83 | 80 |
| csa-k2 | 5.4255 | 13,524,992 | 44,442 | 6.85 | 96 |
| csa-k4 | 5.4414 | 13,344,768 | 43,848 | 6.87 | 128 |
| csa-k8 | 5.4261 | 14,057,472 | 46,198 | 6.91 | 192 |
| csa-k16 | 5.4287 | 13,574,144 | 44,627 | 7.00 | 320 |

## Interpretation

The result is not close:

```text
dense val loss: ~4.37
CSA val loss:   ~5.43
```

Increasing `top_k` did not produce a recovery curve. CSA coverage rose from 80
raw-token equivalents to 320, but validation loss stayed nearly flat.

The most likely explanations are:

1. Plain PyTorch CSA has too much implementation overhead.
2. Dense attention benefits from highly optimized kernels.
3. The small model or short run is not the regime where CSA helps.
4. The selector or compressor may not yet be learning useful compressed memory.

## Debug-Only Simple Experiment

A new simpler experiment was implemented after this pilot:

```text
dense      = read the full causal past
local      = read nearby tokens only
csa        = read nearby tokens plus compressed old memory
```

The current Mac run is only a smoke test. It proves the harness works; it is not
evidence for the paper claim.

## Conclusion

The honest research result is:

```text
Dense attention beats this plain-PyTorch CSA implementation under equal GPU time.
```

The next clean experiment is:

```text
Run dense vs local vs CSA on NVIDIA for 3 seeds, using the same GPU-time budget.
```

That is simpler to teach, easier to explain, and closer to the real question:

```text
Does compressed memory help beyond local attention?
```
