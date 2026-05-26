# Results

The completed paper sweep is summarized in:

- Paper PDF: [research/reports/arch_compare_20260526.pdf](research/reports/arch_compare_20260526.pdf)
- LaTeX source: [research/reports/arch_compare_20260526.tex](research/reports/arch_compare_20260526.tex)
- Final result plots/table: [research/results/forgetting_scaling_latest](research/results/forgetting_scaling_latest)

## Sweep

```text
hardware: 1x RTX 3090, 24GB
model: 18M-class transformer
contexts: 2048, 4096, 8192
budget: 300 active training seconds per run
runs: 33/33 complete
failures: 0
```

## Main Finding

Dense and local attention dominate this unfused PyTorch reference sweep.

Among the compressed-memory policies, the simpler variants are strongest:

```text
age_forgetting
compressed_memory
hierarchical
surprise_retention
```

CSA, predictive routing, token merge, and recurrent state are weaker in this short-budget setup.

## Interpretation

This is not a claim that the memory policies cannot work at scale.

It says:

```text
the reference implementations are correct and runnable,
but the current unfused memory paths do not beat the simple local baseline.
```

The next useful step is either a fused kernel or a synthetic long-memory retrieval task.

