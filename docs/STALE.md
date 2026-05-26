# Stale Files

These files are not used by the current paper pipeline (`forgetting_scaling_sweep.py`, `forgetting_scaling_plot.py`, `test_forgetting_mechanisms.py`, or `paper.pdf`). They are kept in the repo because they are still imported through `models/__init__.py` or `train_llm.py` and removing them would require a refactor pass.

Treat anything below as legacy until the refactor lands. If you fork this repo for a new experiment, you almost certainly do not need to read or touch these.

## Stale experiment scripts

Old, superseded sweeps. Use `experiments/forgetting_scaling_sweep.py` instead.

- `experiments/arch_compare_plot.py`
- `experiments/attention_memory_experiment.py`
- `experiments/attention_policy_experiment.py`
- `experiments/csa_metrics.py`
- `experiments/csa_top_k_sweep.py`
- `experiments/memory_policy_experiment.py`

## Stale configs

- `configs/csa_config.py` — only used by the deleted CSA-only sweeps.
- `configs/memory_policy_config.py` — predecessor of `configs/forgetting_config.py`.

## Stale model modules

Still wired through `models/__init__.py` so they cannot be deleted without an import-graph cleanup, but the current paper does not exercise them.

- `models/forgetting_attention.py` — thin alias kept for backward compatibility with `train_llm.py`.
- `models/novel_attention.py` — exploratory variants (Hebbian, cross-block residual, negative memory, layer decay, multi-resolution) not in the paper.

The age-forgetting variants beyond the linear gate inside `models/memory_policies.py` (exponential, sigmoid, cosine, reciprocal, hard-cutoff) and the `RandomKeyframeAttention` / `PeriodicKeyframeAttention` / `SalienceMemoryAttention` / `UsageRefreshAttention` / `CompetitionMemoryAttention` / `LearnedRouterAttention` classes are also not in the paper sweep, but live in the same file as the active policies and cannot be removed individually.

## Stale tests

These exercise the stale modules above. They still pass and provide regression coverage, but they are not part of the paper's correctness suite.

- `tests/test_compressed_sparse_attention.py`
- `tests/test_forgetting_attention.py`
- `tests/test_memory_policies.py`
- `tests/test_synthetic_data.py`

The paper correctness grid is `tests/test_forgetting_mechanisms.py`.

## Unrelated to the paper

- `benchmarks/` — ARC, GSM8K, HellaSwag eval harnesses. Useful for downstream eval of a trained model but unused by the paper pipeline.
- `train_llm.py` — full LLM training script. Independent of the forgetting sweep; kept for users who want a baseline training entry point.
