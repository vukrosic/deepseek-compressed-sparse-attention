# Compressed Sparse Attention Research Notes

This repo keeps the original dense attention path as the baseline and adds CSA as an opt-in
attention implementation.

## Code Map

- `models/compressed_sparse_attention.py`
  - `TokenCompressor`: equations 9-12 from the DeepSeek-V4 paper.
  - `LightningIndexer`: equations 13-17.
  - `CompressedSparseAttention`: sliding-window entries plus selected compressed entries, then shared key-value MQA from equations 18-19.
  - `GroupedOutputProjection`: grouped output projection from page 11.
- `configs/csa_config.py`
  - Research knobs for block size, top-k selection, local window size, indexer dimensions, and grouped projection.
- `configs/llm_config.py`
  - `attention_impl="dense"` keeps the original baseline.
  - `attention_impl="csa"` enables CSA.
- `tests/test_compressed_sparse_attention.py`
  - CPU checks for the compressor equation, causal block selection, gradients, and a tiny model forward pass.

## Launch Examples

Dense baseline:

```bash
python train_llm.py --attention_impl dense
```

CSA research run:

```bash
python train_llm.py \
  --attention_impl csa \
  --csa_compression_block_size 16 \
  --csa_top_k 8 \
  --csa_sliding_window_size 64 \
  --csa_indexer_heads 4 \
  --csa_output_groups 1
```

## Notes

The implementation is intentionally written in plain PyTorch gathers/top-k operations so it can be
debugged on CPU/MPS first. For NVIDIA experiments, the likely next step is replacing the gather,
top-k, and per-token MQA path with fused kernels while preserving the same module interface and
debug outputs.
