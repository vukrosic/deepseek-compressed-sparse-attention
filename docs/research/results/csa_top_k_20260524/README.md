# CSA top_k Results Bundle

This folder archives the run evidence before the GPU instance is shut down.

## Files

- `all_runs_summary.csv`: flat table for spreadsheet analysis.
- `all_runs_summary.json`: same rows as structured JSON.
- `remote_environment.txt`: GPU, driver, CUDA, PyTorch, and commit metadata.
- `raw_runs/csa_top_k/...`: copied manifests, metrics JSON, and stdout logs. Checkpoints are intentionally excluded.
- `fixed_time_val_loss_vs_top_k.png`: loss curve for the compute-normalized sweep.
- `fixed_time_tokens_seen_vs_top_k.png`: tokens processed under equal GPU time.

## Main Run IDs

- `20260524_054102`: `synthetic_mac_style_gpu_smoke_8192_tokens`
- `20260524_054523`: `real_data_smoke_50k_tokens`
- `20260524_055350`: `fixed_tokens_204800`
- `20260524_055859`: `fixed_time_300s`

## Fixed Tokens Result

All rows saw `204,800` training tokens.

| Run | top_k | Val loss | Tokens/sec | Active sec | Total wall sec | Peak alloc GiB | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 7.0048 | 21032.8 | 9.7 | 10.5 | 6.18 | 2048 |
| csa-k1 | 1 | 6.9797 | 12794.4 | 16.0 | 16.7 | 6.83 | 80 |
| csa-k2 | 2 | 6.9804 | 12908.6 | 15.9 | 16.7 | 6.85 | 96 |
| csa-k4 | 4 | 6.9850 | 12814.5 | 16.0 | 16.8 | 6.87 | 128 |
| csa-k8 | 8 | 6.9828 | 12832.9 | 16.0 | 16.7 | 6.91 | 192 |
| csa-k16 | 16 | 6.9688 | 13146.2 | 15.6 | 16.3 | 7.00 | 320 |

## Fixed GPU Time Result

All rows used about `300` active training seconds.

| Run | top_k | Val loss | Tokens seen | Tokens/sec | Active sec | Total wall sec | Peak alloc GiB | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | - | 4.3713 | 33,275,904 | 109930.2 | 302.7 | 303.4 | 6.18 | 2048 |
| csa-k1 | 1 | 5.4255 | 13,500,416 | 44373.0 | 304.2 | 305.1 | 6.83 | 80 |
| csa-k2 | 2 | 5.4255 | 13,524,992 | 44441.9 | 304.3 | 305.2 | 6.85 | 96 |
| csa-k4 | 4 | 5.4414 | 13,344,768 | 43847.6 | 304.3 | 305.2 | 6.87 | 128 |
| csa-k8 | 8 | 5.4261 | 14,057,472 | 46198.1 | 304.3 | 305.1 | 6.91 | 192 |
| csa-k16 | 16 | 5.4287 | 13,574,144 | 44626.6 | 304.2 | 305.1 | 7.00 | 320 |

## Review Takeaway

The fixed-token sweep made CSA look competitive. The fixed-time sweep reversed that story: dense trained on many more tokens in the same wall-clock budget and achieved much lower validation loss. Increasing CSA `top_k` from `1` to `16` did not produce a recovery curve in this pilot.
