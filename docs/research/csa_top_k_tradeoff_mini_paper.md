# A Small-Scale Study Of top_k In Compressed Sparse Attention

Status: pre-registered mini-paper draft.

This document starts the paper before the results exist. That is intentional.
Writing the question, method, and fairness rules first makes the experiment
harder to fool after the numbers arrive.

## Abstract

Compressed Sparse Attention (CSA) reduces attention cost by compressing old
tokens into summary blocks and selecting only a small number of those blocks for
each query. The DeepSeek-V4 paper reports the CSA architecture and the selected
large-model hyperparameters, but it does not publish a small-model top_k sweep
that shows the quality, speed, and memory tradeoff under limited compute.

We study that tradeoff in an 88M-parameter Transformer implementation of CSA.
Holding model size, dataset, sequence length, optimizer, and training budget
fixed, we vary only the number of selected compressed blocks, top_k. We compare
dense attention against CSA with top_k in {1, 2, 4, 8, 16}. We report validation
loss, throughput, peak VRAM, wall-clock time, and selected-token budget per
query.

Expected contribution: a practical rule for small-model CSA, such as "quality
improves with top_k but saturates after X under this compute budget."

## 1. Question

What is the quality vs compute/memory tradeoff when we vary top_k in Compressed
Sparse Attention?

The specific question is:

```text
For a small 88M model, how much validation loss improvement do we get
as top_k increases, and how much throughput/VRAM do we pay for it?
```

This is not a claim that we can reproduce DeepSeek-V4 scale. It is a small,
controlled measurement of one CSA knob.

## 2. Is This Already Answered By The DeepSeek-V4 Paper?

Partly, but not in the form we need for a seven-day tutorial paper.

The DeepSeek-V4 paper does report chosen CSA settings:

- Flash model: compression rate m = 4 and attention top-k = 512.
- Pro model: compression rate m = 4 and attention top-k = 1024.
- The paper also says a smaller attention top-k than DeepSeek-V3.2 was chosen
  for efficiency on short- and medium-length text.

What the paper does not appear to publish:

- A small-model top_k sweep.
- A curve of validation loss vs top_k.
- A curve of throughput or peak VRAM vs top_k.
- A compute-normalized comparison such as validation loss per GPU-hour.
- A seven-day reproducible teaching setup using a plain PyTorch CSA
  implementation.

So our question is not "what top_k did DeepSeek use?" They answered that.

Our question is:

```text
When a learner implements CSA in a small LLM, what top_k tradeoff do they
actually observe under a fixed compute budget?
```

That is a good tutorial research question because it is narrow, measurable, and
honest.

## 3. Hypotheses

Hypothesis 1:

```text
Increasing top_k will improve validation loss because each query can read more
compressed history.
```

Hypothesis 2:

```text
The improvement will saturate. top_k=16 should not be 16x better than top_k=1.
```

Hypothesis 3:

```text
In plain PyTorch, CSA may be slower than dense attention at short sequence
lengths because compression, top-k selection, and gather operations add overhead.
```

Hypothesis 3 is important. We are measuring the architecture behavior, not
claiming production kernel speed.

## 4. Model And Hardware

Base model:

```text
d_model = 512
n_heads = 8
n_layers = 22
d_ff = 2048
max_seq_len = 2048
parameters ~= 88M
```

Recommended hardware for the seven-day version:

```text
Best practical default: 1x RTX 4090 or 1x RTX 5090
Minimum acceptable:    1x RTX 3090
If renting for margin: 1x 48GB card such as L40S / RTX 6000 Ada
```

Why:

- RTX 3090 has 24GB VRAM and can run the study, but it is the slowest option.
- RTX 4090 also has 24GB VRAM but is much faster, so it is the best
  cost/performance choice if the model fits.
- RTX 5090 has 32GB VRAM, which gives more room for batch size, longer pilots,
  and failed experiments. If price/rental access is reasonable, it is the best
  single consumer GPU choice.
- A 48GB workstation/datacenter card is useful if the goal shifts toward longer
  context or larger batches, but it is not required for the first paper.

## 5. Fairness Rules

top_k increases compute because each query attends to more selected compressed
blocks. Therefore one fairness rule is not enough.

We use two views.

### View A: Fixed Training Tokens

Every run trains on the same number of tokens.

This answers:

```text
If every model sees the same data, which attention setting learns better?
```

This is the cleanest architecture comparison.

Fixed:

- model size
- dataset
- tokenizer
- sequence length
- training tokens
- optimizer
- learning rate schedule
- batch size if it fits
- seed if possible

Changed:

- attention implementation
- top_k for CSA runs

### View B: Fixed GPU-Hours

Every run gets the same wall-clock GPU budget.

This answers:

```text
If I only have N GPU-hours, which setting gives the best validation loss?
```

This is the practical compute comparison.

If time is tight, run View A first. If the signal is interesting, add View B for
the strongest two or three settings.

## 6. Selected-Token Budget

For a CSA run, the rough selected-token budget per query is:

```text
budget = top_k * m + w
```

where:

- m is the compression block size.
- w is the sliding window size.
- top_k is the number of compressed blocks selected.

Example with m = 16 and w = 64:

| top_k | Selected-token budget |
| ---: | ---: |
| 1 | 80 |
| 2 | 96 |
| 4 | 128 |
| 8 | 192 |
| 16 | 320 |

Dense attention at sequence length 2048 can read up to 2048 previous-token
positions. CSA is intentionally reading a much smaller set.

## 7. Experiment Matrix

Primary sweep:

| Run | Attention | top_k | m | w | Train tokens | Seed | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense | dense | all | n/a | n/a | TBD | 1 | baseline |
| csa-k1 | CSA | 1 | 16 | 64 | TBD | 1 | strongest compression |
| csa-k2 | CSA | 2 | 16 | 64 | TBD | 1 | |
| csa-k4 | CSA | 4 | 16 | 64 | TBD | 1 | |
| csa-k8 | CSA | 8 | 16 | 64 | TBD | 1 | default-ish |
| csa-k16 | CSA | 16 | 16 | 64 | TBD | 1 | highest CSA budget |

If seven days are tight, use one seed for all runs. If the signal is promising,
repeat the best, worst, and baseline runs with seeds 2 and 3.

## 8. Metrics

Record these for every run:

| Metric | Why it matters |
| --- | --- |
| validation loss | primary quality metric |
| final perplexity | easier to interpret than loss for readers |
| tokens/sec | practical training speed |
| peak VRAM | memory cost |
| wall-clock time | actual compute spent |
| selected-token budget | architecture budget |
| run command | reproducibility |
| git commit | exact code state |
| dataset path/hash | exact data state |

Do not report only validation loss. CSA is a tradeoff paper.

## 9. Commands

Dense baseline:

```bash
python train_llm.py \
  --attention_impl dense \
  --train_tokens 8000000 \
  --batch_size 8
```

CSA example:

```bash
python train_llm.py \
  --attention_impl csa \
  --csa_compression_block_size 16 \
  --csa_top_k 4 \
  --csa_sliding_window_size 64 \
  --csa_indexer_heads 4 \
  --csa_output_groups 1 \
  --train_tokens 8000000 \
  --batch_size 8
```

The first real run should be a short smoke test, not the full sweep:

```bash
python train_llm.py \
  --attention_impl csa \
  --csa_top_k 1 \
  --train_tokens 200000 \
  --batch_size 4
```

## 10. Results

Fill this after runs finish.

| Run | Val loss | PPL | Tokens/sec | Peak VRAM | Wall time | Selected budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | | | | | | 2048 |
| csa-k1 | | | | | | 80 |
| csa-k2 | | | | | | 96 |
| csa-k4 | | | | | | 128 |
| csa-k8 | | | | | | 192 |
| csa-k16 | | | | | | 320 |

Charts to make:

1. Validation loss vs top_k.
2. Tokens/sec vs top_k.
3. Peak VRAM vs top_k.
4. Validation loss vs selected-token budget.
5. Validation loss vs GPU-hours.

The last chart is the fairness chart.

## 11. Interpretation Template

Use this template after results land:

```text
Increasing top_k from A to B improved validation loss from X to Y, but the gain
from B to C was small. Throughput decreased from P to Q tokens/sec and peak VRAM
increased from R to S GB. Under this budget, top_k=B appears to be the knee of
the curve.
```

If CSA loses to dense attention, say that clearly:

```text
Dense attention still achieved the best validation loss at this sequence length.
However, CSA showed a smooth quality recovery as top_k increased, suggesting the
selector and compressor are functioning. The next experiment should test longer
contexts or better sparse kernels, not claim a speed win at small scale.
```

Negative results are still useful if they teach the tradeoff honestly.

## 12. Seven-Day Schedule

Day 1:

- Run CPU tests.
- Run one dense smoke test and one CSA smoke test.
- Confirm logs capture validation loss, tokens/sec, and wall time.

Day 2:

- Run dense baseline.
- Run csa-k1 and csa-k4.

Day 3:

- Run csa-k2, csa-k8, and csa-k16.
- If one setting crashes from VRAM, reduce batch size and record it.

Day 4:

- Make the first table.
- Make the first two charts.
- Decide whether the signal is worth repeating with extra seeds.

Day 5:

- Repeat only the most informative runs, not everything.
- Suggested repeats: dense, best CSA, worst CSA.

Day 6:

- Write results and discussion.
- Add limitations.
- Add exact commands and hardware.

Day 7:

- Polish the paper.
- Write the tutorial takeaway.
- Choose the next experiment.

## 13. Limitations

This mini-paper does not prove DeepSeek-scale speedups.

Main limitations:

- The model is small.
- The sequence length is much shorter than 1M.
- The implementation uses clear PyTorch, not custom sparse CUDA kernels.
- One seed is not enough for a final scientific claim.
- Fixed-token and fixed-GPU-hour comparisons answer different questions.

That is okay. The paper is valuable if it produces one honest rule under one
well-described budget.

## 14. Expected Final Claim Shape

The final claim should look like this:

```text
On an 88M model trained for X tokens at sequence length 2048, CSA quality
improved as top_k increased from 1 to K, but returns saturated after K. The
best quality/compute tradeoff was top_k=K because it recovered Y% of the dense
baseline quality while using Z selected-token budget and W GPU-hours.
```

Do not write:

```text
CSA is better than dense attention.
```

Write:

```text
Under this small-compute setup, top_k controls a measurable quality/compute
tradeoff, and the knee of the curve was around K.
```

