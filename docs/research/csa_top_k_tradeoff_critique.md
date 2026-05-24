# Critique Of The CSA top_k Mini Paper

This is a useful pilot paper, but not yet a strong research paper.

Its best quality is honesty: it separates fixed-token results from fixed-time
results, reports a negative signal, and avoids claiming that CSA beats dense
attention.

Its biggest weakness is that the current result does not yet tell us whether
CSA is weak, whether this implementation is weak, or whether the experiment is
too small for `top_k` to matter.

## What Works

The question is narrow.

```text
What happens when top_k changes?
```

That is the right size for a seven-day research project. It changes one knob,
not the whole architecture.

The paper also makes the right fairness distinction:

```text
same training tokens != same compute
```

That single distinction saves the paper from a common mistake. In the
fixed-token run, CSA looks competitive. In the fixed-time run, dense wins
clearly. The paper does not hide that.

The tables are also useful because they include:

- validation loss
- tokens seen
- active training time
- total wall time
- throughput
- peak allocated VRAM
- peak reserved VRAM
- attention-vector budget
- raw-token coverage

That makes the result inspectable instead of decorative.

## Major Problems

### 1. The Experiment Is Too Small For A Strong Claim

The model is about `18.4M` parameters, the sequence length is `2048`, and the
training budget is short.

That is fine for a pilot. It is not enough to say much about DeepSeek-scale CSA.

The safe wording is:

```text
In this small plain-PyTorch pilot, top_k did not recover quality under equal GPU
time.
```

The unsafe wording is:

```text
CSA does not work.
```

The current paper mostly avoids the unsafe claim, which is good.

### 2. Fixed GPU Time Is Practical, But Not A True FLOPs Control

The fixed-time sweep is the right headline comparison for a rented-GPU tutorial.
But it is still not a perfect compute comparison.

GPU time mixes together:

- attention math
- Python overhead
- gather/top-k overhead
- dense kernel efficiency
- data loading
- final evaluation
- PyTorch memory behavior

So the result tells us:

```text
What result did this code produce for the same GPU-time budget?
```

It does not tell us:

```text
What result would an optimized CSA kernel produce at the same FLOPs?
```

The paper should keep saying "plain PyTorch implementation" loudly.

### 3. The Dense Baseline Has A Large Kernel Advantage

Dense attention is likely using heavily optimized PyTorch paths. CSA is written
as clear PyTorch logic with compression, indexing, top-k, and gathers.

That means dense may win partly because it is more optimized, not only because
full attention is better.

This is not fatal. It just changes the claim.

Good claim:

```text
Under this repo's plain PyTorch implementation, CSA did not beat dense under
equal GPU time.
```

Bad claim:

```text
Dense attention is architecturally better than CSA.
```

### 4. top_k Did Not Move The Curve

This is the most interesting result and the biggest warning.

Under equal GPU time, all CSA losses cluster near `5.43`:

```text
k1   5.4255
k2   5.4255
k4   5.4414
k8   5.4261
k16  5.4287
```

If `top_k` were working as the main budget knob, we would expect some visible
quality recovery as coverage rises from `80` to `320`.

We did not see that.

Possible explanations:

- The model is too small.
- The run is too short.
- The compressed summaries are not useful yet.
- The selector is not learning useful block choices.
- The implementation has a bug or mismatch with the paper.
- The task at this scale does not need the older compressed context.
- The local window dominates the useful signal.

The next experiment should diagnose this before scaling.

### 5. The Paper Needs Selector And Compressor Evidence

Right now, the paper reports loss and speed. It does not show whether CSA is
actually using its sparse pathway meaningfully.

Before making a stronger claim, log:

- selected block histogram
- selected block distance from the current token
- top-k score entropy
- fraction of repeated selected blocks
- compressor output norm
- indexer gradient norm
- attention mass on local tokens vs compressed summaries

If the selector always chooses near-local blocks, or if scores collapse, then
`top_k` cannot help much.

### 6. The Data Story Is Under-Specified

The paper names `processed_data/speedrun_40M`, but it should record more.

Add:

- dataset source
- exact download command
- number of examples
- validation split rule
- tokenizer
- dataset hash or artifact path

The current setup is probably deterministic enough for a pilot, but a reader
cannot fully reproduce the data state from the paper alone.

### 7. There Is Only One Seed

At this scale, one seed is fragile.

The current fixed-time result is large enough that dense beating CSA is probably
real for this setup. But the exact ordering among CSA rows is not meaningful
yet.

The paper should not interpret:

```text
k4 is worse than k1
```

It should interpret:

```text
CSA rows are roughly tied, and top_k did not produce a clear recovery curve.
```

### 8. The Paper Still Needs Charts

Tables are necessary, but the main result wants one simple plot:

```text
validation loss vs top_k under fixed GPU time
```

The second useful plot:

```text
tokens seen vs top_k under fixed GPU time
```

Together, those show the core story:

```text
CSA sees fewer tokens in the same time, and increasing top_k does not recover
loss in this pilot.
```

## Publication Risk

The current paper is good as:

```text
a research diary
a tutorial artifact
a first pilot result
```

It is not yet good as:

```text
a claim about CSA generally
a claim about DeepSeek's architecture
a claim about optimized sparse attention
```

The safest public framing is:

```text
I implemented a plain-PyTorch version of CSA and ran a first top_k sweep. The
fixed-token view looked misleadingly good, but the fixed-time view showed dense
winning clearly. More importantly, top_k did not move the CSA curve. The next
step is to inspect whether the selector and compressor are actually learning.
```

That is a strong honest takeaway.

## Best Next Revision

Do not make the paper longer.

Improve it by adding three things:

1. A fixed-time loss-vs-top_k plot.
2. A short selector/compressor diagnostics table.
3. A tighter final claim that says this is a pilot diagnostic result.

The best next technical experiment is:

```text
Hold CSA fixed at k=1 and k=16.
Log selected-block behavior and compressed-summary attention.
Train long enough to see whether the selector starts using distant compressed
history.
```

If `k=16` still does not improve after that, the result becomes much more
interesting.
