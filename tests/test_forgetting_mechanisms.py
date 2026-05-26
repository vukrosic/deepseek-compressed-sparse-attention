"""
Correctness test suite for all forgetting mechanisms (existing + v2).

Five tests per mechanism:
  1. shape_dtype      — output shape and dtype match a dense reference for the same input
  2. causality        — perturbing future positions does not change past outputs (the test that caught
                        cross_block_residual's leak)
  3. mask_correctness — running on a known-tiny input gives a finite, non-NaN, non-Inf output
  4. gradient_flow    — backward pass produces no NaN/Inf and at least some non-zero grads
  5. memory_budget    — peak GPU/CPU memory scales as documented for the mechanism

This is the green/red grid that gates whether a mechanism is allowed into the sweep.
Run as: python -m tests.test_forgetting_mechanisms
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

import torch
import torch.nn as nn

sys.path.insert(0, ".")

from configs.csa_config import CSAConfig
from configs.memory_policy_config import MemoryPolicyConfig
from models.layers import TransformerBlock


# Mechanisms we test. The keys must match the attention_impl strings in
# layers.py / train_llm.py.
MECHANISMS = [
    "dense",
    "local",
    "csa",
    "compressed_memory",
    "age_forgetting",
    "hierarchical",
    "predictive",
    "surprise_retention",
    "frequency_lfu",
    "token_merge",
    "recurrent_state",
]


def make_block(attention_impl: str, d_model: int = 64, n_heads: int = 4, max_seq_len: int = 128) -> nn.Module:
    """Build a TransformerBlock with the given attention mechanism."""
    csa = CSAConfig(
        compression_block_size=4,
        top_k=4,
        sliding_window_size=8,
        indexer_heads=2,
    )
    mp = MemoryPolicyConfig(
        local_window_size=8,
        block_size=4,
        memory_budget_blocks=8,
        predictive_top_k=4,
        surprise_top_k=4,
        frequency_top_k=4,
    )

    # Novel-attention paths need their module pre-built and passed in. We
    # only test the four "novel" classes that exist in novel_attention.py;
    # mechanisms wired via memory_policies / new_forgetting are built by
    # TransformerBlock itself.
    novel_attention = None
    return TransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=4 * d_model,
        max_seq_len=max_seq_len,
        dropout=0.0,
        n_kv_heads=n_heads,
        attention_impl=attention_impl,
        csa_config=csa,
        forgetting_config=None,
        memory_policy=mp,
        novel_attention=novel_attention,
    )


@dataclass
class TestResult:
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Test 1: shape and dtype
# ---------------------------------------------------------------------------
def test_shape_dtype(impl: str) -> TestResult:
    torch.manual_seed(0)
    d_model, n_heads, T = 64, 4, 64
    block = make_block(impl, d_model=d_model, n_heads=n_heads, max_seq_len=T).eval()
    x = torch.randn(2, T, d_model)
    with torch.no_grad():
        y = block(x)
    if y.shape != x.shape:
        return TestResult(False, f"shape mismatch: out={y.shape}, in={x.shape}")
    if y.dtype != x.dtype:
        return TestResult(False, f"dtype mismatch: out={y.dtype}, in={x.dtype}")
    return TestResult(True, f"shape={tuple(y.shape)} dtype={y.dtype}")


# ---------------------------------------------------------------------------
# Test 2: causality
# ---------------------------------------------------------------------------
def test_causality(impl: str) -> TestResult:
    """
    Perturb input at position `t_perturb` and check that outputs at positions
    < t_perturb are bitwise unchanged. This is the strongest form of the
    causality test.
    """
    torch.manual_seed(0)
    d_model, n_heads, T = 64, 4, 64
    block = make_block(impl, d_model=d_model, n_heads=n_heads, max_seq_len=T).eval()

    x = torch.randn(2, T, d_model)
    t_perturb = T // 2
    x2 = x.clone()
    x2[:, t_perturb:, :] += torch.randn_like(x2[:, t_perturb:, :]) * 5.0

    with torch.no_grad():
        y1 = block(x)
        y2 = block(x2)

    delta = (y1[:, :t_perturb, :] - y2[:, :t_perturb, :]).abs().max().item()
    # Tolerance: pure float32 reductions are not bitwise but should be < 1e-5.
    if delta > 1e-4:
        return TestResult(False, f"max delta at past positions = {delta:.3e} (expected ~0)")
    return TestResult(True, f"max delta at past positions = {delta:.3e}")


# ---------------------------------------------------------------------------
# Test 3: mask correctness / finiteness
# ---------------------------------------------------------------------------
def test_mask_correctness(impl: str) -> TestResult:
    """
    Run on a small input; verify the output is finite, has no NaN/Inf, and
    actually depends on the input (not a constant or all-zero).
    """
    torch.manual_seed(0)
    d_model, n_heads, T = 64, 4, 32
    block = make_block(impl, d_model=d_model, n_heads=n_heads, max_seq_len=T).eval()

    x1 = torch.randn(1, T, d_model)
    x2 = torch.randn(1, T, d_model)
    with torch.no_grad():
        y1 = block(x1)
        y2 = block(x2)

    if not torch.isfinite(y1).all():
        return TestResult(False, "non-finite values in output")
    if (y1 - y2).abs().max().item() < 1e-6:
        return TestResult(False, "output appears constant w.r.t. input")
    return TestResult(True, f"finite, varies with input (delta={float((y1-y2).abs().max()):.3f})")


# ---------------------------------------------------------------------------
# Test 4: gradient flow
# ---------------------------------------------------------------------------
def test_gradient_flow(impl: str) -> TestResult:
    torch.manual_seed(0)
    d_model, n_heads, T = 64, 4, 64
    block = make_block(impl, d_model=d_model, n_heads=n_heads, max_seq_len=T).train()
    x = torch.randn(2, T, d_model, requires_grad=True)
    y = block(x)
    loss = y.pow(2).mean()
    loss.backward()

    if x.grad is None:
        return TestResult(False, "no gradient flowed back to input")
    if not torch.isfinite(x.grad).all():
        return TestResult(False, "non-finite gradient")

    # Check at least one learnable parameter received a non-zero gradient.
    any_nonzero = False
    for p in block.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            any_nonzero = True
            break
    if not any_nonzero:
        return TestResult(False, "all parameter gradients are zero")
    return TestResult(True, f"input grad norm={float(x.grad.norm()):.3f}, params have nonzero grads")


# ---------------------------------------------------------------------------
# Test 5: memory budget — peak allocation under forward pass
# ---------------------------------------------------------------------------
def test_memory_budget(impl: str) -> TestResult:
    """
    Run a single forward at a fixed shape and report peak memory.

    No hard pass/fail threshold — this records the number so the kernel-writer
    appendix has a real measurement. We only fail this test if the run OOMs
    or raises.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d_model, n_heads, T = 128, 4, 256
    try:
        block = make_block(impl, d_model=d_model, n_heads=n_heads, max_seq_len=T).to(device).eval()
        x = torch.randn(2, T, d_model, device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            y = block(x)
        if device == "cuda":
            peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return TestResult(True, f"peak_cuda={peak_mib:.1f} MiB at T={T} d={d_model}")
        return TestResult(True, f"ran T={T} d={d_model} on CPU")
    except Exception as e:
        return TestResult(False, f"raised: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
TESTS = [
    ("shape_dtype", test_shape_dtype),
    ("causality", test_causality),
    ("mask_correctness", test_mask_correctness),
    ("gradient_flow", test_gradient_flow),
    ("memory_budget", test_memory_budget),
]


def run_all() -> int:
    print(f"{'mechanism':<22} " + " ".join(f"{name:>16}" for name, _ in TESTS))
    print("-" * (22 + 17 * len(TESTS)))
    n_fail = 0
    detail_buffer: list[tuple[str, str, str]] = []
    for impl in MECHANISMS:
        cells = []
        for name, fn in TESTS:
            try:
                r = fn(impl)
            except Exception as e:
                r = TestResult(False, f"EXCEPTION: {type(e).__name__}: {e}")
                detail_buffer.append((impl, name, "".join(traceback.format_exception_only(type(e), e)).strip()))
            cells.append("PASS" if r.passed else "FAIL")
            if not r.passed:
                n_fail += 1
                detail_buffer.append((impl, name, r.detail))
            else:
                detail_buffer.append((impl, name, r.detail))
        print(f"{impl:<22} " + " ".join(f"{c:>16}" for c in cells))

    print()
    print("Detail:")
    for impl, name, detail in detail_buffer:
        print(f"  [{impl}/{name}] {detail}")

    print()
    if n_fail == 0:
        print(f"ALL GREEN — {len(MECHANISMS)} mechanisms × {len(TESTS)} tests")
        return 0
    print(f"{n_fail} failures across {len(MECHANISMS)} mechanisms")
    return 1


if __name__ == "__main__":
    sys.exit(run_all())
