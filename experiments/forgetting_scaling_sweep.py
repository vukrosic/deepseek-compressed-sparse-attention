"""
Forgetting-mechanism scaling sweep — resilient version.

For each (mechanism, context length) we run train_llm.py with a fixed GPU-time
budget. Per-run isolation:
  - Each run is a separate subprocess. An OOM/crash in one mechanism does NOT
    affect the others.
  - All errors are logged to stdout.log and surfaced in summary.json with a
    short reason field.
  - The subprocess has a hard timeout (timeout_factor * max_train_seconds);
    a hung run gets killed without blocking the rest.
  - batch_size scales with context so tokens-per-step is constant:
        batch_size = max(1, base_batch_size * 2048 // ctx)
    This both keeps comparison fair and avoids OOM at long context.

Layout::

    runs/<run_root>/<timestamp>/ctx<ctx>/<arch>/metrics.json
    runs/<run_root>/<timestamp>/ctx<ctx>/<arch>/stdout.log
    runs/<run_root>/<timestamp>/summary.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ARCHS = [
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

DEFAULT_CONTEXTS = [2048, 4096, 8192]


def batch_size_for_ctx(base_batch: int, ctx: int, base_ctx: int = 2048) -> int:
    """Scale batch size so tokens-per-step stays constant.

    Example with base_batch=4, base_ctx=2048:
      ctx=2048 -> 4    (4*2048 = 8192 tokens/step)
      ctx=4096 -> 2    (2*4096 = 8192 tokens/step)
      ctx=8192 -> 1    (1*8192 = 8192 tokens/step)
    """
    bs = max(1, (base_batch * base_ctx) // ctx)
    return bs


def build_command(arch: str, ctx: int, args, output_dir: Path) -> list[str]:
    bs = batch_size_for_ctx(args.batch_size, ctx)
    cmd = [
        args.python, "train_llm.py",
        "--attention_impl", arch,
        "--config_class", args.config_class,
        "--dataset_path", args.dataset_path,
        "--batch_size", str(bs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--compile", args.compile,
        "--warmup", args.warmup,
        "--seed", str(args.seed),
        "--num_workers", "0",  # avoid the multiprocessing-shutdown hang
        "--output_dir", str(output_dir),
        "--max_train_seconds", str(args.max_train_seconds),
        "--train_tokens", str(args.train_tokens),
        "--eval_every", str(args.eval_every),
        "--max_seq_len", str(ctx),
    ]
    return cmd


def classify_error(rc: int, log_path: Path) -> str:
    """Best-effort categorization of why a run failed, by scanning the tail of its log."""
    if rc == 0:
        return "ok"
    if rc < 0:
        return "timeout"
    try:
        tail = log_path.read_text(errors="replace")[-8000:]
    except Exception:
        return f"rc={rc} (no log)"
    if "OutOfMemoryError" in tail or "CUDA out of memory" in tail:
        return "OOM"
    if "RuntimeError" in tail:
        # Pull last RuntimeError line
        for line in reversed(tail.splitlines()):
            if "RuntimeError" in line:
                return f"runtime: {line[:120]}"
    if "Traceback" in tail:
        return f"rc={rc} (traceback in log)"
    return f"rc={rc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_class", default="configs.research_configs.CSAMinimumPaperConfig")
    parser.add_argument("--dataset_path", default="processed_data/speedrun_40M")
    parser.add_argument("--run_root", default="runs/forgetting_scaling")
    parser.add_argument("--max_train_seconds", type=float, default=300.0)
    parser.add_argument("--train_tokens", type=int, default=200_000_000)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Base batch size at base_ctx=2048; scaled down for longer ctx")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--compile", default="false")
    parser.add_argument("--warmup", default="true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--contexts", nargs="+", type=int, default=DEFAULT_CONTEXTS,
                        help="Context lengths to sweep (default: 2048 4096 8192)")
    parser.add_argument("--archs", nargs="+", default=None,
                        help="Override which architectures to run")
    parser.add_argument("--timeout_factor", type=float, default=3.0,
                        help="Per-run subprocess timeout = factor * max_train_seconds")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use for child train_llm.py runs")
    args = parser.parse_args()

    archs = args.archs if args.archs else ARCHS

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.run_root) / timestamp
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "config_class": args.config_class,
        "dataset_path": args.dataset_path,
        "max_train_seconds": args.max_train_seconds,
        "eval_every": args.eval_every,
        "base_batch_size": args.batch_size,
        "seed": args.seed,
        "contexts": args.contexts,
        "archs": archs,
        "runs": [],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    timeout = args.timeout_factor * args.max_train_seconds
    total_runs = len(args.contexts) * len(archs)
    done = 0
    n_ok = 0
    n_fail = 0
    overall_start = time.time()

    for ctx in args.contexts:
        bs = batch_size_for_ctx(args.batch_size, ctx)
        print(f"\n############# context = {ctx} (batch_size = {bs}) #############")
        for arch in archs:
            done += 1
            output_dir = root / f"ctx{ctx}" / arch
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "stdout.log"
            cmd = build_command(arch, ctx, args, output_dir)

            wall_elapsed = time.time() - overall_start
            print(f"\n=== [{done}/{total_runs}] ctx={ctx} arch={arch} bs={bs} (sweep_elapsed={wall_elapsed:.0f}s) ===")
            print(" ".join(cmd))
            started = time.time()
            try:
                with log_path.open("w") as logf:
                    proc = subprocess.run(
                        cmd, cwd=args.repo_root, text=True,
                        stdout=logf, stderr=subprocess.STDOUT, check=False,
                        timeout=timeout,
                    )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = -9
            except Exception as e:
                rc = -1
                with log_path.open("a") as logf:
                    logf.write(f"\n[sweep] subprocess.run raised: {type(e).__name__}: {e}\n")
            elapsed = time.time() - started

            status = classify_error(rc, log_path)
            metrics_exists = (output_dir / "metrics.json").exists()
            if metrics_exists:
                n_ok += 1
            else:
                n_fail += 1

            record = {
                "arch": arch,
                "ctx": ctx,
                "batch_size": bs,
                "returncode": rc,
                "status": status,
                "wall_seconds": elapsed,
                "metrics_path": str(output_dir / "metrics.json"),
                "log_path": str(log_path),
                "metrics_exists": metrics_exists,
            }
            summary["runs"].append(record)
            summary["progress"] = {
                "done": done,
                "total": total_runs,
                "ok": n_ok,
                "failed": n_fail,
                "sweep_elapsed_seconds": time.time() - overall_start,
            }
            (root / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"  status={status} wall={elapsed:.1f}s metrics_exists={metrics_exists} | ok={n_ok}/{done} fail={n_fail}")

    print(f"\n=== Sweep finished in {time.time() - overall_start:.0f}s ===")
    print(f"  {n_ok}/{total_runs} runs produced metrics")
    print(f"  Summary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
