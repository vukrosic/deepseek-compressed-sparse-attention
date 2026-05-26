"""Plot the forgetting-scaling sweep.

Reads a completed or partially completed run from
`experiments/forgetting_scaling_sweep.py` and writes the paper-facing
figures:

  - validation loss vs training tokens
  - validation loss vs wall/active training time
  - final validation loss vs context length
  - throughput and VRAM by context/mechanism

The sweep is time-capped, so final loss alone is not a fair comparison.
These plots make the token-normalized and time-normalized views explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_rows(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        records = read_json(summary_path).get("runs", [])
    else:
        records = [
            {
                "arch": p.parent.name,
                "ctx": int(p.parent.parent.name.removeprefix("ctx")),
                "metrics_path": str(p),
                "metrics_exists": True,
            }
            for p in run_dir.glob("ctx*/**/metrics.json")
        ]

    rows: list[dict] = []
    repo_root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir.parent
    for record in records:
        metrics_path = Path(record["metrics_path"])
        if not metrics_path.is_absolute():
            metrics_path = repo_root / metrics_path
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        final = metrics.get("final_metrics", {})
        history = metrics.get("history", {}) or {}
        rows.append(
            {
                "arch": record["arch"],
                "ctx": int(record["ctx"]),
                "batch_size": record.get("batch_size"),
                "metrics_path": metrics_path,
                "val_loss": final.get("val_loss"),
                "val_accuracy": final.get("val_accuracy"),
                "tokens_seen": metrics.get("tokens_seen"),
                "tokens_per_second": metrics.get("tokens_per_second"),
                "active_seconds": metrics.get("active_training_time_seconds"),
                "wall_seconds": metrics.get("total_wall_time_seconds"),
                "total_parameters": metrics.get("total_parameters"),
                "peak_alloc_gib": metrics.get("peak_cuda_memory_allocated_bytes", 0) / (1024**3),
                "peak_reserved_gib": metrics.get("peak_cuda_memory_reserved_bytes", 0) / (1024**3),
                "history_tokens": history.get("tokens_seen", []) or [],
                "history_losses": history.get("val_losses", []) or [],
                "history_times": history.get("elapsed_times", []) or [],
            }
        )
    return sorted(rows, key=lambda r: (r["ctx"], r["arch"]))


def mechanism_order(rows: list[dict]) -> list[str]:
    preferred = [
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
    present = {r["arch"] for r in rows}
    return [x for x in preferred if x in present] + sorted(present - set(preferred))


def style_for(arch: str, order: list[str]):
    cmap = plt.get_cmap("tab20")
    idx = order.index(arch) if arch in order else 0
    return cmap(idx % 20)


def plot_loss_curves(rows: list[dict], out: Path, x_key: str, xlabel: str, title: str):
    contexts = sorted({r["ctx"] for r in rows})
    order = mechanism_order(rows)
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.2), sharey=True)
    if len(contexts) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, contexts):
        ctx_rows = [r for r in rows if r["ctx"] == ctx]
        for arch in order:
            row = next((r for r in ctx_rows if r["arch"] == arch), None)
            if row is None:
                continue
            xs = row["history_tokens"] if x_key == "tokens" else row["history_times"]
            ys = row["history_losses"]
            if not xs or not ys:
                continue
            xs_plot = [x / 1e6 for x in xs] if x_key == "tokens" else xs
            ax.plot(xs_plot, ys, marker="o", markersize=2.8, linewidth=1.5, label=arch, color=style_for(arch, order))
        ax.set_title(f"context = {ctx}")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("validation loss")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_by_context(rows: list[dict], out: Path):
    order = mechanism_order(rows)
    contexts = sorted({r["ctx"] for r in rows})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for arch in order:
        arch_rows = [r for r in rows if r["arch"] == arch and r["val_loss"] is not None]
        if not arch_rows:
            continue
        xs = [r["ctx"] for r in arch_rows]
        ys = [r["val_loss"] for r in arch_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=arch, color=style_for(arch, order))
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_xlabel("context length")
    ax.set_ylabel("final validation loss after time cap")
    ax.set_title("Final loss vs context length")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bars(rows: list[dict], out: Path, field: str, ylabel: str, title: str):
    order = mechanism_order(rows)
    contexts = sorted({r["ctx"] for r in rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.0 * len(contexts), 4.0), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, contexts):
        ctx_rows = [r for r in rows if r["ctx"] == ctx]
        labels = [a for a in order if any(r["arch"] == a and r.get(field) is not None for r in ctx_rows)]
        values = [next(r[field] for r in ctx_rows if r["arch"] == a) for a in labels]
        ax.bar(range(len(labels)), values, color=[style_for(a, order) for a in labels])
        ax.set_title(f"context = {ctx}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_table(rows: list[dict], out: Path):
    lines = [
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "Context & Mechanism & Val loss & Tokens (M) & Tok/s & Time (s) & VRAM GiB \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["val_loss"] is None:
            continue
        lines.append(
            f"{row['ctx']} & \\texttt{{{row['arch'].replace('_', '\\_')}}} & "
            f"{row['val_loss']:.4f} & "
            f"{row['tokens_seen'] / 1e6:.1f} & "
            f"{row['tokens_per_second'] / 1000:.1f}k & "
            f"{row['active_seconds']:.0f} & "
            f"{row['peak_reserved_gib']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Sweep timestamp directory, e.g. runs/forgetting_scaling/20260526_064522")
    parser.add_argument("--out", type=Path, default=Path("docs/research/results/forgetting_scaling_latest"))
    args = parser.parse_args()

    rows = load_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"No completed metrics found under {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(rows, args.out / "loss_vs_tokens.png", "tokens", "tokens seen (millions)", "Validation loss vs tokens")
    plot_loss_curves(rows, args.out / "loss_vs_time.png", "time", "active training seconds", "Validation loss vs time")
    plot_final_by_context(rows, args.out / "final_loss_vs_context.png")
    plot_bars(rows, args.out / "throughput_by_context.png", "tokens_per_second", "tokens per second", "Throughput by context")
    plot_bars(rows, args.out / "vram_by_context.png", "peak_reserved_gib", "peak reserved VRAM (GiB)", "Peak reserved VRAM by context")
    write_table(rows, args.out / "results_table.tex")

    print(f"Wrote {len(rows)} completed rows to {args.out}")


if __name__ == "__main__":
    main()
