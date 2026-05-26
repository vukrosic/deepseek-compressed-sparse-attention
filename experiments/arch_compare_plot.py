"""
Generate comparison plots + table CSV for the arch_compare sweep.
Reads runs/<run_root>/<timestamp>/<arch>/metrics.json for each arch.
Writes plots and a markdown table next to summary.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "dense": "#203A63",
    "local": "#5B6578",
    "csa": "#F59E0B",
    "compressed_memory": "#10B981",
    "age_forgetting": "#EF4444",
    "hierarchical": "#8B5CF6",
    "predictive": "#06B6D4",
    "entropy_gated_csa": "#F97316",
    "cross_block_residual": "#EC4899",
    "hebbian_co_activation": "#A78BFA",
    "negative_memory": "#84CC16",
    "multi_res_compression": "#F43F5E",
}


def load_runs(run_dir: Path):
    rows = []
    for arch_dir in sorted(run_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        metrics_path = arch_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        rows.append({
            "arch": arch_dir.name,
            "val_loss": m["final_metrics"]["val_loss"],
            "val_acc": m["final_metrics"]["val_accuracy"],
            "val_ppl": m["final_metrics"]["val_perplexity"],
            "tokens_seen": m["tokens_seen"],
            "active_seconds": m["active_training_time_seconds"],
            "tokens_per_sec": m["tokens_per_second"],
            "peak_vram_gib": m["peak_cuda_memory_allocated_bytes"] / (1024**3),
            "peak_vram_reserved_gib": m["peak_cuda_memory_reserved_bytes"] / (1024**3),
            "metadata": m.get("experiment_metadata", {}),
        })
    return rows


def plot_val_loss(rows, out_path: Path):
    archs = [r["arch"] for r in rows]
    losses = [r["val_loss"] for r in rows]
    colors = [COLORS.get(a, "#888") for a in archs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(archs, losses, color=colors)
    ax.set_ylabel("Final validation loss", fontsize=11)
    ax.set_title("Validation loss under equal GPU time", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for bar, v in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(min(losses) * 0.97, max(losses) * 1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_tokens_seen(rows, out_path: Path):
    archs = [r["arch"] for r in rows]
    toks = [r["tokens_seen"] / 1e6 for r in rows]
    colors = [COLORS.get(a, "#888") for a in archs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(archs, toks, color=colors)
    ax.set_ylabel("Tokens processed (millions)", fontsize=11)
    ax.set_title("Tokens processed under equal GPU time", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for bar, v in zip(bars, toks):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_throughput(rows, out_path: Path):
    archs = [r["arch"] for r in rows]
    tps = [r["tokens_per_sec"] / 1000 for r in rows]
    colors = [COLORS.get(a, "#888") for a in archs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(archs, tps, color=colors)
    ax.set_ylabel("Throughput (k tokens/s)", fontsize=11)
    ax.set_title("Training throughput", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for bar, v in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_loss_vs_tokens(rows, out_path: Path):
    """Scatter — final loss vs tokens-seen (each arch one point)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ax.scatter(r["tokens_seen"] / 1e6, r["val_loss"],
                   s=120, color=COLORS.get(r["arch"], "#888"),
                   label=r["arch"], edgecolor="black", linewidth=0.6)
        ax.annotate(r["arch"],
                    (r["tokens_seen"] / 1e6, r["val_loss"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Tokens processed (millions)", fontsize=11)
    ax.set_ylabel("Final validation loss", fontsize=11)
    ax.set_title("Loss vs tokens seen at equal wall-time", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_markdown_table(rows, out_path: Path):
    cols = ["Arch", "Val loss", "Val PPL", "Tokens (M)", "Tok/s (k)", "Active s", "VRAM GiB"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| {arch} | {loss:.4f} | {ppl:.1f} | {tokm:.2f} | {tps:.1f} | {sec:.1f} | {vram:.2f} |".format(
            arch=r["arch"],
            loss=r["val_loss"],
            ppl=r["val_ppl"],
            tokm=r["tokens_seen"] / 1e6,
            tps=r["tokens_per_sec"] / 1000,
            sec=r["active_seconds"],
            vram=r["peak_vram_gib"],
        ))
    out_path.write_text("\n".join(lines) + "\n")


def write_latex_table(rows, out_path: Path):
    lines = [
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Arch & Val loss & Val PPL & Tokens (M) & Tok/s (k) & Active s & VRAM GiB \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append("{arch} & {loss:.4f} & {ppl:.1f} & {tokm:.2f} & {tps:.1f} & {sec:.1f} & {vram:.2f} \\\\".format(
            arch=r["arch"].replace("_", "\\_"),
            loss=r["val_loss"],
            ppl=r["val_ppl"],
            tokm=r["tokens_seen"] / 1e6,
            tps=r["tokens_per_sec"] / 1000,
            sec=r["active_seconds"],
            vram=r["peak_vram_gib"],
        ))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = load_runs(args.run_dir)
    if not rows:
        print(f"No metrics.json files found under {args.run_dir}")
        return

    # Order: dense, local, then alpha sort over rest
    order = ["dense", "local", "csa", "compressed_memory", "age_forgetting",
             "hierarchical", "predictive",
             "entropy_gated_csa", "cross_block_residual",
             "hebbian_co_activation", "negative_memory", "multi_res_compression"]
    rows.sort(key=lambda r: order.index(r["arch"]) if r["arch"] in order else 999)

    out = args.run_dir
    plot_val_loss(rows, out / "val_loss.png")
    plot_tokens_seen(rows, out / "tokens_seen.png")
    plot_throughput(rows, out / "throughput.png")
    plot_loss_vs_tokens(rows, out / "loss_vs_tokens.png")
    write_markdown_table(rows, out / "results.md")
    write_latex_table(rows, out / "results.tex")

    print(f"Wrote 4 plots + results.md + results.tex to {out}")
    print()
    print((out / "results.md").read_text())


if __name__ == "__main__":
    main()
