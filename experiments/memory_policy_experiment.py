from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.research_configs import MemoryFiveMillionConfig
from data.synthetic import SyntheticCausalDataset
from models.llm import MinimalLLM
from utils.runtime import model_dtype_for_device, resolve_device


CONDITIONS = {
    "dense": "dense",
    "csa": "csa",
    "age_forgetting": "age_forgetting",
    "usage_refresh": "usage_refresh",
    "competition": "competition",
    "hierarchical": "hierarchical",
    "predictive": "predictive",
}

CONDITION_ORDER = list(CONDITIONS.keys())

SUMMARY_FIELDS = [
    "condition",
    "seed",
    "returncode",
    "val_loss",
    "val_perplexity",
    "val_accuracy",
    "train_loss",
    "tokens_seen",
    "tokens_per_second",
    "active_training_time_seconds",
    "total_wall_time_seconds",
    "harness_elapsed_seconds",
    "peak_cuda_memory_allocated_gib",
    "peak_cuda_memory_reserved_gib",
    "attention_vector_budget",
    "raw_token_equivalent_coverage",
    "probe_mean_gate",
    "probe_mean_refresh",
    "probe_mean_utility",
    "probe_mean_predictive_score",
    "probe_selected_blocks_mean",
    "probe_hierarchy_levels",
    "metrics_path",
    "stdout_path",
]

CURVE_FIELDS = ["condition", "seed", "age_gate_curve"]


def condition_sort_key(condition: str) -> int:
    try:
        return CONDITION_ORDER.index(condition)
    except ValueError:
        return len(CONDITION_ORDER)


def format_cell(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def parse_seeds(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def build_command(args, attention_impl: str, output_dir: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        "train_llm.py",
        "--config_class",
        "configs.research_configs.MemoryFiveMillionConfig",
        "--attention_impl",
        attention_impl,
        "--train_tokens",
        str(args.train_tokens),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--compile",
        args.compile,
        "--warmup",
        args.warmup,
        "--use_amp",
        args.use_amp,
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
        "--output_dir",
        str(output_dir),
        "--synthetic_data",
        args.synthetic_data,
        "--synthetic_train_sequences",
        str(args.synthetic_train_sequences),
        "--synthetic_val_sequences",
        str(args.synthetic_val_sequences),
        "--synthetic_pattern",
        args.synthetic_pattern,
        "--synthetic_lag",
        str(args.synthetic_lag),
    ]

    if args.max_train_seconds is not None:
        command.extend(["--max_train_seconds", str(args.max_train_seconds)])
    if args.dataset_path:
        command.extend(["--dataset_path", args.dataset_path])
    if args.adamw_lr is not None:
        command.extend(["--adamw_lr", str(args.adamw_lr)])
    if args.muon_lr is not None:
        command.extend(["--muon_lr", str(args.muon_lr)])
    if args.memory_local_window_size is not None:
        command.extend(["--memory_local_window_size", str(args.memory_local_window_size)])
    if args.memory_block_size is not None:
        command.extend(["--memory_block_size", str(args.memory_block_size)])
    if args.memory_budget_blocks is not None:
        command.extend(["--memory_budget_blocks", str(args.memory_budget_blocks)])
    if args.memory_age_decay_rate is not None:
        command.extend(["--memory_age_decay_rate", str(args.memory_age_decay_rate)])
    if args.memory_refresh_strength is not None:
        command.extend(["--memory_refresh_strength", str(args.memory_refresh_strength)])
    if args.memory_gate_floor is not None:
        command.extend(["--memory_gate_floor", str(args.memory_gate_floor)])
    if args.memory_competition_capacity is not None:
        command.extend(["--memory_competition_capacity", str(args.memory_competition_capacity)])
    if args.memory_hierarchy_levels is not None:
        command.extend(["--memory_hierarchy_levels", str(args.memory_hierarchy_levels)])
    if args.memory_hierarchy_branching is not None:
        command.extend(["--memory_hierarchy_branching", str(args.memory_hierarchy_branching)])
    if args.memory_predictive_hidden_dim is not None:
        command.extend(["--memory_predictive_hidden_dim", str(args.memory_predictive_hidden_dim)])
    if args.memory_predictive_top_k is not None:
        command.extend(["--memory_predictive_top_k", str(args.memory_predictive_top_k)])

    return command


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def load_probe_batch(args, config: MemoryFiveMillionConfig):
    if args.synthetic_data != "true":
        return None

    dataset = SyntheticCausalDataset(
        num_sequences=max(1, args.synthetic_val_sequences),
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        pattern=args.synthetic_pattern,
        lag=args.synthetic_lag,
        seed=args.seed_for_probe,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch["input_ids"]


def build_age_curve(debug: dict | None) -> dict[str, float]:
    if not debug:
        return {}

    memory_age = debug.get("memory_age")
    memory_gate = debug.get("memory_gate")
    if memory_age is None or memory_gate is None:
        return {}

    if not torch.is_tensor(memory_age) or not torch.is_tensor(memory_gate):
        return {}

    if memory_age.shape != memory_gate.shape:
        try:
            memory_age = memory_age.expand_as(memory_gate)
        except RuntimeError:
            return {}

    age = memory_age.detach().cpu().reshape(-1).to(torch.long)
    gate = memory_gate.detach().cpu().reshape(-1).to(torch.float32)
    valid = torch.isfinite(gate)
    age = age[valid]
    gate = gate[valid]
    if age.numel() == 0:
        return {}

    curve = {}
    max_age = int(age.max().item())
    for age_idx in range(max_age + 1):
        mask = age == age_idx
        if mask.any():
            curve[str(age_idx)] = float(gate[mask].mean().item())
    return curve


def probe_run(output_dir: Path, args, device: torch.device) -> dict:
    checkpoint_path = output_dir / "model.pt"
    if not checkpoint_path.exists():
        return {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device, dtype=model_dtype_for_device(device, False))
    model.eval()

    input_ids = load_probe_batch(args, config)
    if input_ids is None:
        return {}

    input_ids = input_ids.to(device)
    with torch.no_grad():
        logits, debug = model(input_ids, return_debug=True)

    layer_debugs = debug.get("layer_debugs", [])
    attention_debug = {}
    if layer_debugs:
        first_layer = layer_debugs[0] or {}
        attention_debug = first_layer.get("attention") or {}

    probe = {
        "probe_mean_gate": attention_debug.get("mean_gate"),
        "probe_mean_refresh": attention_debug.get("mean_refresh"),
        "probe_mean_utility": attention_debug.get("mean_utility"),
        "probe_mean_predictive_score": attention_debug.get("mean_predictive_score"),
        "probe_selected_blocks_mean": attention_debug.get("selected_blocks_mean"),
        "probe_hierarchy_levels": attention_debug.get("hierarchy_levels"),
        "probe_attention_vector_budget": attention_debug.get("attention_vector_budget"),
        "probe_raw_token_equivalent_coverage": attention_debug.get("raw_token_equivalent_coverage"),
        "probe_policy": attention_debug.get("policy"),
        "probe_logits_mean": float(logits.float().mean().item()),
        "probe_logits_std": float(logits.float().std().item()),
    }
    if config.attention_impl == "dense":
        probe["probe_attention_vector_budget"] = config.max_seq_len
        probe["probe_raw_token_equivalent_coverage"] = config.max_seq_len
    elif config.attention_impl == "csa":
        probe["probe_attention_vector_budget"] = (
            config.csa.sliding_window_size + config.csa.top_k
        )
        probe["probe_raw_token_equivalent_coverage"] = (
            config.csa.sliding_window_size
            + config.csa.top_k * config.csa.compression_block_size
        )
    elif probe.get("probe_attention_vector_budget") is None:
        probe["probe_attention_vector_budget"] = (
            config.memory_policy.local_window_size + config.memory_policy.memory_budget_blocks
        )
        probe["probe_raw_token_equivalent_coverage"] = (
            config.memory_policy.local_window_size
            + config.memory_policy.memory_budget_blocks * config.memory_policy.block_size
        )
    probe["age_gate_curve"] = build_age_curve(attention_debug)
    return probe


def flatten_record(record: dict, metrics: dict, probe: dict) -> dict:
    final_metrics = metrics.get("final_metrics", {})
    row = {
        "condition": record["condition"],
        "seed": record["seed"],
        "returncode": record["returncode"],
        "harness_elapsed_seconds": record["elapsed_seconds"],
        "metrics_path": record["metrics_path"],
        "stdout_path": record["log_path"],
    }

    for key in SUMMARY_FIELDS:
        if key in row:
            continue
        if key in final_metrics:
            row[key] = final_metrics[key]
        elif key in metrics:
            row[key] = metrics[key]
        elif key == "attention_vector_budget" and "probe_attention_vector_budget" in probe:
            row[key] = probe["probe_attention_vector_budget"]
        elif key == "raw_token_equivalent_coverage" and "probe_raw_token_equivalent_coverage" in probe:
            row[key] = probe["probe_raw_token_equivalent_coverage"]
        elif key in probe:
            row[key] = probe[key]
        elif key in record:
            row[key] = record[key]
        else:
            row[key] = None

    if row["peak_cuda_memory_allocated_gib"] is None and metrics.get("peak_cuda_memory_allocated_bytes") is not None:
        row["peak_cuda_memory_allocated_gib"] = metrics["peak_cuda_memory_allocated_bytes"] / (1024 ** 3)
    if row["peak_cuda_memory_reserved_gib"] is None and metrics.get("peak_cuda_memory_reserved_bytes") is not None:
        row["peak_cuda_memory_reserved_gib"] = metrics["peak_cuda_memory_reserved_bytes"] / (1024 ** 3)
    return row


def numeric_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            values.append(float(value))
    return values


def build_aggregate(rows: list[dict]) -> list[dict]:
    aggregate = []
    for condition in sorted({row["condition"] for row in rows}, key=condition_sort_key):
        condition_rows = [
            row
            for row in rows
            if row["condition"] == condition
            and row["returncode"] == 0
            and row["val_loss"] is not None
        ]
        if not condition_rows:
            continue

        losses = numeric_values(condition_rows, "val_loss")
        toks_per_sec = numeric_values(condition_rows, "tokens_per_second")
        tokens_seen = numeric_values(condition_rows, "tokens_seen")
        peak_alloc = numeric_values(condition_rows, "peak_cuda_memory_allocated_gib")
        peak_reserved = numeric_values(condition_rows, "peak_cuda_memory_reserved_gib")
        probe_gate = numeric_values(condition_rows, "probe_mean_gate")
        probe_refresh = numeric_values(condition_rows, "probe_mean_refresh")
        probe_utility = numeric_values(condition_rows, "probe_mean_utility")
        probe_predictive = numeric_values(condition_rows, "probe_mean_predictive_score")
        probe_selected = numeric_values(condition_rows, "probe_selected_blocks_mean")
        probe_levels = numeric_values(condition_rows, "probe_hierarchy_levels")
        val_std = 0.0
        if len(losses) > 1:
            val_mean = mean(losses)
            val_std = math.sqrt(sum((x - val_mean) ** 2 for x in losses) / (len(losses) - 1))

        aggregate.append(
            {
                "condition": condition,
                "runs": len(condition_rows),
                "val_loss_mean": mean(losses),
                "val_loss_std": val_std,
                "tokens_per_second_mean": mean(toks_per_sec) if toks_per_sec else None,
                "tokens_seen_mean": mean(tokens_seen) if tokens_seen else None,
                "peak_cuda_memory_allocated_gib_mean": mean(peak_alloc) if peak_alloc else None,
                "peak_cuda_memory_reserved_gib_mean": mean(peak_reserved) if peak_reserved else None,
                "attention_vector_budget": condition_rows[0].get("attention_vector_budget"),
                "raw_token_equivalent_coverage": condition_rows[0].get("raw_token_equivalent_coverage"),
                "probe_mean_gate": mean(probe_gate) if probe_gate else None,
                "probe_mean_refresh": mean(probe_refresh) if probe_refresh else None,
                "probe_mean_utility": mean(probe_utility) if probe_utility else None,
                "probe_mean_predictive_score": mean(probe_predictive) if probe_predictive else None,
                "probe_selected_blocks_mean": mean(probe_selected) if probe_selected else None,
                "probe_hierarchy_levels": mean(probe_levels) if probe_levels else None,
            }
        )
    return aggregate


def write_markdown_table(root: Path, aggregate: list[dict]) -> None:
    table_path = root / "summary_table.md"
    lines = [
        "# Memory Policy Experiment Summary",
        "",
        "| Condition | Runs | Val loss | Tok/s | Peak alloc GiB | Peak reserved GiB | Vectors | Coverage | Mean gate | Mean refresh | Mean utility | Mean predict | Selected blocks | Hierarchy levels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(row["runs"]),
                    format_cell(row["val_loss_mean"], 4),
                    format_cell(row["tokens_per_second_mean"], 1),
                    format_cell(row["peak_cuda_memory_allocated_gib_mean"], 2),
                    format_cell(row["peak_cuda_memory_reserved_gib_mean"], 2),
                    format_cell(row["attention_vector_budget"], 0),
                    format_cell(row["raw_token_equivalent_coverage"], 0),
                    format_cell(row["probe_mean_gate"], 4),
                    format_cell(row["probe_mean_refresh"], 4),
                    format_cell(row["probe_mean_utility"], 4),
                    format_cell(row["probe_mean_predictive_score"], 4),
                    format_cell(row["probe_selected_blocks_mean"], 2),
                    format_cell(row["probe_hierarchy_levels"], 1),
                ]
            )
            + " |"
        )
    table_path.write_text("\n".join(lines) + "\n")


def write_plots(root: Path, aggregate: list[dict], probe_rows: list[dict]) -> None:
    if not aggregate:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, FancyArrowPatch
    except Exception as exc:
        (root / "plot_error.txt").write_text(f"Could not import matplotlib: {exc}\n")
        return

    palette = {
        "dense": "#2F4858",
        "age_forgetting": "#F6AE2D",
        "usage_refresh": "#7A3E65",
        "competition": "#3E6A5A",
        "hierarchical": "#5B4E8C",
        "predictive": "#B65D32",
    }

    labels = [row["condition"] for row in aggregate]
    colors = [palette.get(label, "#444444") for label in labels]
    dense_loss = next((row["val_loss_mean"] for row in aggregate if row["condition"] == "dense"), aggregate[0]["val_loss_mean"])
    dense_tps = next((row["tokens_per_second_mean"] for row in aggregate if row["condition"] == "dense"), aggregate[0]["tokens_per_second_mean"])

    def save_bar(values_key: str, title: str, ylabel: str, filename: str, err_key: str | None = None):
        values = [0.0 if row[values_key] is None else row[values_key] for row in aggregate]
        errs = [0.0 if row[err_key] is None else row[err_key] for row in aggregate] if err_key else None
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(labels, values, yerr=errs, capsize=5, color=colors)
        ax.set_title(title)
        ax.set_xlabel("Memory policy")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=180)
        plt.close(fig)

    def add_bar_labels(ax, values, fmt="{:.3f}", dy=0.02):
        for idx, value in enumerate(values):
            ax.text(idx, value + dy, fmt.format(value), ha="center", va="bottom", fontsize=8)

    def save_policy_comparison_summary():
        losses = [row["val_loss_mean"] for row in aggregate]
        loss_deltas = [(loss - dense_loss) * 1000.0 for loss in losses]
        tps = [row["tokens_per_second_mean"] for row in aggregate]
        rel_tps = [value / dense_tps if dense_tps else 0.0 for value in tps]

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
        axes = axes.flatten()

        # Absolute validation loss, zoomed tightly so tiny differences are visible.
        ax = axes[0]
        ax.bar(labels, losses, color=colors)
        for idx, value in enumerate(losses):
            ax.text(idx, value + 0.00005, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        loss_min = min(losses)
        loss_max = max(losses)
        padding = max(0.0006, (loss_max - loss_min) * 2.0)
        ax.set_ylim(loss_min - padding, loss_max + padding)
        ax.set_title("Validation loss (zoomed)")
        ax.set_ylabel("Lower is better")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

        # Loss relative to dense in milli-loss.
        ax = axes[1]
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.bar(labels, loss_deltas, color=colors)
        for idx, value in enumerate(loss_deltas):
            ax.text(idx, value + (0.03 if value >= 0 else -0.08), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        ax.set_title("Validation loss relative to dense")
        ax.set_ylabel("Delta in milli-loss")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

        # Absolute throughput on a log scale.
        ax = axes[2]
        ax.bar(labels, tps, color=colors)
        for idx, value in enumerate(tps):
            ax.text(idx, value * 1.03, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_title("Throughput (log scale)")
        ax.set_ylabel("Tokens per second")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2, which="both")

        # Throughput relative to dense.
        ax = axes[3]
        ax.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.bar(labels, rel_tps, color=colors)
        for idx, value in enumerate(rel_tps):
            ax.text(idx, value + 0.02, f"{value:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.set_title("Throughput relative to dense")
        ax.set_ylabel("Dense = 1.0")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

        fig.suptitle("Memory policy comparison", y=1.02, fontsize=14)
        fig.tight_layout()
        fig.savefig(root / "policy_comparison_summary.png", dpi=200, bbox_inches="tight")
        fig.savefig(root / "policy_comparison_summary.svg", bbox_inches="tight")
        plt.close(fig)

    def save_mechanism_overview():
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes = axes.flatten()
        panel_specs = [
            ("dense", "Dense", "full past", "#2F4858"),
            ("age_forgetting", "Age decay", "older blocks fade", "#F6AE2D"),
            ("usage_refresh", "Refresh", "used blocks stay bright", "#7A3E65"),
            ("competition", "Competition", "top blocks survive", "#3E6A5A"),
            ("hierarchical", "Hierarchy", "summaries of summaries", "#5B4E8C"),
            ("predictive", "Predictive", "future-use score", "#B65D32"),
        ]

        for ax, (key, title, subtitle, color) in zip(axes, panel_specs):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            ax.add_patch(Rectangle((0.3, 4.9), 9.1, 0.75, facecolor="#F3F4F6", edgecolor="#D1D5DB"))
            ax.text(0.6, 5.25, title, fontsize=12, weight="bold", color=color)
            ax.text(3.0, 5.25, subtitle, fontsize=9, color="#4B5563")

            if key == "dense":
                for i in range(8):
                    ax.add_patch(Rectangle((0.6 + i * 1.0, 2.0), 0.8, 1.0, facecolor=color, alpha=0.9))
                ax.text(0.6, 1.2, "every token sees the full causal past", fontsize=9)
            elif key == "age_forgetting":
                for i in range(6):
                    alpha = 0.2 + 0.14 * i
                    ax.add_patch(Rectangle((0.8 + i * 1.25, 2.2), 1.0, 0.8, facecolor=color, alpha=alpha, edgecolor=color))
                ax.text(0.8, 1.2, "recent blocks stay strong; old ones fade", fontsize=9)
            elif key == "usage_refresh":
                for i in range(6):
                    alpha = 0.18 + 0.12 * i
                    ax.add_patch(Rectangle((0.8 + i * 1.25, 2.2), 1.0, 0.8, facecolor=color, alpha=alpha, edgecolor=color))
                ax.add_patch(FancyArrowPatch((4.8, 1.7), (5.9, 2.9), arrowstyle="->", mutation_scale=14, color=color, linewidth=1.5))
                ax.text(0.8, 1.2, "reused blocks get a refresh boost", fontsize=9)
            elif key == "competition":
                for i in range(6):
                    selected = i in {1, 4}
                    face = color if selected else "#D1D5DB"
                    alpha = 0.95 if selected else 0.6
                    ax.add_patch(Rectangle((0.8 + i * 1.25, 2.2), 1.0, 0.8, facecolor=face, edgecolor=color, alpha=alpha))
                    ax.text(1.25 + i * 1.25, 2.55, "keep" if selected else "drop", ha="center", va="center", fontsize=8)
                ax.text(0.8, 1.2, "limited slots compete for survival", fontsize=9)
            elif key == "hierarchical":
                for i in range(4):
                    ax.add_patch(Rectangle((0.8 + i * 1.5, 3.0), 1.1, 0.7, facecolor=color, alpha=0.55))
                for i in range(2):
                    ax.add_patch(Rectangle((1.7 + i * 2.8, 1.9), 1.6, 0.8, facecolor=color, alpha=0.78))
                    ax.add_patch(FancyArrowPatch((1.35 + i * 2.8, 2.9), (2.55 + i * 2.8, 2.25), arrowstyle="->", mutation_scale=14, color=color, linewidth=1.3))
                ax.text(0.8, 1.2, "raw blocks compress into higher summaries", fontsize=9)
            elif key == "predictive":
                scores = [0.18, 0.55, 0.83, 0.31, 0.92, 0.47]
                for i, score in enumerate(scores):
                    selected = score >= 0.55
                    face = color if selected else "#E5E7EB"
                    ax.add_patch(Rectangle((0.8 + i * 1.25, 2.2), 1.0, 0.8, facecolor=face, edgecolor=color, alpha=0.9 if selected else 0.65))
                    ax.text(1.3 + i * 1.25, 2.55, f"{score:.2f}", ha="center", va="center", fontsize=8)
                ax.text(0.8, 1.2, "keep blocks the model predicts will matter", fontsize=9)

        fig.suptitle("Five memory philosophies", y=0.98, fontsize=14)
        fig.tight_layout()
        fig.savefig(root / "memory_mechanisms_overview.png", dpi=200, bbox_inches="tight")
        fig.savefig(root / "memory_mechanisms_overview.svg", bbox_inches="tight")
        plt.close(fig)

    save_bar("val_loss_mean", "Validation loss by memory policy", "Validation loss, lower is better", "val_loss_by_policy.png", "val_loss_std")
    save_bar("tokens_per_second_mean", "Training throughput by memory policy", "Tokens per second, higher is better", "tokens_per_second_by_policy.png")
    save_bar("peak_cuda_memory_allocated_gib_mean", "Peak CUDA/MPS memory by policy", "Peak allocated GiB", "peak_memory_by_policy.png")
    save_bar("attention_vector_budget", "Attention-vector budget by policy", "Vector budget", "attention_budget_by_policy.png")
    save_policy_comparison_summary()
    save_mechanism_overview()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    probe_metrics = [
        ("probe_mean_gate", "Mean gate"),
        ("probe_mean_refresh", "Mean refresh"),
        ("probe_mean_utility", "Mean utility"),
        ("probe_mean_predictive_score", "Mean predictive score"),
    ]
    for ax, (field, title) in zip(axes, probe_metrics):
        values = [row[field] if row[field] is not None else 0.0 for row in aggregate]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Post-run probe metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(root / "memory_probe_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    gate_curve_rows = [row for row in probe_rows if row.get("age_gate_curve")]
    if gate_curve_rows:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for row in gate_curve_rows:
            curve = row["age_gate_curve"]
            ages = sorted(int(age) for age in curve.keys())
            gates = [curve[str(age)] for age in ages]
            ax.plot(ages, gates, marker="o", label=row["condition"])
        ax.set_title("Gate strength by memory age")
        ax.set_xlabel("Memory age (blocks)")
        ax.set_ylabel("Gate value")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "gate_by_age.png", dpi=180)
        plt.close(fig)


def write_summary(root: Path, rows: list[dict], probe_rows: list[dict]) -> list[dict]:
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = build_aggregate(rows)
    with (root / "summary.json").open("w") as f:
        json.dump({"rows": rows, "aggregate": aggregate}, f, indent=2)
    with (root / "probe_records.json").open("w") as f:
        json.dump(probe_rows, f, indent=2)

    write_markdown_table(root, aggregate)
    write_plots(root, aggregate, probe_rows)
    return aggregate


def run_one(args, condition: str, attention_impl: str, seed: int, root: Path) -> dict:
    run_name = f"{condition}-seed{seed}"
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stdout.log"
    command = build_command(args, attention_impl, output_dir, seed)

    started_at = time.time()
    with log_path.open("w") as log_file:
        process = subprocess.run(
            command,
            cwd=args.repo_root,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return {
        "condition": condition,
        "attention_impl": attention_impl,
        "seed": seed,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started_at,
        "command": command,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "metrics_path": str(output_dir / "metrics.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense, CSA, and five memory-policy research sweeps.")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--run_root", default="runs/memory_policy", help="Output root")
    parser.add_argument("--conditions", default="dense,csa,age_forgetting,usage_refresh,competition,hierarchical,predictive", help="Comma-separated conditions to run")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--train_tokens", type=int, default=5_000_000)
    parser.add_argument("--max_train_seconds", type=float)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--adamw_lr", type=float, default=0.002)
    parser.add_argument("--muon_lr", type=float, default=0.01)
    parser.add_argument("--dataset_path")
    parser.add_argument("--synthetic_data", choices=["true", "false"], default="true")
    parser.add_argument("--synthetic_train_sequences", type=int, default=512)
    parser.add_argument("--synthetic_val_sequences", type=int, default=128)
    parser.add_argument("--synthetic_pattern", choices=["copy_lag", "counting"], default="copy_lag")
    parser.add_argument("--synthetic_lag", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--compile", choices=["true", "false"], default="false")
    parser.add_argument("--warmup", choices=["true", "false"], default="false")
    parser.add_argument("--use_amp", choices=["true", "false"], default="false")
    parser.add_argument("--memory_local_window_size", type=int, default=64)
    parser.add_argument("--memory_block_size", type=int, default=4)
    parser.add_argument("--memory_budget_blocks", type=int, default=16)
    parser.add_argument("--memory_age_decay_rate", type=float, default=0.125)
    parser.add_argument("--memory_refresh_strength", type=float, default=0.35)
    parser.add_argument("--memory_gate_floor", type=float, default=0.0)
    parser.add_argument("--memory_competition_capacity", type=int, default=8)
    parser.add_argument("--memory_hierarchy_levels", type=int, default=2)
    parser.add_argument("--memory_hierarchy_branching", type=int, default=4)
    parser.add_argument("--memory_predictive_hidden_dim", type=int, default=32)
    parser.add_argument("--memory_predictive_top_k", type=int, default=8)
    parser.add_argument("--seed_for_probe", type=int, default=12345)
    args = parser.parse_args()

    args.repo_root = str(Path(args.repo_root).resolve())
    root = Path(args.repo_root) / args.run_root / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"

    selected_conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    runs = []
    for condition in selected_conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        runs.append((condition, CONDITIONS[condition]))

    rows: list[dict] = []
    probe_rows: list[dict] = []
    device = resolve_device()

    with manifest_path.open("w") as manifest:
        for seed in parse_seeds(args.seeds):
            for condition, attention_impl in runs:
                record = run_one(args, condition, attention_impl, seed, root)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()

                metrics = read_metrics(Path(record["metrics_path"]))
                probe = probe_run(Path(record["output_dir"]), args, device)
                probe_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "age_gate_curve": probe.get("age_gate_curve", {}),
                    }
                )
                rows.append(flatten_record(record, metrics, probe))
                write_summary(root, rows, probe_rows)

                status = "ok" if record["returncode"] == 0 else f"failed:{record['returncode']}"
                print(f"{record['condition']} seed={seed}: {status} ({record['elapsed_seconds']:.1f}s)")
                if record["returncode"] != 0:
                    print(f"Stopping after failure. See {record['log_path']}")
                    return

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {root / 'summary.csv'}")
    print(f"Wrote table: {root / 'summary_table.md'}")
    print(f"Wrote charts under: {root}")


if __name__ == "__main__":
    main()
