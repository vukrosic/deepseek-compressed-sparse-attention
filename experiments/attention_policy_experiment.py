import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


ALL_CONDITIONS = {
    "dense": ("dense", None),
    "csa": ("csa", 4),
    "forgetting": ("forgetting", None),
}

CONDITION_ORDER = ["dense", "csa", "forgetting"]

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
    "metrics_path",
    "stdout_path",
]


def build_command(args, attention_impl: str, top_k: int | None, output_dir: Path, seed: int) -> list[str]:
    command = [
        sys.executable,
        "train_llm.py",
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
    ]

    if args.config_class:
        command.extend(["--config_class", args.config_class])
    if args.dataset_path:
        command.extend(["--dataset_path", args.dataset_path])
    if args.synthetic_data == "true":
        command.extend(
            [
                "--synthetic_data",
                "true",
                "--synthetic_train_sequences",
                str(args.synthetic_train_sequences),
                "--synthetic_val_sequences",
                str(args.synthetic_val_sequences),
                "--synthetic_pattern",
                args.synthetic_pattern,
                "--synthetic_lag",
                str(args.synthetic_lag),
            ]
        )
    if args.max_train_seconds is not None:
        command.extend(["--max_train_seconds", str(args.max_train_seconds)])
    if args.adamw_lr is not None:
        command.extend(["--adamw_lr", str(args.adamw_lr)])
    if args.muon_lr is not None:
        command.extend(["--muon_lr", str(args.muon_lr)])

    if attention_impl == "csa":
        command.extend(
            [
                "--csa_compression_block_size",
                str(args.csa_compression_block_size),
                "--csa_top_k",
                str(top_k if top_k is not None else args.csa_top_k),
                "--csa_sliding_window_size",
                str(args.csa_sliding_window_size),
                "--csa_indexer_heads",
                str(args.csa_indexer_heads),
                "--csa_output_groups",
                str(args.csa_output_groups),
            ]
        )
    elif attention_impl == "forgetting":
        command.extend(
            [
                "--forgetting_local_window_size",
                str(args.forgetting_local_window_size),
                "--forgetting_memory_block_size",
                str(args.forgetting_memory_block_size),
                "--forgetting_memory_decay_rate",
                str(args.forgetting_memory_decay_rate),
                "--forgetting_gate_floor",
                str(args.forgetting_gate_floor),
            ]
        )

    return command


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def flatten_record(record: dict, metrics: dict) -> dict:
    final_metrics = metrics.get("final_metrics", {})
    metadata = metrics.get("experiment_metadata", metrics.get("metadata", {}))
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
        elif key in metadata:
            row[key] = metadata[key]
        else:
            row[key] = None
    if row["peak_cuda_memory_allocated_gib"] is None and metrics.get("peak_cuda_memory_allocated_bytes") is not None:
        row["peak_cuda_memory_allocated_gib"] = metrics["peak_cuda_memory_allocated_bytes"] / (1024 ** 3)
    if row["peak_cuda_memory_reserved_gib"] is None and metrics.get("peak_cuda_memory_reserved_bytes") is not None:
        row["peak_cuda_memory_reserved_gib"] = metrics["peak_cuda_memory_reserved_bytes"] / (1024 ** 3)
    return row


def condition_sort_key(condition: str) -> int:
    try:
        return CONDITION_ORDER.index(condition)
    except ValueError:
        return len(CONDITION_ORDER)


def numeric_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            values.append(float(value))
    return values


def format_cell(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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
                "attention_vector_budget": condition_rows[0].get("attention_vector_budget"),
                "raw_token_equivalent_coverage": condition_rows[0].get("raw_token_equivalent_coverage"),
            }
        )
    return aggregate


def write_markdown_table(root: Path, aggregate: list[dict]) -> None:
    table_path = root / "summary_table.md"
    lines = [
        "# Attention Policy Experiment Summary",
        "",
        "| Condition | Runs | Val loss mean | Val loss std | Tok/s mean | Tokens seen mean | Peak alloc GiB | Attention vectors | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["condition"]),
                    str(row["runs"]),
                    format_cell(row["val_loss_mean"], 4),
                    format_cell(row["val_loss_std"], 4),
                    format_cell(row["tokens_per_second_mean"], 1),
                    format_cell(row["tokens_seen_mean"], 0),
                    format_cell(row["peak_cuda_memory_allocated_gib_mean"], 2),
                    format_cell(row["attention_vector_budget"], 0),
                    format_cell(row["raw_token_equivalent_coverage"], 0),
                ]
            )
            + " |"
        )
    table_path.write_text("\n".join(lines) + "\n")


def write_plots(root: Path, aggregate: list[dict]) -> None:
    if not aggregate:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (root / "plot_error.txt").write_text(f"Could not import matplotlib: {exc}\n")
        return

    labels = [row["condition"] for row in aggregate]
    colors = ["#2F4858", "#F6AE2D", "#7A3E65"][: len(labels)]

    val_loss = [row["val_loss_mean"] for row in aggregate]
    val_loss_err = [row["val_loss_std"] for row in aggregate]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, val_loss, yerr=val_loss_err, capsize=5, color=colors)
    ax.set_title("Validation loss by attention policy")
    ax.set_xlabel("Attention policy")
    ax.set_ylabel("Validation loss, lower is better")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "val_loss_by_attention.png", dpi=180)
    plt.close(fig)

    tok_rows = [row for row in aggregate if row["tokens_per_second_mean"] is not None]
    if tok_rows:
        labels = [row["condition"] for row in tok_rows]
        tok_per_sec = [row["tokens_per_second_mean"] for row in tok_rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, tok_per_sec, color=colors[: len(labels)])
        ax.set_title("Training throughput by attention policy")
        ax.set_xlabel("Attention policy")
        ax.set_ylabel("Tokens per second, higher is better")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "tokens_per_second_by_attention.png", dpi=180)
        plt.close(fig)


def write_summary(root: Path, rows: list[dict]) -> None:
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = build_aggregate(rows)

    with (root / "summary.json").open("w") as f:
        json.dump({"rows": rows, "aggregate": aggregate}, f, indent=2)
    write_markdown_table(root, aggregate)
    write_plots(root, aggregate)


def run_one(args, condition: str, attention_impl: str, top_k: int | None, seed: int, root: Path) -> dict:
    run_name = f"{condition}-seed{seed}"
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stdout.log"
    command = build_command(args, attention_impl, top_k, output_dir, seed)

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
        "top_k": top_k,
        "seed": seed,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started_at,
        "command": command,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "metrics_path": str(output_dir / "metrics.json"),
    }


def parse_seeds(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense vs CSA vs forgetting attention policy comparisons.")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--run_root", default="runs/attention_policy", help="Output root")
    parser.add_argument("--conditions", default="dense,forgetting", help="Comma-separated conditions to run, e.g. dense,forgetting")
    parser.add_argument("--train_tokens", type=int, default=5_000_000)
    parser.add_argument("--max_train_seconds", type=float)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--adamw_lr", type=float, default=0.0005)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--dataset_path")
    parser.add_argument("--config_class", default="configs.research_configs.CSAFiveMillionConfig")
    parser.add_argument("--synthetic_data", choices=["true", "false"], default="true")
    parser.add_argument("--synthetic_train_sequences", type=int, default=4096)
    parser.add_argument("--synthetic_val_sequences", type=int, default=512)
    parser.add_argument("--synthetic_pattern", choices=["copy_lag", "counting"], default="copy_lag")
    parser.add_argument("--synthetic_lag", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--compile", choices=["true", "false"], default="false")
    parser.add_argument("--warmup", choices=["true", "false"], default="false")
    parser.add_argument("--use_amp", choices=["true", "false"], default="false")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, for example: 42,43,44")
    parser.add_argument("--csa_top_k", type=int, default=4)
    parser.add_argument("--csa_compression_block_size", type=int, default=4)
    parser.add_argument("--csa_sliding_window_size", type=int, default=64)
    parser.add_argument("--csa_indexer_heads", type=int, default=4)
    parser.add_argument("--csa_output_groups", type=int, default=1)
    parser.add_argument("--forgetting_local_window_size", type=int, default=64)
    parser.add_argument("--forgetting_memory_block_size", type=int, default=4)
    parser.add_argument("--forgetting_memory_decay_rate", type=float, default=0.125)
    parser.add_argument("--forgetting_gate_floor", type=float, default=0.0)
    args = parser.parse_args()

    args.repo_root = str(Path(args.repo_root).resolve())
    root = Path(args.repo_root) / args.run_root / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"

    rows = []
    selected_conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    runs = []
    for condition in selected_conditions:
        if condition not in ALL_CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        attention_impl, default_top_k = ALL_CONDITIONS[condition]
        runs.append((condition, attention_impl, default_top_k))

    with manifest_path.open("w") as manifest:
        for seed in parse_seeds(args.seeds):
            for condition, attention_impl, default_top_k in runs:
                top_k = args.csa_top_k if condition == "csa" else default_top_k
                record = run_one(args, condition, attention_impl, top_k, seed, root)
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()
                metrics = read_metrics(Path(record["metrics_path"]))
                rows.append(flatten_record(record, metrics))
                write_summary(root, rows)
                status = "ok" if record["returncode"] == 0 else f"failed:{record['returncode']}"
                print(f"{record['condition']} seed={seed}: {status} ({record['elapsed_seconds']:.1f}s)")
                if record["returncode"] != 0:
                    print(f"Stopping after failure. See {record['log_path']}")
                    return

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {root / 'summary.csv'}")
    print(f"Wrote table: {root / 'summary_table.md'}")
    print(f"Wrote chart: {root / 'val_loss_by_attention.png'}")


if __name__ == "__main__":
    main()
