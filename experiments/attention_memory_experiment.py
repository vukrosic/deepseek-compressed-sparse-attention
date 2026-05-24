import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


CONDITIONS = [
    ("dense", "dense", None),
    ("local", "local", None),
    ("csa", "csa", 4),
]


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


def build_command(args, condition: str, attention_impl: str, top_k: int | None, output_dir: Path, seed: int) -> list[str]:
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
        "--seed",
        str(seed),
        "--num_workers",
        str(args.num_workers),
        "--output_dir",
        str(output_dir),
        "--csa_sliding_window_size",
        str(args.local_window_size),
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
    if attention_impl == "csa":
        command.extend(
            [
                "--csa_compression_block_size",
                str(args.csa_compression_block_size),
                "--csa_top_k",
                str(top_k if top_k is not None else args.csa_top_k),
                "--csa_indexer_heads",
                str(args.csa_indexer_heads),
                "--csa_output_groups",
                str(args.csa_output_groups),
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


def write_summary(root: Path, rows: list[dict]) -> None:
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [
            row for row in rows
            if row["condition"] == condition and row["returncode"] == 0 and row["val_loss"] is not None
        ]
        if not condition_rows:
            continue
        losses = [float(row["val_loss"]) for row in condition_rows]
        toks = [float(row["tokens_per_second"]) for row in condition_rows if row["tokens_per_second"] is not None]
        aggregate.append(
            {
                "condition": condition,
                "runs": len(condition_rows),
                "val_loss_mean": mean(losses),
                "val_loss_std": stdev(losses) if len(losses) > 1 else 0.0,
                "tokens_per_second_mean": mean(toks) if toks else None,
            }
        )

    with (root / "summary.json").open("w") as f:
        json.dump({"rows": rows, "aggregate": aggregate}, f, indent=2)


def run_one(args, condition: str, attention_impl: str, top_k: int | None, seed: int, root: Path) -> dict:
    run_name = f"{condition}-seed{seed}"
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stdout.log"
    command = build_command(args, condition, attention_impl, top_k, output_dir, seed)

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
    parser = argparse.ArgumentParser(description="Run the simple dense vs local vs CSA memory experiment.")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--run_root", default="runs/attention_memory", help="Output root")
    parser.add_argument("--train_tokens", type=int, default=200_000)
    parser.add_argument("--max_train_seconds", type=float)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataset_path")
    parser.add_argument("--config_class")
    parser.add_argument("--synthetic_data", choices=["true", "false"], default="false")
    parser.add_argument("--synthetic_train_sequences", type=int, default=256)
    parser.add_argument("--synthetic_val_sequences", type=int, default=64)
    parser.add_argument("--synthetic_pattern", choices=["copy_lag", "counting"], default="copy_lag")
    parser.add_argument("--synthetic_lag", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--compile", choices=["true", "false"], default="false")
    parser.add_argument("--warmup", choices=["true", "false"], default="false")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds, for example: 42,43,44")
    parser.add_argument("--local_window_size", type=int, default=64)
    parser.add_argument("--csa_top_k", type=int, default=4)
    parser.add_argument("--csa_compression_block_size", type=int, default=16)
    parser.add_argument("--csa_indexer_heads", type=int, default=4)
    parser.add_argument("--csa_output_groups", type=int, default=1)
    args = parser.parse_args()

    args.repo_root = str(Path(args.repo_root).resolve())
    root = Path(args.repo_root) / args.run_root / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"

    rows = []
    with manifest_path.open("w") as manifest:
        for seed in parse_seeds(args.seeds):
            for condition, attention_impl, default_top_k in CONDITIONS:
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


if __name__ == "__main__":
    main()
