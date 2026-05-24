import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PILOT_RUNS = [
    ("dense", None),
    ("csa", 1),
    ("csa", 8),
]

FULL_RUNS = [
    ("dense", None),
    ("csa", 1),
    ("csa", 16),
    ("csa", 4),
    ("csa", 2),
    ("csa", 8),
]


def build_command(args, attention_impl: str, top_k: int | None, output_dir: Path) -> list[str]:
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
        str(args.seed),
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
    if attention_impl == "csa":
        command.extend(
            [
                "--csa_compression_block_size",
                str(args.csa_compression_block_size),
                "--csa_top_k",
                str(top_k),
                "--csa_sliding_window_size",
                str(args.csa_sliding_window_size),
                "--csa_indexer_heads",
                str(args.csa_indexer_heads),
                "--csa_output_groups",
                str(args.csa_output_groups),
            ]
        )

    return command


def run_one(args, attention_impl: str, top_k: int | None, root: Path) -> dict:
    run_name = "dense" if attention_impl == "dense" else f"csa-k{top_k}"
    output_dir = root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "stdout.log"
    command = build_command(args, attention_impl, top_k, output_dir)

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

    record = {
        "run_name": run_name,
        "attention_impl": attention_impl,
        "top_k": top_k,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started_at,
        "command": command,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "metrics_path": str(output_dir / "metrics.json"),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSA top-k pilot or sweep.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--run_root", default="runs/csa_top_k", help="Output root")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csa_compression_block_size", type=int, default=16)
    parser.add_argument("--csa_sliding_window_size", type=int, default=64)
    parser.add_argument("--csa_indexer_heads", type=int, default=4)
    parser.add_argument("--csa_output_groups", type=int, default=1)
    args = parser.parse_args()

    args.repo_root = str(Path(args.repo_root).resolve())
    root = Path(args.repo_root) / args.run_root / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"

    runs = PILOT_RUNS if args.mode == "pilot" else FULL_RUNS
    records = []
    with manifest_path.open("w") as manifest:
        for attention_impl, top_k in runs:
            record = run_one(args, attention_impl, top_k, root)
            records.append(record)
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if record["returncode"] != 0:
                break

    print(f"Wrote manifest: {manifest_path}")
    for record in records:
        status = "ok" if record["returncode"] == 0 else f"failed:{record['returncode']}"
        print(f"{record['run_name']}: {status} ({record['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
