import argparse
import time
import os
import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import LLMConfig
from configs.dataset_config import DataConfig
from training.trainer import train_minimal_llm
from utils.helpers import set_seed, format_time
from utils.logger import setup_logging
from utils.runtime import resolve_device


# Worker init function to ensure each worker has a deterministic seed
# Global seed used by worker_init_fn (set in main)
_GLOBAL_SEED = 42

def worker_init_fn(worker_id):
    worker_seed = _GLOBAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_system_info():
    device = resolve_device()
    print(f"Device: {device.type.upper()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    elif device.type == "mps":
        print("GPU: Apple Metal / MPS")
    print(f"PyTorch: {torch.__version__}\n")


def _regroup_to_seq_len(ds, target_len: int):
    """
    Re-chunk a pre-tokenized dataset to a different sequence length on the fly.

    Used when MAX_SEQ_LEN_OVERRIDE is set and differs from the prep length:
    we concatenate the existing chunks, then split into target_len blocks.
    Drops the last partial block. Returns a HuggingFace Dataset with
    columns input_ids and labels at the new length.
    """
    import numpy as np
    from datasets import Dataset

    arr_ids = np.concatenate([np.asarray(r["input_ids"]) for r in ds])
    total = arr_ids.shape[0]
    n_full = total // target_len
    if n_full == 0:
        raise RuntimeError(
            f"Not enough tokens ({total}) to form even one chunk of length {target_len}"
        )
    arr_ids = arr_ids[: n_full * target_len].reshape(n_full, target_len)
    out = Dataset.from_dict({"input_ids": arr_ids, "labels": arr_ids.copy()})
    out.set_format(type="torch", columns=["input_ids", "labels"])
    return out


def _maybe_regroup(train_ds, val_ds):
    """If MAX_SEQ_LEN_OVERRIDE is set, regroup both splits to that length."""
    import os
    override = os.environ.get("MAX_SEQ_LEN_OVERRIDE")
    if override is None:
        return train_ds, val_ds
    target = int(override)
    # Only regroup if needed: peek at first row length.
    first = train_ds[0]["input_ids"]
    cur_len = first.shape[-1] if hasattr(first, "shape") else len(first)
    if cur_len == target:
        return train_ds, val_ds
    print(f"Regrouping datasets from len={cur_len} → len={target}")
    return _regroup_to_seq_len(train_ds, target), _regroup_to_seq_len(val_ds, target)


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    # CASE 0: Dataset path is already a processed on-disk dataset
    # We check if the path passed in data_cfg.dataset_path is a directory containing a dataset dict
    if os.path.isdir(data_cfg.dataset_path):
        # Check for metadata to validate max_seq_len consistency
        metadata_path = os.path.join(data_cfg.dataset_path, "prep_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    prep_metadata = json.load(f)
                    prep_max_seq = prep_metadata.get("max_seq_len")
                    if prep_max_seq and prep_max_seq != data_cfg.seq_length:
                        print("\n" + "="*70)
                        print("⚠️  WARNING: max_seq_len MISMATCH DETECTED!")
                        print("="*70)
                        print(f"  Data was prepared with max_seq_len: {prep_max_seq}")
                        print(f"  Current config has max_seq_len:     {data_cfg.seq_length}")
                        print(f"\n  This WILL cause RoPE cache shape mismatch errors!")
                        print(f"  Please re-run data preparation with current max_seq_len:")
                        print(f"    python data/prepare_mix_data.py --target_tokens 25_000_000")
                        print("="*70 + "\n")
                        override = os.environ.get("MAX_SEQ_LEN_OVERRIDE")
                        if override is not None:
                            print(f"max_seq_len mismatch (prepared={prep_max_seq}, config={data_cfg.seq_length}); "
                                  f"will regroup chunks on-the-fly via MAX_SEQ_LEN_OVERRIDE={override}")
                        else:
                            raise ValueError(f"max_seq_len mismatch: prepared={prep_max_seq}, config={data_cfg.seq_length}. Run: python data/prepare_mix_data.py --target_tokens 25_000_000 or adjust the number of tokens")
                    else:
                        print(f"✓ Validated: Data prepared with max_seq_len={prep_max_seq}")
            except json.JSONDecodeError:
                print("Warning: Could not read prep_metadata.json")
        
        # Heuristic: check if it has dataset_dict.json or state.json or just load it
        try:
            print(f"Loading pre-processed dataset from {data_cfg.dataset_path}...")
            # We assume it's a dataset with "input_ids" and "labels"
            ds = load_from_disk(data_cfg.dataset_path)
            
            # Set format to torch for preprocessed datasets
            if hasattr(ds, 'set_format'):
                # Single dataset
                if "input_ids" in ds.column_names and "labels" in ds.column_names:
                    ds.set_format(type="torch", columns=["input_ids", "labels"])
            
            # If it's a DatasetDict (train, val), return it
            if isinstance(ds, dict) or hasattr(ds, "keys"):
                if "train" in ds and "val" in ds:
                    # Set format for both splits
                    if hasattr(ds["train"], 'set_format'):
                        ds["train"].set_format(type="torch", columns=["input_ids", "labels"])
                        ds["val"].set_format(type="torch", columns=["input_ids", "labels"])
                    return _maybe_regroup(ds["train"], ds["val"])
                elif "train" in ds:
                    # Splitting manually if only train exists
                    print("Found only 'train' split. Creating validation split...")
                    splitted = ds["train"].train_test_split(test_size=0.1, seed=42)
                    # Set format for both splits
                    splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
                    splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
                    return _maybe_regroup(splitted["train"], splitted["test"])
            
            # If it's a single Dataset (just rows)
            print("Loaded single dataset. Splitting into train/val...")
            splitted = ds.train_test_split(test_size=0.1, seed=42)
            # Set format for both splits
            splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
            splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
            return splitted["train"], splitted["test"]

        except Exception as e:

            # Fallback: try loading "train" and "val" subdirectories directly
            try:
                train_path = os.path.join(data_cfg.dataset_path, "train")
                val_path = os.path.join(data_cfg.dataset_path, "val")
                if os.path.exists(train_path) and os.path.exists(val_path):
                    print(f"Loading separate train/val datasets from {data_cfg.dataset_path}...")
                    train_ds = load_from_disk(train_path)
                    val_ds = load_from_disk(val_path)
                    
                    if hasattr(train_ds, 'set_format'):
                        train_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    if hasattr(val_ds, 'set_format'):
                        val_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    return train_ds, val_ds
            except Exception as e2:
                print(f"Sub-directory load failed: {e2}")

            print(f"Could not load as direct dataset ({e}). Falling back to HF loading...")

    # cache_dir provided via argument
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    # Define what config parameters invalidate the cache
    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    # 1. Try to load valid cache
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")
    
    # 2. Rebuild cache
    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
    
    # Ensure directory exists immediately
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and split
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )
    
    # Streaming requires taking samples explicitly
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val
    
    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Tokenize and save
    print("Tokenizing train set...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    train_ds.save_to_disk(train_cache)
    
    print("Tokenizing validation set...")
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)
    val_ds.save_to_disk(val_cache)

    # Save cache info
    os.makedirs(cache_dir, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(config_state, f, indent=2)
    print("Saved dataset cache info.")

    return train_ds, val_ds


def infer_vocab_size_from_dataset(train_ds, val_ds=None, multiple: int = 64) -> int:
    """
    Infer the minimum embedding size needed for a preprocessed token dataset.

    The returned size is rounded up so the embedding table stays a bit more
    GPU-friendly than a raw max-token+1 value.
    """
    max_token = -1
    for ds in (train_ds, val_ds):
        if ds is None:
            continue
        for i in range(len(ds)):
            tokens = ds[i]["input_ids"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if len(tokens) == 0:
                continue
            row_max = max(tokens)
            if row_max > max_token:
                max_token = row_max

    if max_token < 0:
        return 0

    needed = max_token + 1
    if multiple > 1:
        needed = ((needed + multiple - 1) // multiple) * multiple
    return needed


def main():
    global _GLOBAL_SEED
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting training")

    print_system_info()
    parser = argparse.ArgumentParser(description="Train MoE Model")
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adamw_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--train_tokens", type=int, default=1_000_000_000, help="Override train_tokens")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--config_class", type=str, help="Python path to config class (e.g., configs.llm_config.LLMConfig)")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint file to load weights from")
    parser.add_argument("--compile", type=str, help="Whether to compile the model (true/false)")
    parser.add_argument("--dataset_path", type=str, help="Path to preprocessed dataset directory")
    parser.add_argument("--synthetic_data", choices=["true", "false"], default="false", help="Use a deterministic synthetic dataset for local smoke runs")
    parser.add_argument("--synthetic_train_sequences", type=int, default=256, help="Synthetic train sequence count")
    parser.add_argument("--synthetic_val_sequences", type=int, default=64, help="Synthetic validation sequence count")
    parser.add_argument("--synthetic_pattern", choices=["copy_lag", "counting"], default="copy_lag", help="Synthetic sequence pattern")
    parser.add_argument("--synthetic_lag", type=int, default=32, help="Lag used by the copy_lag synthetic pattern")
    parser.add_argument("--eval_every", type=int, help="Override eval_every steps")
    parser.add_argument("--save_every", type=int, help="Override save_every steps")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient_accumulation_steps")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency in steps")
    parser.add_argument("--max_train_seconds", type=float, help="Stop after this many active training seconds")
    parser.add_argument("--warmup", type=str, default="true", help="Whether to perform untimed compilation warmup (true/false)")
    parser.add_argument("--use_amp", type=str, help="Whether to use mixed precision (true/false)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--attention_impl",
        choices=[
            "dense",
            "local",
            "csa",
            "compressed_memory",
            "forgetting",
            "age_forgetting",
            "age_forgetting_exponential",
            "age_forgetting_sigmoid",
            "age_forgetting_cosine",
            "age_forgetting_reciprocal",
            "age_forgetting_hard_cutoff",
            "random_keyframe",
            "periodic_keyframe",
            "learned_router",
            "salience_memory",
            "usage_refresh",
            "competition",
            "hierarchical",
            "predictive",
            "surprise_retention",
            "frequency_lfu",
            "token_merge",
            "recurrent_state",
            "entropy_gated_csa",
            "cross_block_residual",
            "negative_memory",
            "hebbian_co_activation",
            "multi_res_compression",
        ],
        help="Attention implementation to train",
    )
    parser.add_argument("--csa_compression_block_size", type=int, help="CSA compression block size m")
    parser.add_argument("--csa_top_k", type=int, help="CSA number of compressed blocks selected per token")
    parser.add_argument("--csa_sliding_window_size", type=int, help="CSA local sliding window size")
    parser.add_argument("--csa_indexer_heads", type=int, help="CSA lightning indexer query heads")
    parser.add_argument("--csa_query_compression_dim", type=int, help="CSA low-rank query dimension")
    parser.add_argument("--csa_indexer_dim", type=int, help="CSA indexer key/query head dimension")
    parser.add_argument("--csa_output_groups", type=int, help="CSA grouped output projection groups")
    parser.add_argument("--csa_group_hidden_dim", type=int, help="CSA per-group hidden projection dimension")
    parser.add_argument("--forgetting_local_window_size", type=int, help="Forgetting attention local window size")
    parser.add_argument("--forgetting_memory_block_size", type=int, help="Forgetting memory block size")
    parser.add_argument("--forgetting_memory_decay_rate", type=float, help="Forgetting memory decay per block")
    parser.add_argument("--forgetting_gate_floor", type=float, help="Minimum forgetting gate value")
    parser.add_argument("--memory_local_window_size", type=int, help="Shared memory local window size")
    parser.add_argument("--memory_block_size", type=int, help="Shared memory block size")
    parser.add_argument("--memory_budget_blocks", type=int, help="Shared memory budget in blocks")
    parser.add_argument("--memory_age_decay_rate", type=float, help="Age-based decay rate")
    parser.add_argument("--memory_refresh_strength", type=float, help="Usage refresh strength")
    parser.add_argument("--memory_gate_floor", type=float, help="Minimum memory gate value")
    parser.add_argument("--memory_competition_capacity", type=int, help="Top-k capacity for competition memory")
    parser.add_argument("--memory_periodic_stride", type=int, help="Stride for periodic keyframe routing")
    parser.add_argument("--memory_router_hidden_dim", type=int, help="Hidden dim for learned router")
    parser.add_argument("--memory_router_top_k", type=int, help="Top-k blocks for learned router")
    parser.add_argument("--memory_hierarchy_levels", type=int, help="Hierarchy depth for recursive summaries")
    parser.add_argument("--memory_hierarchy_branching", type=int, help="Hierarchy branching factor")
    parser.add_argument("--memory_predictive_hidden_dim", type=int, help="Hidden dim for predictive scorer")
    parser.add_argument("--memory_predictive_top_k", type=int, help="Top-k blocks for predictive memory")
    parser.add_argument("--max_seq_len", type=int,
                        help="Override config.max_seq_len. If the dataset was prepared at a different chunk size, "
                             "chunks will be assembled / truncated on-the-fly via concat_to_seq_len.")

    args = parser.parse_args()

    # Set global seed for reproducibility
    _GLOBAL_SEED = args.seed
    set_seed(args.seed)
    config_seed = args.seed
    print(f"Random seed: {args.seed}")

    # Load Config
    if args.config_class:
        import importlib
        try:
            module_name, class_name = args.config_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ConfigClass = getattr(module, class_name)
            print(f"Loading config from {args.config_class}")
            config = ConfigClass()
        except Exception as e:
            print(f"Error loading config class {args.config_class}: {e}")
            raise e
    else:
        # Default config
        config = LLMConfig()
    config.seed = config_seed

    # Override config with args
    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.train_tokens is not None:
        config.train_tokens = args.train_tokens
    if args.compile is not None:
        config.compile_model = (args.compile.lower() == "true")
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        config.log_every = args.log_every
    if args.max_train_seconds is not None:
        config.max_train_seconds = args.max_train_seconds
    if args.use_amp is not None:
        config.use_amp = (args.use_amp.lower() == "true")
    if args.attention_impl is not None:
        config.attention_impl = args.attention_impl
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
        os.environ["MAX_SEQ_LEN_OVERRIDE"] = str(args.max_seq_len)
        print(f"max_seq_len override → {args.max_seq_len}")
    if args.csa_compression_block_size is not None:
        config.csa.compression_block_size = args.csa_compression_block_size
    if args.csa_top_k is not None:
        config.csa.top_k = args.csa_top_k
    if args.csa_sliding_window_size is not None:
        config.csa.sliding_window_size = args.csa_sliding_window_size
    if args.csa_indexer_heads is not None:
        config.csa.indexer_heads = args.csa_indexer_heads
    if args.csa_query_compression_dim is not None:
        config.csa.query_compression_dim = args.csa_query_compression_dim
    if args.csa_indexer_dim is not None:
        config.csa.indexer_dim = args.csa_indexer_dim
    if args.csa_output_groups is not None:
        config.csa.output_groups = args.csa_output_groups
    if args.csa_group_hidden_dim is not None:
        config.csa.group_hidden_dim = args.csa_group_hidden_dim
    if args.forgetting_local_window_size is not None:
        config.forgetting.local_window_size = args.forgetting_local_window_size
        config.memory_policy.local_window_size = args.forgetting_local_window_size
    if args.forgetting_memory_block_size is not None:
        config.forgetting.memory_block_size = args.forgetting_memory_block_size
        config.memory_policy.block_size = args.forgetting_memory_block_size
    if args.forgetting_memory_decay_rate is not None:
        config.forgetting.memory_decay_rate = args.forgetting_memory_decay_rate
        config.memory_policy.age_decay_rate = args.forgetting_memory_decay_rate
    if args.forgetting_gate_floor is not None:
        config.forgetting.gate_floor = args.forgetting_gate_floor
        config.memory_policy.gate_floor = args.forgetting_gate_floor
    if args.memory_local_window_size is not None:
        config.memory_policy.local_window_size = args.memory_local_window_size
    if args.memory_block_size is not None:
        config.memory_policy.block_size = args.memory_block_size
    if args.memory_budget_blocks is not None:
        config.memory_policy.memory_budget_blocks = args.memory_budget_blocks
    if args.memory_age_decay_rate is not None:
        config.memory_policy.age_decay_rate = args.memory_age_decay_rate
    if args.memory_refresh_strength is not None:
        config.memory_policy.refresh_strength = args.memory_refresh_strength
    if args.memory_gate_floor is not None:
        config.memory_policy.gate_floor = args.memory_gate_floor
    if args.memory_competition_capacity is not None:
        config.memory_policy.competition_capacity = args.memory_competition_capacity
    if args.memory_periodic_stride is not None:
        config.memory_policy.periodic_stride = args.memory_periodic_stride
    if args.memory_router_hidden_dim is not None:
        config.memory_policy.router_hidden_dim = args.memory_router_hidden_dim
    if args.memory_router_top_k is not None:
        config.memory_policy.router_top_k = args.memory_router_top_k
    if args.memory_hierarchy_levels is not None:
        config.memory_policy.hierarchy_levels = args.memory_hierarchy_levels
    if args.memory_hierarchy_branching is not None:
        config.memory_policy.hierarchy_branching = args.memory_hierarchy_branching
    if args.memory_predictive_hidden_dim is not None:
        config.memory_policy.predictive_hidden_dim = args.memory_predictive_hidden_dim
    if args.memory_predictive_top_k is not None:
        config.memory_policy.predictive_top_k = args.memory_predictive_top_k
    config.__post_init__()
    
    # Define custom milestones for validation curves and autosetup logging.
    # If the caller explicitly asked for eval_every, keep that override and do
    # not replace it with milestone-based logging.
    if args.eval_every is None:
        # For 8M benchmark (approx 488 steps)
        if config.train_tokens <= 8000000:
            config.eval_milestones = (0, 50, 100, 150, 200, 300, 400)
            config.log_every = 50
            config.eval_every = None  # Only use milestones
        # For 20M benchmark (approx 1220 steps)
        elif config.train_tokens <= 20000000:
            config.eval_milestones = (0, 100, 250, 500, 750, 1000)
            config.log_every = 100
            config.eval_every = None
        # For 100M benchmark (approx 6100 steps)
        elif config.train_tokens <= 100000000:
            config.eval_milestones = (0, 500, 1000, 2000, 3000, 4000, 5000)
            config.log_every = 250
            config.eval_every = None
        # For 1B benchmark (approx 61000 steps)
        else:
            config.eval_milestones = (0, 1000, 5000, 10000, 20000, 30000, 40000, 50000)
            config.log_every = 1000
            config.eval_every = None
    
    # Allow command line override ONLY if explicitly provided (argparse default check)
    if args.log_every != 100: # 100 is the default in parser
        config.log_every = args.log_every
    
    use_warmup = (args.warmup.lower() == "true")

    
    output_dir = args.output_dir

    # Calculate required documents dynamically
    # Assume avg 1000 tokens per doc (conservative estimate)
    # Safety factor 2.0 to ensure enough data
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    total_tokens_needed = config.train_tokens
    calc_num_docs = int((total_tokens_needed / avg_tokens_per_doc) * safety_factor)
    
    # For very short runs (debugging), we verify we have at least some docs.
    if calc_num_docs < 100:
        calc_num_docs = 100
        
    print(f"Dynamic Data Calculation:")
    print(f"  Batch: {config.batch_size}, Seq: {config.max_seq_len}, Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Target tokens: {total_tokens_needed:,}")
    print(f"  Est. docs needed (factor {safety_factor}): {calc_num_docs:,}")
    
    num_docs = calc_num_docs

    if args.synthetic_data == "true":
        from data.synthetic import SyntheticCausalDataset

        print("Loading deterministic synthetic dataset for local smoke run...")
        train_ds = SyntheticCausalDataset(
            num_sequences=args.synthetic_train_sequences,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            pattern=args.synthetic_pattern,
            lag=args.synthetic_lag,
            seed=args.seed,
        )
        val_ds = SyntheticCausalDataset(
            num_sequences=args.synthetic_val_sequences,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            pattern=args.synthetic_pattern,
            lag=args.synthetic_lag,
            seed=args.seed + 10_000,
        )
    else:
        print("Loading dataset with Hugging Face Datasets API...")
        data_cfg = DataConfig(
            dataset_path=args.dataset_path if args.dataset_path else "auto",
            seq_length=config.max_seq_len,
            num_samples=num_docs,
            cache_dir="./hf_cache",
        )

        # Show which dataset was resolved (especially useful for auto-detection)
        if not args.dataset_path:
            print(f"📂 Auto-detected dataset: {data_cfg.dataset_path}")

        tokenizer = None
        if not os.path.isdir(data_cfg.dataset_path):
            from data.loader import setup_tokenizer

            # Only inspect the tokenizer when we are preparing raw text.
            tokenizer = setup_tokenizer(data_cfg)
            config.vocab_size = tokenizer.vocab_size

        # Prepare datasets (handles caching automatically)
        train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

        if os.path.isdir(data_cfg.dataset_path):
            inferred_vocab_size = infer_vocab_size_from_dataset(train_ds, val_ds)
            if inferred_vocab_size > config.vocab_size:
                print(
                    f"⚠️  Inferred dataset vocab size {inferred_vocab_size:,} "
                    f"exceeds model vocab size {config.vocab_size:,}. "
                    f"Expanding model vocab to match."
                )
                config.vocab_size = inferred_vocab_size

    device = resolve_device()
    
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    if args.num_workers > 0:
        loader_args["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)


    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"attention: {config.attention_impl}")
    age_gate_impls = {
        "forgetting",
        "age_forgetting",
        "age_forgetting_exponential",
        "age_forgetting_sigmoid",
        "age_forgetting_cosine",
        "age_forgetting_reciprocal",
        "age_forgetting_hard_cutoff",
        "random_keyframe",
        "periodic_keyframe",
        "learned_router",
        "salience_memory",
    }
    if config.attention_impl in {"local", "csa", "compressed_memory", *age_gate_impls, "usage_refresh", "competition", "hierarchical", "predictive", "surprise_retention", "frequency_lfu", "token_merge"}:
        if config.attention_impl == "local":
            print(f"local window: {config.memory_policy.local_window_size}")
    if config.attention_impl == "csa":
        print(
            "csa: "
            f"m={config.csa.compression_block_size}, "
            f"top_k={config.csa.top_k}, "
            f"window={config.csa.sliding_window_size}, "
            f"indexer_heads={config.csa.indexer_heads}, "
            f"groups={config.csa.output_groups}"
        )
    if config.attention_impl == "compressed_memory":
        print(
            "compressed_memory: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}"
        )
    if config.attention_impl in age_gate_impls:
        print(
            "forgetting: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"decay={config.memory_policy.age_decay_rate}, "
            f"gate_floor={config.memory_policy.gate_floor}"
        )
    if config.attention_impl == "usage_refresh":
        print(
            "usage_refresh: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"decay={config.memory_policy.age_decay_rate}, "
            f"refresh={config.memory_policy.refresh_strength}"
        )
    if config.attention_impl == "competition":
        print(
            "competition: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"capacity={config.memory_policy.competition_capacity}"
        )
    if config.attention_impl == "random_keyframe":
        print(
            "random_keyframe: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}"
        )
    if config.attention_impl == "periodic_keyframe":
        print(
            "periodic_keyframe: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"stride={config.memory_policy.periodic_stride}"
        )
    if config.attention_impl == "learned_router":
        print(
            "learned_router: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"hidden={config.memory_policy.router_hidden_dim}, "
            f"top_k={config.memory_policy.router_top_k}"
        )
    if config.attention_impl == "salience_memory":
        print(
            "salience_memory: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}"
        )
    if config.attention_impl == "hierarchical":
        print(
            "hierarchical: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"levels={config.memory_policy.hierarchy_levels}, "
            f"branching={config.memory_policy.hierarchy_branching}"
        )
    if config.attention_impl == "predictive":
        print(
            "predictive: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"hidden={config.memory_policy.predictive_hidden_dim}, "
            f"top_k={config.memory_policy.predictive_top_k}"
        )
    if config.attention_impl == "surprise_retention":
        print(
            "surprise_retention: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"hidden={config.memory_policy.surprise_hidden_dim}, "
            f"top_k={config.memory_policy.surprise_top_k}"
        )
    if config.attention_impl == "frequency_lfu":
        print(
            "frequency_lfu: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"top_k={config.memory_policy.frequency_top_k}"
        )
    if config.attention_impl == "token_merge":
        print(
            "token_merge: "
            f"window={config.memory_policy.local_window_size}, "
            f"block={config.memory_policy.block_size}, "
            f"budget={config.memory_policy.memory_budget_blocks}, "
            f"merge_ratio={config.memory_policy.token_merge_ratio}"
        )
    if config.attention_impl == "recurrent_state":
        print("recurrent_state: linear attention, no block memory")
    if config.attention_impl in {
        "entropy_gated_csa",
        "cross_block_residual",
        "negative_memory",
        "hebbian_co_activation",
        "multi_res_compression",
    }:
        print(f"novel attention: {config.attention_impl}")
    print(f"train tokens: {config.train_tokens:,}")
    if getattr(config, "max_train_seconds", None) is not None:
        print(f"max train seconds: {config.max_train_seconds}")
    print(f"batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    # Build novel attention module instances for attention_impl strings that use them
    novel_attention_modules = None
    if config.attention_impl in {
        "entropy_gated_csa",
        "cross_block_residual",
        "negative_memory",
        "hebbian_co_activation",
        "multi_res_compression",
    }:
        from models.novel_attention import (
            EntropyGatedCSA,
            CrossBlockResidualAttention,
            NegativeMemoryAttention,
            HebbianCoActivationAttention,
            MultiResolutionCompressionAttention,
        )

        # Shared CSA config values
        csa = config.csa
        mp = config.memory_policy

        if config.attention_impl == "entropy_gated_csa":
            novel_attention_modules = {
                "entropy_gated_csa": EntropyGatedCSA(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    compression_block_size=csa.compression_block_size,
                    top_k=csa.top_k,
                    sliding_window_size=csa.sliding_window_size,
                    indexer_heads=csa.indexer_heads,
                    query_compression_dim=csa.query_compression_dim,
                    indexer_dim=csa.indexer_dim,
                    output_groups=csa.output_groups,
                    group_hidden_dim=csa.group_hidden_dim,
                )
            }
            print(f"novel attention: {config.attention_impl} (block_size={csa.compression_block_size}, top_k={csa.top_k})")
        elif config.attention_impl == "cross_block_residual":
            novel_attention_modules = {
                "cross_block_residual": CrossBlockResidualAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    local_window_size=mp.local_window_size,
                    block_size=mp.block_size,
                    memory_budget_blocks=mp.memory_budget_blocks,
                )
            }
            print(f"novel attention: {config.attention_impl} (window={mp.local_window_size}, block={mp.block_size})")
        elif config.attention_impl == "negative_memory":
            novel_attention_modules = {
                "negative_memory": NegativeMemoryAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    local_window_size=mp.local_window_size,
                    block_size=mp.block_size,
                    memory_budget_blocks=mp.memory_budget_blocks,
                    negative_pool_size=8,
                    suppression_strength=0.5,
                )
            }
            print(f"novel attention: {config.attention_impl}")
        elif config.attention_impl == "hebbian_co_activation":
            novel_attention_modules = {
                "hebbian_co_activation": HebbianCoActivationAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    local_window_size=mp.local_window_size,
                    block_size=mp.block_size,
                    memory_budget_blocks=mp.memory_budget_blocks,
                )
            }
            print(f"novel attention: {config.attention_impl}")
        elif config.attention_impl == "multi_res_compression":
            novel_attention_modules = {
                "multi_res_compression": MultiResolutionCompressionAttention(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    max_seq_len=config.max_seq_len,
                    compression_block_sizes=(4, 8, 16),
                    top_k_per_resolution=4,
                    sliding_window_size=csa.sliding_window_size,
                    indexer_heads=csa.indexer_heads,
                )
            }
            print(f"novel attention: {config.attention_impl}")

    train_minimal_llm(
        config,
        train_loader,
        val_loader,
        output_dir=output_dir,
        load_weights_path=args.load_checkpoint,
        novel_attention_modules=novel_attention_modules,
    )


if __name__ == "__main__":
    main()
