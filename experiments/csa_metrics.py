import platform
import subprocess
from typing import Any, Dict

import torch


def attention_budget(config) -> Dict[str, float | int | None]:
    seq_len = config.max_seq_len
    if config.attention_impl == "dense":
        return {
            "attention_vector_budget": seq_len / 2,
            "attention_vector_budget_max": seq_len,
            "raw_token_equivalent_coverage": seq_len,
            "dense_avg_causal_budget": seq_len / 2,
        }

    csa = config.csa
    if config.attention_impl == "local":
        window = min(csa.sliding_window_size, seq_len)
        return {
            "attention_vector_budget": window,
            "attention_vector_budget_max": window,
            "raw_token_equivalent_coverage": window,
            "dense_avg_causal_budget": seq_len / 2,
        }

    if config.attention_impl in {
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
        "compressed_memory",
        "usage_refresh",
        "competition",
        "hierarchical",
        "predictive",
    }:
        policy = config.memory_policy
        window = min(policy.local_window_size, seq_len)
        memory_blocks = min(policy.memory_budget_blocks, max(0, (seq_len - window + policy.block_size - 1) // policy.block_size))
        if config.attention_impl == "competition":
            memory_blocks = min(memory_blocks, policy.competition_capacity)
        if config.attention_impl == "predictive":
            memory_blocks = min(memory_blocks, policy.predictive_top_k)
        return {
            "attention_vector_budget": window + memory_blocks,
            "attention_vector_budget_max": window + memory_blocks,
            "raw_token_equivalent_coverage": window + memory_blocks * policy.block_size,
            "dense_avg_causal_budget": seq_len / 2,
        }

    return {
        "attention_vector_budget": csa.sliding_window_size + csa.top_k,
        "attention_vector_budget_max": csa.sliding_window_size + csa.top_k,
        "raw_token_equivalent_coverage": csa.sliding_window_size
        + csa.top_k * csa.compression_block_size,
        "dense_avg_causal_budget": seq_len / 2,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def hardware_metadata() -> Dict[str, Any]:
    gpu_name = None
    gpu_memory_total_bytes = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_total_bytes = props.total_memory
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name = "Apple MPS"

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_memory_total_bytes": gpu_memory_total_bytes,
    }


def build_run_metadata(config) -> Dict[str, Any]:
    csa = config.csa
    policy = config.memory_policy
    memory_impls = {
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
        "compressed_memory",
        "usage_refresh",
        "competition",
        "hierarchical",
        "predictive",
    }
    metadata = {
        "git_commit": git_commit(),
        "seed": getattr(config, "seed", None),
        "attention_impl": config.attention_impl,
        "train_tokens_target": config.train_tokens,
        "max_train_seconds": getattr(config, "max_train_seconds", None),
        "max_seq_len": config.max_seq_len,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "csa_top_k": csa.top_k if config.attention_impl == "csa" else None,
        "csa_compression_block_size": csa.compression_block_size
        if config.attention_impl == "csa"
        else None,
        "csa_sliding_window_size": csa.sliding_window_size
        if config.attention_impl in {"local", "csa"}
        else None,
        "csa_indexer_heads": csa.indexer_heads if config.attention_impl == "csa" else None,
        "memory_local_window_size": policy.local_window_size
        if config.attention_impl in memory_impls
        else None,
        "memory_block_size": policy.block_size
        if config.attention_impl in memory_impls
        else None,
        "memory_budget_blocks": policy.memory_budget_blocks
        if config.attention_impl in memory_impls
        else None,
        "memory_age_decay_rate": policy.age_decay_rate
        if config.attention_impl in memory_impls
        else None,
        "memory_gate_floor": policy.gate_floor
        if config.attention_impl in memory_impls
        else None,
    }
    metadata.update(attention_budget(config))
    metadata.update(hardware_metadata())
    return metadata
