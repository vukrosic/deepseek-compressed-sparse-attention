from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_device() -> torch.device:
    """Pick the best available accelerator in one place."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_dtype_for_device(device: torch.device, use_amp: bool = True) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if use_amp else torch.float32
    if device.type == "mps":
        return torch.float16 if use_amp else torch.float32
    return torch.float32


def autocast_for_device(device: torch.device, enabled: bool = True):
    if not enabled or device.type == "cpu":
        return nullcontext()

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: torch.device) -> tuple[int, int]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    return 0, 0
