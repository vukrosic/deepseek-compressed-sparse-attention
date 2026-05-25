from dataclasses import dataclass


@dataclass
class ForgettingConfig:
    """Research knobs for gated compressed memory attention."""

    local_window_size: int = 64
    memory_block_size: int = 4
    memory_decay_rate: float = 0.125
    gate_floor: float = 0.0

    def __post_init__(self):
        if self.local_window_size <= 0:
            raise ValueError("local_window_size must be positive")
        if self.memory_block_size <= 0:
            raise ValueError("memory_block_size must be positive")
        if self.memory_decay_rate < 0:
            raise ValueError("memory_decay_rate must be non-negative")
        if not (0.0 <= self.gate_floor <= 1.0):
            raise ValueError("gate_floor must be between 0 and 1")
