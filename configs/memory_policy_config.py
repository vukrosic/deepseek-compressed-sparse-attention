from dataclasses import dataclass


@dataclass
class MemoryPolicyConfig:
    """Shared research knobs for the five memory philosophies."""

    local_window_size: int = 64
    block_size: int = 4
    memory_budget_blocks: int = 16

    age_decay_rate: float = 0.125
    refresh_strength: float = 0.35
    gate_floor: float = 0.0

    competition_capacity: int = 16
    hierarchy_levels: int = 2
    hierarchy_branching: int = 4

    predictive_hidden_dim: int = 32
    predictive_top_k: int = 8

    def __post_init__(self):
        if self.local_window_size <= 0:
            raise ValueError("local_window_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.memory_budget_blocks <= 0:
            raise ValueError("memory_budget_blocks must be positive")
        if self.age_decay_rate < 0:
            raise ValueError("age_decay_rate must be non-negative")
        if self.refresh_strength < 0:
            raise ValueError("refresh_strength must be non-negative")
        if not (0.0 <= self.gate_floor <= 1.0):
            raise ValueError("gate_floor must be between 0 and 1")
        if self.competition_capacity <= 0:
            raise ValueError("competition_capacity must be positive")
        if self.hierarchy_levels <= 0:
            raise ValueError("hierarchy_levels must be positive")
        if self.hierarchy_branching <= 1:
            raise ValueError("hierarchy_branching must be greater than 1")
        if self.predictive_hidden_dim <= 0:
            raise ValueError("predictive_hidden_dim must be positive")
        if self.predictive_top_k <= 0:
            raise ValueError("predictive_top_k must be positive")
