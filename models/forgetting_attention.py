from __future__ import annotations

from .memory_policies import AgeForgettingAttention


class ForgettingAttention(AgeForgettingAttention):
    """Backward-compatible alias for the age-based forgetting policy."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        memory_block_size: int = 4,
        memory_decay_rate: float = 0.125,
        gate_floor: float = 0.0,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            local_window_size=local_window_size,
            block_size=memory_block_size,
            memory_budget_blocks=max(1, max_seq_len // memory_block_size),
            age_decay_rate=memory_decay_rate,
            gate_floor=gate_floor,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )

    def forward(self, x, return_debug: bool = False):
        output = super().forward(x, return_debug=return_debug)
        if not return_debug:
            return output

        logits, debug = output
        if "memory_gate" in debug and "retention_gate" not in debug:
            debug["retention_gate"] = debug["memory_gate"]
        return logits, debug
