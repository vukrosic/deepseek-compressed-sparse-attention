from dataclasses import dataclass
from typing import Optional


@dataclass
class CSAConfig:
    """Research knobs for Compressed Sparse Attention."""

    compression_block_size: int = 16
    top_k: int = 8
    sliding_window_size: int = 64
    indexer_heads: int = 4
    query_compression_dim: Optional[int] = None
    indexer_dim: Optional[int] = None
    output_groups: int = 1
    group_hidden_dim: Optional[int] = None

    def __post_init__(self):
        assert self.compression_block_size > 0, "compression_block_size must be positive"
        assert self.top_k >= 0, "top_k must be non-negative"
        assert self.sliding_window_size > 0, "sliding_window_size must be positive"
        assert self.indexer_heads > 0, "indexer_heads must be positive"
        assert self.output_groups > 0, "output_groups must be positive"
