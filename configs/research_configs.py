from dataclasses import dataclass, field

from configs.csa_config import CSAConfig
from configs.llm_config import LLMConfig


@dataclass
class CSAMacSmokeConfig(LLMConfig):
    """Small enough to run quick dense/CSA checks on a Mac CPU."""

    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 128
    vocab_size: int = 1024
    compile_model: bool = False
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8192
    muon_lr: float = 0.006
    adamw_lr: float = 0.0015
    weight_decay: float = 0.01
    use_amp: bool = False
    csa: CSAConfig = field(
        default_factory=lambda: CSAConfig(
            compression_block_size=8,
            top_k=2,
            sliding_window_size=16,
            indexer_heads=2,
            query_compression_dim=32,
            indexer_dim=16,
            output_groups=1,
        )
    )


@dataclass
class CSAMinimumPaperConfig(LLMConfig):
    """Lowest-cost real-text config that should still produce a useful curve."""

    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 8
    d_ff: int = 1024
    max_seq_len: int = 2048
    batch_size: int = 8
    train_tokens: int = 8_000_000
    csa: CSAConfig = field(
        default_factory=lambda: CSAConfig(
            compression_block_size=16,
            top_k=4,
            sliding_window_size=64,
            indexer_heads=4,
            output_groups=1,
        )
    )
