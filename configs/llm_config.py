from dataclasses import dataclass, field
from typing import Optional, Tuple
from configs.csa_config import CSAConfig
from configs.forgetting_config import ForgettingConfig
from configs.memory_policy_config import MemoryPolicyConfig


@dataclass
class LLMConfig:
    # Model architecture (88M Params)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 22
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      

    # Attention implementation
    # "dense" keeps the original baseline. "local" keeps only a sliding window.
    # "csa" enables local attention plus compressed old memory.
    # "compressed_memory" keeps local attention plus ungated compressed memory.
    # "forgetting" / "age_forgetting" keeps local attention plus gated compressed memory.
    # "age_forgetting_exponential", "age_forgetting_sigmoid",
    # "age_forgetting_cosine", "age_forgetting_reciprocal", and
    # "age_forgetting_hard_cutoff" are extra gate variants on the same base.
    # "random_keyframe", "periodic_keyframe", "learned_router", and
    # "salience_memory" are structurally different routing mechanisms over the
    # same local+memory base.
    # "usage_refresh", "competition", "hierarchical", and "predictive" are
    # alternate memory philosophies that share the same local+compressed base.
    attention_impl: str = "dense"
    csa: CSAConfig = field(default_factory=CSAConfig)
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    memory_policy: MemoryPolicyConfig = field(default_factory=MemoryPolicyConfig)
    
    # Data params
    # ⚠️ WARNING: For simplicity, I recomend not changing max_seq_len
    # If you change max_seq_len, you MUST re-run data preparation!
    # The data preparation script chunks data at this exact length, and the RoPE
    # cache is initialized with this value. Mismatches will cause runtime errors.
    # Run: python data/prepare_mix_data.py --target_tokens 25_000_000
    # you may change the number of tokens
    max_seq_len: int = 2048  # check the warning above
    vocab_size: int = 49152  
    
    # Base Training Defaults
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8000000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.attention_impl in {
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
        }, "attention_impl must be a supported attention policy"
        self.csa.__post_init__()
        self.forgetting.__post_init__()
        self.memory_policy.__post_init__()
