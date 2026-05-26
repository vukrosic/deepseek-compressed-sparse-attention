import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import SquaredReLUFeedForward
from .compressed_sparse_attention import CompressedSparseAttention


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        seq_len = x_BTHD.size(1)
        if seq_len > self.cos_cached.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds rotary cache {self.cos_cached.size(0)}"
            )
        cos = torch.repeat_interleave(
            self.cos_cached[:seq_len].to(device=x_BTHD.device, dtype=x_BTHD.dtype),
            2,
            dim=-1,
        ).unsqueeze(0).unsqueeze(2)
        sin = torch.repeat_interleave(
            self.sin_cached[:seq_len].to(device=x_BTHD.device, dtype=x_BTHD.dtype),
            2,
            dim=-1,
        ).unsqueeze(0).unsqueeze(2)
        return (x_BTHD * cos) + (_rotate_half(x_BTHD) * sin)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        
        # ============ MERGED QKVO PROJECTION ============
        # Instead of 4 separate Linear layers, use single merged projection
        q_size = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = d_model
        
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size  # Q + K + V sizes
        
        # Single parameter tensor for all projections
        # Shape: [Q_size + K_size + V_size + O_size, d_model]
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
        # ================================================
        
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        
        # Split the result into Q, K, V
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ================================================
        
        # Reshape to multi-head format
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply RoPE
        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))
        
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        
        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:])


class LocalSlidingWindowAttention(MultiHeadAttention):
    """Dense projection path with causal attention restricted to a local window."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        window_size: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))

        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        positions = torch.arange(seq_len, device=x.device)
        query_positions = positions.view(seq_len, 1)
        key_positions = positions.view(1, seq_len)
        local_mask = (key_positions <= query_positions) & (
            key_positions >= query_positions - self.window_size + 1
        )

        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=local_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:])


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    # -------------------------------------------------------------------------
    # Novel attention mechanisms (novel_attention.py) — wire in via TransformerBlock
    # by passing the module instance directly as `attention_module`:
    #
    #   from models.novel_attention import EntropyGatedCSA, CrossBlockResidualAttention
    #   block = TransformerBlock(..., attention_impl="entropy_gated_csa",
    #                            novel_attention=EntropyGatedCSA(d_model=64, n_heads=4, ...))
    #
    # Supported attention_impl strings for novel mechanisms:
    #   "entropy_gated_csa"       → EntropyGatedCSA
    #   "cross_block_residual"    → CrossBlockResidualAttention
    #   "negative_memory"          → NegativeMemoryAttention
    #   "hebbian_co_activation"    → HebbianCoActivationAttention
    #   "multi_res_compression"    → MultiResolutionCompressionAttention
    #
    # GradientRetentionWrapper and LayerDecayAttention are wrappers — construct them
    # first, then pass as `attention_module=GradientRetentionWrapper(base, ...)` or
    # `attention_module=LayerDecayAttention(base, layer_idx=2, total_layers=12)`.
    # -------------------------------------------------------------------------

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        attention_impl: str = "dense",
        csa_config = None,
        forgetting_config = None,
        memory_policy = None,
        novel_attention = None,
    ):
        super().__init__()

        if attention_impl == "dense":
            self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        elif attention_impl == "local":
            window_size = getattr(memory_policy, "local_window_size", None)
            if window_size is None and csa_config is not None:
                window_size = csa_config.sliding_window_size
            if window_size is None:
                raise ValueError("memory_policy.local_window_size is required when attention_impl='local'")
            self.attention = LocalSlidingWindowAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                window_size=window_size,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "csa":
            if csa_config is None:
                raise ValueError("csa_config is required when attention_impl='csa'")
            self.attention = CompressedSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                compression_block_size=csa_config.compression_block_size,
                top_k=csa_config.top_k,
                sliding_window_size=csa_config.sliding_window_size,
                indexer_heads=csa_config.indexer_heads,
                query_compression_dim=csa_config.query_compression_dim,
                indexer_dim=csa_config.indexer_dim,
                output_groups=csa_config.output_groups,
                group_hidden_dim=csa_config.group_hidden_dim,
                dropout=dropout,
            )
        elif attention_impl == "compressed_memory":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='compressed_memory'")
            from .memory_policies import CompressedMemoryNoGateAttention

            self.attention = CompressedMemoryNoGateAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl in {
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
        }:
            if memory_policy is not None:
                from .memory_policies import (
                    AgeForgettingAttention,
                    AgeForgettingCosineAttention,
                    AgeForgettingExponentialAttention,
                    AgeForgettingHardCutoffAttention,
                    AgeForgettingReciprocalAttention,
                    AgeForgettingSigmoidAttention,
                    LearnedRouterAttention,
                    PeriodicKeyframeAttention,
                    RandomKeyframeAttention,
                    SalienceMemoryAttention,
                )

                attention_cls = {
                    "forgetting": AgeForgettingAttention,
                    "age_forgetting": AgeForgettingAttention,
                    "age_forgetting_exponential": AgeForgettingExponentialAttention,
                    "age_forgetting_sigmoid": AgeForgettingSigmoidAttention,
                    "age_forgetting_cosine": AgeForgettingCosineAttention,
                    "age_forgetting_reciprocal": AgeForgettingReciprocalAttention,
                    "age_forgetting_hard_cutoff": AgeForgettingHardCutoffAttention,
                    "random_keyframe": RandomKeyframeAttention,
                    "periodic_keyframe": PeriodicKeyframeAttention,
                    "learned_router": LearnedRouterAttention,
                    "salience_memory": SalienceMemoryAttention,
                }[attention_impl]

                attention_kwargs = dict(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    local_window_size=memory_policy.local_window_size,
                    block_size=memory_policy.block_size,
                    memory_budget_blocks=memory_policy.memory_budget_blocks,
                    dropout=dropout,
                    n_kv_heads=n_kv_heads,
                )
                if attention_impl in {
                    "forgetting",
                    "age_forgetting",
                    "age_forgetting_exponential",
                    "age_forgetting_sigmoid",
                    "age_forgetting_cosine",
                    "age_forgetting_reciprocal",
                    "age_forgetting_hard_cutoff",
                }:
                    attention_kwargs.update(
                        age_decay_rate=memory_policy.age_decay_rate,
                        gate_floor=memory_policy.gate_floor,
                    )
                elif attention_impl == "periodic_keyframe":
                    attention_kwargs.update(periodic_stride=memory_policy.periodic_stride)
                elif attention_impl == "learned_router":
                    attention_kwargs.update(
                        router_hidden_dim=memory_policy.router_hidden_dim,
                        router_top_k=memory_policy.router_top_k,
                    )
                elif attention_impl == "salience_memory":
                    pass

                self.attention = attention_cls(**attention_kwargs)
            else:
                if forgetting_config is None:
                    raise ValueError("forgetting_config is required when attention_impl='forgetting'")
                from .forgetting_attention import ForgettingAttention

                self.attention = ForgettingAttention(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    local_window_size=forgetting_config.local_window_size,
                    memory_block_size=forgetting_config.memory_block_size,
                    memory_decay_rate=forgetting_config.memory_decay_rate,
                    gate_floor=forgetting_config.gate_floor,
                    dropout=dropout,
                    n_kv_heads=n_kv_heads,
                )
        elif attention_impl == "usage_refresh":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='usage_refresh'")
            from .memory_policies import UsageRefreshAttention

            self.attention = UsageRefreshAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                age_decay_rate=memory_policy.age_decay_rate,
                refresh_strength=memory_policy.refresh_strength,
                gate_floor=memory_policy.gate_floor,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "competition":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='competition'")
            from .memory_policies import CompetitionMemoryAttention

            self.attention = CompetitionMemoryAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                competition_capacity=memory_policy.competition_capacity,
                age_decay_rate=memory_policy.age_decay_rate,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "hierarchical":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='hierarchical'")
            from .memory_policies import HierarchicalSummarizationAttention

            self.attention = HierarchicalSummarizationAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                hierarchy_levels=memory_policy.hierarchy_levels,
                hierarchy_branching=memory_policy.hierarchy_branching,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "predictive":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='predictive'")
            from .memory_policies import PredictiveImportanceAttention

            self.attention = PredictiveImportanceAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                predictive_hidden_dim=memory_policy.predictive_hidden_dim,
                predictive_top_k=memory_policy.predictive_top_k,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "surprise_retention":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='surprise_retention'")
            from .new_forgetting import SurpriseRetentionAttention

            self.attention = SurpriseRetentionAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                surprise_hidden_dim=memory_policy.surprise_hidden_dim,
                surprise_top_k=memory_policy.surprise_top_k,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "frequency_lfu":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='frequency_lfu'")
            from .new_forgetting import FrequencyLFUAttention

            self.attention = FrequencyLFUAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                frequency_top_k=memory_policy.frequency_top_k,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "token_merge":
            if memory_policy is None:
                raise ValueError("memory_policy is required when attention_impl='token_merge'")
            from .new_forgetting import TokenMergeAttention

            self.attention = TokenMergeAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                local_window_size=memory_policy.local_window_size,
                block_size=memory_policy.block_size,
                memory_budget_blocks=memory_policy.memory_budget_blocks,
                merge_ratio=memory_policy.token_merge_ratio,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "recurrent_state":
            from .new_forgetting import RecurrentStateAttention

            self.attention = RecurrentStateAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
                n_kv_heads=n_kv_heads,
            )
        elif attention_impl == "entropy_gated_csa":
            if novel_attention is None:
                raise ValueError("novel_attention required for entropy_gated_csa")
            self.attention = novel_attention
        elif attention_impl == "cross_block_residual":
            if novel_attention is None:
                raise ValueError("novel_attention required for cross_block_residual")
            self.attention = novel_attention
        elif attention_impl == "negative_memory":
            if novel_attention is None:
                raise ValueError("novel_attention required for negative_memory")
            self.attention = novel_attention
        elif attention_impl == "hebbian_co_activation":
            if novel_attention is None:
                raise ValueError("novel_attention required for hebbian_co_activation")
            self.attention = novel_attention
        elif attention_impl == "multi_res_compression":
            if novel_attention is None:
                raise ValueError("novel_attention required for multi_res_compression")
            self.attention = novel_attention
        else:
            raise ValueError(f"Unknown attention_impl: {attention_impl}")

        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_debug: bool = False):
        # Self-attention
        norm_x = self.norm1(x)
        if return_debug:
            try:
                attn_result = self.attention(norm_x, return_debug=True)
            except TypeError:
                attn_result = self.attention(norm_x)
        else:
            attn_result = self.attention(norm_x)

        if isinstance(attn_result, tuple):
            attn_out, attn_debug = attn_result
        else:
            attn_out, attn_debug = attn_result, None

        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        if return_debug:
            return x, {"attention": attn_debug}
        return x
