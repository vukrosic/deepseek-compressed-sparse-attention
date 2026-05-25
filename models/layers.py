import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward
from .compressed_sparse_attention import CompressedSparseAttention


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=10000
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


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
    ):
        super().__init__()

        if attention_impl == "dense":
            self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        elif attention_impl == "local":
            if csa_config is None:
                raise ValueError("csa_config is required when attention_impl='local'")
            self.attention = LocalSlidingWindowAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                window_size=csa_config.sliding_window_size,
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
        elif attention_impl == "forgetting":
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
        else:
            raise ValueError(f"Unknown attention_impl: {attention_impl}")

        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
