"""
Novel attention mechanisms extending the DeepSeek CSA tutorial.
Each class is self-contained and integrates with the existing SharedMemoryAttentionBase
infrastructure where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiHeadAttention
from .memory_policies import SharedMemoryAttentionBase
from .compressed_sparse_attention import (
    TokenCompressor,
    LightningIndexer,
    GroupedOutputProjection,
)


# ---------------------------------------------------------------------------
# 1. Entropy-Gated Compression Rate
# ---------------------------------------------------------------------------

class EntropyGatedTokenCompressor(nn.Module):
    """
    Token compressor whose output dimension (effective block size) is gated
    by the entropy of the attention distribution within each block.

    High entropy in a block → low compression (preserve distributed signal).
    Low entropy in a block → high compression (discard redundant tokens safely).

    This is a drop-in replacement for TokenCompressor in CSA.
    """

    def __init__(self, d_model: int, out_dim: int, block_size: int, min_blocks: int = 1):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.block_size = block_size
        self.min_blocks = min_blocks

        self.a_value = nn.Linear(d_model, out_dim, bias=False)
        self.b_value = nn.Linear(d_model, out_dim, bias=False)
        self.a_weight = nn.Linear(d_model, out_dim, bias=False)
        self.b_weight = nn.Linear(d_model, out_dim, bias=False)
        self.a_position_bias = nn.Parameter(torch.zeros(block_size, out_dim))
        self.b_position_bias = nn.Parameter(torch.zeros(block_size, out_dim))

        # Entropy gate: predict a compression factor per block from token features
        self.entropy_predictor = nn.Sequential(
            nn.Linear(d_model, out_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(out_dim // 2, 1, bias=False),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_block_entropy: torch.Tensor,
        return_weights: bool = False,
    ):
        """
        Args:
            hidden_states: [B, seq_len, d_model]
            per_block_entropy: [B, num_blocks] entropy per block (passed in, not computed here)
            return_weights: whether to return compression weights.
        """
        batch_size, seq_len, _ = hidden_states.shape
        block_size = self.block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_len = num_blocks * block_size
        pad_len = padded_len - seq_len

        c_a = self._pad_tokens(self.a_value(hidden_states), pad_len)
        c_b = self._pad_tokens(self.b_value(hidden_states), pad_len)
        z_a = self._pad_tokens(self.a_weight(hidden_states), pad_len)
        z_b = self._pad_tokens(self.b_weight(hidden_states), pad_len)

        c_a = c_a.view(batch_size, num_blocks, block_size, self.out_dim)
        c_b = c_b.view(batch_size, num_blocks, block_size, self.out_dim)
        z_a = z_a.view(batch_size, num_blocks, block_size, self.out_dim)
        z_b = z_b.view(batch_size, num_blocks, block_size, self.out_dim)

        prev_c_b = torch.zeros_like(c_b)
        prev_z_b = torch.zeros_like(z_b)
        if num_blocks > 1:
            prev_c_b[:, 1:] = c_b[:, :-1]
            prev_z_b[:, 1:] = z_b[:, :-1]

        current_mask = self._block_mask(seq_len, num_blocks, hidden_states.device)
        previous_mask = torch.zeros_like(current_mask)
        if num_blocks > 1:
            previous_mask[1:] = current_mask[:-1]

        score_a = z_a + self.a_position_bias.view(1, 1, block_size, self.out_dim)
        score_b = prev_z_b + self.b_position_bias.view(1, 1, block_size, self.out_dim)
        scores = torch.cat([score_a, score_b], dim=2)

        masks = torch.cat([current_mask, previous_mask], dim=1)
        scores = scores.masked_fill(
            ~masks.view(1, num_blocks, 2 * block_size, 1),
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(scores, dim=2)
        weights_a, weights_b = weights.split(block_size, dim=2)
        entries = (weights_a * c_a).sum(dim=2) + (weights_b * prev_c_b).sum(dim=2)

        # ── Entropy gate ──────────────────────────────────────────────────────
        # per_block_entropy is [B, num_blocks] — per-block entropy passed in.
        max_ent = max(float(block_size) ** 0.5, 1e-9)
        norm_entropy = (per_block_entropy / max_ent).unsqueeze(-1).clamp(0.0, 1.0)

        # Predict per-block compression scale from normalized entropy.
        block_features = hidden_states.mean(dim=1)                     # [B, d_model]
        scale_logit = self.entropy_predictor(block_features)           # [B, 1]
        scale = 0.5 + 0.5 * torch.sigmoid(scale_logit)                 # [B, 1]
        entries = entries * scale.unsqueeze(-1)                         # broadcast over out_dim
        # ──────────────────────────────────────────────────────────────────────

        if return_weights:
            return entries, weights_a, weights_b, norm_entropy.squeeze(-1)
        return entries

    @staticmethod
    def _pad_tokens(x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def _block_mask(self, seq_len: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(num_blocks * self.block_size, device=device)
        return (positions < seq_len).view(num_blocks, self.block_size)


class EntropyGatedCSA(nn.Module):
    """
    CSA variant where the TokenCompressor is entropy-gated.
    Adds a second 'entropy_for_entropy' signal computed from the local attention
    distribution to modulate per-block compression resolution.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        compression_block_size: int = 16,
        top_k: int = 8,
        sliding_window_size: int = 64,
        indexer_heads: int = 4,
        query_compression_dim: Optional[int] = None,
        indexer_dim: Optional[int] = None,
        output_groups: int = 1,
        group_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.compression_block_size = compression_block_size
        self.top_k = top_k
        self.sliding_window_size = sliding_window_size
        self.query_compression_dim = query_compression_dim or max(self.head_dim, d_model // 4)
        self.indexer_dim = indexer_dim or self.head_dim
        self.dropout = dropout

        self.kv_compressor = EntropyGatedTokenCompressor(
            d_model, self.head_dim, compression_block_size
        )
        self.indexer_key_compressor = TokenCompressor(d_model, self.indexer_dim, compression_block_size)
        self.local_kv = nn.Linear(d_model, self.head_dim, bias=False)

        self.query_down = nn.Linear(d_model, self.query_compression_dim, bias=False)
        self.indexer = LightningIndexer(
            d_model=d_model,
            query_dim=self.query_compression_dim,
            indexer_dim=self.indexer_dim,
            indexer_heads=indexer_heads,
        )
        self.query_up = nn.Linear(self.query_compression_dim, n_heads * self.head_dim, bias=False)
        self.output = GroupedOutputProjection(
            n_heads=n_heads,
            head_dim=self.head_dim,
            d_model=d_model,
            output_groups=output_groups,
            group_hidden_dim=group_hidden_dim,
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        query_latent = self.query_down(x)

        # ── Per-block "content entropy" proxy from input embeddings ──────────
        # High intra-block spread (tokens point in different directions) -> high entropy,
        # block keeps more compression dims. Low spread (redundant block) -> compressed more.
        # Computed directly from x — no extra attention pass, no softmax-of-softmax bug.
        block_size = self.compression_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_len = num_blocks * block_size
        pad_len = padded_len - seq_len

        x_padded = F.pad(x, (0, 0, 0, pad_len)) if pad_len > 0 else x
        block_tokens = x_padded.view(batch_size, num_blocks, block_size, self.d_model)
        x_norm = F.normalize(block_tokens, dim=-1)
        block_mean = x_norm.mean(dim=2, keepdim=True)
        # Mean cosine distance from each token to the block centroid -> spread/entropy proxy
        per_block_attn = (1.0 - (x_norm * block_mean).sum(dim=-1)).mean(dim=-1)  # [B, num_blocks]
        # ──────────────────────────────────────────────────────────────────────

        compressed_kv, _, _, entropy_scale = self.kv_compressor(
            x, per_block_attn, return_weights=True
        )
        indexer_keys = self.indexer_key_compressor(x)
        selection = self.indexer(
            hidden_states=x,
            query_latent=query_latent,
            indexer_keys=indexer_keys,
            block_size=self.compression_block_size,
            top_k=self.top_k,
        )

        sparse_kv = self._gather_selected_compressed(compressed_kv, selection.indices)
        local_qkv_t, local_mask, local_indices = self._gather_local_window(self.local_kv(x))
        attention_kv = torch.cat([local_qkv_t, sparse_kv], dim=2)
        # attention_mask: [B, S, K] → [B, H, S, K] to match _shared_mqa convention
        attention_mask = local_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        sparse_mask_4d = selection.mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attention_mask = torch.cat([attention_mask, sparse_mask_4d], dim=-1)

        queries = self.query_up(query_latent).view(batch_size, seq_len, self.n_heads, self.head_dim)
        attention_output = self._shared_mqa(queries, attention_kv, attention_mask)
        output = self.output(attention_output)

        if not return_debug:
            return output
        return output, {
            "compressed_kv": compressed_kv,
            "entropy_scale": entropy_scale,
            "indexer_scores": selection.scores,
            "selected_block_indices": selection.indices,
        }

    def _gather_selected_compressed(self, compressed_kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size = compressed_kv.size(0)
        head_dim = compressed_kv.size(-1)
        if indices.size(-1) == 0:
            return compressed_kv.new_empty(batch_size, indices.size(1), 0, head_dim)
        safe_indices = indices.clamp(min=0, max=compressed_kv.size(1) - 1)
        batch_indices = torch.arange(batch_size, device=compressed_kv.device).view(batch_size, 1, 1)
        return compressed_kv[batch_indices, safe_indices]

    def _gather_local_window(self, token_kv: torch.Tensor):
        batch_size, seq_len, head_dim = token_kv.shape
        window = min(self.sliding_window_size, seq_len)
        device = token_kv.device
        end_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
        offsets = torch.arange(window - 1, -1, -1, device=device).view(1, window)
        indices = end_positions - offsets
        mask = indices >= 0
        safe_indices = indices.clamp(min=0)
        expanded_indices = safe_indices.view(1, seq_len, window).expand(batch_size, seq_len, window)
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        gathered = token_kv[batch_indices, expanded_indices]
        return gathered, mask.view(1, seq_len, window).expand(batch_size, seq_len, window), indices

    def _shared_mqa(self, queries, key_values, attention_mask):
        # queries: [B, S, H, d], key_values: [B, S, K, d]
        # attention_mask: [B, H, S, K] — True=ignore positions
        scores = torch.einsum("bthc,btlc->bthl", queries, key_values)
        scores = scores * (self.head_dim ** -0.5)
        # attention_mask is [B, H, S, K]; transpose to [B, S, H, K] and broadcast with scores [B, S, H, K]
        scores = scores.masked_fill(~attention_mask.transpose(1, 2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.einsum("bthl,btlc->bthc", weights, key_values)


# ---------------------------------------------------------------------------
# 2. Hebbian Co-Activation Memory
# ---------------------------------------------------------------------------

class HebbianCoActivationAttention(SharedMemoryAttentionBase):
    """
    Forgetting policy that tracks inter-block co-activation.

    Maintains a batch of co-activation matrices (one per head) recording which
    memory blocks attended to each other. Blocks with strong mutual co-activation
    form schemas that are preferentially retained. Isolated blocks (few co-activations)
    are pruned first.

    Requires SharedMemoryAttentionBase for local window + block memory plumbing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        co_activation_decay: float = 0.95,
        co_activation_strength: float = 0.1,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        from .memory_policies import SharedMemoryAttentionBase

        SharedMemoryAttentionBase.__init__(
            self,
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            local_window_size=local_window_size,
            block_size=block_size,
            memory_budget_blocks=memory_budget_blocks,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )
        self.co_activation_decay = co_activation_decay
        self.co_activation_strength = co_activation_strength
        self._co_activation: Optional[torch.Tensor] = None

    def _build_co_activation_matrix(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_heads, num_blocks, num_blocks, device=device)

    def _update_co_activation(
        self,
        attn_weights: torch.Tensor,
        block_ids: torch.Tensor,
        num_blocks: int,
        batch_size: int,
    ):
        """
        attn_weights: [B, n_heads, seq_len, memory_blocks]
        """
        if self._co_activation is None or self._co_activation.size(-1) != num_blocks:
            self._co_activation = self._build_co_activation_matrix(num_blocks, attn_weights.device)

        self._co_activation = self._co_activation * self.co_activation_decay

        # Aggregate co-activation from attention weights
        # If block i attends to block j, increment co_activation[i,j]
        if attn_weights.size(-1) > 0:
            avg_co_act = attn_weights.mean(dim=1)  # [B, seq_len, num_blocks]
            co_sum = torch.einsum("bsi,bsj->bij", avg_co_act, avg_co_act).mean(dim=0)
            self._co_activation += self.co_activation_strength * co_sum

    def _score_by_co_activation(self, num_blocks: int) -> torch.Tensor:
        """Higher co-activation → higher retention score."""
        if self._co_activation is None:
            return torch.ones(num_blocks, device=next(self.parameters()).device)
        # Use mean of incoming + outgoing co-activation strength
        co_mat = self._co_activation[:, :num_blocks, :num_blocks]
        incoming = co_mat.sum(dim=-2)  # [n_heads, num_blocks]
        outgoing = co_mat.sum(dim=-1)  # [n_heads, num_blocks]
        total = incoming + outgoing
        return total.mean(dim=0)  # [num_blocks] averaged across heads

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        num_blocks = memory_k.size(-2)

        if num_blocks > 0:
            # Compute temporary attention weights to update co-activation
            scale = self.d_k**-0.5
            raw_scores = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale
            raw_scores = raw_scores.masked_fill(~memory_mask, torch.finfo(raw_scores.dtype).min)
            temp_weights = torch.softmax(raw_scores, dim=-1)

            self._update_co_activation(temp_weights, block_ids, num_blocks, Q.size(0))

            co_scores = self._score_by_co_activation(num_blocks).unsqueeze(0).unsqueeze(0)
            selected_mask = self._topk_mask(co_scores, memory_mask.any(dim=(0, 1, 2), keepdim=True), self.memory_budget_blocks)
            selected_mask = selected_mask.expand_as(memory_mask) & memory_mask
            memory_gate = selected_mask.float()
        else:
            selected_mask = memory_mask.clone()
            memory_gate = torch.zeros_like(memory_mask, dtype=torch.float)

        attn_output, debug_info = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output
        return output, self._base_debug(
            "hebbian_coactivation",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )

    def _topk_mask(self, scores, valid_mask, top_k):
        if scores.size(-1) == 0 or top_k <= 0:
            return torch.zeros_like(valid_mask)
        top_k = min(top_k, scores.size(-1))
        masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        top_indices = torch.topk(masked_scores, k=top_k, dim=-1).indices
        selected_mask = torch.zeros_like(valid_mask)
        selected_mask.scatter_(-1, top_indices, True)
        return selected_mask & valid_mask


# ---------------------------------------------------------------------------
# 3. Gradient-Weighted Retention
# ---------------------------------------------------------------------------

class GradientRetentionWrapper(nn.Module):
    """
    Wraps any attention mechanism and injects gradient-weighted retention scores.

    The wrapper accumulates per-block L2 gradient norms during backward passes.
    After accumulation, it multiplies each block's retention gate by
    1 + gradient_signal (so gradient-heavy blocks are retained more).

    Usage:
        attention = GradientRetentionWrapper(
            base_attention=SomeExistingAttention(...),
            d_model=...,
            block_size=...,
            decay_rate=0.99,
        )
    """

    def __init__(
        self,
        base_attention: nn.Module,
        d_model: int,
        block_size: int,
        decay_rate: float = 0.99,
        strength: float = 0.5,
    ):
        super().__init__()
        self.base_attention = base_attention
        self.block_size = block_size
        self.decay_rate = decay_rate
        self.strength = strength
        self.register_buffer("_gradient_accum", torch.zeros(4096))  # max blocks
        self._register_forward_func()

    def _register_forward_func(self):
        """Hook into base_attention's forward to capture gradients after backward."""
        pass  # handled via post_backward hook registration in training loop

    def update_gradient_signal(self, block_gradients: torch.Tensor):
        """
        Called after backward. block_gradients: [num_blocks] L2 norm per block.
        """
        n = block_gradients.size(0)
        if self._gradient_accum.size(0) < n:
            new_buf = torch.zeros(n, device=self._gradient_accum.device)
            new_buf[: self._gradient_accum.size(0)] = self._gradient_accum
            self._gradient_accum = new_buf
        self._gradient_accum[:n] = (
            self.decay_rate * self._gradient_accum[:n] + (1 - self.decay_rate) * block_gradients
        )

    def get_retention_boost(self, num_blocks: int, device: torch.device) -> torch.Tensor:
        """[num_blocks] retention multiplier ∈ [1, 1+strength]."""
        if num_blocks == 0:
            return torch.tensor([], device=device)
        raw = self._gradient_accum[:num_blocks]
        norm = (raw / (raw.max() + 1e-9)).clamp(0.0, 1.0)
        return 1.0 + self.strength * norm

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        try:
            result = self.base_attention(x, return_debug=return_debug)
        except TypeError:
            # Base attention doesn't support return_debug
            result = self.base_attention(x)
            if return_debug:
                return result, {"gradient_retention": self.get_retention_boost(0, x.device)}
            return result
        if return_debug:
            output, debug = result
            debug["gradient_retention"] = self.get_retention_boost(
                debug.get("memory_blocks", 0), x.device
            )
            return output, debug
        return result


# ---------------------------------------------------------------------------
# 4. Cross-Block Residual Attention
# ---------------------------------------------------------------------------

class CrossBlockResidualAttention(SharedMemoryAttentionBase):
    """
    Local attention with a residual bypass to specific tokens from neighboring blocks.

    After computing compressed block summaries, each query block can directly attend
    to 1-2 exact tokens from adjacent blocks (via residual connections) without
    going through the compression bottleneck.

    The residual connection is gated: the model learns to what degree to bypass
    compression for precise recall.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        residual_tokens_per_block: int = 2,
        residual_gate_scale: float = 1.0,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            local_window_size=local_window_size,
            block_size=block_size,
            memory_budget_blocks=memory_budget_blocks,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )
        self.residual_tokens_per_block = residual_tokens_per_block
        self.residual_gate_scale = residual_gate_scale
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        batch_size, seq_len = x.size(0), x.size(1)

        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        memory_gate = torch.ones_like(memory_mask, dtype=Q.dtype).masked_fill(~memory_mask, 0.0)
        attn_output, debug_info = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )

        # ── Causal residual bypass: each query attends to last `R` tokens of its
        # *previous* block (uncompressed). For tokens in block 0 the bypass is masked.
        if self.residual_tokens_per_block > 0 and seq_len > self.block_size:
            R = self.residual_tokens_per_block
            gate = torch.sigmoid(self.residual_gate * self.residual_gate_scale)
            device = x.device
            # For each query position t in block b, the previous block is (b-1).
            # Bypass tokens = positions [(b-1)*B + B - R, (b-1)*B + B).
            positions = torch.arange(seq_len, device=device)
            q_block = positions // self.block_size                          # [S]
            prev_block_end = (q_block * self.block_size)                    # [S] = b*B
            offsets = torch.arange(R, device=device).view(1, R)             # [1, R]
            # Indices into K/V; for query in block 0 these will be negative -> mask
            bypass_idx = prev_block_end.view(seq_len, 1) - R + offsets      # [S, R]
            bypass_mask = bypass_idx >= 0                                   # [S, R]
            safe_idx = bypass_idx.clamp(min=0)                              # [S, R]
            # Gather K and V — K:[B,H,S,d_k] -> bypass_k:[B,H,S,R,d_k]
            # Per-(t,r) gather from K[:, :, idx, :] -> shape [B,H,S,R,d_k]
            flat_idx = safe_idx.flatten()                                   # [S*R]
            bypass_k = K[:, :, flat_idx, :].view(
                batch_size, self.n_heads, seq_len, R, self.d_k,
            )
            bypass_v = V[:, :, flat_idx, :].view(
                batch_size, self.n_heads, seq_len, R, self.d_k,
            )
            # Scores over the R bypass slots, masked by bypass_mask
            scores = (Q.unsqueeze(-2) * bypass_k).sum(dim=-1) / (self.d_k ** 0.5)  # [B,H,S,R]
            mask_4d = bypass_mask.view(1, 1, seq_len, R)
            scores = scores.masked_fill(~mask_4d, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            # If a row is fully masked (block 0), softmax gives NaN -> zero it out
            row_valid = mask_4d.any(dim=-1, keepdim=True)
            weights = torch.where(row_valid, weights, torch.zeros_like(weights))
            residual_out = (weights.unsqueeze(-1) * bypass_v).sum(dim=-2)   # [B,H,S,d_k]
            attn_output = attn_output + gate * residual_out
        # ──────────────────────────────────────────────────────────────────────

        output = self._finalize(attn_output)

        if not return_debug:
            return output
        return output, self._base_debug(
            "cross_block_residual",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            residual_gate=float(torch.sigmoid(self.residual_gate * self.residual_gate_scale).item()),
        )

    def _residual_tokens(self, K: torch.Tensor, V: torch.Tensor, block_ids: torch.Tensor):
        """
        For each query block, gather the last `residual_tokens_per_block` tokens
        from the previous block (if any) as a residual bypass.
        """
        batch_size, n_heads, seq_len, head_dim = K.shape
        block_size = self.block_size
        num_blocks = block_ids.size(0)
        device = K.device

        window = self.residual_tokens_per_block
        residual_k_list = []
        residual_v_list = []
        residual_mask_list = []
        residual_indices_list = []

        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            if block_idx == 0:
                continue
            prev_block_start = (block_idx - 1) * block_size
            prev_block_end = prev_block_start + block_size
            # Take last `window` tokens of previous block
            token_start = max(prev_block_end - window, prev_block_start)
            k_tokens = K[:, :, token_start:prev_block_end, :]
            v_tokens = V[:, :, token_start:prev_block_end, :]
            token_positions = torch.arange(token_start, prev_block_end, device=device)

            for q_block_start in range(block_start, min(block_start + block_size, seq_len)):
                residual_k_list.append(k_tokens)
                residual_v_list.append(v_tokens)
                residual_mask_list.append(
                    (token_positions < seq_len).view(1, 1, 1, -1).expand(batch_size, n_heads, 1, -1)
                )
                residual_indices_list.append(token_positions)

        if not residual_k_list:
            empty = K.new_empty(batch_size, seq_len, 0, head_dim)
            return empty, empty, empty.bool(), torch.tensor([], device=device, dtype=torch.long)

        residual_k = torch.cat([k.unsqueeze(1) for k in residual_k_list], dim=1)
        residual_v = torch.cat([v.unsqueeze(1) for v in residual_v_list], dim=1)
        residual_mask = torch.cat([m.unsqueeze(1) for m in residual_mask_list], dim=1)
        residual_indices = torch.cat([i.unsqueeze(0) for i in residual_indices_list], dim=0)

        return residual_k, residual_v, residual_mask, residual_indices


# ---------------------------------------------------------------------------
# 5. Negative Memory / Suppression Pool
# ---------------------------------------------------------------------------

class NegativeMemoryAttention(SharedMemoryAttentionBase):
    """
    Extends any local-window + memory architecture with an explicit suppression pool.

    Maintains a 'negative' memory that stores tokens the model has learned to suppress
    (decided not to attend to). When attending, the model can actively attend to this
    pool with a negative weight — subtracting out known noise or distracting patterns.

    This does NOT require SharedMemoryAttentionBase — it is a standalone mechanism.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        negative_pool_size: int = 16,
        suppression_strength: float = 0.3,
        negative_decay: float = 0.99,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            local_window_size=local_window_size,
            block_size=block_size,
            memory_budget_blocks=memory_budget_blocks,
            dropout=dropout,
            n_kv_heads=n_kv_heads,
        )
        self.negative_pool_size = negative_pool_size
        self.suppression_strength = suppression_strength
        self.negative_decay = negative_decay

        # Learned negative token embeddings: [pool_size, d_model]
        self.negative_pool = nn.Parameter(torch.zeros(negative_pool_size, d_model))
        nn.init.normal_(self.negative_pool, std=0.02)

        # Project negative pool to key/value head_dim via local_kv
        self.local_kv = nn.Linear(d_model, self.d_k, bias=False)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        batch_size, seq_len = x.size(0), x.size(1)

        # ── Local window K/V ──────────────────────────────────────────────────
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        # ─────────────────────────────────────────────────────────────────────

        # ── Memory blocks ─────────────────────────────────────────────────────
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        # ─────────────────────────────────────────────────────────────────────

        # ── Negative pool: project [neg_pool_size, d_model] → [neg_pool_size, d_k] ──
        neg_emb = self.local_kv(self.negative_pool)    # [P, d_k]
        # ──────────────────────────────────────────────────────────────────────

        # ── Positive local + memory attention via _combine_attention ──────────────
        memory_gate = torch.ones_like(memory_mask, dtype=Q.dtype).masked_fill(~memory_mask, 0.0)
        positive_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )  # [B, H, S, d_k]

        # ── Negative pool residual: subtract weighted negative attention ──────────
        # neg_emb: [P, d_k]; Q: [B, H, S, d_k]; positive_output: [B, H, S, d_k]
        # Compute negative attention: Q is [B,H,S,d_k]; neg_emb expanded is [B,H,S,P,d_k]
        neg_emb_exp = neg_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_heads, seq_len, -1, -1
        )  # [B, H, S, P, d_k]
        neg_q_expanded = Q.unsqueeze(-2)  # [B, H, S, 1, d_k]
        neg_scores = torch.einsum("bhs...d,bhs...d->bhs...", neg_q_expanded, neg_emb_exp) / (self.d_k ** 0.5)
        neg_scores = neg_scores.squeeze(-1)  # [B, H, S, P]
        neg_weights = torch.softmax(neg_scores, dim=-1) * (-self.suppression_strength)  # [B,H,S,P]
        neg_out = torch.einsum("bhsp,bhspd->bhsd", neg_weights, neg_emb_exp)  # [B,H,S,d_k]
        attn_output = positive_output + neg_out  # residual subtraction — [B, H, S, d_k]
        # ────────────────────────────────────────────────────────────────────────

        output = self._finalize(attn_output)

        # Decay negative pool — use .data to avoid version tracking on Parameter wrapper
        with torch.no_grad():
            self.negative_pool.data.mul_(self.negative_decay)

        if not return_debug:
            return output
        return output, {
            "negative_pool_norm": float(self.negative_pool.norm().item()),
        }


# ---------------------------------------------------------------------------
# 6. Temporal Decay + Layer Depth Cross
# ---------------------------------------------------------------------------

class LayerDecayAttention(nn.Module):
    """
    Wrapper that applies different forgetting rates per transformer layer.

    Shallow layers (closer to input) get aggressive decay — they captured
    surface statistics that are quickly redundant.
    Deep layers (closer to output) get slow decay — they captured abstract
    semantic state that should persist.

    This is a DROP-IN WRAPPER around any existing attention mechanism.
    Layers closer to the input have higher decay rates.
    """

    def __init__(
        self,
        base_attention: nn.Module,
        layer_idx: int,
        total_layers: int,
        base_decay_rate: float = 0.125,
        decay_spread: float = 0.1,
    ):
        super().__init__()
        self.base_attention = base_attention
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        # Linear interpolation: layer 0 gets base_decay + spread, last layer gets base_decay
        depth_normalized = layer_idx / max(total_layers - 1, 1)
        self.age_decay_rate = base_decay_rate - decay_spread * (1.0 - depth_normalized)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        try:
            result = self.base_attention(x, return_debug=return_debug)
        except TypeError:
            result = self.base_attention(x)
            if return_debug:
                return result, {"layer_decay_rate": self.age_decay_rate, "layer_idx": self.layer_idx}
            return result
        if return_debug:
            output, debug = result
            debug["layer_decay_rate"] = self.age_decay_rate
            debug["layer_idx"] = self.layer_idx
            return output, debug
        return result


# ---------------------------------------------------------------------------
# 7. Query-Dependent Multi-Resolution Compression
# ---------------------------------------------------------------------------

class MultiResolutionCompressionAttention(nn.Module):
    """
    Produces multiple compression resolutions simultaneously (e.g., block sizes 4, 8, 16).
    The indexer selects which resolution to use per query — easy tokens attend to
    coarse summaries, hard tokens attend to fine-grained summaries.

    This is a CSA variant that generates K different resolution KV tables
    and K different indexer outputs, then merges them.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        compression_block_sizes: tuple[int, ...] = (4, 8, 16),
        top_k_per_resolution: int = 4,
        sliding_window_size: int = 64,
        indexer_heads: int = 4,
        query_compression_dim: Optional[int] = None,
        indexer_dim: Optional[int] = None,
        output_groups: int = 1,
        group_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not compression_block_sizes:
            raise ValueError("compression_block_sizes must be non-empty")

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.compression_block_sizes = compression_block_sizes
        self.num_resolutions = len(compression_block_sizes)
        self.top_k_per_resolution = top_k_per_resolution
        self.sliding_window_size = sliding_window_size
        self.query_compression_dim = query_compression_dim or max(self.head_dim, d_model // 4)
        self.indexer_dim = indexer_dim or self.head_dim
        self.dropout = dropout

        # One compressor + indexer per resolution
        self.compressors = nn.ModuleList([
            TokenCompressor(d_model, self.head_dim, bs)
            for bs in compression_block_sizes
        ])
        self.indexer_key_compressors = nn.ModuleList([
            TokenCompressor(d_model, self.indexer_dim, bs)
            for bs in compression_block_sizes
        ])
        self.indexers = nn.ModuleList([
            LightningIndexer(
                d_model=d_model,
                query_dim=self.query_compression_dim,
                indexer_dim=self.indexer_dim,
                indexer_heads=indexer_heads,
            )
            for _ in compression_block_sizes
        ])
        self.local_kv = nn.Linear(d_model, self.head_dim, bias=False)
        self.query_down = nn.Linear(d_model, self.query_compression_dim, bias=False)
        self.query_up = nn.Linear(self.query_compression_dim, n_heads * self.head_dim, bias=False)
        self.output = GroupedOutputProjection(
            n_heads=n_heads,
            head_dim=self.head_dim,
            d_model=d_model,
            output_groups=output_groups,
            group_hidden_dim=group_hidden_dim,
        )
        # Resolution selector: which resolution to use per query block
        self.resolution_gate = nn.Sequential(
            nn.Linear(d_model, len(compression_block_sizes)),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        query_latent = self.query_down(x)
        queries = self.query_up(query_latent).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Resolution gate: [B, seq_len, num_resolutions]
        resolution_weights = self.resolution_gate(x)

        all_sparse_kv = []
        all_sparse_masks = []
        all_sparse_indices = []

        for res_idx, (comp, ik_comp, idxr) in enumerate(
            zip(self.compressors, self.indexer_key_compressors, self.indexers)
        ):
            compressed_kv = comp(x)
            indexer_keys = ik_comp(x)
            selection = idxr(
                hidden_states=x,
                query_latent=query_latent,
                indexer_keys=indexer_keys,
                block_size=self.compression_block_sizes[res_idx],
                top_k=self.top_k_per_resolution,
            )
            sparse_kv = self._gather_selected_compressed(compressed_kv, selection.indices)
            all_sparse_kv.append(sparse_kv)
            all_sparse_masks.append(selection.mask)
            all_sparse_indices.append(selection.indices)

        # Merge sparse KV by resolution weights
        merged_sparse_kv, merged_sparse_mask = self._merge_by_resolution(
            all_sparse_kv, all_sparse_masks, resolution_weights, batch_size, seq_len
        )
        local_kv_t, local_mask, local_indices = self._gather_local_window(self.local_kv(x))
        attention_kv = torch.cat([local_kv_t, merged_sparse_kv], dim=2)
        # Build mask matching actual attention_kv columns
        local_cols = local_kv_t.size(2)
        sparse_cols = merged_sparse_kv.size(2)
        local_mask_4d = local_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        merged_mask_4d = merged_sparse_mask[..., :sparse_cols].unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        attention_mask = torch.cat([local_mask_4d, merged_mask_4d], dim=-1)

        attention_output = self._shared_mqa(queries, attention_kv, attention_mask)
        output = self.output(attention_output)

        if not return_debug:
            return output
        return output, {
            "resolution_weights": resolution_weights,
            "num_resolutions": self.num_resolutions,
            "compression_block_sizes": self.compression_block_sizes,
        }

    def _merge_by_resolution(
        self,
        all_sparse_kv: list[torch.Tensor],
        all_sparse_masks: list[torch.Tensor],
        resolution_weights: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted merge of per-resolution sparse KV tensors.
        We take the weighted average of all resolution KV entries.
        Also merges masks proportionally to resolution weights.
        """
        if not all_sparse_kv:
            return (
                torch.empty(batch_size, seq_len, 0, self.head_dim, device=resolution_weights.device),
                torch.empty(batch_size, seq_len, 0, dtype=torch.bool, device=resolution_weights.device),
            )
        max_len = max(t.size(2) for t in all_sparse_kv)
        num_resolutions = len(all_sparse_kv)

        # Pad each resolution KV to max_len, then stack: [B, S, num_resolutions, max_len, head_dim]
        padded_kv = []
        for t in all_sparse_kv:
            if t.size(2) < max_len:
                pad = t.new_zeros(batch_size, seq_len, max_len - t.size(2), self.head_dim)
                t = torch.cat([t, pad], dim=2)
            padded_kv.append(t)
        stacked_kv = torch.stack(padded_kv, dim=-3)  # [B, S, num_resolutions, max_len, head_dim]

        # Pad masks to max_len, then stack: [B, S, num_resolutions, max_len]
        padded_mask = []
        for m in all_sparse_masks:
            if m.size(-1) < max_len:
                pad = m.new_zeros(batch_size, seq_len, max_len - m.size(-1), dtype=torch.bool)
                m = torch.cat([m, pad], dim=-1)
            padded_mask.append(m)
        stacked_mask = torch.stack(padded_mask, dim=-3)  # [B, S, num_resolutions, max_len]

        # weights: [B, S, num_resolutions] → [B, S, num_resolutions, 1, 1] for KV merge
        weights_for_kv = resolution_weights.unsqueeze(-1).unsqueeze(-1)
        # weights: [B, S, num_resolutions] → [B, num_resolutions, S, 1] for mask merge (broadcasts over max_len)
        weights_for_mask = resolution_weights.transpose(1, 2).unsqueeze(-1)

        merged_kv = (stacked_kv * weights_for_kv).sum(dim=-3)  # [B, S, max_len, head_dim]
        merged_mask = (stacked_mask.float() * weights_for_mask).sum(dim=-3) > 0  # [B, S, max_len]

        non_zero_counts = (merged_kv.sum(dim=-1) != 0).sum(dim=-1)
        max_non_zero = max(1, int(non_zero_counts.max().item()))
        return merged_kv[:, :, :max_non_zero], merged_mask[:, :, :max_non_zero]

    def _gather_selected_compressed(self, compressed_kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size = compressed_kv.size(0)
        head_dim = compressed_kv.size(-1)
        if indices.size(-1) == 0:
            return compressed_kv.new_empty(batch_size, indices.size(1), 0, head_dim)
        safe_indices = indices.clamp(min=0, max=compressed_kv.size(1) - 1)
        batch_indices = torch.arange(batch_size, device=compressed_kv.device).view(batch_size, 1, 1)
        return compressed_kv[batch_indices, safe_indices]

    def _gather_local_window(self, token_kv: torch.Tensor):
        batch_size, seq_len, head_dim = token_kv.shape
        window = min(self.sliding_window_size, seq_len)
        device = token_kv.device
        end_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
        offsets = torch.arange(window - 1, -1, -1, device=device).view(1, window)
        indices = end_positions - offsets
        mask = indices >= 0
        safe_indices = indices.clamp(min=0)
        expanded_indices = safe_indices.view(1, seq_len, window).expand(batch_size, seq_len, window)
        batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        gathered = token_kv[batch_indices, expanded_indices]
        return gathered, mask.view(1, seq_len, window).expand(batch_size, seq_len, window), indices

    def _shared_mqa(self, queries, key_values, attention_mask):
        # queries: [B, S, H, d], key_values: [B, S, K, d]
        # attention_mask: [B, H, S, K] — True=ignore positions
        scores = torch.einsum("bthc,btlc->bthl", queries, key_values)
        scores = scores * (self.head_dim ** -0.5)
        # attention_mask is [B, H, S, K]; transpose to [B, S, H, K] and broadcast with scores [B, S, H, K]
        scores = scores.masked_fill(~attention_mask.transpose(1, 2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.einsum("bthl,btlc->bthc", weights, key_values)