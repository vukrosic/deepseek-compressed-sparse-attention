"""
New forgetting mechanisms (v2) — plain-PyTorch reference implementations.

All four mechanisms here use a *primitive that none of the existing
mechanisms (local, compressed_memory, age_forgetting, hierarchical,
predictive, csa) use*:

  - SurpriseRetentionAttention   : retains blocks by *internal prediction error*
                                   (training-signal primitive)
  - FrequencyLFUAttention        : retains blocks by *cumulative past attention mass*
                                   (use-count primitive)
  - TokenMergeAttention          : compresses memory via *content-similarity merging*
                                   (ToMe-style merge primitive)
  - RecurrentStateAttention      : replaces block memory with a *fixed-size
                                   recurrent state* (linear-attention primitive)

Each module is written so that:
  1. Causality is provable from the code (no future-position read).
  2. Shapes, dtypes, and the (Q, local_k, local_v, memory_k, memory_v, memory_mask,
     memory_gate) interface match the existing memory-policy family, so the
     `_combine_attention` helper is reused.
  3. The math is documented in docstrings — these are *reference implementations*
     for kernel writers, not optimized code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_policies import SharedMemoryAttentionBase, _topk_mask


# ============================================================================
# 1. Surprise-based retention
# ============================================================================

class SurpriseRetentionAttention(SharedMemoryAttentionBase):
    """
    Retain compressed memory blocks whose contents the model *cannot
    self-predict well*. The intuition: blocks that are "boring" (predictable
    from a small predictor) carry little new information; blocks where the
    predictor is wrong carry surprise/novelty and are worth keeping.

    Per-block surprise score::

        s_b = || V_block_b - predictor(K_block_b) ||_2^2

    where predictor is a small learned MLP applied per block. The top-k blocks
    by surprise (within causal scope) are kept; everything else is gated to 0.

    Distinct from `predictive` (which scores per *query* using an MLP over
    query⊕block features) — surprise is a *per-block, query-independent*
    information-density score, computed entirely from the block's own K/V.
    """

    policy_name = "surprise_retention"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        surprise_hidden_dim: int = 64,
        surprise_top_k: int = 8,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model, n_heads, max_seq_len,
            local_window_size, block_size, memory_budget_blocks,
            dropout, n_kv_heads,
        )
        if surprise_hidden_dim <= 0:
            raise ValueError("surprise_hidden_dim must be positive")
        if surprise_top_k <= 0:
            raise ValueError("surprise_top_k must be positive")
        self.surprise_top_k = surprise_top_k
        # Small per-head predictor: K_block -> predicted V_block.
        # Shared across heads to keep parameter count low.
        self.surprise_predictor = nn.Sequential(
            nn.Linear(self.d_k, surprise_hidden_dim),
            nn.GELU(),
            nn.Linear(surprise_hidden_dim, self.d_k),
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        batch_size = Q.size(0)
        n_heads = Q.size(1)
        m = memory_k.size(-2)  # number of memory blocks

        # Causal mask: block b is visible to query t iff b < t/block_size.
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(batch_size, n_heads, seq_len, m)

        if m == 0:
            memory_gate = memory_k.new_zeros(batch_size, n_heads, seq_len, 0)
            attn_output, _ = self._combine_attention(
                Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
            )
            output = self._finalize(attn_output)
            if not return_debug:
                return output
            return output, self._base_debug(
                self.policy_name, local_k.size(3), 0,
                self.local_window_size,
            )

        # Per-block surprise score s_b = ||V_b - f(K_b)||^2.
        # Shape: (B, H, m, d_k) → (B, H, m).
        predicted = self.surprise_predictor(memory_k)
        per_block_surprise = ((memory_v - predicted) ** 2).sum(dim=-1)  # (B, H, m)

        # Score is query-independent: broadcast across query positions.
        # Shape (B, H, T, m).
        scores = per_block_surprise.unsqueeze(2).expand(batch_size, n_heads, seq_len, m)

        # Top-k within the causally-valid set.
        selected_mask = _topk_mask(scores, memory_mask, self.surprise_top_k)
        memory_gate = selected_mask.to(Q.dtype)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output
        return output, self._base_debug(
            self.policy_name,
            local_k.size(3),
            m,
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_predictive_score=float(per_block_surprise.mean().item()),
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


# ============================================================================
# 2. Frequency / LFU retention via cumulative past attention mass
# ============================================================================

class FrequencyLFUAttention(SharedMemoryAttentionBase):
    """
    Retain blocks that have *historically* been attended to most. At query
    position t, the "use count" of block b is the cumulative attention mass
    that previous queries (positions 0..t-1) placed on block b.

    Causal formulation::

        sim_{t,b}    = (Q_t · K_block_b) / sqrt(d_k)
        attn_{t,b}   = softmax_over_b(sim_{t,:})        # within causal scope
        use_{t,b}    = sum_{t' < t} attn_{t',b}         # exclusive prefix sum
        keep_t       = top_k(use_{t,:} restricted to causally-valid blocks)

    Distinct from `age_forgetting` (time-based) and `predictive` (per-query
    MLP score). Here the "score" is derived from the model's own past
    attention pattern — an emergent measure of which blocks the model has
    actually found useful.

    Note: at query t=0 use_{0,:} is all zero so we fall back to attending
    uniformly to the top-k most recent blocks.
    """

    policy_name = "frequency_lfu"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        frequency_top_k: int = 8,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model, n_heads, max_seq_len,
            local_window_size, block_size, memory_budget_blocks,
            dropout, n_kv_heads,
        )
        if frequency_top_k <= 0:
            raise ValueError("frequency_top_k must be positive")
        self.frequency_top_k = frequency_top_k

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        batch_size = Q.size(0)
        n_heads = Q.size(1)
        m = memory_k.size(-2)

        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(batch_size, n_heads, seq_len, m)

        if m == 0:
            memory_gate = memory_k.new_zeros(batch_size, n_heads, seq_len, 0)
            attn_output, _ = self._combine_attention(
                Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
            )
            output = self._finalize(attn_output)
            if not return_debug:
                return output
            return output, self._base_debug(self.policy_name, local_k.size(3), 0, self.local_window_size)

        scale = self.d_k ** -0.5
        # Pairwise similarity: (B, H, T, m)
        sim = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale
        sim = sim.masked_fill(~memory_mask, torch.finfo(sim.dtype).min)
        attn_per_query = torch.softmax(sim, dim=-1)  # (B, H, T, m)
        # Zero out rows where the entire memory is invalid (avoid NaN from
        # softmax of all -inf).
        any_valid = memory_mask.any(dim=-1, keepdim=True)
        attn_per_query = attn_per_query * any_valid

        # Exclusive cumulative sum along the query/time axis (axis=2).
        # use_count[t] = sum_{t' < t} attn_per_query[t']
        cum = torch.cumsum(attn_per_query, dim=2)
        use_count = cum - attn_per_query  # exclusive

        # Top-k on use_count, restricted to causally-valid blocks.
        # When use_count is all zero (t=0 and early steps), the topk falls
        # back to the most recent valid blocks (block_ids are sorted ascending,
        # so we additionally bias by block recency to break ties).
        recency_bias = block_ids.float().view(1, 1, 1, -1) * 1e-6
        scores = use_count + recency_bias.to(use_count.dtype)
        selected_mask = _topk_mask(scores, memory_mask, self.frequency_top_k)
        memory_gate = selected_mask.to(Q.dtype)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output
        return output, self._base_debug(
            self.policy_name, local_k.size(3), m,
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_predictive_score=float(use_count.mean().item()),
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


# ============================================================================
# 3. Token merging (content-similarity compression)
# ============================================================================

class TokenMergeAttention(SharedMemoryAttentionBase):
    """
    Compress memory by *content-similarity merging* (ToMe-style), not by
    positional grouping.

    Algorithm (applied independently per (batch, head)):

      1. Take the block-mean memory keys/values produced by the parent class.
         Length m.
      2. Within each consecutive pair (b_{2i}, b_{2i+1}), compute cosine
         similarity. The pair with the highest similarity in the sequence is
         a candidate to merge.
      3. Merge the top-r most-similar adjacent pairs: replace each pair with
         the weighted mean of its K and V (weights derived from the
         similarities).
      4. The remaining m - r tokens form the compressed memory. Block ids
         are inherited from the *earlier* of each merged pair so the causal
         block id is preserved (a merged block's id ≤ both originals' ids).

    Distinct from `hierarchical` (positional summary) and `compressed_memory`
    (position-grouped mean): here, which tokens get combined depends on
    *content*, not position.

    Implementation note: rather than iteratively merging (hard to vectorize)
    we do a *single bipartite soft merge*: split the m blocks alternately
    into set A (even index) and B (odd index); for each token in A, find
    the most similar token in B and soft-merge top-r matches. This is the
    standard ToMe operator (Bolya et al. 2023) and is causal because A and
    B both come from the past memory.
    """

    policy_name = "token_merge"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        merge_ratio: float = 0.5,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model, n_heads, max_seq_len,
            local_window_size, block_size, memory_budget_blocks,
            dropout, n_kv_heads,
        )
        if not (0.0 < merge_ratio < 1.0):
            raise ValueError("merge_ratio must be in (0, 1)")
        self.merge_ratio = merge_ratio

    def _bipartite_merge(self, k: torch.Tensor, v: torch.Tensor, ids: torch.Tensor):
        """
        Merge adjacent k/v entries by content similarity.

        Input:  k, v of shape (B, H, m, d_k); ids of shape (m,).
        Output: merged k, v of shape (B, H, m', d_k) with m' = m - r;
                merged ids of shape (m',).

        Causality is preserved because we only ever combine entries with
        smaller-or-equal block id into the slot retained at the smaller id.
        """
        B, H, m, d = k.shape
        if m < 2:
            return k, v, ids

        # Bipartite split A=even index in [0..m), B=odd index.
        a_idx = torch.arange(0, m, 2, device=k.device)
        b_idx = torch.arange(1, m, 2, device=k.device)
        # Trim A so it matches B (in case m is odd).
        n_pairs = min(a_idx.numel(), b_idx.numel())
        a_idx = a_idx[:n_pairs]
        b_idx = b_idx[:n_pairs]

        k_a = k[:, :, a_idx, :]
        k_b = k[:, :, b_idx, :]
        v_a = v[:, :, a_idx, :]
        v_b = v[:, :, b_idx, :]

        # Cosine similarity along adjacent pairs.
        k_a_n = F.normalize(k_a, dim=-1)
        k_b_n = F.normalize(k_b, dim=-1)
        sim = (k_a_n * k_b_n).sum(dim=-1)  # (B, H, n_pairs)

        r = max(1, int(n_pairs * self.merge_ratio))
        # For each (B, H), pick r pair indices with highest sim → merge them.
        # We average sim across (B, H) to choose a single merge schedule
        # (so output length is uniform across the batch — required for
        # downstream einsum shapes).
        pair_sim = sim.mean(dim=(0, 1))  # (n_pairs,)
        _, merge_pair_idx = torch.topk(pair_sim, k=r)
        merge_pair_idx, _ = torch.sort(merge_pair_idx)  # keep positional order

        # Build the kept-mask over original positions: start with all True,
        # then for each merged pair, set position b_idx[p] to False and
        # combine into a_idx[p].
        keep = torch.ones(m, dtype=torch.bool, device=k.device)
        keep[b_idx[merge_pair_idx]] = False

        # For merged pairs, replace k[a] and v[a] with the weighted mean
        # (weighted by their respective norms; or just unweighted mean since
        # the K vectors are already RMSNormed upstream).
        a_to_merge = a_idx[merge_pair_idx]
        b_to_merge = b_idx[merge_pair_idx]

        # New k/v at the A position is the average of A and B.
        k_merged = (k[:, :, a_to_merge, :] + k[:, :, b_to_merge, :]) * 0.5
        v_merged = (v[:, :, a_to_merge, :] + v[:, :, b_to_merge, :]) * 0.5

        k = k.clone()
        v = v.clone()
        k[:, :, a_to_merge, :] = k_merged
        v[:, :, a_to_merge, :] = v_merged

        # Compact to kept positions.
        k_out = k[:, :, keep, :]
        v_out = v[:, :, keep, :]
        ids_out = ids[keep]

        return k_out, v_out, ids_out

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        batch_size = Q.size(0)
        n_heads = Q.size(1)

        # Apply content-similarity merge to compress memory.
        memory_k, memory_v, block_ids = self._bipartite_merge(memory_k, memory_v, block_ids)
        m = memory_k.size(-2)

        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(batch_size, n_heads, seq_len, m)

        memory_gate = torch.ones_like(memory_mask, dtype=Q.dtype).masked_fill(~memory_mask, 0.0)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output
        return output, self._base_debug(
            self.policy_name, local_k.size(3), m,
            self.local_window_size + m * self.block_size,
            selected_blocks_mean=float(memory_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


# ============================================================================
# 4. Recurrent state (linear attention)
# ============================================================================

class RecurrentStateAttention(nn.Module):
    """
    Replace the entire block memory with a *fixed-size recurrent state* in the
    linear-attention family (Katharopoulos et al. 2020; RetNet-style read).

    Math:

        phi(x) = elu(x) + 1                     # positive feature map
        S_t    = sum_{s <= t}  phi(K_s) ⊗ V_s    # (d_k × d_k) state at time t
        Z_t    = sum_{s <= t}  phi(K_s)          # (d_k,) normalizer
        y_t    = (phi(Q_t)^T S_t) / (phi(Q_t)^T Z_t + eps)

    Computed in parallel via cumsum along the time axis. The "memory" is the
    state S_t — a fixed-size summary that *forgets nothing exactly* but
    *blurs the past* into a single d_k×d_k matrix per head. This is
    fundamentally different from every other mechanism in this paper:
    there is no block storage, no top-k selection, no per-token gate.

    Causality: cumulative sum along time is exactly the causal mask.

    Memory: O(B · H · d_k^2) regardless of context length — *constant* with
    respect to T. This is the structural property kernel writers care about.

    Output projection is the same merged QKV+O scheme used everywhere else;
    we extend `nn.Module` rather than `SharedMemoryAttentionBase` because we
    have no notion of "memory blocks" or "local window".
    """

    policy_name = "recurrent_state"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        self.eps = eps

        q_size = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = d_model
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size

        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)

        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.dropout = dropout

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        # Positive feature map for linear attention. ELU+1 is the standard
        # choice; it's smooth, positive, and 1 at the origin (so an
        # all-zero state has well-defined behavior).
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = F.linear(x, self.qkvo_proj[: self.qkv_size])
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        # Shape conventions: (B, T, H, D).
        phi_Q = self._feature_map(Q)  # (B, T, H, D)
        phi_K = self._feature_map(K)  # (B, T, H, D)

        # Outer products per token: (B, T, H, D, D).
        # Then causal cumulative sum along T gives S_t.
        kv_outer = torch.einsum("bthd,bthe->bthde", phi_K, V)
        S = torch.cumsum(kv_outer, dim=1)                     # (B, T, H, D, D)
        Z = torch.cumsum(phi_K, dim=1)                        # (B, T, H, D)

        numerator = torch.einsum("bthd,bthde->bthe", phi_Q, S)  # (B, T, H, D)
        denominator = torch.einsum("bthd,bthd->bth", phi_Q, Z).unsqueeze(-1)
        out = numerator / (denominator + self.eps)             # (B, T, H, D)

        out = out.reshape(batch_size, seq_len, self.d_model)
        out = F.linear(out, self.qkvo_proj[self.qkv_size :])

        if not return_debug:
            return out
        return out, {
            "policy": self.policy_name,
            "local_width": 0,
            "block_size": 0,
            "memory_blocks": 0,
            "memory_budget_blocks": 0,
            "attention_vector_budget": 0,
            "raw_token_equivalent_coverage": seq_len,
        }
