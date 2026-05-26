from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiHeadAttention


@dataclass
class MemoryPolicyDebug:
    policy: str
    local_width: int
    block_size: int
    memory_blocks: int
    memory_budget_blocks: int
    mean_gate: float | None = None
    mean_random_score: float | None = None
    mean_periodic_score: float | None = None
    mean_route_score: float | None = None
    mean_salience_score: float | None = None
    mean_refresh: float | None = None
    mean_utility: float | None = None
    mean_predictive_score: float | None = None
    hierarchy_levels: int | None = None
    selected_blocks_mean: float | None = None
    attention_vector_budget: int | None = None
    raw_token_equivalent_coverage: int | None = None


def _gate_horizon(rate: float) -> float:
    return max(1.0, 1.0 / max(rate, 1e-9))


def _age_gate_values(
    age: torch.Tensor,
    mode: str,
    age_decay_rate: float,
    gate_floor: float,
) -> torch.Tensor:
    """Build a bounded gate from an integer age tensor.

    All modes stay in the same family: 1.0 for fresh memory, smaller values for
    older memory, and an explicit floor when the gate should not collapse all
    the way to zero.
    """
    age = age.float()
    mode = mode.lower()
    if mode == "linear":
        gate = 1.0 - age_decay_rate * age
    elif mode == "exponential":
        gate = torch.exp(-age_decay_rate * age)
    elif mode == "reciprocal":
        gate = 1.0 / (1.0 + age_decay_rate * age)
    elif mode == "cosine":
        horizon = _gate_horizon(age_decay_rate)
        progress = (age / horizon).clamp(0.0, 1.0)
        gate = 0.5 * (1.0 + torch.cos(torch.pi * progress))
    elif mode == "sigmoid":
        horizon = _gate_horizon(age_decay_rate)
        gate = torch.sigmoid((horizon - age) * max(age_decay_rate, 1e-9) * 4.0)
    elif mode == "hard_cutoff":
        horizon = int(round(_gate_horizon(age_decay_rate)))
        gate = torch.where(age <= float(horizon), torch.ones_like(age), torch.zeros_like(age))
    else:
        raise ValueError(f"Unknown age gate mode: {mode}")

    if gate_floor > 0.0:
        gate = torch.clamp(gate, min=gate_floor, max=1.0)
    else:
        gate = torch.clamp(gate, min=0.0, max=1.0)
    return gate


def _topk_mask(scores: torch.Tensor, valid_mask: torch.Tensor, top_k: int) -> torch.Tensor:
    if scores.size(-1) == 0 or top_k <= 0:
        return torch.zeros_like(valid_mask)

    top_k = min(top_k, scores.size(-1))
    masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    top_indices = torch.topk(masked_scores, k=top_k, dim=-1).indices
    selected_mask = torch.zeros_like(valid_mask)
    selected_mask.scatter_(-1, top_indices, True)
    return selected_mask & valid_mask


class SharedMemoryAttentionBase(MultiHeadAttention):
    """Shared local-window + block-memory attention plumbing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        if local_window_size <= 0:
            raise ValueError("local_window_size must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if memory_budget_blocks <= 0:
            raise ValueError("memory_budget_blocks must be positive")

        self.local_window_size = local_window_size
        self.block_size = block_size
        self.memory_budget_blocks = memory_budget_blocks
        self._gate_eps = 1e-9

    def _project_qkv(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = F.linear(x, self.qkvo_proj[: self.qkv_size])
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)

        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))

        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        return Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

    def _local_window(self, tensor: torch.Tensor):
        batch_size, n_heads, seq_len, head_dim = tensor.shape
        window = min(self.local_window_size, seq_len)
        device = tensor.device

        positions = torch.arange(seq_len, device=device).view(seq_len, 1)
        offsets = torch.arange(window - 1, -1, -1, device=device).view(1, window)
        indices = positions - offsets
        mask = indices >= 0
        safe_indices = indices.clamp(min=0)

        tensor_bt = tensor.transpose(1, 2)
        expanded = tensor_bt.unsqueeze(2).expand(batch_size, seq_len, window, n_heads, head_dim)
        gather_index = safe_indices.view(1, seq_len, window, 1, 1).expand(
            batch_size, seq_len, window, n_heads, head_dim
        )
        gathered = torch.gather(expanded, dim=1, index=gather_index)
        gathered = gathered.permute(0, 3, 1, 2, 4).contiguous()
        return gathered, mask.view(1, 1, seq_len, window), indices

    def _block_mean(self, tensor: torch.Tensor):
        batch_size, n_heads, seq_len, head_dim = tensor.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        padded_len = num_blocks * self.block_size
        pad_len = padded_len - seq_len

        if pad_len > 0:
            tensor = F.pad(tensor, (0, 0, 0, pad_len))

        blocks = tensor.view(batch_size, n_heads, num_blocks, self.block_size, head_dim).mean(dim=3)
        block_ids = torch.arange(num_blocks, device=tensor.device)
        return blocks, block_ids

    def _recent_blocks(self, K: torch.Tensor, V: torch.Tensor):
        blocks_k, block_ids = self._block_mean(K)
        blocks_v, _ = self._block_mean(V)
        budget = min(self.memory_budget_blocks, blocks_k.size(-2))
        start = max(0, blocks_k.size(-2) - budget)
        return (
            blocks_k[:, :, start:],
            blocks_v[:, :, start:],
            block_ids[start:],
            blocks_k.size(-2),
        )

    def _base_debug(
        self,
        policy: str,
        local_width: int,
        memory_blocks: int,
        coverage_tokens: int,
        **extra: object,
    ) -> dict:
        payload = {
            "policy": policy,
            "local_width": local_width,
            "block_size": self.block_size,
            "memory_blocks": memory_blocks,
            "memory_budget_blocks": self.memory_budget_blocks,
            "attention_vector_budget": local_width + memory_blocks,
            "raw_token_equivalent_coverage": coverage_tokens,
        }
        payload.update(extra)
        return payload

    def _combine_attention(
        self,
        Q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        local_mask: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_gate: Optional[torch.Tensor] = None,
    ):
        scale = self.d_k**-0.5
        local_scores = torch.einsum("bhtd,bhtwd->bhtw", Q, local_k) * scale

        if memory_k.size(-2) == 0:
            local_scores = local_scores.masked_fill(~local_mask, torch.finfo(local_scores.dtype).min)
            local_weights = torch.softmax(local_scores, dim=-1)
            attn_output = torch.einsum("bhtw,bhtwd->bhtd", local_weights, local_v)
            return attn_output, {
                "local_width": local_scores.size(-1),
                "memory_mask": memory_mask,
                "memory_gate": None,
                "memory_weights": None,
            }

        memory_scores = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale

        if memory_gate is None:
            memory_gate = torch.ones_like(memory_scores, dtype=memory_scores.dtype)

        raw_memory_gate = memory_gate
        gate_mask = memory_mask & (raw_memory_gate > 0)
        log_memory_gate = raw_memory_gate.clamp_min(self._gate_eps)

        local_scores = local_scores.masked_fill(~local_mask, torch.finfo(local_scores.dtype).min)
        memory_scores = memory_scores + torch.log(log_memory_gate)
        memory_scores = memory_scores.masked_fill(~gate_mask, torch.finfo(memory_scores.dtype).min)

        combined_scores = torch.cat([local_scores, memory_scores], dim=-1)
        combined_weights = torch.softmax(combined_scores, dim=-1)

        local_width = local_scores.size(-1)
        local_weights = combined_weights[..., :local_width]
        memory_weights = combined_weights[..., local_width:]

        attn_output = torch.einsum("bhtw,bhtwd->bhtd", local_weights, local_v)
        attn_output = attn_output + torch.einsum("bhtm,bhmd->bhtd", memory_weights, memory_v)
        return attn_output, {
            "local_width": local_width,
            "memory_mask": memory_mask,
            "memory_gate": raw_memory_gate,
            "memory_weights": memory_weights,
        }

    def _finalize(self, attn_output: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, head_dim = attn_output.shape
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size :])


class AgeGateAttentionBase(SharedMemoryAttentionBase):
    gate_mode: str = "linear"
    policy_name: str = "age_forgetting"

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        age_decay_rate: float = 0.125,
        gate_floor: float = 0.0,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        self.age_decay_rate = age_decay_rate
        self.gate_floor = gate_floor

    def _build_memory_gate(self, age: torch.Tensor) -> torch.Tensor:
        return _age_gate_values(age, self.gate_mode, self.age_decay_rate, self.gate_floor)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        age = (query_blocks.view(1, 1, seq_len, 1) - block_ids.view(1, 1, 1, -1)).clamp_min(0)
        memory_gate = self._build_memory_gate(age)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        memory_gate = memory_gate.expand_as(memory_mask).masked_fill(~memory_mask, 0.0)
        age = age.expand_as(memory_gate)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_gate = memory_gate[memory_mask]
        mean_gate = float(valid_gate.mean().item()) if valid_gate.numel() else 0.0
        return output, self._base_debug(
            self.policy_name,
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_gate=mean_gate,
            selected_blocks_mean=float(memory_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            memory_age=age,
        )


class AgeForgettingAttention(AgeGateAttentionBase):
    gate_mode = "linear"
    policy_name = "age_forgetting"


class AgeForgettingExponentialAttention(AgeGateAttentionBase):
    gate_mode = "exponential"
    policy_name = "age_forgetting_exponential"


class AgeForgettingSigmoidAttention(AgeGateAttentionBase):
    gate_mode = "sigmoid"
    policy_name = "age_forgetting_sigmoid"


class AgeForgettingCosineAttention(AgeGateAttentionBase):
    gate_mode = "cosine"
    policy_name = "age_forgetting_cosine"


class AgeForgettingReciprocalAttention(AgeGateAttentionBase):
    gate_mode = "reciprocal"
    policy_name = "age_forgetting_reciprocal"


class AgeForgettingHardCutoffAttention(AgeGateAttentionBase):
    gate_mode = "hard_cutoff"
    policy_name = "age_forgetting_hard_cutoff"


class RandomKeyframeAttention(SharedMemoryAttentionBase):
    """Select a random subset of memory blocks within the fixed budget."""

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        if memory_k.size(-2) == 0:
            selected_mask = memory_mask.clone()
            random_scores = memory_k.new_zeros(Q.size(0), Q.size(1), seq_len, 0)
        else:
            random_scores = torch.rand(
                Q.size(0), Q.size(1), seq_len, memory_k.size(-2), device=x.device, dtype=Q.dtype
            )
            selected_mask = _topk_mask(random_scores, memory_mask, self.memory_budget_blocks)

        memory_gate = selected_mask.float()
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_scores = random_scores[selected_mask]
        return output, self._base_debug(
            "random_keyframe",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_random_score=float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


class PeriodicKeyframeAttention(SharedMemoryAttentionBase):
    """Select fixed keyframes at a configurable stride."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        periodic_stride: int = 4,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        if periodic_stride <= 0:
            raise ValueError("periodic_stride must be positive")
        self.periodic_stride = periodic_stride

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        periodic_mask = (block_ids % self.periodic_stride == 0).view(1, 1, 1, -1).expand_as(memory_mask)
        candidate_mask = memory_mask & periodic_mask
        candidate_mask = torch.where(candidate_mask.any(dim=-1, keepdim=True), candidate_mask, memory_mask)

        if memory_k.size(-2) == 0:
            selected_mask = memory_mask.clone()
            periodic_scores = memory_k.new_zeros(Q.size(0), Q.size(1), seq_len, 0)
        else:
            periodic_scores = block_ids.view(1, 1, 1, -1).expand_as(memory_mask).float()
            selected_mask = _topk_mask(periodic_scores, candidate_mask, self.memory_budget_blocks)

        memory_gate = selected_mask.float()
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_scores = periodic_scores[selected_mask]
        return output, self._base_debug(
            "periodic_keyframe",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_periodic_score=float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


class LearnedRouterAttention(SharedMemoryAttentionBase):
    """Score memory blocks with a learned router over content features."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        router_hidden_dim: int = 64,
        router_top_k: int = 8,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        if router_hidden_dim <= 0:
            raise ValueError("router_hidden_dim must be positive")
        if router_top_k <= 0:
            raise ValueError("router_top_k must be positive")
        self.router_hidden_dim = router_hidden_dim
        self.router_top_k = router_top_k
        self.router = nn.Sequential(
            nn.Linear(2 * self.d_k + 2, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        if memory_k.size(-2) == 0:
            selected_mask = memory_mask.clone()
            router_scores = memory_k.new_zeros(Q.size(0), Q.size(1), seq_len, 0)
        else:
            query_features = Q.unsqueeze(-2).expand(-1, -1, -1, memory_k.size(-2), -1)
            memory_features = memory_k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            key_norm = memory_k.norm(dim=-1).unsqueeze(2).expand(-1, -1, seq_len, -1).unsqueeze(-1)
            value_norm = memory_v.norm(dim=-1).unsqueeze(2).expand(-1, -1, seq_len, -1).unsqueeze(-1)
            features = torch.cat([query_features, memory_features, key_norm, value_norm], dim=-1)
            router_scores = self.router(features).squeeze(-1)
            selected_mask = _topk_mask(router_scores, memory_mask, self.router_top_k)

        score_gate = torch.sigmoid(router_scores) if router_scores.numel() else router_scores
        memory_gate = selected_mask.float() * score_gate
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_scores = router_scores[selected_mask]
        return output, self._base_debug(
            "learned_router",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_route_score=float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


class SalienceMemoryAttention(SharedMemoryAttentionBase):
    """Select memory by key/value salience rather than age."""

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        if memory_k.size(-2) == 0:
            selected_mask = memory_mask.clone()
            salience_scores = memory_k.new_zeros(Q.size(0), Q.size(1), seq_len, 0)
        else:
            salience_scores = memory_k.norm(dim=-1).unsqueeze(2) + memory_v.norm(dim=-1).unsqueeze(2)
            salience_scores = salience_scores.expand(-1, -1, seq_len, -1)
            selected_mask = _topk_mask(salience_scores, memory_mask, self.memory_budget_blocks)

        memory_gate = selected_mask.float()
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_scores = salience_scores[selected_mask]
        return output, self._base_debug(
            "salience_memory",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_salience_score=float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


class CompressedMemoryNoGateAttention(SharedMemoryAttentionBase):
    """Local attention plus compressed block memory with no decay gate."""

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, _ = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        memory_gate = torch.ones_like(memory_mask, dtype=Q.dtype).masked_fill(~memory_mask, 0.0)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_gate = memory_gate[memory_mask]
        return output, self._base_debug(
            "compressed_memory",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_gate=float(valid_gate.mean().item()) if valid_gate.numel() else 0.0,
            selected_blocks_mean=float(memory_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
        )


class UsageRefreshAttention(SharedMemoryAttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        age_decay_rate: float = 0.125,
        refresh_strength: float = 0.35,
        gate_floor: float = 0.0,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        self.age_decay_rate = age_decay_rate
        self.refresh_strength = refresh_strength
        self.gate_floor = gate_floor

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        age = (query_blocks.view(1, 1, seq_len, 1) - block_ids.view(1, 1, 1, -1)).clamp_min(0)
        age_gate = torch.clamp(1.0 - self.age_decay_rate * age.float(), min=self.gate_floor, max=1.0)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        age_gate = age_gate.expand_as(memory_mask).masked_fill(~memory_mask, 0.0)
        age = age.expand_as(age_gate)

        scale = self.d_k**-0.5
        similarity = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale
        if memory_k.size(-2) == 0:
            refresh_signal = similarity.new_zeros(similarity.shape)
        else:
            refresh_signal = torch.softmax(similarity.masked_fill(~memory_mask, torch.finfo(similarity.dtype).min), dim=-1)
        memory_gate = torch.clamp(age_gate + self.refresh_strength * refresh_signal, min=self.gate_floor, max=1.0)
        memory_gate = memory_gate.masked_fill(~memory_mask, 0.0)

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_gate = memory_gate[memory_mask]
        valid_refresh = refresh_signal[memory_mask]
        return output, self._base_debug(
            "usage_refresh",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_gate=float(valid_gate.mean().item()) if valid_gate.numel() else 0.0,
            mean_refresh=float(valid_refresh.mean().item()) if valid_refresh.numel() else 0.0,
            selected_blocks_mean=float(memory_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            memory_age=age,
        )


class CompetitionMemoryAttention(SharedMemoryAttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        competition_capacity: int = 16,
        age_decay_rate: float = 0.125,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        if competition_capacity <= 0:
            raise ValueError("competition_capacity must be positive")
        self.competition_capacity = competition_capacity
        self.age_decay_rate = age_decay_rate

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        age = (query_blocks.view(1, 1, seq_len, 1) - block_ids.view(1, 1, 1, -1)).clamp_min(0)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))
        age_gate = torch.clamp(1.0 - self.age_decay_rate * age.float(), min=0.0, max=1.0)
        age_gate = age_gate.expand_as(memory_mask).masked_fill(~memory_mask, 0.0)
        age = age.expand_as(age_gate)

        scale = self.d_k**-0.5
        similarity = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale
        utility = similarity + torch.log(age_gate.clamp_min(self._gate_eps))
        utility = utility.masked_fill(~memory_mask, torch.finfo(utility.dtype).min)

        if memory_k.size(-2) == 0:
            selected_mask = memory_mask.clone()
        else:
            top_k = min(self.competition_capacity, memory_k.size(-2))
            top_indices = torch.topk(utility, k=top_k, dim=-1).indices
            selected_mask = torch.zeros_like(memory_mask)
            selected_mask.scatter_(-1, top_indices, True)
            selected_mask = selected_mask & memory_mask

        memory_gate = selected_mask.float() * age_gate
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        selected_util = utility[selected_mask]
        return output, self._base_debug(
            "competition",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_utility=float(selected_util.mean().item()) if selected_util.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            memory_age=age,
        )


class HierarchicalSummarizationAttention(SharedMemoryAttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        hierarchy_levels: int = 2,
        hierarchy_branching: int = 4,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        if hierarchy_levels <= 0:
            raise ValueError("hierarchy_levels must be positive")
        if hierarchy_branching <= 1:
            raise ValueError("hierarchy_branching must be greater than 1")
        self.hierarchy_levels = hierarchy_levels
        self.hierarchy_branching = hierarchy_branching

    def _hierarchy_levels(self, K: torch.Tensor, V: torch.Tensor):
        current_k, current_v, block_ids, total_blocks = self._recent_blocks(K, V)
        levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
        current_span = self.block_size
        levels.append((current_k, current_v, block_ids, current_span))

        for _ in range(1, self.hierarchy_levels):
            if current_k.size(-2) <= 1:
                break

            groups = (current_k.size(-2) + self.hierarchy_branching - 1) // self.hierarchy_branching
            pad_len = groups * self.hierarchy_branching - current_k.size(-2)
            if pad_len > 0:
                current_k = F.pad(current_k, (0, 0, 0, pad_len))
                current_v = F.pad(current_v, (0, 0, 0, pad_len))
                padded_ids = F.pad(block_ids, (0, pad_len), value=int(block_ids[-1].item()) if block_ids.numel() else 0)
            else:
                padded_ids = block_ids

            current_k = current_k.view(
                current_k.size(0), current_k.size(1), groups, self.hierarchy_branching, current_k.size(-1)
            ).mean(dim=3)
            current_v = current_v.view(
                current_v.size(0), current_v.size(1), groups, self.hierarchy_branching, current_v.size(-1)
            ).mean(dim=3)
            block_ids = padded_ids.view(groups, self.hierarchy_branching)[:, -1]
            current_span *= self.hierarchy_branching
            levels.append((current_k, current_v, block_ids, current_span))

        return levels, total_blocks

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        levels, total_blocks = self._hierarchy_levels(K, V)

        memory_k = torch.cat([level_k for level_k, _, _, _ in levels], dim=2) if levels else K[:, :, :0]
        memory_v = torch.cat([level_v for _, level_v, _, _ in levels], dim=2) if levels else V[:, :, :0]
        memory_end_ids = torch.cat([ids for _, _, ids, _ in levels], dim=0) if levels else torch.empty(0, device=x.device, dtype=torch.long)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        if memory_end_ids.numel():
            memory_mask = memory_end_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        else:
            memory_mask = torch.zeros(Q.size(0), Q.size(1), Q.size(2), 0, device=x.device, dtype=torch.bool)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        level_weights = []
        level_sizes = []
        for level_idx, (level_k, _, _, _) in enumerate(levels):
            level_sizes.append(level_k.size(-2))
            level_weight = 1.0 / (level_idx + 1)
            level_weights.append(level_k.new_full((level_k.size(-2),), level_weight))
        memory_gate = torch.cat(level_weights, dim=0) if level_weights else memory_k.new_zeros(0)
        if memory_gate.numel():
            memory_gate = memory_gate.view(1, 1, 1, -1).expand_as(memory_mask).clone()
            memory_gate = memory_gate.masked_fill(~memory_mask, 0.0)
        else:
            memory_gate = memory_mask.float()

        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        valid_gate = memory_gate[memory_mask]
        return output, self._base_debug(
            "hierarchical",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_gate=float(valid_gate.mean().item()) if valid_gate.numel() else 0.0,
            hierarchy_levels=len(levels),
            selected_blocks_mean=float(memory_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            memory_age=(query_blocks.view(1, 1, seq_len, 1) - memory_end_ids.view(1, 1, 1, -1)).clamp_min(0)
            if memory_end_ids.numel()
            else torch.zeros(Q.size(0), Q.size(1), Q.size(2), 0, device=x.device),
        )


class PredictiveImportanceAttention(SharedMemoryAttentionBase):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        local_window_size: int,
        block_size: int,
        memory_budget_blocks: int,
        predictive_hidden_dim: int = 32,
        predictive_top_k: int = 8,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__(
            d_model,
            n_heads,
            max_seq_len,
            local_window_size,
            block_size,
            memory_budget_blocks,
            dropout,
            n_kv_heads,
        )
        if predictive_hidden_dim <= 0:
            raise ValueError("predictive_hidden_dim must be positive")
        if predictive_top_k <= 0:
            raise ValueError("predictive_top_k must be positive")
        self.predictive_hidden_dim = predictive_hidden_dim
        self.predictive_top_k = predictive_top_k
        self.predictor = nn.Sequential(
            nn.Linear(2 * self.d_k + 1, predictive_hidden_dim),
            nn.GELU(),
            nn.Linear(predictive_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        Q, K, V = self._project_qkv(x)
        local_k, local_mask, local_indices = self._local_window(K)
        local_v, _, _ = self._local_window(V)
        memory_k, memory_v, block_ids, total_blocks = self._recent_blocks(K, V)

        seq_len = x.size(1)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        memory_mask = block_ids.view(1, 1, 1, -1) < query_blocks.view(1, 1, seq_len, 1)
        memory_mask = memory_mask.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2))

        if memory_k.size(-2) == 0:
            score_gate = memory_k.new_zeros(Q.size(0), Q.size(1), Q.size(2), 0)
            selected_mask = memory_mask.clone()
        else:
            query_features = Q.unsqueeze(-2).expand(-1, -1, -1, memory_k.size(-2), -1)
            memory_features = memory_k.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            age = (query_blocks.view(1, 1, seq_len, 1) - block_ids.view(1, 1, 1, -1)).clamp_min(0).float()
            if memory_k.size(-2) > 1:
                age = age / float(memory_k.size(-2) - 1)
            age = age.expand(Q.size(0), Q.size(1), seq_len, memory_k.size(-2)).unsqueeze(-1)
            features = torch.cat([query_features, memory_features, age], dim=-1)
            scores = self.predictor(features).squeeze(-1)
            score_gate = torch.sigmoid(scores)
            scores = scores.masked_fill(~memory_mask, torch.finfo(scores.dtype).min)
            top_k = min(self.predictive_top_k, memory_k.size(-2))
            top_indices = torch.topk(scores, k=top_k, dim=-1).indices
            selected_mask = torch.zeros_like(memory_mask)
            selected_mask.scatter_(-1, top_indices, True)
            selected_mask = selected_mask & memory_mask

        memory_gate = selected_mask.float() * score_gate
        attn_output, _ = self._combine_attention(
            Q, local_k, local_v, local_mask, memory_k, memory_v, memory_mask, memory_gate
        )
        output = self._finalize(attn_output)

        if not return_debug:
            return output

        selected_scores = score_gate[selected_mask]
        return output, self._base_debug(
            "predictive",
            local_k.size(3),
            memory_k.size(-2),
            self.local_window_size + min(total_blocks, self.memory_budget_blocks) * self.block_size,
            mean_predictive_score=float(selected_scores.mean().item()) if selected_scores.numel() else 0.0,
            selected_blocks_mean=float(selected_mask.sum(dim=-1).float().mean().item()),
            memory_gate=memory_gate,
            memory_mask=memory_mask,
            memory_age=(query_blocks.view(1, 1, seq_len, 1) - block_ids.view(1, 1, 1, -1)).clamp_min(0),
        )
