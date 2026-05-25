from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiHeadAttention


class ForgettingAttention(MultiHeadAttention):
    """
    Dense projections plus a fixed local window and gated compressed memory.

    Old tokens are compressed into fixed-size blocks. The older a block is
    relative to the query, the smaller its retention gate becomes. A gate of
    0 removes the block entirely, 1 keeps it fully, and intermediate values
    down-weight its attention mass.
    """

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
        super().__init__(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        if local_window_size <= 0:
            raise ValueError("local_window_size must be positive")
        if memory_block_size <= 0:
            raise ValueError("memory_block_size must be positive")
        if memory_decay_rate < 0:
            raise ValueError("memory_decay_rate must be non-negative")
        if not (0.0 <= gate_floor <= 1.0):
            raise ValueError("gate_floor must be between 0 and 1")

        self.local_window_size = local_window_size
        self.memory_block_size = memory_block_size
        self.memory_decay_rate = memory_decay_rate
        self.gate_floor = gate_floor
        self._gate_eps = 1e-9

    def forward(self, x: torch.Tensor, return_debug: bool = False):
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

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        local_k, local_mask, local_indices = self._gather_local_window(K)
        local_v, _, _ = self._gather_local_window(V)

        memory_k, memory_v, memory_mask, retention_gate, memory_indices = self._build_memory_bank(K, V)

        scale = self.d_k**-0.5
        local_scores = torch.einsum("bhtd,bhtwd->bhtw", Q, local_k) * scale
        memory_scores = torch.einsum("bhtd,bhmd->bhtm", Q, memory_k) * scale

        gate_mask = memory_mask & (retention_gate > 0)
        retention_gate = retention_gate.masked_fill(~gate_mask, 0.0)
        memory_scores = memory_scores + torch.log(retention_gate.clamp_min(self._gate_eps))

        local_scores = local_scores.masked_fill(~local_mask, torch.finfo(local_scores.dtype).min)
        memory_scores = memory_scores.masked_fill(~gate_mask, torch.finfo(memory_scores.dtype).min)

        combined_scores = torch.cat([local_scores, memory_scores], dim=-1)
        combined_weights = torch.softmax(combined_scores, dim=-1)

        local_width = local_scores.size(-1)
        local_weights = combined_weights[..., :local_width]
        memory_weights = combined_weights[..., local_width:]

        attn_output = torch.einsum("bhtw,bhtwd->bhtd", local_weights, local_v)
        attn_output = attn_output + torch.einsum("bhtm,bhmd->bhtd", memory_weights, memory_v)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = F.linear(attn_output, self.qkvo_proj[self.qkv_size :])

        if not return_debug:
            return output

        return output, {
            "local_indices": local_indices,
            "memory_indices": memory_indices,
            "memory_mask": memory_mask,
            "retention_gate": retention_gate,
            "local_width": local_width,
        }

    def _gather_local_window(self, tensor: torch.Tensor):
        batch_size, n_heads, seq_len, head_dim = tensor.shape
        window = min(self.local_window_size, seq_len)
        device = tensor.device

        positions = torch.arange(seq_len, device=device).view(seq_len, 1)
        offsets = torch.arange(window - 1, -1, -1, device=device).view(1, window)
        indices = positions - offsets
        mask = indices >= 0
        safe_indices = indices.clamp(min=0)

        tensor_bt = tensor.transpose(1, 2)  # [B, T, H, D]
        expanded = tensor_bt.unsqueeze(2).expand(batch_size, seq_len, window, n_heads, head_dim)
        gather_index = safe_indices.view(1, seq_len, window, 1, 1).expand(
            batch_size, seq_len, window, n_heads, head_dim
        )
        gathered = torch.gather(expanded, dim=1, index=gather_index)
        gathered = gathered.permute(0, 3, 1, 2, 4).contiguous()
        return gathered, mask.view(1, 1, seq_len, window), indices

    def _build_memory_bank(self, K: torch.Tensor, V: torch.Tensor):
        batch_size, n_heads, seq_len, head_dim = K.shape
        block_size = self.memory_block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_len = num_blocks * block_size
        pad_len = padded_len - seq_len

        if pad_len > 0:
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        K = K.view(batch_size, n_heads, num_blocks, block_size, head_dim).mean(dim=3)
        V = V.view(batch_size, n_heads, num_blocks, block_size, head_dim).mean(dim=3)

        query_blocks = torch.arange(seq_len, device=K.device) // block_size
        memory_blocks = torch.arange(num_blocks, device=K.device)

        memory_mask = memory_blocks.view(1, 1, 1, num_blocks) < query_blocks.view(1, 1, seq_len, 1)
        age = (query_blocks.view(1, 1, seq_len, 1) - memory_blocks.view(1, 1, 1, num_blocks)).clamp_min(0)
        retention_gate = torch.clamp(1.0 - self.memory_decay_rate * age.float(), min=self.gate_floor, max=1.0)
        retention_gate = retention_gate.masked_fill(~memory_mask, 0.0)

        return K, V, memory_mask, retention_gate, memory_blocks
