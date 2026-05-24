from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressionOutput(NamedTuple):
    entries: torch.Tensor
    weights_a: torch.Tensor
    weights_b: torch.Tensor


class IndexSelection(NamedTuple):
    indices: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor


class TokenCompressor(nn.Module):
    """
    Token-level compressor from DeepSeek-V4 CSA equations 9-12.

    For compressed block i, C^a comes from the current block and C^b comes from
    the previous block. The first previous block is masked as negative infinity.
    """

    def __init__(self, d_model: int, out_dim: int, block_size: int):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.block_size = block_size

        self.a_value = nn.Linear(d_model, out_dim, bias=False)
        self.b_value = nn.Linear(d_model, out_dim, bias=False)
        self.a_weight = nn.Linear(d_model, out_dim, bias=False)
        self.b_weight = nn.Linear(d_model, out_dim, bias=False)
        self.a_position_bias = nn.Parameter(torch.zeros(block_size, out_dim))
        self.b_position_bias = nn.Parameter(torch.zeros(block_size, out_dim))

    def forward(self, hidden_states: torch.Tensor, return_weights: bool = False):
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

        if return_weights:
            return CompressionOutput(entries=entries, weights_a=weights_a, weights_b=weights_b)
        return entries

    @staticmethod
    def _pad_tokens(x: torch.Tensor, pad_len: int) -> torch.Tensor:
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def _block_mask(self, seq_len: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(num_blocks * self.block_size, device=device)
        return (positions < seq_len).view(num_blocks, self.block_size)


class LightningIndexer(nn.Module):
    """Low-rank top-k compressed block selector from equations 13-17."""

    def __init__(self, d_model: int, query_dim: int, indexer_dim: int, indexer_heads: int):
        super().__init__()
        self.indexer_dim = indexer_dim
        self.indexer_heads = indexer_heads
        self.query_up = nn.Linear(query_dim, indexer_heads * indexer_dim, bias=False)
        self.head_weight = nn.Linear(d_model, indexer_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_latent: torch.Tensor,
        indexer_keys: torch.Tensor,
        block_size: int,
        top_k: int,
    ) -> IndexSelection:
        batch_size, seq_len, _ = hidden_states.shape
        num_blocks = indexer_keys.size(1)
        device = hidden_states.device

        if top_k == 0:
            empty_indices = torch.empty(batch_size, seq_len, 0, dtype=torch.long, device=device)
            empty_mask = torch.empty(batch_size, seq_len, 0, dtype=torch.bool, device=device)
            empty_scores = hidden_states.new_full(
                (batch_size, seq_len, num_blocks),
                torch.finfo(hidden_states.dtype).min,
            )
            return IndexSelection(indices=empty_indices, mask=empty_mask, scores=empty_scores)

        indexer_queries = self.query_up(query_latent).view(
            batch_size, seq_len, self.indexer_heads, self.indexer_dim
        )
        head_weights = self.head_weight(hidden_states)

        per_head_scores = torch.einsum("bthc,bsc->bths", indexer_queries, indexer_keys)
        scores = (head_weights.unsqueeze(-1) * F.relu(per_head_scores)).sum(dim=2)

        query_blocks = torch.arange(seq_len, device=device) // block_size
        block_ids = torch.arange(num_blocks, device=device)
        causal_mask = block_ids.view(1, num_blocks) < query_blocks.view(seq_len, 1)
        scores = scores.masked_fill(~causal_mask.view(1, seq_len, num_blocks), torch.finfo(scores.dtype).min)

        actual_top_k = min(top_k, num_blocks)
        top_scores, top_indices = torch.topk(scores, k=actual_top_k, dim=-1)
        top_mask = torch.isfinite(top_scores) & (top_scores > torch.finfo(top_scores.dtype).min)
        return IndexSelection(indices=top_indices, mask=top_mask, scores=scores)


class GroupedOutputProjection(nn.Module):
    """Grouped output projection from page 11 of the DeepSeek-V4 paper."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        d_model: int,
        output_groups: int = 1,
        group_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if n_heads % output_groups != 0:
            raise ValueError("n_heads must be divisible by output_groups")
        if d_model % output_groups != 0 and group_hidden_dim is None:
            raise ValueError("d_model must be divisible by output_groups when group_hidden_dim is not set")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.output_groups = output_groups
        self.heads_per_group = n_heads // output_groups
        self.group_input_dim = self.heads_per_group * head_dim
        self.group_hidden_dim = group_hidden_dim or (d_model // output_groups)

        self.group_projections = nn.ModuleList(
            [nn.Linear(self.group_input_dim, self.group_hidden_dim, bias=False) for _ in range(output_groups)]
        )
        self.output_projection = nn.Linear(output_groups * self.group_hidden_dim, d_model, bias=False)

    def forward(self, attention_output: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = attention_output.shape
        grouped = attention_output.view(
            batch_size,
            seq_len,
            self.output_groups,
            self.heads_per_group * self.head_dim,
        )
        projected = [
            projection(grouped[:, :, group_idx])
            for group_idx, projection in enumerate(self.group_projections)
        ]
        return self.output_projection(torch.cat(projected, dim=-1))


class CompressedSparseAttention(nn.Module):
    """
    Research implementation of Compressed Sparse Attention.

    This favors clarity and debuggability over custom CUDA kernels. It is meant
    to define the math cleanly before replacing gathers/top-k with GPU kernels.
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

        self.kv_compressor = TokenCompressor(d_model, self.head_dim, compression_block_size)
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

        compressed_kv = self.kv_compressor(x)
        indexer_keys = self.indexer_key_compressor(x)
        query_latent = self.query_down(x)
        selection = self.indexer(
            hidden_states=x,
            query_latent=query_latent,
            indexer_keys=indexer_keys,
            block_size=self.compression_block_size,
            top_k=self.top_k,
        )

        sparse_kv = self._gather_selected_compressed(compressed_kv, selection.indices)
        local_kv, local_mask, local_indices = self._gather_local_window(self.local_kv(x))
        attention_kv = torch.cat([local_kv, sparse_kv], dim=2)
        attention_mask = torch.cat([local_mask, selection.mask], dim=2)

        queries = self.query_up(query_latent).view(batch_size, seq_len, self.n_heads, self.head_dim)
        attention_output = self._shared_mqa(queries, attention_kv, attention_mask)
        output = self.output(attention_output)

        if not return_debug:
            return output

        return output, {
            "compressed_kv": compressed_kv,
            "indexer_scores": selection.scores,
            "selected_block_indices": selection.indices,
            "selected_block_mask": selection.mask,
            "local_indices": local_indices,
            "attention_mask": attention_mask,
        }

    def _gather_selected_compressed(self, compressed_kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_size, _, head_dim = compressed_kv.shape
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

    def _shared_mqa(
        self,
        queries: torch.Tensor,
        key_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("bthc,btlc->bthl", queries, key_values)
        scores = scores * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~attention_mask.unsqueeze(2), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.einsum("bthl,btlc->bthc", weights, key_values)
