import unittest

import torch

from configs.csa_config import CSAConfig
from configs.llm_config import LLMConfig
from models.compressed_sparse_attention import CompressedSparseAttention, TokenCompressor
from models.llm import MinimalLLM


class TokenCompressorTest(unittest.TestCase):
    def test_overlapped_compression_matches_paper_equations(self):
        compressor = TokenCompressor(d_model=1, out_dim=1, block_size=2)
        with torch.no_grad():
            compressor.a_value.weight.fill_(1.0)
            compressor.b_value.weight.fill_(10.0)
            compressor.a_weight.weight.zero_()
            compressor.b_weight.weight.zero_()
            compressor.a_position_bias.zero_()
            compressor.b_position_bias.zero_()

        hidden = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        output = compressor(hidden, return_weights=True)

        self.assertEqual(output.entries.shape, (1, 2, 1))
        torch.testing.assert_close(output.entries[0, :, 0], torch.tensor([1.5, 9.25]))


class CompressedSparseAttentionTest(unittest.TestCase):
    def test_sparse_selection_is_causal_by_compressed_block(self):
        torch.manual_seed(0)
        attention = CompressedSparseAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            compression_block_size=2,
            top_k=2,
            sliding_window_size=3,
            indexer_heads=2,
            query_compression_dim=8,
            indexer_dim=4,
            output_groups=2,
        )
        hidden = torch.randn(1, 7, 16)

        _, debug = attention(hidden, return_debug=True)
        indices = debug["selected_block_indices"][0]
        mask = debug["selected_block_mask"][0]

        for token_idx in range(hidden.size(1)):
            allowed_blocks = token_idx // 2
            valid_indices = indices[token_idx][mask[token_idx]]
            if allowed_blocks == 0:
                self.assertEqual(valid_indices.numel(), 0)
            else:
                self.assertTrue(torch.all(valid_indices < allowed_blocks))

    def test_forward_backward_on_cpu(self):
        torch.manual_seed(1)
        attention = CompressedSparseAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            compression_block_size=3,
            top_k=2,
            sliding_window_size=4,
            indexer_heads=2,
            query_compression_dim=8,
            indexer_dim=4,
            output_groups=2,
        )
        hidden = torch.randn(2, 9, 16, requires_grad=True)

        output = attention(hidden)
        self.assertEqual(output.shape, hidden.shape)

        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_minimal_llm_uses_csa_attention(self):
        config = LLMConfig(
            d_model=16,
            n_heads=4,
            n_kv_heads=4,
            n_layers=1,
            d_ff=32,
            max_seq_len=12,
            vocab_size=64,
            attention_impl="csa",
            csa=CSAConfig(
                compression_block_size=3,
                top_k=2,
                sliding_window_size=4,
                indexer_heads=2,
                query_compression_dim=8,
                indexer_dim=4,
                output_groups=2,
            ),
        )
        model = MinimalLLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, config.max_seq_len, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
