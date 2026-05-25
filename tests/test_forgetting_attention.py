import unittest

import torch

from configs.llm_config import LLMConfig
from configs.research_configs import CSAFiveMillionConfig
from models.forgetting_attention import ForgettingAttention
from models.llm import MinimalLLM


class ForgettingAttentionTest(unittest.TestCase):
    def test_gate_values_follow_distance_decay(self):
        attention = ForgettingAttention(
            d_model=8,
            n_heads=2,
            max_seq_len=8,
            local_window_size=1,
            memory_block_size=1,
            memory_decay_rate=0.5,
            gate_floor=0.0,
            dropout=0.0,
        )
        hidden = torch.randn(1, 4, 8)

        _, debug = attention(hidden, return_debug=True)
        gate = debug["retention_gate"][0, 0]

        self.assertTrue(torch.allclose(gate[0], torch.zeros_like(gate[0])))
        self.assertAlmostEqual(gate[1, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(gate[2, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(gate[2, 1].item(), 0.5, places=5)

    def test_forward_backward_on_cpu(self):
        attention = ForgettingAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            memory_block_size=2,
            memory_decay_rate=0.25,
            gate_floor=0.0,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16, requires_grad=True)

        output = attention(hidden)
        self.assertEqual(output.shape, hidden.shape)

        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_minimal_llm_uses_forgetting_attention(self):
        config = LLMConfig(
            d_model=16,
            n_heads=4,
            n_kv_heads=4,
            n_layers=1,
            d_ff=32,
            max_seq_len=12,
            vocab_size=64,
            attention_impl="forgetting",
        )
        model = MinimalLLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, config.max_seq_len))

        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, config.max_seq_len, config.vocab_size))


class FiveMillionPresetTest(unittest.TestCase):
    def test_preset_is_close_to_five_million_parameters(self):
        config = CSAFiveMillionConfig()
        model = MinimalLLM(config)
        total_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(total_params, 4_500_000)
        self.assertLess(total_params, 5_800_000)


if __name__ == "__main__":
    unittest.main()
