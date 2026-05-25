import unittest

import torch

from configs.llm_config import LLMConfig
from configs.memory_policy_config import MemoryPolicyConfig
from models.llm import MinimalLLM
from models.memory_policies import (
    AgeForgettingAttention,
    CompetitionMemoryAttention,
    HierarchicalSummarizationAttention,
    PredictiveImportanceAttention,
    UsageRefreshAttention,
)


class AgeForgettingAttentionTest(unittest.TestCase):
    def test_gate_values_follow_distance_decay(self):
        attention = AgeForgettingAttention(
            d_model=8,
            n_heads=2,
            max_seq_len=8,
            local_window_size=1,
            block_size=1,
            memory_budget_blocks=4,
            age_decay_rate=0.5,
            gate_floor=0.0,
            dropout=0.0,
        )
        hidden = torch.randn(1, 4, 8)

        _, debug = attention(hidden, return_debug=True)
        gate = debug["memory_gate"][0, 0]

        self.assertTrue(torch.allclose(gate[0], torch.zeros_like(gate[0])))
        self.assertAlmostEqual(gate[1, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(gate[2, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(gate[2, 1].item(), 0.5, places=5)

    def test_forward_backward_on_cpu(self):
        attention = AgeForgettingAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            age_decay_rate=0.25,
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


class UsageRefreshAttentionTest(unittest.TestCase):
    def test_forward_and_debug_metrics_exist(self):
        attention = UsageRefreshAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            age_decay_rate=0.25,
            refresh_strength=0.35,
            gate_floor=0.0,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)

        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_refresh", debug)
        self.assertIn("mean_gate", debug)


class CompetitionMemoryAttentionTest(unittest.TestCase):
    def test_selects_no_more_than_capacity(self):
        attention = CompetitionMemoryAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            competition_capacity=2,
            age_decay_rate=0.25,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)

        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_utility", debug)
        self.assertLessEqual(debug["selected_blocks_mean"], 2.0 + 1e-5)


class HierarchicalSummarizationAttentionTest(unittest.TestCase):
    def test_hierarchy_reports_levels(self):
        attention = HierarchicalSummarizationAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            hierarchy_levels=2,
            hierarchy_branching=2,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)

        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("hierarchy_levels", debug)
        self.assertGreaterEqual(debug["hierarchy_levels"], 1)


class PredictiveImportanceAttentionTest(unittest.TestCase):
    def test_predictive_policy_returns_scores(self):
        attention = PredictiveImportanceAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            predictive_hidden_dim=8,
            predictive_top_k=2,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)

        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_predictive_score", debug)
        self.assertIn("selected_blocks_mean", debug)


class MemoryFiveMillionRoutingTest(unittest.TestCase):
    def _model_for_impl(self, attention_impl: str) -> MinimalLLM:
        config = LLMConfig(
            d_model=16,
            n_heads=4,
            n_kv_heads=4,
            n_layers=1,
            d_ff=32,
            max_seq_len=12,
            vocab_size=64,
            attention_impl=attention_impl,
            memory_policy=MemoryPolicyConfig(
                local_window_size=4,
                block_size=2,
                memory_budget_blocks=4,
                age_decay_rate=0.25,
                refresh_strength=0.35,
                competition_capacity=2,
                hierarchy_levels=2,
                hierarchy_branching=2,
                predictive_hidden_dim=8,
                predictive_top_k=2,
            ),
        )
        return MinimalLLM(config)

    def test_new_attention_impls_route_through_minimal_llm(self):
        input_ids = torch.randint(0, 64, (2, 12))
        for attention_impl in [
            "age_forgetting",
            "usage_refresh",
            "competition",
            "hierarchical",
            "predictive",
        ]:
            model = self._model_for_impl(attention_impl)
            logits = model(input_ids)
            self.assertEqual(logits.shape, (2, 12, 64))

    def test_preset_is_close_to_five_million_parameters(self):
        from configs.research_configs import MemoryFiveMillionConfig

        config = MemoryFiveMillionConfig()
        model = MinimalLLM(config)
        total_params = sum(p.numel() for p in model.parameters())

        self.assertGreater(total_params, 4_500_000)
        self.assertLess(total_params, 5_900_000)


if __name__ == "__main__":
    unittest.main()
