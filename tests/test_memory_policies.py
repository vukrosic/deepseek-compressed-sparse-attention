import unittest

import torch

from configs.llm_config import LLMConfig
from configs.memory_policy_config import MemoryPolicyConfig
from experiments.csa_metrics import attention_budget, build_run_metadata
from models.llm import MinimalLLM
from models.memory_policies import (
    AgeForgettingAttention,
    AgeForgettingCosineAttention,
    AgeForgettingExponentialAttention,
    AgeForgettingHardCutoffAttention,
    AgeForgettingReciprocalAttention,
    AgeForgettingSigmoidAttention,
    CompressedMemoryNoGateAttention,
    CompetitionMemoryAttention,
    HierarchicalSummarizationAttention,
    LearnedRouterAttention,
    PeriodicKeyframeAttention,
    RandomKeyframeAttention,
    PredictiveImportanceAttention,
    SalienceMemoryAttention,
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

    def test_zero_gate_removes_memory_from_softmax(self):
        attention = AgeForgettingAttention(
            d_model=4,
            n_heads=1,
            max_seq_len=4,
            local_window_size=1,
            block_size=1,
            memory_budget_blocks=1,
            dropout=0.0,
        )
        Q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        local_k = torch.zeros(1, 1, 1, 1, 4)
        local_v = torch.ones(1, 1, 1, 1, 4)
        local_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        memory_k = torch.tensor([[[[100.0, 0.0, 0.0, 0.0]]]])
        memory_v = torch.full((1, 1, 1, 4), 100.0)
        memory_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        memory_gate = torch.zeros(1, 1, 1, 1)

        output, debug = attention._combine_attention(
            Q,
            local_k,
            local_v,
            local_mask,
            memory_k,
            memory_v,
            memory_mask,
            memory_gate,
        )

        self.assertTrue(torch.allclose(output, local_v.squeeze(3)))
        self.assertTrue(torch.all(debug["memory_weights"] == 0))

    def test_alternative_age_gates_stay_monotone(self):
        variants = [
            AgeForgettingAttention,
            AgeForgettingExponentialAttention,
            AgeForgettingSigmoidAttention,
            AgeForgettingCosineAttention,
            AgeForgettingReciprocalAttention,
            AgeForgettingHardCutoffAttention,
        ]
        hidden = torch.randn(1, 10, 8)

        for variant in variants:
            with self.subTest(variant=variant.__name__):
                attention = variant(
                    d_model=8,
                    n_heads=2,
                    max_seq_len=10,
                    local_window_size=1,
                    block_size=1,
                    memory_budget_blocks=6,
                    age_decay_rate=0.5,
                    gate_floor=0.0,
                    dropout=0.0,
                )
                output, debug = attention(hidden, return_debug=True)
                self.assertEqual(output.shape, hidden.shape)

                age = debug["memory_age"][debug["memory_mask"]]
                gate = debug["memory_gate"][debug["memory_mask"]]
                self.assertGreater(gate.numel(), 0)

                age_values = torch.unique(age).tolist()
                age_values.sort()
                age_means = []
                for age_value in age_values:
                    mask = age == age_value
                    age_means.append(float(gate[mask].mean().item()))

                self.assertTrue(
                    all(
                        age_means[idx] >= age_means[idx + 1] - 1e-6
                        for idx in range(len(age_means) - 1)
                    )
                )


class StructuralMemoryAttentionTest(unittest.TestCase):
    def test_random_keyframe_attention_runs(self):
        attention = RandomKeyframeAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)
        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_random_score", debug)

    def test_periodic_keyframe_attention_runs(self):
        attention = PeriodicKeyframeAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            periodic_stride=2,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)
        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_periodic_score", debug)

    def test_learned_router_attention_runs(self):
        attention = LearnedRouterAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            router_hidden_dim=8,
            router_top_k=2,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)
        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_route_score", debug)

    def test_salience_memory_attention_runs(self):
        attention = SalienceMemoryAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)
        output, debug = attention(hidden, return_debug=True)
        self.assertEqual(output.shape, hidden.shape)
        self.assertIn("mean_salience_score", debug)


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


class CompressedMemoryNoGateAttentionTest(unittest.TestCase):
    def test_valid_memory_has_unit_gate(self):
        attention = CompressedMemoryNoGateAttention(
            d_model=16,
            n_heads=4,
            max_seq_len=12,
            local_window_size=4,
            block_size=2,
            memory_budget_blocks=4,
            dropout=0.0,
        )
        hidden = torch.randn(2, 9, 16)

        output, debug = attention(hidden, return_debug=True)

        self.assertEqual(output.shape, hidden.shape)
        valid_gate = debug["memory_gate"][debug["memory_mask"]]
        self.assertTrue(torch.all(valid_gate == 1))
        self.assertEqual(debug["mean_gate"], 1.0)


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
                periodic_stride=2,
                router_hidden_dim=8,
                router_top_k=2,
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
            "age_forgetting_exponential",
            "age_forgetting_sigmoid",
            "age_forgetting_cosine",
            "age_forgetting_reciprocal",
            "age_forgetting_hard_cutoff",
            "random_keyframe",
            "periodic_keyframe",
            "learned_router",
            "salience_memory",
            "compressed_memory",
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

    def test_age_forgetting_metadata_uses_memory_policy_budget(self):
        config = LLMConfig(
            d_model=16,
            n_heads=4,
            n_kv_heads=4,
            n_layers=1,
            d_ff=32,
            max_seq_len=256,
            vocab_size=64,
            attention_impl="age_forgetting",
            memory_policy=MemoryPolicyConfig(
                local_window_size=64,
                block_size=4,
                memory_budget_blocks=16,
            ),
        )

        budget = attention_budget(config)
        metadata = build_run_metadata(config)

        self.assertEqual(budget["attention_vector_budget"], 80)
        self.assertEqual(budget["raw_token_equivalent_coverage"], 128)
        self.assertEqual(metadata["memory_local_window_size"], 64)
        self.assertEqual(metadata["memory_block_size"], 4)
        self.assertEqual(metadata["memory_budget_blocks"], 16)


if __name__ == "__main__":
    unittest.main()
