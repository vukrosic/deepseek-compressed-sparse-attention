import unittest

import torch

from data.synthetic import SyntheticCausalDataset


class SyntheticCausalDatasetTest(unittest.TestCase):
    def test_counting_pattern_is_deterministic(self):
        dataset = SyntheticCausalDataset(
            num_sequences=2,
            seq_len=8,
            vocab_size=32,
            pattern="counting",
            seed=123,
        )

        first = dataset[0]["input_ids"]
        again = dataset[0]["input_ids"]
        torch.testing.assert_close(first, again)
        torch.testing.assert_close(dataset[0]["labels"], first)

    def test_copy_lag_pattern_uses_previous_tokens(self):
        dataset = SyntheticCausalDataset(
            num_sequences=1,
            seq_len=12,
            vocab_size=64,
            pattern="copy_lag",
            lag=4,
            seed=7,
        )

        input_ids = dataset[0]["input_ids"]
        expected_tail = (input_ids[:-4] + 1) % 64
        torch.testing.assert_close(input_ids[4:], expected_tail)


if __name__ == "__main__":
    unittest.main()
