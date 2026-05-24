from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticCausalDataset(Dataset):
    """Tiny deterministic language-modeling dataset for local smoke runs."""

    def __init__(
        self,
        num_sequences: int,
        seq_len: int,
        vocab_size: int,
        pattern: str = "copy_lag",
        lag: int = 32,
        seed: int = 0,
    ) -> None:
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        if seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if vocab_size <= 4:
            raise ValueError("vocab_size must be greater than 4")
        if lag <= 0:
            raise ValueError("lag must be positive")

        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pattern = pattern
        self.lag = min(lag, seq_len - 1)
        self.seed = seed

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + idx)

        if self.pattern == "counting":
            offset = torch.randint(0, self.vocab_size, (1,), generator=generator).item()
            input_ids = (torch.arange(self.seq_len) + offset) % self.vocab_size
        elif self.pattern == "copy_lag":
            input_ids = torch.empty(self.seq_len, dtype=torch.long)
            prefix = torch.randint(
                0,
                self.vocab_size,
                (self.lag,),
                generator=generator,
                dtype=torch.long,
            )
            input_ids[: self.lag] = prefix
            for pos in range(self.lag, self.seq_len):
                input_ids[pos] = (input_ids[pos - self.lag] + 1) % self.vocab_size
        else:
            raise ValueError(f"Unknown synthetic pattern: {self.pattern}")

        return {"input_ids": input_ids, "labels": input_ids.clone()}
