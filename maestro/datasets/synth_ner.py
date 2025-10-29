"""Synthetic sequence labelling datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, tokens: np.ndarray, tags: np.ndarray):
        self.tokens = torch.from_numpy(tokens).long()
        self.tags = torch.from_numpy(tags).long()

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[index], self.tags[index]


@dataclass
class NERConfig:
    vocab_size: int
    num_tags: int
    train_size: int
    val_size: int
    probe_size: int
    noise: float
    sequence_length: int = 16


def _generate_sequences(config: NERConfig, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = rng.integers(0, config.vocab_size, size=(n, config.sequence_length))
    tags = rng.integers(0, config.num_tags, size=(n, config.sequence_length))
    span_len = max(1, config.sequence_length // 4)
    for i in range(n):
        start = rng.integers(0, config.sequence_length - span_len + 1)
        tag = rng.integers(1, config.num_tags)
        tags[i, start : start + span_len] = tag
    # Flip some tags to inject noise
    mask = rng.random(size=tags.shape) < config.noise
    tags[mask] = rng.integers(0, config.num_tags, size=mask.sum())
    return tokens, tags


def build_ner_dataset(name: str, config: NERConfig, seed: int) -> Dict[str, Dataset]:
    rng = np.random.default_rng(seed)
    train_tokens, train_tags = _generate_sequences(config, rng, config.train_size)
    val_tokens, val_tags = _generate_sequences(config, rng, config.val_size)
    probe_tokens, probe_tags = _generate_sequences(config, rng, config.probe_size)
    return {
        "train": SequenceDataset(train_tokens, train_tags),
        "val": SequenceDataset(val_tokens, val_tags),
        "probe": SequenceDataset(probe_tokens, probe_tags),
        "metadata": {
            "num_tags": config.num_tags,
            "vocab_size": config.vocab_size,
            "sequence_length": config.sequence_length,
            "name": name,
        },
    }
