"""Synthetic classification datasets for MAESTRO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


@dataclass
class ClassificationConfig:
    feature_dim: int
    num_classes: int
    train_size: int
    val_size: int
    probe_size: int
    noise: float
    imbalance: float


def _sample_dataset(
    config: ClassificationConfig, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    centres = rng.normal(size=(config.num_classes, config.feature_dim))
    cov = np.eye(config.feature_dim)

    def sample(n: int) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []
        for cls in range(config.num_classes):
            cls_n = int(n / config.num_classes)
            cls_n = max(1, int(cls_n * (1.0 - config.imbalance * rng.random())))
            points = rng.multivariate_normal(centres[cls], cov, size=cls_n)
            noise = rng.normal(scale=config.noise, size=points.shape)
            data.append(points + noise)
            labels.append(np.full(cls_n, cls))
        x = np.concatenate(data, axis=0)
        y = np.concatenate(labels, axis=0)
        perm = rng.permutation(x.shape[0])
        return x[perm], y[perm]

    train_x, train_y = sample(config.train_size)
    val_x, val_y = sample(config.val_size)
    probe_x, probe_y = sample(config.probe_size)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "probe_x": probe_x,
        "probe_y": probe_y,
    }


def build_classification_dataset(
    name: str, config: ClassificationConfig, seed: int
) -> Dict[str, Dataset]:
    rng = np.random.default_rng(seed)
    arrays = _sample_dataset(config, rng)
    return {
        "train": ClassificationDataset(arrays["train_x"], arrays["train_y"]),
        "val": ClassificationDataset(arrays["val_x"], arrays["val_y"]),
        "probe": ClassificationDataset(arrays["probe_x"], arrays["probe_y"]),
        "metadata": {
            "num_classes": config.num_classes,
            "feature_dim": config.feature_dim,
            "name": name,
        },
    }
