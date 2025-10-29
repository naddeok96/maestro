"""Synthetic detection datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, images: np.ndarray, boxes: List[np.ndarray]):
        self.images = torch.from_numpy(images).float()
        self.boxes = [torch.from_numpy(b).float() for b in boxes]

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.boxes[index]


@dataclass
class DetectionConfig:
    image_size: int
    train_size: int
    val_size: int
    probe_size: int
    max_objects: int
    noise: float


def _render_boxes(config: DetectionConfig, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    images = np.zeros((n, 1, config.image_size, config.image_size), dtype=np.float32)
    all_boxes: List[np.ndarray] = []
    for i in range(n):
        count = rng.integers(1, config.max_objects + 1)
        boxes = []
        for _ in range(count):
            x1 = rng.uniform(0.0, config.image_size * 0.6)
            y1 = rng.uniform(0.0, config.image_size * 0.6)
            w = rng.uniform(config.image_size * 0.1, config.image_size * 0.4)
            h = rng.uniform(config.image_size * 0.1, config.image_size * 0.4)
            x2 = min(config.image_size - 1.0, x1 + w)
            y2 = min(config.image_size - 1.0, y1 + h)
            boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
            images[i, 0, int(y1) : int(y2), int(x1) : int(x2)] = 1.0
        noise = rng.normal(scale=config.noise, size=images[i].shape)
        images[i] = np.clip(images[i] + noise, 0.0, 1.0)
        all_boxes.append(np.stack(boxes, axis=0))
    return images, all_boxes


def build_detection_dataset(name: str, config: DetectionConfig, seed: int) -> Dict[str, Dataset]:
    rng = np.random.default_rng(seed)
    train_images, train_boxes = _render_boxes(config, rng, config.train_size)
    val_images, val_boxes = _render_boxes(config, rng, config.val_size)
    probe_images, probe_boxes = _render_boxes(config, rng, config.probe_size)
    return {
        "train": DetectionDataset(train_images, train_boxes),
        "val": DetectionDataset(val_images, val_boxes),
        "probe": DetectionDataset(probe_images, probe_boxes),
        "metadata": {"image_size": config.image_size, "name": name},
    }
