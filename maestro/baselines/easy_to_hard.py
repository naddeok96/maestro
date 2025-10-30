"""Easy-to-hard curriculum baseline."""

from __future__ import annotations

import numpy as np


def easy_to_hard_action(step: int, num_steps: int, num_datasets: int) -> dict:
    order = np.linspace(0, 1, num_datasets)
    focus = min(num_datasets - 1, int(step / max(1, num_steps) * num_datasets))
    mixture = np.zeros(num_datasets, dtype=np.float32)
    mixture[focus] = 1.0
    return {
        "w": mixture,
        "eta": np.array([1e-3], dtype=np.float32),
        "u": np.array([0.1], dtype=np.float32),
    }
