"""Uniform baseline scheduler."""

from __future__ import annotations

import numpy as np


def uniform_action(num_datasets: int, eta: float = 1e-3, usage: float = 0.1) -> dict:
    mixture = np.ones(num_datasets, dtype=np.float32) / num_datasets
    return {
        "w": mixture,
        "eta": np.array([eta], dtype=np.float32),
        "u": np.array([usage], dtype=np.float32),
    }
