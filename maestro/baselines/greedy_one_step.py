"""Greedy one-step gain baseline."""
from __future__ import annotations

import numpy as np


def greedy_action(metrics: dict) -> dict:
    accuracies = np.array([m.get("accuracy", 0.0) for m in metrics.values()], dtype=np.float32)
    target = np.argmin(accuracies)
    mixture = np.zeros_like(accuracies)
    mixture[target] = 1.0
    return {"w": mixture, "eta": np.array([1e-3], dtype=np.float32), "u": np.array([0.1], dtype=np.float32)}
