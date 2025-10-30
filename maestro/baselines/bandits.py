"""Simple LinUCB-style bandit baseline."""

from __future__ import annotations

import numpy as np


def linucb_action(estimates: np.ndarray, step: int) -> dict:
    confidence = 1.0 / np.sqrt(step + 1.0)
    scores = estimates + confidence
    idx = int(np.argmax(scores))
    mixture = np.zeros_like(estimates)
    mixture[idx] = 1.0
    return {
        "w": mixture.astype(np.float32),
        "eta": np.array([1e-3], dtype=np.float32),
        "u": np.array([0.1], dtype=np.float32),
    }
