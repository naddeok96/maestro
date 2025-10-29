"""Simple baseline policies."""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Chooses actions uniformly at random."""

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return int(self.rng.integers(0, self.action_space.n))
