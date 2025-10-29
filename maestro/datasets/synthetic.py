"""Synthetic datasets for Maestro."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class Task:
    """A single tutoring task."""

    difficulty: float
    concept_vector: np.ndarray
    label: int


class SyntheticCurriculum:
    """Collection of synthetic tasks parameterised by a difficulty schedule."""

    def __init__(
        self,
        dim: int = 4,
        difficulty_levels: Iterable[float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.difficulty_levels = list(difficulty_levels or np.linspace(0.1, 1.0, 5))
        self.rng = np.random.default_rng(seed)
        self._concepts = self.rng.normal(size=(len(self.difficulty_levels), dim))

    def sample_task(self, rng: np.random.Generator | None = None) -> Task:
        rng = rng or self.rng
        idx = rng.integers(0, len(self.difficulty_levels))
        difficulty = float(self.difficulty_levels[idx])
        concept = self._concepts[idx]
        noise = rng.normal(scale=0.1, size=self.dim)
        concept_vector = concept + noise
        # Labels follow logistic function of difficulty and concept magnitude
        logit = concept_vector.dot(concept) - difficulty
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = int(rng.random() < prob)
        return Task(difficulty=difficulty, concept_vector=concept_vector, label=label)

    def describe(self) -> Dict[str, float]:
        return {
            "dim": float(self.dim),
            "avg_difficulty": float(np.mean(self.difficulty_levels)),
            "var_difficulty": float(np.var(self.difficulty_levels)),
        }

    def curriculum_sequence(self, length: int, rng: np.random.Generator | None = None) -> List[Task]:
        rng = rng or self.rng
        return [self.sample_task(rng=rng) for _ in range(length)]
