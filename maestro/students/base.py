"""Student models used in Maestro."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from maestro.datasets.synthetic import Task


class StudentModel(Protocol):
    """Protocol for student models."""

    name: str

    def evaluate(
        self, task: Task, rng: np.random.Generator | None = None
    ) -> tuple[float, float, float]:
        """Return reward, cost, accuracy for the task."""


@dataclass
class BaseStudent:
    name: str
    skill: float
    assessment_cost: float
    noise: float = 0.1

    def evaluate(
        self, task: Task, rng: np.random.Generator | None = None
    ) -> tuple[float, float, float]:
        rng = rng or np.random.default_rng()
        mastery = np.clip(self.skill - task.difficulty, -2.0, 2.0)
        logit = mastery + float(task.concept_vector.mean())
        prob_correct = 1.0 / (1.0 + np.exp(-logit))
        noisy_prob = np.clip(prob_correct + rng.normal(scale=self.noise), 0.0, 1.0)
        accuracy = float(noisy_prob)
        reward = accuracy - self.assessment_cost
        cost = float(np.clip(self.assessment_cost + rng.normal(scale=0.01), 0.0, 1.0))
        return reward, cost, accuracy
