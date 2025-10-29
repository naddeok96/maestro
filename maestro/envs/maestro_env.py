"""Gymnasium environment for Maestro meta-tutoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.envs.observation import ObservationBuilder, ProbeResult
from maestro.students.base import StudentModel
from maestro.students.registry import load_students
from maestro.utils.cmdp import BudgetManager


@dataclass
class MaestroEnvConfig:
    """Configuration for :class:`MaestroEnv`."""

    curriculum: SyntheticCurriculum
    max_steps: int = 50
    probes_per_step: int = 8
    budget: float = 10.0
    seed: Optional[int] = None


class MaestroEnv(gym.Env):
    """Environment that simulates teaching policies under resource budgets.

    The agent observes summaries of probing students on synthetic tasks and must
    choose the next student to deploy. Rewards balance student accuracy against
    assessment cost subject to a CMDP-style budget handled by
    :class:`~maestro.utils.cmdp.BudgetManager`.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: MaestroEnvConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.curriculum = config.curriculum
        self.students: List[StudentModel] = load_students(self.curriculum)
        self.budget = BudgetManager(total_budget=config.budget)
        self.builder = ObservationBuilder(
            students=self.students,
            curriculum=self.curriculum,
            probes_per_step=config.probes_per_step,
            rng=self.rng,
        )
        self._step_count = 0

        # Discrete choice over students.
        self.action_space = gym.spaces.Discrete(len(self.students))

        # Observation is a concatenated float vector of probe summaries.
        # Each probe result has (estimated_accuracy, estimated_cost, support).
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.students) * 3,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.builder.reset_rng(self.rng)
        self.budget.reset()
        self.builder.reset()
        self._step_count = 0
        observation = self._get_observation()
        return observation, {"budget": self.budget.remaining}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Invalid student index: {action}")
        if self.budget.is_depleted:
            # If the budget is depleted terminate immediately.
            return self._get_observation(), 0.0, True, False, {
                "budget": self.budget.remaining,
                "info": "budget_depleted",
            }

        student = self.students[action]
        task = self.curriculum.sample_task(rng=self.rng)
        reward, cost, accuracy = student.evaluate(task, rng=self.rng)
        self.budget.consume(cost)
        self.builder.update_with_outcome(action, accuracy=accuracy, cost=cost)
        self._step_count += 1

        observation = self._get_observation()
        terminated = self._step_count >= self.config.max_steps or self.budget.is_depleted
        truncated = False
        info = {
            "budget": self.budget.remaining,
            "cost": cost,
            "accuracy": accuracy,
            "student": student.name,
        }
        return observation, reward, terminated, truncated, info

    # Gymnasium API compatibility shim
    def render(self) -> str:
        probe_strings = [
            f"{probe.student_name}: acc={probe.estimated_accuracy:.2f} cost={probe.estimated_cost:.2f}"
            for probe in self.builder.latest_probes
        ]
        return " | ".join(probe_strings)

    def _get_observation(self) -> np.ndarray:
        probes: Iterable[ProbeResult] = self.builder.build()
        vector = np.concatenate(
            [np.array([p.estimated_accuracy, p.estimated_cost, p.support], dtype=np.float32) for p in probes]
        )
        return vector.astype(np.float32)

    @property
    def latest_probes(self) -> List[ProbeResult]:
        return list(self.builder.latest_probes)
