"""Observation and probing utilities for the Maestro environment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from maestro.datasets.synthetic import SyntheticCurriculum
from maestro.students.base import StudentModel


@dataclass
class ProbeResult:
    """Summary statistics from probing a student."""

    student_name: str
    estimated_accuracy: float
    estimated_cost: float
    support: float


class ObservationBuilder:
    """Maintains running probe summaries for each student.

    The builder performs Monte Carlo probing of each student using tasks sampled
    from the curriculum. Probes are aggregated into order-invariant
    :class:`ProbeResult` objects used by :class:`maestro.envs.maestro_env.MaestroEnv`.
    """

    def __init__(
        self,
        students: Sequence[StudentModel],
        curriculum: SyntheticCurriculum,
        probes_per_step: int,
        rng: np.random.Generator,
    ) -> None:
        self.students = list(students)
        self.curriculum = curriculum
        self.probes_per_step = probes_per_step
        self.rng = rng
        self._history: List[ProbeResult] = []
        self._support_counts = np.zeros(len(self.students), dtype=np.float32)
        self._accuracy_sums = np.zeros(len(self.students), dtype=np.float32)
        self._cost_sums = np.zeros(len(self.students), dtype=np.float32)
        self._latest: List[ProbeResult] = []

    def reset(self) -> None:
        self._history.clear()
        self._support_counts.fill(0.0)
        self._accuracy_sums.fill(0.0)
        self._cost_sums.fill(0.0)
        self._latest.clear()
        self._latest = self._probe_all()

    def reset_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def update_with_outcome(self, student_index: int, accuracy: float, cost: float) -> None:
        self._support_counts[student_index] += 1.0
        self._accuracy_sums[student_index] += accuracy
        self._cost_sums[student_index] += cost
        self._latest = self._probe_all()

    def build(self) -> Iterable[ProbeResult]:
        if not self._latest:
            self._latest = self._probe_all()
        return tuple(self._latest)

    @property
    def latest_probes(self) -> Sequence[ProbeResult]:
        return tuple(self._latest)

    def _probe_all(self) -> List[ProbeResult]:
        probes: List[ProbeResult] = []
        for idx, student in enumerate(self.students):
            accuracy_sum = self._accuracy_sums[idx]
            cost_sum = self._cost_sums[idx]
            support = max(self._support_counts[idx], 1.0)
            # Monte Carlo sample additional tasks to keep probes fresh.
            sampled_accuracies = []
            sampled_costs = []
            for _ in range(self.probes_per_step):
                task = self.curriculum.sample_task(rng=self.rng)
                reward, cost, accuracy = student.evaluate(task, rng=self.rng)
                sampled_accuracies.append(accuracy)
                sampled_costs.append(cost)
            blended_accuracy = (accuracy_sum + float(np.mean(sampled_accuracies))) / support
            blended_cost = (cost_sum + float(np.mean(sampled_costs))) / support
            probes.append(
                ProbeResult(
                    student_name=student.name,
                    estimated_accuracy=float(np.clip(blended_accuracy, 0.0, 1.0)),
                    estimated_cost=float(np.clip(blended_cost, 0.0, 1.0)),
                    support=float(support / (self.probes_per_step + support)),
                )
            )
        # Sort for permutation invariance of observations
        probes.sort(key=lambda p: p.student_name)
        return probes
