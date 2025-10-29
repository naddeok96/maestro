"""Diagnostics and metrics for Maestro experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from maestro.envs.observation import ProbeResult


@dataclass
class RolloutMetrics:
    rewards: List[float]
    costs: List[float]
    accuracies: List[float]

    @property
    def cumulative_reward(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def average_cost(self) -> float:
        return float(np.mean(self.costs)) if self.costs else 0.0

    @property
    def average_accuracy(self) -> float:
        return float(np.mean(self.accuracies)) if self.accuracies else 0.0


def compute_markov_diagnostic(series: Iterable[float]) -> float:
    """Return the lag-one auto-correlation as a proxy for Markovity."""

    series = np.asarray(list(series), dtype=np.float32)
    if len(series) < 2:
        return 0.0
    mean = float(series.mean())
    numerator = float(np.sum((series[:-1] - mean) * (series[1:] - mean)))
    denominator = float(np.sum((series - mean) ** 2)) + 1e-6
    return numerator / denominator


def check_probe_invariance(probes: Iterable[ProbeResult]) -> bool:
    """Ensure permutation invariance by comparing sorted student names."""

    names = [p.student_name for p in probes]
    return names == sorted(names)


def summarize_rollout(history: List[dict]) -> RolloutMetrics:
    rewards = [step["reward"] for step in history]
    costs = [step["info"]["cost"] for step in history if "cost" in step.get("info", {})]
    accuracies = [step["info"]["accuracy"] for step in history if "accuracy" in step.get("info", {})]
    return RolloutMetrics(rewards=rewards, costs=costs, accuracies=accuracies)
