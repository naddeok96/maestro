"""Lightweight MAESTRO policy stub for external controllers.

This module provides a self-contained policy that mimics the scheduling
interface of the trained MAESTRO agent.  It is intentionally simple yet
deterministic so that external scripts – such as the YOLOv8 track and the
large-model transfer experiment – can query curriculum decisions without
depending on heavy RL checkpoints.  The implementation follows the design
sketched in the paper: mixture weights favour datasets with higher probe
losses, while the learning-rate and usage controls anneal over the course of
the training horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


@dataclass
class MaestroPolicyConfig:
    """Configuration for :class:`MaestroPolicy`.

    Attributes
    ----------
    min_lr_scale:
        Lower bound for the learning rate multiplier predicted by the policy.
    max_lr_scale:
        Upper bound for the learning rate multiplier.
    min_usage:
        Minimum fraction of the remaining budget that the policy will allocate
        during a segment.
    max_usage:
        Maximum fraction of the remaining budget for early segments.
    temperature:
        Temperature parameter used when converting probe scores into a
        probability distribution.  Higher values produce smoother mixture
        weights.
    epsilon:
        Small constant added for numerical stability when all probe scores are
        identical.
    """

    min_lr_scale: float = 0.5
    max_lr_scale: float = 1.5
    min_usage: float = 0.2
    max_usage: float = 0.7
    temperature: float = 1.0
    epsilon: float = 1e-6


class MaestroPolicy:
    """Deterministic controller that emulates MAESTRO scheduling outputs.

    The policy only relies on summary statistics ("probes") from each dataset
    and the remaining training budget.  It outputs a tuple ``(w, eta_scale, u)``
    where ``w`` are the dataset mixture weights, ``eta_scale`` is the
    multiplicative learning-rate factor, and ``u`` controls the usage of the
    remaining budget.  This mirrors the interface used by the real MAESTRO
    teacher and allows the research scripts to stay agnostic to the underlying
    RL implementation.
    """

    def __init__(self, config: MaestroPolicyConfig | None = None) -> None:
        self.config = config or MaestroPolicyConfig()

    # ------------------------------------------------------------------
    def _score_dataset(self, probes: Mapping[str, float]) -> float:
        """Return an importance score for a dataset based on probe statistics."""

        loss = float(probes.get("loss_mean", 0.0))
        entropy = float(probes.get("entropy_mean", 0.0))
        loss_iqr = float(probes.get("loss_iqr", 0.0))
        grad_norm = float(probes.get("grad_norm_log", 0.0))
        score = 0.6 * loss + 0.2 * entropy + 0.1 * loss_iqr + 0.1 * grad_norm
        return max(self.config.epsilon, score)

    # ------------------------------------------------------------------
    def _normalise_weights(self, scores: Iterable[float]) -> np.ndarray:
        scores_arr = np.asarray(list(scores), dtype=np.float64)
        if scores_arr.size == 0:
            return scores_arr
        scaled = scores_arr / max(self.config.epsilon, self.config.temperature)
        scaled = scaled - scaled.max()
        weights = np.exp(scaled)
        weights /= max(self.config.epsilon, weights.sum())
        return weights.astype(np.float32)

    # ------------------------------------------------------------------
    def _schedule_controls(self, t: int, horizon: int) -> Tuple[float, float]:
        """Compute learning-rate and usage controls for the given segment."""

        progress = float(t - 1) / max(1.0, float(horizon))
        eta = self.config.max_lr_scale - progress * (
            self.config.max_lr_scale - self.config.min_lr_scale
        )
        usage = self.config.max_usage - progress * (
            self.config.max_usage - self.config.min_usage
        )
        return float(np.clip(eta, self.config.min_lr_scale, self.config.max_lr_scale)), float(
            np.clip(usage, self.config.min_usage, self.config.max_usage)
        )

    # ------------------------------------------------------------------
    def get_action(
        self,
        probes: Dict[str, Mapping[str, float]],
        budget_remaining: int,
        t: int,
        horizon: int,
    ) -> Tuple[Dict[str, float], float, float]:
        """Compute the scheduling decision for the current segment.

        Parameters
        ----------
        probes:
            Nested mapping ``dataset -> feature -> value`` describing the probe
            statistics gathered from quick evaluations.
        budget_remaining:
            Number of training examples (or images) that are still available in
            the overall budget.  The policy only uses the value to ensure that
            the usage fraction does not result in empty segments.
        t:
            One-indexed index of the current segment.
        horizon:
            Total number of segments in the run.

        Returns
        -------
        Tuple[Dict[str, float], float, float]
            Mixture weights per dataset, the learning-rate multiplier, and the
            usage fraction of the remaining budget.
        """

        dataset_keys = list(probes.keys())
        if not dataset_keys:
            raise ValueError("No datasets provided to MaestroPolicy")

        scores = [self._score_dataset(probes[name]) for name in dataset_keys]
        weights = self._normalise_weights(scores)
        eta_scale, usage_fraction = self._schedule_controls(t, horizon)

        if budget_remaining <= 0:
            usage_fraction = 0.0

        weight_map = {name: float(weight) for name, weight in zip(dataset_keys, weights)}
        return weight_map, float(eta_scale), float(usage_fraction)


__all__ = ["MaestroPolicy", "MaestroPolicyConfig"]

