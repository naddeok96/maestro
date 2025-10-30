"""Schedulers and helper utilities for optimizer parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerSettings:
    learning_rate: float
    weight_decay: float
    momentum: float


def clamp_learning_rate(value: float, eta_min: float, eta_max: float) -> float:
    return max(eta_min, min(eta_max, value))
