"""Gradient utilities for MAESTRO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import torch


@dataclass
class GradientProjector:
    """Random projection used to compress gradients to a fixed dimension."""

    input_dim: int
    output_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.matrix = torch.from_numpy(
            rng.normal(scale=1.0 / np.sqrt(self.output_dim), size=(self.output_dim, self.input_dim))
        ).float()

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        vector = vector.view(-1).float()
        if vector.numel() < self.input_dim:
            padded = torch.zeros(self.input_dim, dtype=torch.float32, device=vector.device)
            padded[: vector.numel()] = vector
            vector = padded
        elif vector.numel() > self.input_dim:
            vector = vector[: self.input_dim]
        return torch.matmul(self.matrix.to(vector.device), vector)


def flatten_parameters(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Return a detached flattened tensor of the provided parameters."""

    return torch.cat([p.detach().view(-1) for p in params])


def flatten_gradients(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Return flattened gradients, treating missing gradients as zeros."""

    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads)


class ExponentialMovingAverage:
    """EMA tracker with optional debiasing."""

    def __init__(self, beta: float, debias: bool = False):
        self.beta = beta
        self.debias = debias
        self.buffer: Optional[torch.Tensor] = None
        self.power = 1.0

    def update(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        if self.buffer is None:
            self.buffer = value.clone()
        else:
            self.buffer.mul_(self.beta).add_(value * (1.0 - self.beta))
        if self.debias:
            self.power *= self.beta
            return self.buffer / (1.0 - self.power)
        return self.buffer

    def get(self) -> Optional[torch.Tensor]:
        return None if self.buffer is None else self.buffer.detach().clone()


class RobustScalarNormalizer:
    """Median/IQR EMA based normaliser for scalar features."""

    def __init__(self, beta: float = 0.9, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.median_ema = 0.0
        self.iqr_ema = 1.0
        self.initialised = False

    def update(self, value: float) -> float:
        if not self.initialised:
            self.median_ema = value
            self.iqr_ema = 1.0
            self.initialised = True
        else:
            self.median_ema = self.beta * self.median_ema + (1.0 - self.beta) * value
            deviation = abs(value - self.median_ema)
            self.iqr_ema = self.beta * self.iqr_ema + (1.0 - self.beta) * max(deviation, self.eps)
        return (value - self.median_ema) / (self.iqr_ema + self.eps)


def parameter_iterator(module: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    for p in module.parameters():
        yield p


def gradient_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    num = torch.dot(a, b)
    denom = torch.norm(a) * torch.norm(b) + eps
    if denom.item() == 0.0:
        return 0.0
    return (num / denom).item()


def l2_norm(vector: torch.Tensor, eps: float = 1e-8) -> float:
    return float(torch.norm(vector).item() + eps)


def parameter_change(previous: torch.Tensor, current: torch.Tensor, eps: float = 1e-8) -> float:
    delta = current - previous
    return float(torch.norm(delta) / (torch.norm(previous) + eps))
