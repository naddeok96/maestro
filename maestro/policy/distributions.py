"""Distribution helpers for policy heads."""
from __future__ import annotations

import torch


def simplex_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def bounded_scalar(logit: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(logit)
