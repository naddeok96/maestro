"""Feature extraction helpers for students."""
from __future__ import annotations

import torch


def flatten_batch(batch) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return torch.cat([flatten_batch(item) for item in batch], dim=-1)
    if isinstance(batch, dict):
        return torch.cat([flatten_batch(v) for v in batch.values()], dim=-1)
    if isinstance(batch, torch.Tensor):
        return batch.view(batch.size(0), -1)
    raise TypeError(f"Unsupported batch type: {type(batch)}")
