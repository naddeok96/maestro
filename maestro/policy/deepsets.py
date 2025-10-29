"""DeepSets encoder for dataset descriptors."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(nn.ReLU())
        last_dim = hidden
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, phi_dim: int = 64, rho_dim: int = 64):
        super().__init__()
        self.phi = _mlp(input_dim, (phi_dim,), phi_dim)
        self.rho = _mlp(phi_dim, (rho_dim,), rho_dim)

    def forward(self, descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if descriptors.dim() != 2:
            raise ValueError("Descriptors must be of shape [N, D]")
        encoded = self.phi(descriptors)
        pooled = encoded.mean(dim=0, keepdim=True)
        summary = self.rho(pooled)
        return summary.squeeze(0), encoded
