"""Policy heads for MAESTRO."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .distributions import bounded_scalar, simplex_from_logits


def _build_mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class MixtureHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, 1)

    def forward(self, encoded: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        expanded_context = context.unsqueeze(0).expand(encoded.size(0), -1)
        logits = self.net(torch.cat([encoded, expanded_context], dim=-1)).squeeze(-1)
        return logits


class ScalarHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dims, 1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class PolicyHeads(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        context_dim: int,
        mix_hidden: Sequence[int] = (128, 128),
        scalar_hidden: Sequence[int] = (64, 64),
        eta_bounds: tuple[float, float] = (1e-5, 1e-2),
    ) -> None:
        super().__init__()
        self.mixture_head = MixtureHead(descriptor_dim + context_dim, mix_hidden)
        self.lr_head = ScalarHead(context_dim, scalar_hidden)
        self.usage_head = ScalarHead(context_dim, scalar_hidden)
        self.eta_bounds = eta_bounds

    def forward(
        self,
        encoded_datasets: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.mixture_head(encoded_datasets, context)
        mixture = simplex_from_logits(logits)
        lr_logit = self.lr_head(context)
        usage_logit = self.usage_head(context)
        eta = bounded_scalar(lr_logit, self.eta_bounds[0], self.eta_bounds[1])
        usage = torch.sigmoid(usage_logit)
        return mixture, eta.squeeze(-1), usage.squeeze(-1)
