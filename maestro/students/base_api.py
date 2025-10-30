"""Student model API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

import torch
from torch.utils.data import DataLoader

from maestro.utils import OptimizerSettings


class AbstractStudent(Protocol):
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def configure_optimizer(self, settings: OptimizerSettings) -> None: ...

    def step_on_minibatch(self, batch) -> Dict[str, float]: ...

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]: ...

    def feature_embed(self, batch) -> torch.Tensor: ...

    @property
    def device(self) -> torch.device: ...


@dataclass
class StudentState:
    model: AbstractStudent
    optimizer: torch.optim.Optimizer
