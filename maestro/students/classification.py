"""Simple classification student."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from maestro.utils import OptimizerSettings, flatten_gradients


@dataclass
class ClassificationStudent(nn.Module):
    input_dim: int
    num_classes: int

    def __post_init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes),
        )
        self._optimizer: torch.optim.Optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_optimizer(self, settings: OptimizerSettings) -> None:
        self._optimizer = torch.optim.SGD(
            self.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def step_on_minibatch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self._optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        grad_vec = flatten_gradients(list(self.parameters())).detach().cpu()
        self._optimizer.step()
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc, "grad_norm": float(grad_vec.norm().item()), "grad_vector": grad_vec}

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.forward(inputs)
                loss = self.loss_fn(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                total_correct += (logits.argmax(dim=-1) == targets).sum().item()
                total += inputs.size(0)
        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {"loss": total_loss / total, "accuracy": total_correct / total}

    def feature_embed(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, _ = batch
        inputs = inputs.to(self.device)
        with torch.no_grad():
            for layer in self.net[:-1]:
                inputs = layer(inputs)
        return inputs.detach()
