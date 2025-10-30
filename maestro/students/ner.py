"""Synthetic NER student."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from maestro.utils import OptimizerSettings, flatten_gradients


def _make_mask(tags: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(tags, dtype=torch.float32)


@dataclass
class NERStudent(nn.Module):
    vocab_size: int
    num_tags: int
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.encoder = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.decoder = nn.Linear(self.hidden_dim, self.num_tags)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_optimizer(self, settings: OptimizerSettings) -> None:
        self._optimizer = torch.optim.Adam(
            self.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        encoded, _ = self.encoder(emb)
        logits = self.decoder(encoded)
        return logits

    def step_on_minibatch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        self.train()
        tokens, tags = batch
        tokens = tokens.to(self.device)
        tags = tags.to(self.device)
        self._optimizer.zero_grad()
        logits = self.forward(tokens)
        loss = self.loss_fn(logits.view(-1, self.num_tags), tags.view(-1))
        mask = _make_mask(tags).view(-1).to(self.device)
        loss = (loss * mask).mean()
        loss.backward()
        grad_vec = flatten_gradients(list(self.parameters())).detach().cpu()
        self._optimizer.step()
        preds = logits.argmax(dim=-1)
        accuracy = (preds == tags).float().mean().item()
        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "grad_norm": float(grad_vec.norm().item()),
            "grad_vector": grad_vec,
        }

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        with torch.no_grad():
            for tokens, tags in loader:
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                logits = self.forward(tokens)
                per_token_loss = self.loss_fn(logits.view(-1, self.num_tags), tags.view(-1))
                total_loss += per_token_loss.sum().item()
                total_tokens += tags.numel()
                total_correct += (logits.argmax(dim=-1) == tags).sum().item()
        if total_tokens == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {"loss": total_loss / total_tokens, "accuracy": total_correct / total_tokens}

    def feature_embed(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tokens, _ = batch
        tokens = tokens.to(self.device)
        with torch.no_grad():
            emb = self.embedding(tokens)
            encoded, _ = self.encoder(emb)
        return encoded.mean(dim=1)
