"""Probe utilities for dataset-level descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from maestro.datasets import DatasetSpec
from maestro.envs.metrics import ece_5
from maestro.utils import GradientProjector, RobustScalarNormalizer


@dataclass
class ProbeManager:
    student: torch.nn.Module
    datasets: List[DatasetSpec]
    probe_size: int
    projector: GradientProjector
    device: torch.device
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.normalisers: Dict[str, List[RobustScalarNormalizer]] = {
            spec.name: [RobustScalarNormalizer() for _ in range(8)] for spec in self.datasets
        }
        self.probe_loaders = {
            spec.name: DataLoader(spec.probe, batch_size=self.probe_size, shuffle=True)
            for spec in self.datasets
        }
        self._iters = {name: iter(loader) for name, loader in self.probe_loaders.items()}

    def _next_probe_batch(self, name: str):
        try:
            return next(self._iters[name])
        except StopIteration:
            self._iters[name] = iter(self.probe_loaders[name])
            return next(self._iters[name])

    def compute_descriptors(self, prev_grad: torch.Tensor, grad_ema: torch.Tensor) -> Dict[str, torch.Tensor]:
        descriptors: Dict[str, torch.Tensor] = {}
        param_list = list(self.student.parameters())
        for spec in self.datasets:
            batch = self._next_probe_batch(spec.name)
            if spec.task_type == "classification":
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.student(inputs)
                loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                nll_mean = loss.mean().item()
                q25, q75 = np.percentile(loss.detach().cpu().numpy(), [25, 75])
                nll_iqr = float(q75 - q25)
                probs = torch.softmax(logits, dim=-1)
                entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean().item()
                ece = ece_5(logits, targets)
                grad = torch.autograd.grad(loss.mean(), param_list, retain_graph=False)
                grad_vec = torch.cat([g.view(-1) for g in grad])
            elif spec.task_type == "ner":
                tokens, tags = batch
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                logits = self.student(tokens)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tags.view(-1), reduction="none"
                )
                nll_mean = loss.mean().item()
                q25, q75 = np.percentile(loss.detach().cpu().numpy(), [25, 75])
                nll_iqr = float(q75 - q25)
                probs = torch.softmax(logits, dim=-1)
                entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).mean().item()
                ece = ece_5(logits.view(-1, logits.size(-1)), tags.view(-1))
                grad = torch.autograd.grad(loss.mean(), param_list, retain_graph=False)
                grad_vec = torch.cat([g.view(-1) for g in grad])
            else:  # detection
                images, boxes = batch
                images = images.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                pred = self.student(images)
                scores = pred[..., 0]
                logits = torch.sigmoid(scores)
                # simple L1 loss between first box and GT
                loss_components = []
                l1 = torch.nn.SmoothL1Loss(reduction="none")
                for i, gt in enumerate(boxes):
                    if gt.numel() == 0:
                        continue
                    loss_components.append(l1(pred[i, 0, 1:], gt[0]))
                if loss_components:
                    loss_tensor = torch.stack(loss_components).mean(dim=-1)
                else:
                    loss_tensor = torch.zeros(images.size(0), device=self.device)
                nll_mean = loss_tensor.mean().item()
                q25, q75 = np.percentile(loss_tensor.detach().cpu().numpy(), [25, 75])
                nll_iqr = float(q75 - q25)
                entropy = float(-(logits * torch.log(logits + 1e-8)).mean().item())
                ece = float(torch.abs(logits.mean() - torch.ones_like(logits).mean()).item())
                loss = loss_tensor.mean()
                grad = torch.autograd.grad(loss, param_list, retain_graph=False)
                grad_vec = torch.cat([g.view(-1) for g in grad])

            grad_proj = self.projector.project(grad_vec.detach().cpu())
            grad_norm = float(grad_vec.norm().item())
            cos = torch.nn.functional.cosine_similarity(grad_proj, grad_ema, dim=0).item()

            # Diversity via effective rank
            with torch.no_grad():
                if spec.task_type == "classification":
                    features = self.student.feature_embed((inputs, targets))
                elif spec.task_type == "ner":
                    features = self.student.feature_embed((tokens, tags))
                else:
                    features = self.student.feature_embed((images, boxes))
            features = features.detach().cpu().numpy()
            cov = np.cov(features, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-8)
            eff_rank = float((eigvals.sum() ** 2) / (np.square(eigvals).sum()))
            log_eff_rank = float(np.log(eff_rank + 1e-8))
            size = float(np.log(len(spec.train) + 1.0))

            values = [
                nll_mean,
                nll_iqr,
                entropy,
                ece,
                float(np.log(grad_norm + 1e-8)),
                cos,
                log_eff_rank,
                size,
            ]
            normalised = [norm.update(val) for norm, val in zip(self.normalisers[spec.name], values)]
            descriptors[spec.name] = torch.tensor(normalised, dtype=torch.float32)
        return descriptors
