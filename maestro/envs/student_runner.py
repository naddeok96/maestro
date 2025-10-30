"""Handles student updates given teacher actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from maestro.datasets import DatasetSpec
from maestro.envs.metrics import macro_average
from maestro.utils import (
    ExponentialMovingAverage,
    GradientProjector,
    OptimizerSettings,
    flatten_parameters,
    parameter_change,
)

from .probes import ProbeManager


@dataclass
class SegmentOutput:
    dataset_metrics: Dict[str, Dict[str, float]]
    macro_accuracy: float
    train_loss: float
    val_loss: float
    grad_projection: torch.Tensor
    grad_ema: torch.Tensor
    grad_cosine: float
    grad_norm: float
    grad_norm_ema: float
    param_change: float
    descriptors: Dict[str, torch.Tensor]
    usage: int
    batches: int
    lr: float
    momentum: float
    weight_decay: float
    mixture: Sequence[float]


class StudentRunner:
    def __init__(
        self,
        student: torch.nn.Module,
        datasets: List[DatasetSpec],
        batch_size: int,
        probe_size: int,
        grad_project_dim: int,
        grad_ema_beta: float,
        grad_norm_alpha: float,
        seed: int,
        device: torch.device,
    ) -> None:
        self.student = student.to(device)
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        param_count = flatten_parameters(list(self.student.parameters())).numel()
        self.projector = GradientProjector(param_count, grad_project_dim, seed=seed)
        self.grad_ema = ExponentialMovingAverage(grad_ema_beta)
        self.grad_norm_ema = ExponentialMovingAverage(grad_norm_alpha)
        self.prev_grad = torch.zeros(grad_project_dim)
        self.prev_params = flatten_parameters(list(self.student.parameters())).detach().cpu()
        self.probe_manager = ProbeManager(
            student=self.student,
            datasets=self.datasets,
            probe_size=probe_size,
            projector=self.projector,
            device=device,
            seed=seed,
        )
        self.eval_loaders = {
            spec.name: DataLoader(spec.val, batch_size=batch_size)
            for spec in self.datasets
        }

    def _sample_batch(self, dataset: DatasetSpec):
        indices = self.rng.integers(0, len(dataset.train), size=self.batch_size)
        items = [dataset.train[int(i)] for i in indices]
        if dataset.task_type == "classification":
            xs = torch.stack([item[0] for item in items])
            ys = torch.stack([item[1] for item in items])
            return xs.to(self.device), ys.to(self.device)
        if dataset.task_type == "ner":
            tokens = torch.stack([item[0] for item in items])
            tags = torch.stack([item[1] for item in items])
            return tokens.to(self.device), tags.to(self.device)
        if dataset.task_type == "detection":
            images = torch.stack([item[0] for item in items])
            boxes = [item[1] for item in items]
            return images.to(self.device), [b.to(self.device) for b in boxes]
        raise ValueError(f"Unsupported task type: {dataset.task_type}")

    def run_segment(
        self,
        mixture: Sequence[float],
        usage_batches: int,
        settings: OptimizerSettings,
    ) -> SegmentOutput:
        mixture = np.array(mixture, dtype=np.float32)
        mixture = mixture / mixture.sum()
        batches = max(0, usage_batches)
        self.student.configure_optimizer(settings)

        grad_accum = []
        loss_accum = []
        for _ in range(batches):
            dataset_idx = int(self.rng.choice(len(self.datasets), p=mixture))
            batch = self._sample_batch(self.datasets[dataset_idx])
            metrics = self.student.step_on_minibatch(batch)
            if metrics.get("grad_vector") is not None:
                grad_accum.append(metrics["grad_vector"].float())
            loss_accum.append(metrics.get("loss", 0.0))
        if grad_accum:
            grad_stack = torch.stack(grad_accum)
            grad_mean = grad_stack.mean(dim=0)
        else:
            grad_mean = torch.zeros(self.projector.matrix.shape[1])
        grad_proj = self.projector.project(grad_mean)
        ema = self.grad_ema.update(grad_proj.cpu())
        grad_norm = float(grad_mean.norm().item())
        grad_norm_ema = float(self.grad_norm_ema.update(torch.tensor([grad_norm])).item())
        grad_cos = torch.nn.functional.cosine_similarity(grad_proj, self.prev_grad, dim=0).item()
        self.prev_grad = grad_proj.detach().cpu()

        current_params = flatten_parameters(list(self.student.parameters())).detach().cpu()
        param_delta = parameter_change(self.prev_params, current_params)
        self.prev_params = current_params

        dataset_metrics = {
            spec.name: self.student.eval_on_loader(self.eval_loaders[spec.name])
            for spec in self.datasets
        }
        macro_acc = macro_average({name: metrics.get("accuracy", 0.0) for name, metrics in dataset_metrics.items()})
        val_loss = float(np.mean([metrics.get("loss", 0.0) for metrics in dataset_metrics.values()]))
        train_loss = float(np.mean(loss_accum)) if loss_accum else 0.0

        descriptors = self.probe_manager.compute_descriptors(self.prev_grad, ema)

        return SegmentOutput(
            dataset_metrics=dataset_metrics,
            macro_accuracy=macro_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            grad_projection=grad_proj.detach().cpu(),
            grad_ema=ema.detach().cpu(),
            grad_cosine=grad_cos,
            grad_norm=grad_norm,
            grad_norm_ema=grad_norm_ema,
            param_change=param_delta,
            descriptors=descriptors,
            usage=batches * self.batch_size,
            batches=batches,
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
            mixture=mixture.tolist(),
        )
