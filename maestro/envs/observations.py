"""Observation construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from maestro.datasets import DatasetSpec
from maestro.utils import estimate_flops

from .student_runner import SegmentOutput


@dataclass
class Observation:
    g_data: np.ndarray
    g_model: np.ndarray
    g_progress: np.ndarray
    descriptors: np.ndarray
    dataset_names: List[str]


class ObservationBuilder:
    def __init__(
        self,
        datasets: List[DatasetSpec],
        total_budget: float,
        ablations: Dict[str, bool] | None = None,
    ) -> None:
        self.datasets = datasets
        self.total_budget = total_budget
        self.ablations = ablations or {}
        self.reset()

    def reset(self) -> None:
        self.previous_val_loss = 0.0
        self.previous_gradient_cos = 0.0
        self.previous_macro = 0.0
        self.prev_grad_norm = 0.0

    def _model_features(self, student: torch.nn.Module) -> np.ndarray:
        params = [p.detach().view(-1) for p in student.parameters()]
        param_count = float(sum(p.numel() for p in params))
        log_param_count = float(np.log(param_count + 1.0))
        sample_shape = None
        for spec in self.datasets:
            if spec.task_type == "classification":
                sample_shape = (spec.metadata["feature_dim"],)
                break
            if spec.task_type == "ner":
                sample_shape = (spec.metadata["sequence_length"],)
                break
            if spec.task_type == "detection":
                sample_shape = (
                    1,
                    spec.metadata["image_size"],
                    spec.metadata["image_size"],
                )
                break
        if sample_shape is None:
            sample_shape = (1,)
        try:
            flops = estimate_flops(student, sample_shape)
        except Exception:
            flops = 1e6
        log_flops = float(np.log(flops + 1.0))
        depth = float(len(list(student.modules())))
        layer_sizes = [float(np.log(p.numel() + 1.0)) for p in params if p.numel() > 0]
        median_log_width = float(np.median(layer_sizes)) if layer_sizes else 0.0
        weights = torch.cat(params)
        median = weights.median().item()
        deviation = torch.abs(weights - median)
        thresh = (
            torch.quantile(deviation, 0.75).item() if deviation.numel() > 0 else 0.0
        )
        mask = (torch.abs(weights - median) < thresh + 1e-8).float()
        sparsity = float(mask.mean().item())
        skip_density = 0.0
        return np.array(
            [
                log_param_count,
                log_flops,
                depth,
                median_log_width,
                sparsity,
                skip_density,
            ],
            dtype=np.float32,
        )

    def _progress_features(
        self,
        step_index: int,
        horizon: int,
        remaining_budget: float,
        segment: SegmentOutput,
    ) -> np.ndarray:
        frac_time = step_index / max(1, horizon)
        frac_budget = remaining_budget / max(1.0, self.total_budget)
        eta = segment.lr
        momentum = segment.momentum
        log_weight_decay = float(np.log(segment.weight_decay + 1e-8))
        val_loss = segment.val_loss
        train_minus_val = segment.train_loss - segment.val_loss
        slope = val_loss - self.previous_val_loss
        upr = segment.param_change
        grad_cos = segment.grad_cosine
        log_grad_norm = float(np.log(segment.grad_norm_ema + 1e-8))
        self.previous_val_loss = val_loss
        self.previous_macro = segment.macro_accuracy
        self.prev_grad_norm = segment.grad_norm
        return np.array(
            [
                frac_time,
                frac_budget,
                eta,
                momentum,
                log_weight_decay,
                val_loss,
                train_minus_val,
                slope,
                upr,
                grad_cos,
                log_grad_norm,
            ],
            dtype=np.float32,
        )

    def build(
        self,
        student: torch.nn.Module,
        step_index: int,
        horizon: int,
        remaining_budget: float,
        segment: SegmentOutput,
    ) -> Observation:
        dataset_names = [spec.name for spec in self.datasets]
        descriptor_list = [segment.descriptors[name] for name in dataset_names]
        descriptors = torch.stack(descriptor_list).numpy()
        # Expose the mean descriptor as g_data; the DeepSets summary is computed
        # inside the policy to match the paper's \rho(mean(\phi(z))) aggregation.
        g_data = descriptors.mean(axis=0)
        g_model = self._model_features(student)
        g_progress = self._progress_features(
            step_index, horizon, remaining_budget, segment
        )
        if self.ablations.get("drop_grad_cosine", False):
            if g_progress.size >= 10:
                g_progress[9] = 0.0
        if self.ablations.get("drop_progress_block", False):
            g_progress = np.zeros_like(g_progress)
        if self.ablations.get("drop_model_block", False):
            g_model = np.zeros_like(g_model)
        if self.ablations.get("drop_data_block", False):
            g_data = np.zeros_like(g_data)
        return Observation(
            g_data=g_data.astype(np.float32),
            g_model=g_model.astype(np.float32),
            g_progress=g_progress.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            dataset_names=dataset_names,
        )
