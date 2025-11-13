"""Toy detection student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from maestro.envs.metrics import mean_average_precision
from maestro.utils import OptimizerSettings, flatten_gradients


@dataclass(eq=False)
class DetectionStudent(nn.Module):
    image_size: int
    num_channels: int = 3
    max_predictions: int = 2

    def __post_init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(self.num_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, self.max_predictions * 5)
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.SmoothL1Loss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_optimizer(self, settings: OptimizerSettings) -> None:
        self._optimizer = torch.optim.Adam(
            self.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        logits = self.head(features.view(features.size(0), -1))
        return logits.view(images.size(0), self.max_predictions, 5)

    def step_on_minibatch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        images, targets = batch
        images = images.to(self.device)
        target_boxes = [t.to(self.device) for t in targets]
        self._optimizer.zero_grad()
        pred = self.forward(images)
        scores = pred[..., 0]
        boxes = pred[..., 1:]
        loss = 0.0
        bce_targets = torch.zeros_like(scores)
        for i, gt in enumerate(target_boxes):
            count = min(gt.shape[0], self.max_predictions)
            if count > 0:
                bce_targets[i, :count] = 1.0
                loss += self.l1(boxes[i, :count], gt[:count])
        loss = loss + self.bce(scores, bce_targets)
        loss.backward()
        grad_vec = flatten_gradients(list(self.parameters())).detach().cpu()
        self._optimizer.step()
        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_vec.norm().item()),
            "grad_vector": grad_vec,
        }

    def _predict_boxes(
        self, images: torch.Tensor
    ) -> List[List[Tuple[float, torch.Tensor]]]:
        pred = self.forward(images)
        scores = torch.sigmoid(pred[..., 0])
        boxes = pred[..., 1:]
        out: List[List[Tuple[float, torch.Tensor]]] = []
        for s, b in zip(scores, boxes):
            pairs = []
            for score, box in zip(s, b):
                pairs.append((float(score.item()), box.detach().cpu()))
            out.append(pairs)
        return out

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate with mAP@0.5 IoU for publication-quality detection metric."""
        self.eval()
        all_preds: List[List[Tuple[float, torch.Tensor]]] = []
        all_targets: List[List[torch.Tensor]] = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                preds = self._predict_boxes(images)  # list per image of (score, box)
                all_preds.extend(preds)
                all_targets.extend([t.detach().cpu() for t in targets])
        if not all_targets:
            return {"loss": 0.0, "accuracy": 0.0}
        map50 = mean_average_precision(all_preds, all_targets, iou_threshold=0.5)
        return {"loss": 0.0, "accuracy": float(map50)}

    def feature_embed(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        images, _ = batch
        images = images.to(self.device)
        with torch.no_grad():
            features = self.backbone(images)
        return features.view(features.size(0), -1)
