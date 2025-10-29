"""Metrics utilities used across experiments."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def macro_average(per_dataset: Dict[str, float]) -> float:
    if not per_dataset:
        return 0.0
    return float(sum(per_dataset.values()) / len(per_dataset))


def ece_5(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    bins = torch.linspace(0.0, 1.0, steps=6, device=logits.device)
    total = logits.shape[0]
    ece = 0.0
    for idx in range(5):
        lower, upper = bins[idx], bins[idx + 1]
        mask = (confidences >= lower) & (confidences < upper)
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == targets[mask]).float().mean().item()
        conf = confidences[mask].mean().item()
        ece += mask.float().mean().item() * abs(acc - conf)
    return float(ece)


def detection_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area + 1e-8
    return float(inter_area / union)


def mean_average_precision(
    predictions: Sequence[Sequence[Tuple[float, torch.Tensor]]],
    targets: Sequence[Sequence[torch.Tensor]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute mean average precision for a toy detection task."""

    aps: List[float] = []
    for preds, gts in zip(predictions, targets):
        if len(gts) == 0:
            aps.append(1.0)
            continue
        preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
        tp = np.zeros(len(preds_sorted))
        fp = np.zeros(len(preds_sorted))
        used = np.zeros(len(gts), dtype=bool)
        for i, (score, box) in enumerate(preds_sorted):
            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                iou = detection_iou(box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_threshold and not used[best_idx]:
                tp[i] = 1.0
                used[best_idx] = True
            else:
                fp[i] = 1.0
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recalls = cumulative_tp / (len(gts) + 1e-8)
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)
        ap = 0.0
        for thr in np.linspace(0.0, 1.0, 11):
            mask = recalls >= thr
            prec = np.max(precisions[mask]) if np.any(mask) else 0.0
            ap += prec / 11.0
        aps.append(float(ap))
    if not aps:
        return 0.0
    return float(sum(aps) / len(aps))
