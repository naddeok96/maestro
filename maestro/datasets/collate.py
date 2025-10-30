"""Custom collate functions for MAESTRO datasets."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def detection_collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Collate function that preserves variable-length box annotations.

    Detection datasets yield an image tensor and a tensor of bounding boxes per
    sample. The number of boxes can vary across the batch, so the default
    PyTorch collate (which stacks tensors) fails. This helper stacks images
    while returning the list of box tensors untouched.
    """

    if not batch:
        raise ValueError("detection_collate received an empty batch")

    images, boxes = zip(*batch)
    stacked_images = torch.stack(tuple(images), dim=0)
    return stacked_images, [box for box in boxes]
