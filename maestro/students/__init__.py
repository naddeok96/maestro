"""Student exports and registry."""

from __future__ import annotations

from typing import List

from maestro.datasets import DatasetSpec

from .base_api import AbstractStudent
from .classification import ClassificationStudent
from .detection import DetectionStudent
from .ner import NERStudent

__all__ = [
    "ClassificationStudent",
    "DetectionStudent",
    "NERStudent",
    "build_student",
]


def build_student(dataset_specs: List[DatasetSpec]) -> AbstractStudent:
    """Instantiate a student matching the first dataset's task type."""

    if not dataset_specs:
        raise ValueError("No datasets provided")
    task = dataset_specs[0].task_type
    if any(spec.task_type != task for spec in dataset_specs):
        raise ValueError("All datasets must share the same task type")
    metadata = dataset_specs[0].metadata
    if task == "classification":
        return ClassificationStudent(
            in_channels=int(metadata.get("num_channels", 1)),
            num_classes=int(metadata["num_classes"]),
        )
    if task == "ner":
        return NERStudent(
            vocab_size=int(metadata["vocab_size"]),
            num_tags=int(metadata["num_tags"]),
        )
    if task == "detection":
        return DetectionStudent(
            image_size=int(metadata["image_size"]),
            num_channels=int(metadata.get("num_channels", 3)),
        )
    raise ValueError(f"Unknown task type: {task}")
