"""Dataset registry and factories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .synth_classification import ClassificationConfig, build_classification_dataset
from .synth_detection import DetectionConfig, build_detection_dataset
from .synth_ner import NERConfig, build_ner_dataset


@dataclass
class DatasetSpec:
    name: str
    task_type: str
    train: object
    val: object
    probe: object
    metadata: Dict[str, object]


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def build_from_config(
    path: str,
    seed: int,
    *,
    num_datasets: Optional[int] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> List[DatasetSpec]:
    """Build dataset specifications from a YAML configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    seed:
        Base random seed used for dataset materialization.
    num_datasets:
        Optional override for the number of datasets to instantiate.
    overrides:
        Optional mapping of configuration keys to override inside the
        ``datasets`` section (e.g., ``noise`` or ``imbalance``).
    """
    cfg = _load_yaml(Path(path))
    task = cfg["task_family"]
    datasets_cfg = dict(cfg["datasets"])
    if num_datasets is not None:
        datasets_cfg["count"] = int(num_datasets)
    if overrides:
        datasets_cfg.update(overrides)
    specs: List[DatasetSpec] = []
    for index in range(int(datasets_cfg.get("count", 1))):
        dataset_seed = seed + index * 17
        name = f"{task}_{index}"
        if task == "classification":
            data = build_classification_dataset(
                name,
                ClassificationConfig(
                    feature_dim=datasets_cfg["feature_dim"],
                    num_classes=datasets_cfg["num_classes"],
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    noise=datasets_cfg.get("noise", 0.0),
                    imbalance=datasets_cfg.get("imbalance", 0.0),
                ),
                dataset_seed,
            )
            specs.append(
                DatasetSpec(
                    name=name,
                    task_type="classification",
                    train=data["train"],
                    val=data["val"],
                    probe=data["probe"],
                    metadata=data["metadata"],
                )
            )
        elif task == "ner":
            data = build_ner_dataset(
                name,
                NERConfig(
                    vocab_size=datasets_cfg["vocab_size"],
                    num_tags=datasets_cfg["num_tags"],
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    noise=datasets_cfg.get("noise", 0.0),
                ),
                dataset_seed,
            )
            specs.append(
                DatasetSpec(
                    name=name,
                    task_type="ner",
                    train=data["train"],
                    val=data["val"],
                    probe=data["probe"],
                    metadata=data["metadata"],
                )
            )
        elif task == "detection":
            data = build_detection_dataset(
                name,
                DetectionConfig(
                    image_size=datasets_cfg["image_size"],
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    max_objects=datasets_cfg["max_objects"],
                    noise=datasets_cfg.get("noise", 0.0),
                ),
                dataset_seed,
            )
            specs.append(
                DatasetSpec(
                    name=name,
                    task_type="detection",
                    train=data["train"],
                    val=data["val"],
                    probe=data["probe"],
                    metadata=data["metadata"],
                )
            )
        else:
            raise ValueError(f"Unknown task family: {task}")
    return specs
