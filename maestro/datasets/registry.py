"""Dataset registry and factories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .real_classification import ClassificationConfig, build_classification_dataset
from .real_detection import DetectionConfig, build_detection_dataset
from .real_ner import NERConfig, build_ner_dataset


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


def _resolve_dataset_kind(
    cfg: Dict[str, object],
    index: int,
    *,
    singular_key: str,
    plural_key: str,
    default: str,
) -> str:
    if plural_key in cfg and cfg[plural_key]:
        kinds = list(cfg[plural_key])
        if not kinds:
            raise ValueError(f"{plural_key} provided but empty")
        return str(kinds[index % len(kinds)])
    return str(cfg.get(singular_key, default))


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
            dataset_kind = _resolve_dataset_kind(
                datasets_cfg,
                index,
                singular_key="dataset_kind",
                plural_key="dataset_kinds",
                default="cifar10",
            )
            data = build_classification_dataset(
                name,
                ClassificationConfig(
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    dataset_kind=dataset_kind,
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
            if "dataset_kind" not in datasets_cfg and "dataset_name" in datasets_cfg:
                datasets_cfg["dataset_kind"] = datasets_cfg["dataset_name"]
            dataset_kind = _resolve_dataset_kind(
                datasets_cfg,
                index,
                singular_key="dataset_kind",
                plural_key="dataset_kinds",
                default="conll2003",
            )
            data = build_ner_dataset(
                name,
                NERConfig(
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    noise=datasets_cfg.get("noise", 0.0),
                    dataset_kind=dataset_kind,
                    max_sequence_length=datasets_cfg.get("sequence_length", 32),
                    entity_injection_prob=datasets_cfg.get(
                        "entity_injection_prob", 0.0
                    ),
                    focus_entities=datasets_cfg.get("focus_entities"),
                    lowercase=bool(datasets_cfg.get("lowercase", True)),
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
            dataset_kind = _resolve_dataset_kind(
                datasets_cfg,
                index,
                singular_key="dataset_kind",
                plural_key="dataset_kinds",
                default="voc",
            )
            data = build_detection_dataset(
                name,
                DetectionConfig(
                    train_size=datasets_cfg["train_size"],
                    val_size=datasets_cfg["val_size"],
                    probe_size=datasets_cfg["probe_size"],
                    max_objects=datasets_cfg["max_objects"],
                    image_size=datasets_cfg["image_size"],
                    dataset_kind=dataset_kind,
                    categories=datasets_cfg.get("categories"),
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
