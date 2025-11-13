# 🚀 Run-All Pipeline Bundle

_Root: `/home/naddeok5/maestro`_

<details>
<summary>Table of contents</summary>

- [configs/meta_train/lofo_classification.yaml](#file-configsmeta_trainlofo_classificationyaml)
- [configs/meta_train/small_cpu_debug.yaml](#file-configsmeta_trainsmall_cpu_debugyaml)
- [configs/publication/classification_pub.yaml](#file-configspublicationclassification_pubyaml)
- [configs/publication/detection_pub.yaml](#file-configspublicationdetection_pubyaml)
- [configs/publication/main_suite.yaml](#file-configspublicationmain_suiteyaml)
- [configs/publication/ner_pub.yaml](#file-configspublicationner_pubyaml)
- [configs/tasks/classification.yaml](#file-configstasksclassificationyaml)
- [configs/tasks/detection.yaml](#file-configstasksdetectionyaml)
- [configs/tasks/ner.yaml](#file-configstasksneryaml)
- [maestro/__init__.py](#file-maestro__init__py)
- [maestro/datasets/__init__.py](#file-maestrodatasets__init__py)
- [maestro/datasets/collate.py](#file-maestrodatasetscollatepy)
- [maestro/datasets/real_classification.py](#file-maestrodatasetsreal_classificationpy)
- [maestro/datasets/real_detection.py](#file-maestrodatasetsreal_detectionpy)
- [maestro/datasets/real_ner.py](#file-maestrodatasetsreal_nerpy)
- [maestro/datasets/registry.py](#file-maestrodatasetsregistrypy)
- [maestro/students/__init__.py](#file-maestrostudents__init__py)
- [maestro/students/base_api.py](#file-maestrostudentsbase_apipy)
- [maestro/students/classification.py](#file-maestrostudentsclassificationpy)
- [maestro/students/detection.py](#file-maestrostudentsdetectionpy)
- [maestro/students/features.py](#file-maestrostudentsfeaturespy)
- [maestro/students/ner.py](#file-maestrostudentsnerpy)
- [maestro/utils/__init__.py](#file-maestroutils__init__py)
- [maestro/utils/logging.py](#file-maestroutilsloggingpy)
- [maestro/utils/seeding.py](#file-maestroutilsseedingpy)
- [requirements.txt](#file-requirementstxt)
- [scripts/download_datasets.sh](#file-scriptsdownload_datasetssh)
- [scripts/export_learning_curves.py](#file-scriptsexport_learning_curvespy)
- [scripts/generate_ood_grid.py](#file-scriptsgenerate_ood_gridpy)
- [scripts/generate_tables.py](#file-scriptsgenerate_tablespy)
- [scripts/make_publication_figures.py](#file-scriptsmake_publication_figurespy)
- [scripts/run_all.sh](#file-scriptsrun_allsh)
- [scripts/run_comparative.py](#file-scriptsrun_comparativepy)
- [scripts/run_eval.py](#file-scriptsrun_evalpy)
- [scripts/run_markov_diag.py](#file-scriptsrun_markov_diagpy)
- [scripts/run_n_invariance.py](#file-scriptsrun_n_invariancepy)
- [train_baselines.py](#file-train_baselinespy)
- [train_maestro_teacher.py](#file-train_maestro_teacherpy)
- [train_maestro_yolo.py](#file-train_maestro_yolopy)

</details>

---

## `configs/meta_train/lofo_classification.yaml` <a id="file-configsmeta_trainlofo_classificationyaml"></a>

- Size: 285B

```yaml
defaults: [../defaults]
run:
  id: lofo_classification
  total_episodes: 5
  dry_run: true
tasks:
  - configs/tasks/classification.yaml
eval_tasks:
  - configs/tasks/detection.yaml
teacher:
  hidden_dim: 64
  dataset_head_dim: 64
  policy_hidden: [128, 128]
  value_hidden: [128, 128]
```


---

## `configs/meta_train/small_cpu_debug.yaml` <a id="file-configsmeta_trainsmall_cpu_debugyaml"></a>

- Size: 258B

```yaml
defaults: [../defaults]
run:
  id: debug_run
  total_episodes: 3
  dry_run: false
tasks:
  - configs/tasks/classification.yaml
  - configs/tasks/ner.yaml
teacher:
  hidden_dim: 64
  dataset_head_dim: 64
  policy_hidden: [128, 128]
  value_hidden: [128, 128]
```


---

## `configs/publication/classification_pub.yaml` <a id="file-configspublicationclassification_pubyaml"></a>

- Size: 193B

```yaml
task_family: classification
datasets:
  count: 4
  dataset_kinds: [cifar10, cifar10, mnist, fashion_mnist]
  train_size: 20000
  val_size: 5000
  probe_size: 5000
  noise: 0.1
  imbalance: 0.4
```


---

## `configs/publication/detection_pub.yaml` <a id="file-configspublicationdetection_pubyaml"></a>

- Size: 257B

```yaml
task_family: detection
datasets:
  count: 3
  dataset_kinds: [voc, coco_tiny, shapes]
  image_size: 320
  train_size: 1500
  val_size: 400
  probe_size: 400
  max_objects: 6
  categories: ["person", "car", "bus", "truck", "bicycle", "dog", "cat", "bottle"]
```


---

## `configs/publication/main_suite.yaml` <a id="file-configspublicationmain_suiteyaml"></a>

- Size: 1KB

```yaml
defaults: [../defaults]
run:
  id: publication_main
  total_episodes: 5000
  checkpoint_interval: 100
  eval_interval: 50
seed: 42
horizon: 100
batch_size: 128
initial_budget: 65536

tasks:
  - configs/publication/classification_pub.yaml
  - configs/publication/ner_pub.yaml
  - configs/publication/detection_pub.yaml

teacher:
  hidden_dim: 256
  dataset_head_dim: 256
  policy_hidden: [512, 512, 256, 128]
  value_hidden: [512, 512, 256, 128]
  mixture_bias_init: 0.5

probe:
  size: 512
  grad_project_dim: 16384
  grad_ema_beta: 0.97
  grad_norm_alpha: 0.97

ppo:
  gamma: 0.995
  gae_lambda: 0.98
  clip_ratio: 0.2
  learning_rate: 3.0e-5
  minibatch_size: 2048
  rollout_length: 8192
  epochs: 15
  entropy_coef_mix: 0.015
  entropy_coef_u: 0.004
  barrier_kappa: 1.0e-4
  barrier_kappa_prime: 2.0e-3
  value_coef: 0.5
  max_grad_norm: 1.0
  cmdp_lambda_lr: 1.0e-4
  cmdp_target_fraction: 0.90

ppo_exploration:
  warmup_episodes: 300
  entropy_mix_warmup: 0.02
  entropy_mix_final: 0.01
  entropy_u_warmup: 0.006
  entropy_u_final: 0.002
  barrier_u_warmup: 0.002
  barrier_u_final: 0.001
  

optimizer:
  weight_decay: 1.0e-5
  momentum: 0.95
  eta_min: 1.0e-6
  eta_max: 1.0e-1

logging:
  output_dir: outputs/publication_$(date +%Y%m%d)
  tensorboard: false
  csv_interval: 1
  checkpoint_best: true
```


---

## `configs/publication/ner_pub.yaml` <a id="file-configspublicationner_pubyaml"></a>

- Size: 261B

```yaml
task_family: ner
datasets:
  count: 4
  dataset_kinds: [conll2003]
  train_size: 6000
  val_size: 2000
  probe_size: 2000
  sequence_length: 64
  noise: 0.1
  entity_injection_prob: 0.15
  focus_entities: ["B-ORG", "B-PER", "B-LOC", "B-MISC"]
  lowercase: true
```


---

## `configs/tasks/classification.yaml` <a id="file-configstasksclassificationyaml"></a>

- Size: 184B

```yaml
task_family: classification
datasets:
  count: 3
  dataset_kinds: [cifar10, mnist, fashion_mnist]
  train_size: 4096
  val_size: 1024
  probe_size: 1024
  noise: 0.05
  imbalance: 0.3
```


---

## `configs/tasks/detection.yaml` <a id="file-configstasksdetectionyaml"></a>

- Size: 219B

```yaml
task_family: detection
datasets:
  count: 2
  dataset_kinds: [voc, shapes]
  image_size: 256
  train_size: 320
  val_size: 120
  probe_size: 120
  max_objects: 5
  categories: ["person", "dog", "cat", "car", "bicycle"]
```


---

## `configs/tasks/ner.yaml` <a id="file-configstasksneryaml"></a>

- Size: 249B

```yaml
task_family: ner
datasets:
  count: 2
  dataset_kinds: [conll2003]
  train_size: 2000
  val_size: 800
  probe_size: 800
  sequence_length: 48
  noise: 0.05
  entity_injection_prob: 0.1
  focus_entities: ["B-ORG", "B-PER", "B-LOC"]
  lowercase: true
```


---

## `maestro/__init__.py` <a id="file-maestro__init__py"></a>

- Size: 141B

```python
"""MAESTRO experiments package."""

from .envs.maestro_env import MaestroEnv, MaestroEnvConfig

__all__ = ["MaestroEnv", "MaestroEnvConfig"]
```


---

## `maestro/datasets/__init__.py` <a id="file-maestrodatasets__init__py"></a>

- Size: 185B

```python
"""Dataset exports."""

from .collate import detection_collate
from .registry import DatasetSpec, build_from_config

__all__ = ["DatasetSpec", "build_from_config", "detection_collate"]
```


---

## `maestro/datasets/collate.py` <a id="file-maestrodatasetscollatepy"></a>

- Size: 865B

```python
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
```


---

## `maestro/datasets/real_classification.py` <a id="file-maestrodatasetsreal_classificationpy"></a>

- Size: 11KB

```python
"""Realistic image classification datasets for MAESTRO."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


__all__ = ["ClassificationConfig", "build_classification_dataset"]


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MAESTRO_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "data",
    )
)


_DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "image_size": 32,
    },
    "mnist": {
        "cls": datasets.MNIST,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "image_size": 28,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "num_classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
        "channels": 1,
        "image_size": 28,
    },
}


@dataclass
class ClassificationConfig:
    """Configuration controlling the realistic classification builder."""

    dataset_kind: str
    train_size: int
    val_size: int
    probe_size: int
    noise: float = 0.0
    imbalance: float = 0.0


class SubsampledDataset(Dataset):
    """Dataset wrapper that applies deterministic transforms per base index."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        labels: np.ndarray,
        *,
        dataset_kind: str,
        transform_noise: float,
        domain_seed: int,
        image_size: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.labels = torch.from_numpy(labels).long()
        self.dataset_kind = dataset_kind
        self.transform_noise = transform_noise
        self.domain_seed = domain_seed
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        img, _ = self.base_dataset[base_idx]
        if not isinstance(img, Image.Image):
            raise TypeError(
                f"Expected PIL.Image from torchvision dataset, got {type(img)}"
            )
        tensor = apply_domain_transform(
            img,
            dataset_kind=self.dataset_kind,
            noise=self.transform_noise,
            seed=self.domain_seed + base_idx * 1009,
            image_size=self.image_size,
        )
        label = self.labels[idx]
        return tensor, label


def _dataset_root(kind: str) -> Path:
    root = DEFAULT_DATA_ROOT / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_torchvision_dataset(kind: str, train: bool) -> Dataset:
    if kind not in _DATASET_REGISTRY:
        raise ValueError(f"Unsupported classification dataset kind: {kind}")
    meta = _DATASET_REGISTRY[kind]
    dataset_cls = meta["cls"]  # type: ignore[index]
    root = _dataset_root(kind)
    allow_download = os.environ.get("MAESTRO_ALLOW_DATASET_DOWNLOAD", "0") == "1"
    try:
        return dataset_cls(root=root, train=train, download=False)  # type: ignore[call-arg]
    except RuntimeError as exc:
        if allow_download:
            return dataset_cls(root=root, train=train, download=True)  # type: ignore[call-arg]
        raise RuntimeError(
            f"{exc}. Run scripts/download_datasets.sh to fetch {kind}, "
            "or set MAESTRO_ALLOW_DATASET_DOWNLOAD=1 to allow on-demand downloads."
        ) from exc


def _targets_as_numpy(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError(
            f"Dataset {dataset.__class__.__name__} does not expose targets."
        )
    if isinstance(targets, list):
        return np.asarray(targets, dtype=np.int64)
    if torch.is_tensor(targets):
        return targets.detach().cpu().numpy().astype(np.int64)
    return np.asarray(targets, dtype=np.int64)


def create_imbalanced_indices(
    targets: Sequence[int],
    size: int,
    imbalance: float,
    seed: int,
) -> List[int]:
    """Sample indices with controllable class imbalance."""

    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets, dtype=np.int64)
    unique = np.unique(targets_arr)
    if unique.size == 0:
        raise ValueError("Targets array is empty")
    order = np.argsort(unique)
    unique = unique[order]
    ranks = {cls: rank for rank, cls in enumerate(unique)}
    tail_strength = max(0.0, float(imbalance))
    class_weights = np.array(
        [np.exp(-tail_strength * ranks[int(cls)]) for cls in unique], dtype=np.float32
    )
    class_weights /= class_weights.sum()
    weights = np.array([class_weights[ranks[int(cls)]] for cls in targets_arr])
    weights /= weights.sum()
    replace = size > len(targets_arr)
    indices = rng.choice(len(targets_arr), size=size, replace=replace, p=weights)
    return indices.astype(int).tolist()


def inject_label_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, rng: np.random.Generator
) -> np.ndarray:
    if noise_rate <= 0.0:
        return labels
    noisy = labels.copy()
    mask = rng.random(size=noisy.shape) < noise_rate
    if not mask.any():
        return noisy
    noisy_classes = rng.integers(0, num_classes, size=int(mask.sum()))
    noisy[mask] = noisy_classes
    return noisy


def apply_domain_transform(
    image: Image.Image,
    *,
    dataset_kind: str,
    noise: float,
    seed: int,
    image_size: int,
) -> torch.Tensor:
    """Apply deterministic, domain-specific augmentations."""

    rng = np.random.default_rng(seed)
    if dataset_kind == "cifar10":
        image = image.convert("RGB")
        if image_size != 32:
            image = image.resize((image_size, image_size), Image.BILINEAR)
        pad = 4
        padded = TF.pad(image, padding=pad, fill=0)
        max_offset = pad * 2
        top = int(rng.integers(0, max_offset + 1))
        left = int(rng.integers(0, max_offset + 1))
        image = padded.crop((left, top, left + image_size, top + image_size))
        if rng.random() < 0.5:
            image = TF.hflip(image)
        brightness = 1.0 + rng.uniform(-0.2, 0.2) * (1.0 + noise)
        contrast = 1.0 + rng.uniform(-0.2, 0.2) * (1.0 + noise)
        saturation = 1.0 + rng.uniform(-0.1, 0.1) * (1.0 + noise * 0.5)
        hue = rng.uniform(-0.02, 0.02) * (1.0 + noise * 0.25)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)
        tensor = TF.to_tensor(image)
        mean = torch.tensor(_DATASET_REGISTRY["cifar10"]["mean"]).view(-1, 1, 1)
        std = torch.tensor(_DATASET_REGISTRY["cifar10"]["std"]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    if dataset_kind in ("mnist", "fashion_mnist"):
        image = image.convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BILINEAR)
        angle = float(rng.uniform(-25.0, 25.0) * (1.0 + noise * 0.5))
        scale = float(rng.uniform(0.9, 1.1 + noise * 0.15))
        translate = (
            int(rng.integers(-3, 4)),
            int(rng.integers(-3, 4)),
        )
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=(0.0, 0.0),
        )
        if dataset_kind == "fashion_mnist":
            brightness = 1.0 + rng.uniform(-0.3, 0.3) * (1.0 + noise)
            contrast = 1.0 + rng.uniform(-0.3, 0.3) * (1.0 + noise)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
        tensor = TF.to_tensor(image)
        stats_key = "mnist" if dataset_kind == "mnist" else "fashion_mnist"
        mean = torch.tensor(_DATASET_REGISTRY[stats_key]["mean"]).view(-1, 1, 1)
        std = torch.tensor(_DATASET_REGISTRY[stats_key]["std"]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    raise ValueError(f"Unknown dataset kind for transform: {dataset_kind}")


def _sample_split_indices(size: int, total: int, rng: np.random.Generator) -> List[int]:
    replace = size > total
    indices = rng.choice(total, size=size, replace=replace)
    return indices.astype(int).tolist()


def build_classification_dataset(
    name: str,
    config: ClassificationConfig,
    seed: int,
) -> Dict[str, Dataset]:
    """Instantiate realistic classification datasets and metadata."""

    dataset_kind = config.dataset_kind
    if dataset_kind not in _DATASET_REGISTRY:
        raise ValueError(f"Unsupported classification dataset: {dataset_kind}")
    meta = _DATASET_REGISTRY[dataset_kind]
    base_train = _load_torchvision_dataset(dataset_kind, train=True)
    base_test = _load_torchvision_dataset(dataset_kind, train=False)
    rng = np.random.default_rng(seed)
    train_indices = create_imbalanced_indices(
        _targets_as_numpy(base_train),
        config.train_size,
        config.imbalance,
        seed + 11,
    )
    train_labels = np.asarray(_targets_as_numpy(base_train))[train_indices]
    train_labels = inject_label_noise(
        train_labels,
        config.noise,
        int(meta["num_classes"]),
        rng,
    )
    val_indices = _sample_split_indices(
        config.val_size, len(base_test), rng
    )
    probe_indices = _sample_split_indices(
        config.probe_size, len(base_test), rng
    )
    val_targets = np.asarray(_targets_as_numpy(base_test))[val_indices]
    probe_targets = np.asarray(_targets_as_numpy(base_test))[probe_indices]
    image_size = int(meta["image_size"])
    train_ds = SubsampledDataset(
        base_train,
        train_indices,
        train_labels,
        dataset_kind=dataset_kind,
        transform_noise=config.noise,
        domain_seed=seed,
        image_size=image_size,
    )
    val_ds = SubsampledDataset(
        base_test,
        val_indices,
        val_targets,
        dataset_kind=dataset_kind,
        transform_noise=0.0,
        domain_seed=seed + 101,
        image_size=image_size,
    )
    probe_ds = SubsampledDataset(
        base_test,
        probe_indices,
        probe_targets,
        dataset_kind=dataset_kind,
        transform_noise=0.0,
        domain_seed=seed + 211,
        image_size=image_size,
    )
    metadata = {
        "num_classes": int(meta["num_classes"]),
        "num_channels": int(meta["channels"]),
        "image_size": image_size,
        "input_shape": (
            int(meta["channels"]),
            image_size,
            image_size,
        ),
        "input_type": "image",
        "dataset_kind": dataset_kind,
        "name": name,
    }
    return {"train": train_ds, "val": val_ds, "probe": probe_ds, "metadata": metadata}
```


---

## `maestro/datasets/real_detection.py` <a id="file-maestrodatasetsreal_detectionpy"></a>

- Size: 13KB

```python
"""Realistic detection dataset builders (VOC/COCO/shapes-on-backgrounds)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection, VOCDetection


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MAESTRO_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "data",
    )
)
VOC_ROOT = DEFAULT_DATA_ROOT / "voc"
COCO_ROOT = DEFAULT_DATA_ROOT / "coco"

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(-1, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(-1, 1, 1)

DEFAULT_COCO_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "dog",
    "cat",
    "chair",
    "bottle",
    "truck",
    "bus",
    "bird",
]


__all__ = ["DetectionConfig", "build_detection_dataset"]


@dataclass
class DetectionConfig:
    dataset_kind: str
    train_size: int
    val_size: int
    probe_size: int
    max_objects: int
    image_size: int
    categories: Sequence[str] | None = None


@dataclass
class DetectionRecord:
    image_path: Path | None
    boxes: torch.Tensor
    orig_size: Tuple[int, int]
    source: str
    tensor: torch.Tensor | None = None


class DetectionDataset(Dataset):
    """Lazy dataset that loads/resizes images on demand."""

    def __init__(self, records: Sequence[DetectionRecord], image_size: int) -> None:
        self.records = list(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if record.tensor is not None:
            return record.tensor.clone(), record.boxes.clone()
        if record.image_path is None:
            raise ValueError("Detection record missing image tensor and path.")
        image = Image.open(record.image_path).convert("RGB")
        tensor, boxes = _prepare_image_and_boxes(
            image,
            record.boxes,
            orig_size=record.orig_size,
            image_size=self.image_size,
        )
        return tensor, boxes


def _prepare_image_and_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    *,
    orig_size: Tuple[int, int],
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = image.resize((image_size, image_size), Image.BILINEAR)
    width, height = orig_size
    width = max(1, int(width))
    height = max(1, int(height))
    scale_x = image_size / float(width)
    scale_y = image_size / float(height)
    scaled = boxes.clone()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    scaled = torch.clamp(scaled, 0.0, float(image_size))
    tensor = TF.to_tensor(image)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, scaled


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"{description} not found at {path}. "
            "Run scripts/download_datasets.sh first."
        )


def _voc_split_path(image_set: str) -> Path:
    return VOC_ROOT / "VOCdevkit" / "VOC2007" / "ImageSets" / "Main" / f"{image_set}.txt"


def _load_voc_dataset(image_set: str) -> VOCDetection:
    _ensure_exists(VOC_ROOT, "VOC root directory")
    try:
        return VOCDetection(
            root=str(VOC_ROOT),
            year="2007",
            image_set=image_set,
            download=False,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}. Ensure PASCAL VOC files exist under {VOC_ROOT}."
        ) from exc


def _parse_voc_boxes(
    target: Dict,
    categories: Sequence[str] | None,
    max_objects: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    objects = target["annotation"].get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    boxes: List[List[float]] = []
    for obj in objects:
        name = obj.get("name")
        if categories and name not in categories:
            continue
        bbox = obj.get("bndbox", {})
        try:
            x1 = float(bbox["xmin"])
            y1 = float(bbox["ymin"])
            x2 = float(bbox["xmax"])
            y2 = float(bbox["ymax"])
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
    boxes = boxes[:max_objects]
    size = target["annotation"]["size"]
    width = int(size["width"])
    height = int(size["height"])
    return torch.as_tensor(boxes, dtype=torch.float32), (width, height)


def _collect_voc_records(
    image_set: str,
    categories: Sequence[str] | None,
    max_objects: int,
) -> List[DetectionRecord]:
    dataset = _load_voc_dataset(image_set)
    records: List[DetectionRecord] = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        boxes, size = _parse_voc_boxes(target, categories, max_objects)
        if boxes.numel() == 0:
            continue
        image_path = Path(dataset.images[idx])
        records.append(
            DetectionRecord(
                image_path=image_path,
                boxes=boxes,
                orig_size=size,
                source="voc",
            )
        )
    if not records:
        raise RuntimeError(
            f"No usable VOC samples found for split '{image_set}'. "
            "Consider adjusting `categories` or run download script."
        )
    return records


def _coco_annotation_file(split: str) -> Tuple[Path, Path]:
    images = COCO_ROOT / "images" / split
    annotations = COCO_ROOT / "annotations" / f"instances_{split}.json"
    _ensure_exists(images, f"COCO images ({split})")
    _ensure_exists(annotations, f"COCO annotations ({split})")
    return images, annotations


def _coco_category_lookup(dataset: CocoDetection) -> Dict[int, str]:
    cats = dataset.coco.loadCats(dataset.coco.getCatIds())
    return {cat["id"]: cat["name"] for cat in cats}


def _collect_coco_records(
    split: str,
    size: int,
    categories: Sequence[str] | None,
    max_objects: int,
    rng: np.random.Generator,
) -> List[DetectionRecord]:
    images, ann = _coco_annotation_file(split)
    dataset = CocoDetection(root=str(images), annFile=str(ann))
    cat_lookup = _coco_category_lookup(dataset)
    whitelist = set(categories or DEFAULT_COCO_CATEGORIES)
    records: List[DetectionRecord] = []
    seen = set()
    attempts = 0
    max_attempts = len(dataset) * 5
    while len(records) < size and attempts < max_attempts:
        idx = int(rng.integers(0, len(dataset)))
        if idx in seen:
            attempts += 1
            continue
        seen.add(idx)
        image, annotations = dataset[idx]
        boxes: List[List[float]] = []
        for ann_data in annotations:
            label = cat_lookup.get(ann_data.get("category_id"))
            if whitelist and label not in whitelist:
                continue
            bbox = ann_data.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            boxes.append([x1, y1, x1 + w, y1 + h])
        if not boxes:
            attempts += 1
            continue
        boxes_arr = torch.as_tensor(boxes[:max_objects], dtype=torch.float32)
        if boxes_arr.numel() == 0:
            attempts += 1
            continue
        img_meta = dataset.coco.loadImgs(dataset.ids[idx])[0]
        image_path = Path(dataset.root) / img_meta["file_name"]
        records.append(
            DetectionRecord(
                image_path=image_path,
                boxes=boxes_arr,
                orig_size=(img_meta["width"], img_meta["height"]),
                source="coco",
            )
        )
    if len(records) < size:
        raise RuntimeError(
            f"Unable to collect {size} COCO samples for split '{split}'. "
            "Try running download_datasets.sh or reducing split sizes."
        )
    return records[:size]


def _shape_backgrounds() -> List[Path]:
    images_dir = VOC_ROOT / "VOCdevkit" / "VOC2007" / "JPEGImages"
    _ensure_exists(images_dir, "VOC JPEGImages for synthetic shapes")
    return sorted(images_dir.glob("*.jpg"))


def _render_shapes_records(
    total: int,
    image_size: int,
    max_objects: int,
    rng: np.random.Generator,
) -> List[DetectionRecord]:
    backgrounds = _shape_backgrounds()
    if not backgrounds:
        raise RuntimeError("No VOC backgrounds available for synthetic shapes.")
    records: List[DetectionRecord] = []
    for idx in range(total):
        bg_path = backgrounds[idx % len(backgrounds)]
        image = Image.open(bg_path).convert("RGB").resize(
            (image_size, image_size), Image.BILINEAR
        )
        draw = ImageDraw.Draw(image, "RGBA")
        count = int(rng.integers(1, max_objects + 1))
        boxes: List[List[float]] = []
        for _ in range(count):
            w = float(rng.uniform(image_size * 0.1, image_size * 0.4))
            h = float(rng.uniform(image_size * 0.1, image_size * 0.4))
            x1 = float(rng.uniform(0, image_size - w))
            y1 = float(rng.uniform(0, image_size - h))
            x2 = x1 + w
            y2 = y1 + h
            color = (
                int(rng.integers(32, 220)),
                int(rng.integers(32, 220)),
                int(rng.integers(32, 220)),
                200,
            )
            if rng.random() < 0.5:
                draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=3)
            else:
                draw.ellipse([x1, y1, x2, y2], outline=color[:3], width=3)
            boxes.append([x1, y1, x2, y2])
        tensor = TF.to_tensor(image)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        records.append(
            DetectionRecord(
                image_path=None,
                boxes=torch.as_tensor(boxes, dtype=torch.float32),
                orig_size=(image_size, image_size),
                source="shapes",
                tensor=tensor,
            )
        )
    return records


def _split_records(
    records: Sequence[DetectionRecord],
    sizes: Tuple[int, int, int],
    rng: np.random.Generator,
) -> Tuple[List[DetectionRecord], List[DetectionRecord], List[DetectionRecord]]:
    total = sum(sizes)
    if total > len(records):
        raise ValueError("Not enough records to split into train/val/probe.")
    indices = list(range(len(records)))
    rng.shuffle(indices)
    start = 0
    splits: List[List[DetectionRecord]] = []
    for size in sizes:
        end = start + size
        splits.append([records[i] for i in indices[start:end]])
        start = end
    return splits[0], splits[1], splits[2]


def build_detection_dataset(
    name: str,
    config: DetectionConfig,
    seed: int,
) -> Dict[str, Dataset]:
    """Build detection datasets backed by VOC/COCO or shapes on VOC backgrounds."""

    rng = np.random.default_rng(seed)
    dataset_kind = config.dataset_kind
    image_size = config.image_size
    if dataset_kind == "voc":
        train_records_full = _collect_voc_records("trainval", config.categories, config.max_objects)
        eval_records_full = _collect_voc_records("test", config.categories, config.max_objects)
        train_records = _sample_records(train_records_full, config.train_size, rng)
        val_records = _sample_records(eval_records_full, config.val_size, rng)
        probe_records = _sample_records(eval_records_full, config.probe_size, rng)
    elif dataset_kind == "coco_tiny":
        train_records = _collect_coco_records(
            "train2017",
            config.train_size,
            categories=config.categories,
            max_objects=config.max_objects,
            rng=rng,
        )
        val_records = _collect_coco_records(
            "val2017",
            config.val_size,
            categories=config.categories,
            max_objects=config.max_objects,
            rng=rng,
        )
        probe_records = _collect_coco_records(
            "val2017",
            config.probe_size,
            categories=config.categories,
            max_objects=config.max_objects,
            rng=rng,
        )
    elif dataset_kind == "shapes":
        total = config.train_size + config.val_size + config.probe_size
        records = _render_shapes_records(
            total,
            image_size=image_size,
            max_objects=config.max_objects,
            rng=rng,
        )
        train_records, val_records, probe_records = _split_records(
            records,
            (config.train_size, config.val_size, config.probe_size),
            rng,
        )
    else:
        raise ValueError(f"Unsupported detection dataset kind: {dataset_kind}")
    dataset = {
        "train": DetectionDataset(train_records, image_size),
        "val": DetectionDataset(val_records, image_size),
        "probe": DetectionDataset(probe_records, image_size),
        "metadata": {
            "image_size": image_size,
            "num_channels": 3,
            "input_shape": (3, image_size, image_size),
            "input_type": "image",
            "dataset_kind": dataset_kind,
            "name": name,
        },
    }
    return dataset


def _sample_records(
    records: Sequence[DetectionRecord],
    size: int,
    rng: np.random.Generator,
) -> List[DetectionRecord]:
    if not records:
        raise ValueError("No detection records to sample from.")
    replace = size > len(records)
    indices = rng.choice(len(records), size=size, replace=replace)
    return [records[int(i)] for i in indices]
```


---

## `maestro/datasets/real_ner.py` <a id="file-maestrodatasetsreal_nerpy"></a>

- Size: 8KB

```python
"""Realistic NER datasets backed by CoNLL-style corpora."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from datasets import DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover - handled via requirements
    raise ImportError(
        "The 'datasets' package is required for the NER builder. "
        "Please install it via `pip install datasets` or run "
        "`scripts/download_datasets.sh`."
    ) from exc


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "MAESTRO_DATA_ROOT",
        Path(__file__).resolve().parents[2] / "data",
    )
)
HF_CACHE = Path(
    os.environ.get(
        "MAESTRO_HF_CACHE",
        DEFAULT_DATA_ROOT / "hf_cache",
    )
)
HF_CACHE.mkdir(parents=True, exist_ok=True)


__all__ = ["NERConfig", "build_ner_dataset"]


@dataclass
class NERConfig:
    dataset_kind: str
    train_size: int
    val_size: int
    probe_size: int
    noise: float
    max_sequence_length: int = 32
    entity_injection_prob: float = 0.0
    focus_entities: Sequence[str] | None = None
    lowercase: bool = True


class TokenSequenceDataset(Dataset):
    """Simple tensor-backed dataset for token/tag sequences."""

    def __init__(self, tokens: np.ndarray, tags: np.ndarray) -> None:
        self.tokens = torch.from_numpy(tokens).long()
        self.tags = torch.from_numpy(tags).long()

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int):
        return self.tokens[index], self.tags[index]


_VOCAB_CACHE: Dict[Tuple[str, bool], Dict[str, int]] = {}


def _build_vocab(
    corpus: DatasetDict, lowercase: bool, *, min_freq: int = 1
) -> Dict[str, int]:
    cache_key = (corpus["train"].info.builder_name, lowercase)
    if cache_key in _VOCAB_CACHE:
        return _VOCAB_CACHE[cache_key]
    counter: Dict[str, int] = {}
    for split in corpus.values():
        tokens_list = split["tokens"]
        for tokens in tokens_list:
            for token in tokens:
                key = token.lower() if lowercase else token
                counter[key] = counter.get(key, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        if token not in vocab:
            vocab[token] = len(vocab)
    _VOCAB_CACHE[cache_key] = vocab
    return vocab


def _encode_example(
    tokens: Sequence[str],
    tags: Sequence[int],
    vocab: Dict[str, int],
    *,
    lowercase: bool,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    token_ids = np.full(max_length, vocab["<pad>"], dtype=np.int64)
    tag_ids = np.zeros(max_length, dtype=np.int64)
    limit = min(len(tokens), max_length)
    for idx in range(limit):
        token = tokens[idx].lower() if lowercase else tokens[idx]
        token_ids[idx] = vocab.get(token, vocab["<unk>"])
        tag_ids[idx] = int(tags[idx])
    return token_ids, tag_ids


def _inject_entities(
    tag_ids: np.ndarray,
    rng: np.random.Generator,
    num_tags: int,
) -> None:
    if tag_ids.size == 0:
        return
    span = max(1, int(tag_ids.size * 0.2))
    start = int(rng.integers(0, max(1, tag_ids.size - span + 1)))
    end = min(tag_ids.size, start + span)
    new_tag = int(rng.integers(1, num_tags))
    tag_ids[start:end] = new_tag


def _apply_sequence_noise(
    tag_ids: np.ndarray, noise_rate: float, rng: np.random.Generator, num_tags: int
) -> np.ndarray:
    if noise_rate <= 0.0:
        return tag_ids
    noisy = tag_ids.copy()
    mask = rng.random(size=noisy.shape) < noise_rate
    if mask.any():
        noisy[mask] = rng.integers(0, num_tags, size=int(mask.sum()))
    return noisy


def _collect_examples(
    split,
    focus_ids: Sequence[int] | None,
) -> List[Dict[str, Sequence]]:
    examples: List[Dict[str, Sequence]] = []
    for tokens, tags in zip(split["tokens"], split["ner_tags"]):
        if focus_ids:
            if not any(tag in focus_ids for tag in tags):
                continue
        examples.append({"tokens": tokens, "ner_tags": tags})
    return examples if examples else [
        {"tokens": tokens, "ner_tags": tags}
        for tokens, tags in zip(split["tokens"], split["ner_tags"])
    ]


def _sample_examples(
    examples: Sequence[Dict[str, Sequence]],
    size: int,
    rng: np.random.Generator,
) -> List[Dict[str, Sequence]]:
    if not examples:
        raise ValueError("No examples available to sample.")
    replace = size > len(examples)
    indices = rng.choice(len(examples), size=size, replace=replace)
    return [examples[int(i)] for i in indices]


def _resolve_focus_ids(
    focus_entities: Sequence[str] | None,
    tag_names: Sequence[str],
) -> List[int] | None:
    if not focus_entities:
        return None
    name_to_id = {name: idx for idx, name in enumerate(tag_names)}
    ids = [name_to_id[name] for name in focus_entities if name in name_to_id]
    return ids or None


def _encode_split(
    examples: Sequence[Dict[str, Sequence]],
    size: int,
    rng: np.random.Generator,
    *,
    vocab: Dict[str, int],
    lowercase: bool,
    max_length: int,
    num_tags: int,
    noise: float,
    entity_injection_prob: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sampled = _sample_examples(examples, size, rng)
    tokens = np.zeros((len(sampled), max_length), dtype=np.int64)
    tags = np.zeros((len(sampled), max_length), dtype=np.int64)
    for idx, example in enumerate(sampled):
        token_ids, tag_ids = _encode_example(
            example["tokens"],
            example["ner_tags"],
            vocab,
            lowercase=lowercase,
            max_length=max_length,
        )
        if entity_injection_prob > 0.0 and rng.random() < entity_injection_prob:
            _inject_entities(tag_ids, rng, num_tags)
        if noise > 0.0:
            tag_ids = _apply_sequence_noise(tag_ids, noise, rng, num_tags)
        tokens[idx] = token_ids
        tags[idx] = tag_ids
    return tokens, tags


def build_ner_dataset(
    name: str,
    config: NERConfig,
    seed: int,
) -> Dict[str, Dataset]:
    """Materialise NER datasets from a HuggingFace corpus."""

    corpus = load_dataset(config.dataset_kind, cache_dir=str(HF_CACHE))
    tag_feature = corpus["train"].features["ner_tags"].feature
    tag_names = list(tag_feature.names)
    num_tags = len(tag_names)
    vocab = _build_vocab(corpus, lowercase=config.lowercase)
    rng = np.random.default_rng(seed)
    focus_ids = _resolve_focus_ids(config.focus_entities, tag_names)
    train_examples = _collect_examples(corpus["train"], focus_ids)
    val_examples = _collect_examples(corpus["validation"], focus_ids)
    test_examples = _collect_examples(corpus["test"], focus_ids)
    train_tokens, train_tags = _encode_split(
        train_examples,
        config.train_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=config.noise,
        entity_injection_prob=config.entity_injection_prob,
    )
    val_tokens, val_tags = _encode_split(
        val_examples,
        config.val_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=0.0,
        entity_injection_prob=0.0,
    )
    probe_tokens, probe_tags = _encode_split(
        test_examples,
        config.probe_size,
        rng,
        vocab=vocab,
        lowercase=config.lowercase,
        max_length=config.max_sequence_length,
        num_tags=num_tags,
        noise=0.0,
        entity_injection_prob=0.0,
    )
    metadata = {
        "num_tags": num_tags,
        "vocab_size": len(vocab),
        "sequence_length": config.max_sequence_length,
        "input_shape": (config.max_sequence_length,),
        "input_type": "text",
        "dataset_kind": config.dataset_kind,
        "focus_entities": list(config.focus_entities) if config.focus_entities else None,
        "name": name,
    }
    return {
        "train": TokenSequenceDataset(train_tokens, train_tags),
        "val": TokenSequenceDataset(val_tokens, val_tags),
        "probe": TokenSequenceDataset(probe_tokens, probe_tags),
        "metadata": metadata,
    }
```


---

## `maestro/datasets/registry.py` <a id="file-maestrodatasetsregistrypy"></a>

- Size: 6KB

```python
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
```


---

## `maestro/students/__init__.py` <a id="file-maestrostudents__init__py"></a>

- Size: 1KB

```python
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
```


---

## `maestro/students/base_api.py` <a id="file-maestrostudentsbase_apipy"></a>

- Size: 764B

```python
"""Student model API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

import torch
from torch.utils.data import DataLoader

from maestro.utils import OptimizerSettings


class AbstractStudent(Protocol):
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def configure_optimizer(self, settings: OptimizerSettings) -> None: ...

    def step_on_minibatch(self, batch) -> Dict[str, float]: ...

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]: ...

    def feature_embed(self, batch) -> torch.Tensor: ...

    @property
    def device(self) -> torch.device: ...


@dataclass
class StudentState:
    model: AbstractStudent
    optimizer: torch.optim.Optimizer
```


---

## `maestro/students/classification.py` <a id="file-maestrostudentsclassificationpy"></a>

- Size: 3KB

```python
"""Image-based classification student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from maestro.utils import OptimizerSettings, flatten_gradients


@dataclass(eq=False)
class ClassificationStudent(nn.Module):
    in_channels: int
    num_classes: int

    def __post_init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, self.num_classes)
        self._optimizer: torch.optim.Optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3, momentum=0.9
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def configure_optimizer(self, settings: OptimizerSettings) -> None:
        self._optimizer = torch.optim.SGD(
            self.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(inputs)
        logits = self.head(feats.view(feats.size(0), -1))
        return logits

    def step_on_minibatch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        self.train()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self._optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        grad_vec = flatten_gradients(list(self.parameters())).detach().cpu()
        self._optimizer.step()
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        return {
            "loss": loss.item(),
            "accuracy": acc,
            "grad_norm": float(grad_vec.norm().item()),
            "grad_vector": grad_vec,
        }

    def eval_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.forward(inputs)
                loss = self.loss_fn(logits, targets)
                total_loss += loss.item() * inputs.size(0)
                total_correct += (logits.argmax(dim=-1) == targets).sum().item()
                total += inputs.size(0)
        if total == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {"loss": total_loss / total, "accuracy": total_correct / total}

    def feature_embed(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, _ = batch
        inputs = inputs.to(self.device)
        with torch.no_grad():
            feats = self.encoder(inputs)
        return feats.view(feats.size(0), -1).detach()
```


---

## `maestro/students/detection.py` <a id="file-maestrostudentsdetectionpy"></a>

- Size: 4KB

```python
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
```


---

## `maestro/students/features.py` <a id="file-maestrostudentsfeaturespy"></a>

- Size: 512B

```python
"""Feature extraction helpers for students."""

from __future__ import annotations

import torch


def flatten_batch(batch) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        return torch.cat([flatten_batch(item) for item in batch], dim=-1)
    if isinstance(batch, dict):
        return torch.cat([flatten_batch(v) for v in batch.values()], dim=-1)
    if isinstance(batch, torch.Tensor):
        return batch.view(batch.size(0), -1)
    raise TypeError(f"Unsupported batch type: {type(batch)}")
```


---

## `maestro/students/ner.py` <a id="file-maestrostudentsnerpy"></a>

- Size: 4KB

```python
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


@dataclass(eq=False)
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
        self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )

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

    def step_on_minibatch(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
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
                per_token_loss = self.loss_fn(
                    logits.view(-1, self.num_tags), tags.view(-1)
                )
                total_loss += per_token_loss.sum().item()
                total_tokens += tags.numel()
                total_correct += (logits.argmax(dim=-1) == tags).sum().item()
        if total_tokens == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": total_loss / total_tokens,
            "accuracy": total_correct / total_tokens,
        }

    def feature_embed(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        tokens, _ = batch
        tokens = tokens.to(self.device)
        with torch.no_grad():
            emb = self.embedding(tokens)
            encoded, _ = self.encoder(emb)
        return encoded.mean(dim=1)
```


---

## `maestro/utils/__init__.py` <a id="file-maestroutils__init__py"></a>

- Size: 1007B

```python
"""Utility exports for convenience."""

from .flops import estimate_flops
from .grads import (
    ExponentialMovingAverage,
    GradientProjector,
    RobustScalarNormalizer,
    flatten_gradients,
    flatten_parameters,
    gradient_cosine,
    l2_norm,
    parameter_change,
)
from .logging import MetricsLogger, RunPaths
from .schedules import OptimizerSettings, clamp_learning_rate
from .seeding import seed_everything
from .serialization import load_checkpoint, save_checkpoint

from .wandb import init_wandb_run, log_checkpoint, log_metrics

__all__ = [
    "ExponentialMovingAverage",
    "GradientProjector",
    "RobustScalarNormalizer",
    "flatten_gradients",
    "flatten_parameters",
    "gradient_cosine",
    "l2_norm",
    "parameter_change",
    "estimate_flops",
    "MetricsLogger",
    "RunPaths",
    "OptimizerSettings",
    "clamp_learning_rate",
    "seed_everything",
    "load_checkpoint",
    "save_checkpoint",
    "init_wandb_run",
    "log_checkpoint",
    "log_metrics",
]
```


---

## `maestro/utils/logging.py` <a id="file-maestroutilsloggingpy"></a>

- Size: 1KB

```python
"""Logging helpers for MAESTRO runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MetricsLogger:
    output_dir: Path
    csv_filename: str = "metrics.csv"
    json_filename: str = "metrics.json"
    csv_fieldnames: Optional[list[str]] = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / self.csv_filename
        self.json_path = self.output_dir / self.json_filename
        self.rows: list[Dict[str, float]] = []

    def log_row(self, row: Dict[str, float]) -> None:
        if self.csv_fieldnames is None:
            self.csv_fieldnames = list(row.keys())
        self.rows.append(row)
        with self.csv_path.open("a", newline="") as handle:
            writer: csv.DictWriter[str] = csv.DictWriter(
                handle, fieldnames=self.csv_fieldnames
            )
            if self.csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    def flush_json(self) -> None:
        with self.json_path.open("w") as handle:
            json.dump(self.rows, handle, indent=2)


@dataclass
class RunPaths:
    base: Path
    run_id: str

    def resolve(self) -> Path:
        path = self.base / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
```


---

## `maestro/utils/seeding.py` <a id="file-maestroutilsseedingpy"></a>

- Size: 308B

```python
"""Global seeding utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```


---

## `requirements.txt` <a id="file-requirementstxt"></a>

- Size: 2KB

```
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
beautifulsoup4==4.14.2
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.3.0
cloudpickle==3.1.1
contourpy==1.3.3
cycler==0.12.1
datasets==3.1.0
Farama-Notifications==0.0.4
filelock==3.20.0
fonttools==4.60.1
fsspec==2025.10.0
gdown==5.2.0
gitdb==4.0.12
GitPython==3.1.45
gymnasium==1.2.1
hydra-core==1.3.2
idna==3.11
Jinja2==3.1.6
joblib==1.5.2
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.7
mpmath==1.3.0
networkx==3.5
numpy==2.2.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
omegaconf==2.3.0
opencv-python==4.12.0.88
packaging==25.0
pandas==2.3.3
pillow==12.0.0
platformdirs==4.5.0
polars==1.35.1
polars-runtime-32==1.35.1
protobuf==6.33.0
psutil==7.1.2
ptflops==0.7.5
pydantic==2.12.3
pydantic_core==2.41.4
pycocotools==2.0.8
pyparsing==3.2.5
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
requests==2.32.5
scikit-learn==1.7.2
scipy==1.16.3
sentry-sdk==2.43.0
setuptools==80.9.0
six==1.17.0
smmap==5.0.2
soupsieve==2.8
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
triton==3.5.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
ultralytics==8.3.223
ultralytics-thop==2.0.18
urllib3==2.5.0
wandb==0.22.3
```


---

## `scripts/download_datasets.sh` <a id="file-scriptsdownload_datasetssh"></a>

- Size: 41KB

```bash
#!/usr/bin/env bash
# Stable, strict, **idempotent** dataset fetcher + YOLO converter
# - Artistic datasets via gdown (Clipart/Watercolor/Comic) with auto-normalize
# - VOC via torchvision downloader
# - Skips work when artifacts already exist (safe to re-run)
#
# Usage:
#   bash scripts/download_datasets.sh [DATA_DIR]
#
# Env:
#   PYTHON_BIN=python3 VAL_FRACTION=0.2 DEBUG=1
#   # Optional direct URLs (tried before gdown IDs):
#   CLIPART1K_URL=..., WATERCOLOR2K_URL=..., COMIC2K_URL=...
#   # Optional custom Google Drive IDs (defaults set below):
#   CLIPART1K_GDRIVE_ID=..., WATERCOLOR2K_GDRIVE_ID=..., COMIC2K_GDRIVE_ID=...

set -Eeuo pipefail
shopt -s inherit_errexit

# ----------------- Global diagnostics -----------------
trap 'echo "[FATAL] ${BASH_SOURCE[0]}:$LINENO => $BASH_COMMAND" >&2' ERR
trap 'echo "[FATAL] interrupted (SIGINT/SIGTERM)" >&2; exit 2' INT TERM
: "${DEBUG:=0}"
if [[ "${DEBUG}" == "1" ]]; then
  export BASH_XTRACEFD=2
  export PS4='+ [${EPOCHREALTIME}] ${BASH_SOURCE##*/}:${LINENO}: '
  set -x
fi

# ----------------- Config -----------------
DATA_DIR_INPUT="${1:-$PWD/data}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"   # autosplit ratio for train-only sets
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Default Google Drive IDs
: "${CLIPART1K_GDRIVE_ID:=1LvxwCOfUa-OklIvBJhB8zJlochjJiPFS}"
: "${WATERCOLOR2K_GDRIVE_ID:=1fa2L6oaPSjZ1_WqlTmIp6i2RbdR2y1Pw}"
: "${COMIC2K_GDRIVE_ID:=1bZtVWcxxFrijE_ALvNPjH1MXIKio6BIr}"

# ----------------- Utils -----------------
die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
log() { echo -e "$*"; }

# Merge directory A/* into B/ (rsync if present, else cp -a)
merge_dir() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "$src/" "$dst/" || cp -a "$src/." "$dst/"
    else
      mkdir -p "$dst"
      cp -a "$src/." "$dst/"
    fi
  fi
}

# GNU/BSD size helper
file_size() {
  local f="$1"
  if command -v stat >/dev/null 2>&1; then
    stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || wc -c <"$f"
  else
    wc -c <"$f"
  fi
}

is_valid_zip() { unzip -tq "$1" >/dev/null 2>&1; }
is_valid_tar() { tar -tf "$1" >/dev/null 2>&1; }

have_voc_triplet() {
  # Return 0 if CWD has JPEGImages/Annotations/ImageSets
  [[ -d JPEGImages && -d Annotations && -d ImageSets ]]
}

# ----------------- Resolve working dir (fail-loud) -----------------
DATA_DIR="$(
  cd "$(dirname "$DATA_DIR_INPUT")" \
  && mkdir -p "$(basename "$DATA_DIR_INPUT")" \
  && cd "$(basename "$DATA_DIR_INPUT")" \
  && pwd -P
)" || die "Failed to resolve DATA_DIR from: $DATA_DIR_INPUT"
[[ -w "$DATA_DIR" ]] || die "DATA_DIR not writable: $DATA_DIR"

cd "$DATA_DIR"
ROOT_DIR="$(pwd -P)"
log "[*] Using DATA_DIR: $ROOT_DIR"

# ----------------- Preconditions -----------------
need_cmd "$PYTHON_BIN"
need_cmd unzip
need_cmd tar

# ----------------- Downloader -----------------
fetch_file() {
  # fetch_file <outfile> <url1> [url2] ...
  local outfile="$1"; shift
  local urls=("$@")
  [[ "${#urls[@]}" -gt 0 ]] || die "fetch_file: no URLs provided for $outfile"

  if [[ -s "$outfile" ]]; then
    case "$outfile" in
      *.zip) if is_valid_zip "$outfile"; then log "[ok] Already valid: $outfile"; return 0; fi ;;
      *.tar|*.tar.gz|*.tgz) if is_valid_tar "$outfile"; then log "[ok] Already valid: $outfile"; return 0; fi ;;
      *) log "[ok] Already present: $outfile"; return 0 ;;
    esac
    log "[!] Existing $outfile invalid, refetching…"
    rm -f "$outfile"
  fi

  rm -f "$outfile".part 2>/dev/null || true
  for u in "${urls[@]}"; do
    [[ -z "$u" ]] && continue
    log "[*] Fetching $outfile from: $u"
    if command -v wget >/dev/null 2>&1; then
      if ! wget -c --tries=3 --timeout=60 -O "$outfile".part "$u"; then
        log "[!] wget failed: $u"; continue
      fi
    elif command -v curl >/dev/null 2>&1; then
      if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 600 -o "$outfile".part "$u"; then
        log "[!] curl failed: $u"; continue
      fi
    else
      die "Neither wget nor curl found for download."
    fi

    local sz; sz=$(file_size "$outfile".part 2>/dev/null || echo 0)
    if [[ "$sz" -le 1024 ]]; then
      log "[!] Download too small ($sz bytes), trying next mirror"
      rm -f "$outfile".part
      continue
    fi

    local ok=false
    case "$outfile" in
      *.zip) is_valid_zip "$outfile".part && ok=true ;;
      *.tar|*.tar.gz|*.tgz) is_valid_tar "$outfile".part && ok=true ;;
      *) ok=true ;;
    esac

    if [[ "$ok" == "true" ]]; then
      mv "$outfile".part "$outfile"
      log "[ok] Valid: $outfile"
      return 0
    fi

    log "[!] Archive invalid; trying next mirror"
    rm -f "$outfile".part
  done

  die "All mirrors failed for $outfile"
}

# gdown helper (Drive ID)
gdown_fetch () {
  # gdown_fetch <outfile> <gdrive_file_id>
  local out="$1"; local fid="${2:-}"
  [[ -n "$fid" ]] || return 1
  if "$PYTHON_BIN" - "$fid" "$out".part <<'PY'
import sys
fid, out = sys.argv[1], sys.argv[2]
try:
    import gdown
except Exception:
    sys.exit(2)
url = f"https://drive.google.com/uc?id={fid}"
ok = gdown.download(url, out, quiet=False)
sys.exit(0 if ok else 1)
PY
  then
    mv "$out".part "$out"
    return 0
  else
    rm -f "$out".part >/dev/null 2>&1 || true
    return 1
  fi
}

extract_here() {
  local archive="$1"
  [[ -f "$archive" ]] || die "Archive not found: $archive"
  case "$archive" in
    *.zip) unzip -n "$archive" >/dev/null ;;
    *.tar) tar -xf "$archive" ;;
    *.tar.gz|*.tgz) tar -xzf "$archive" ;;
    *) die "Unknown archive format: $archive" ;;
  esac
}

# Find nested VOC root (dir containing JPEGImages+Annotations+ImageSets) and normalize into CWD
normalize_voc_here() {
  local found=""
  while IFS= read -r -d '' d; do
    if [[ -d "$d/JPEGImages" && -d "$d/Annotations" && -d "$d/ImageSets" ]]; then
      found="$d"; break
    fi
  done < <(find . -type d -print0)

  if [[ -z "$found" && -d ./clipart && -d ./clipart/JPEGImages && -d ./clipart/Annotations && -d ./clipart/ImageSets ]]; then
    found="./clipart"
  fi

  local base="$(basename "$(pwd -P)")"
  if [[ -z "$found" && -d "./$base" && -d "./$base/JPEGImages" && -d "./$base/Annotations" && -d "./$base/ImageSets" ]]; then
    found="./$base"
  fi

  if [[ -n "$found" && "$found" != "." ]]; then
    log "[*] Normalizing VOC layout from: $found -> $(pwd -P)"
    for sub in JPEGImages Annotations ImageSets; do
      if [[ -d "$sub" ]]; then
        merge_dir "$found/$sub" "$sub"
      else
        mv "$found/$sub" .
      fi
    done
  fi
}

# ----------------- Python deps (loud on failure) -----------------
log "[*] Checking Python deps (pillow, pyyaml, numpy, gdown)"
if ! "$PYTHON_BIN" - <<'PY'
import sys
try:
    import PIL, yaml, numpy  # noqa
    from PIL import Image    # noqa
    import gdown             # noqa
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then
  log "[*] Installing pillow pyyaml numpy gdown …"
  "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir pillow pyyaml numpy gdown
fi

log "[*] Verifying torchvision + datasets availability"
if ! "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torchvision  # noqa
    import datasets     # noqa
except Exception:
    sys.exit(1)
sys.exit(0)
PY
then
  log "[*] Installing HuggingFace datasets …"
  "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir datasets
fi

log "[*] Preparing torchvision classification datasets (MNIST/Fashion/CIFAR-10)"
env DATA_ROOT_FOR_DL="$ROOT_DIR" PYTHONPATH="" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from torchvision import datasets

root = Path(os.environ["DATA_ROOT_FOR_DL"])
pairs = [
    ("mnist", datasets.MNIST),
    ("fashion_mnist", datasets.FashionMNIST),
    ("cifar10", datasets.CIFAR10),
]
for name, cls in pairs:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    for split in (True, False):
        cls(root=str(target), train=split, download=True)
print("[ok] Torchvision classification datasets cached.")
PY

log "[*] Caching CoNLL-2003 via HuggingFace datasets (NER corpus)"
HF_CACHE_DIR="$ROOT_DIR/hf_cache"
mkdir -p "$HF_CACHE_DIR"
env HF_CACHE_DIR="$HF_CACHE_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

cache = Path(os.environ["HF_CACHE_DIR"])
cache.mkdir(parents=True, exist_ok=True)
load_dataset("conll2003", cache_dir=str(cache))
print(f"[ok] conll2003 cached under {cache}")
PY

log "[i] Tier-2 hooks: run this script again after adding CIFAR-100/Tiny ImageNet configs."

# ----------------- Small sets (artistic via gdown, with auto-normalize) -----------------
log "[*] Preparing artistic datasets via Google Drive (Clipart1k, Watercolor2k, Comic2k)"

# Helper to fetch (URL override first, then gdown ID), then unzip & auto-normalize VOC layout
fetch_unzip_voc() {
  # fetch_unzip_voc <folder_name> <outfile.zip> <URL_env> <GDRIVE_ID>
  local folder="$1"; local zipname="$2"; local url_env="$3"; local gid="$4"
  mkdir -p "$folder"
  pushd "$folder" >/dev/null

  if have_voc_triplet; then
    log "[ok] $folder already prepared, skipping download/extract"
    popd >/dev/null; return 0
  fi

  if [[ -n "$url_env" ]]; then
    fetch_file "$zipname" "$url_env"
  else
    gdown_fetch "$zipname" "$gid" || die "Failed to download $folder via gdown"
  fi
  extract_here "$zipname" || true
  rm -f "$zipname"

  normalize_voc_here
  have_voc_triplet || die "$folder structure missing {JPEGImages,Annotations,ImageSets}"
  popd >/dev/null
  log "[ok] $folder ready"
}

# Clipart1k, Watercolor2k, Comic2k
fetch_unzip_voc "clipart1k"    "clipart.zip"    "${CLIPART1K_URL:-}"    "$CLIPART1K_GDRIVE_ID"
fetch_unzip_voc "watercolor2k" "watercolor.zip" "${WATERCOLOR2K_URL:-}" "$WATERCOLOR2K_GDRIVE_ID"
fetch_unzip_voc "comic2k"      "comic.zip"      "${COMIC2K_URL:-}"      "$COMIC2K_GDRIVE_ID"

# ----------------- Penn-Fudan (mask dataset) -----------------
mkdir -p pennfudan
pushd pennfudan >/dev/null
if [[ -d PennFudanPed/PNGImages && -d PennFudanPed/PedMasks ]]; then
  log "[ok] PennFudan already extracted, skipping"
else
  fetch_file PennFudanPed.zip "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
  extract_here PennFudanPed.zip
  [[ -d PennFudanPed/PNGImages && -d PennFudanPed/PedMasks ]] || die "PennFudan structure unexpected"
fi
popd >/dev/null

# ----------------- KITTI 2D: images + labels -----------------
mkdir -p kitti_raw
pushd kitti_raw >/dev/null
if [[ -d training/image_2 && -d training/label_2 ]]; then
  log "[ok] KITTI raw already extracted, skipping"
else
  fetch_file data_object_image_2.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
  fetch_file data_object_label_2.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
  extract_here data_object_image_2.zip
  extract_here data_object_label_2.zip
fi
popd >/dev/null

mkdir -p kitti
if [[ -d kitti_raw/training/image_2 && -d kitti_raw/training/label_2 ]]; then
  ln -sfn "$ROOT_DIR/kitti_raw/training/image_2" "$ROOT_DIR/kitti/image_2"
  ln -sfn "$ROOT_DIR/kitti_raw/training/label_2" "$ROOT_DIR/kitti/label_2"
else
  die "KITTI training folders not found after extraction"
fi

# ----------------- COCO 2017 -----------------
log "[*] Preparing COCO 2017"
mkdir -p coco
pushd coco >/dev/null
if [[ -d images/train2017 && -d images/val2017 && -f annotations/instances_train2017.json ]]; then
  log "[ok] COCO2017 already present, skipping download"
else
  mkdir -p images annotations
  fetch_file train2017.zip "http://images.cocodataset.org/zips/train2017.zip"
  fetch_file val2017.zip   "http://images.cocodataset.org/zips/val2017.zip"
  fetch_file annotations_trainval2017.zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  extract_here train2017.zip
  extract_here val2017.zip
  extract_here annotations_trainval2017.zip
  mv -n train2017 images/train2017 || true
  mv -n val2017   images/val2017   || true
fi
popd >/dev/null

# ----------------- LVIS v1 (annotations only) -----------------
log "[*] Preparing LVIS v1 (annotations only)"
mkdir -p lvis
pushd lvis >/dev/null
if [[ -f lvis_v1_train.json && -f lvis_v1_val.json ]]; then
  log "[ok] LVIS jsons already present, skipping"
else
  fetch_file lvis_v1_train.json.zip "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
  fetch_file lvis_v1_val.json.zip   "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
  extract_here lvis_v1_train.json.zip
  extract_here lvis_v1_val.json.zip
  [[ -f lvis_v1_train.json && -f lvis_v1_val.json ]] || die "LVIS jsons missing after unzip"
fi
popd >/dev/null



# ---- COCO compatibility links for summary (generic train/val) ----
mkdir -p "$ROOT_DIR/coco/images" "$ROOT_DIR/coco/labels"
ln -sfn "$ROOT_DIR/coco/images/train2017" "$ROOT_DIR/coco/images/train"
ln -sfn "$ROOT_DIR/coco/images/val2017"   "$ROOT_DIR/coco/images/val"
ln -sfn "$ROOT_DIR/coco/labels/train2017" "$ROOT_DIR/coco/labels/train" || true
ln -sfn "$ROOT_DIR/coco/labels/val2017"   "$ROOT_DIR/coco/labels/val"   || true

# ---- LVIS labels: create direct train/val dirs (no symlinks for consistency) ----
mkdir -p "$ROOT_DIR/lvis/labels/train" "$ROOT_DIR/lvis/labels/val" "$ROOT_DIR/lvis/images"

# ---- Ensure LVIS image dirs are *real directories* (not symlinks), then materialize subsets ----
for d in "$ROOT_DIR/lvis/images/train" "$ROOT_DIR/lvis/images/val"; do
  if [[ -L "$d" ]]; then rm -f "$d"; fi
  mkdir -p "$d"
done

log "[*] Linking LVIS image subsets from COCO"
"$PYTHON_BIN" - <<'PY'
import json, os, sys
from pathlib import Path

ROOT = Path(".").resolve()
coco_train = ROOT/"coco/images/train2017"
coco_val   = ROOT/"coco/images/val2017"
lvis_dir   = ROOT/"lvis"
out_train  = lvis_dir/"images/train"
out_val    = lvis_dir/"images/val"

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def link_subset(lvis_json: Path, out_dir: Path):
    ensure(out_dir)
    if not lvis_json.exists():
        print(f"[skip] {lvis_json} missing")
        return 0
    
    print(f"[*] Processing {lvis_json.name}...")
    data = json.loads(lvis_json.read_text())
    imgs = data.get("images", [])
    ok = 0
    failed = 0
    
    for im in imgs:
        # LVIS uses coco_url instead of file_name
        fn = im.get("file_name")
        if not fn:
            coco_url = im.get("coco_url", "")
            if coco_url:
                fn = Path(coco_url).name
            else:
                # Fallback: use id
                img_id = im.get("id")
                if img_id:
                    fn = f"{int(img_id):012d}.jpg"
                else:
                    failed += 1
                    continue
        
        dst = out_dir / fn
        if dst.exists():
            ok += 1
            continue
        
        # Find source in COCO
        src = None
        for base in (coco_train, coco_val):
            cand = base / fn
            if cand.exists():
                src = cand
                break
        
        if src is None:
            # Try basename fallback
            basefn = Path(fn).name
            for base in (coco_train, coco_val):
                cand = base / basefn
                if cand.exists():
                    src = cand
                    break
        
        if src is None:
            failed += 1
            continue
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src.resolve(), dst)
            ok += 1
        except Exception:
            import shutil
            try:
                shutil.copy2(src, dst)
                ok += 1
            except Exception:
                failed += 1
    
    print(f"[ok] LVIS subset -> {out_dir.name}: {ok} images linked")
    if failed > 0:
        print(f"[!] Failed to link {failed} images")
    return ok

n_tr = link_subset(lvis_dir/"lvis_v1_train.json", out_train)
n_va = link_subset(lvis_dir/"lvis_v1_val.json",   out_val)

if n_tr == 0 and n_va == 0:
    print("[FATAL] No LVIS images were linked!", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$ROOT_DIR/lvis/labels"

# link or copy labels/train2017 -> labels/train
if [[ -d "$ROOT_DIR/lvis/labels/train2017" ]]; then
  ln -sfn "$ROOT_DIR/lvis/labels/train2017" "$ROOT_DIR/lvis/labels/train" 2>/dev/null \
    || { rm -rf "$ROOT_DIR/lvis/labels/train"; cp -a "$ROOT_DIR/lvis/labels/train2017" "$ROOT_DIR/lvis/labels/train"; }
fi

# link or copy labels/val2017 -> labels/val
if [[ -d "$ROOT_DIR/lvis/labels/val2017" ]]; then
  ln -sfn "$ROOT_DIR/lvis/labels/val2017" "$ROOT_DIR/lvis/labels/val" 2>/dev/null \
    || { rm -rf "$ROOT_DIR/lvis/labels/val"; cp -a "$ROOT_DIR/lvis/labels/val2017" "$ROOT_DIR/lvis/labels/val"; }
fi


# ----------------- PASCAL VOC 2007+2012 via Hugging Face (HF-only, robust normalize) -----------------
log "[*] Preparing PASCAL VOC 2007+2012"

mkdir -p voc
pushd voc >/dev/null

if [[ -d VOCdevkit/VOC2007 && -d VOCdevkit/VOC2012 ]]; then
  log "[ok] VOCdevkit already present, skipping download"
else
  : "${PYTHON_BIN:=${PYTHON_BIN:-python3}}"
  : "${VOC_HF_REPO:=HuggingFaceM4/pascal_voc}"   # dataset repo with the original tarballs
  : "${VOC_HF_REVISION:=main}"
  : "${VOC_2007_ARCHIVE:=voc2007.tar.gz}"        # 2007 trainval + test
  : "${VOC_2012_ARCHIVE:=voc2012.tar.gz}"        # 2012 trainval

  need_cmd tar

  hf_download() {
    # hf_download <repo_id> <filename> <revision> <outpath>
    local repo="$1" fn="$2" rev="$3" out="$4"
    "$PYTHON_BIN" - "$repo" "$fn" "$rev" "$out" <<'PY'
import sys, os, shutil
repo, fn, rev, out = sys.argv[1:5]
try:
    from huggingface_hub import hf_hub_download
except Exception:
    sys.exit(2)  # library missing
try:
    path = hf_hub_download(repo_id=repo, filename=fn, revision=rev, repo_type="dataset")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.abspath(path) != os.path.abspath(out):
        shutil.copy2(path, out)
    print("[ok] hf_hub:", fn)
    sys.exit(0)
except Exception as e:
    print("[!] hf_hub failed:", e)
    sys.exit(1)
PY
    return $?
  }

  # Ensure huggingface_hub is installed
  if ! "$PYTHON_BIN" - <<'PY'
try:
    import huggingface_hub  # noqa
except Exception:
    raise SystemExit(1)
PY
  then
    log "[*] Installing huggingface_hub …"
    "$PYTHON_BIN" -m pip install --upgrade --no-cache-dir huggingface_hub || die "Failed to install huggingface_hub"
  fi

  # Download archives if missing/invalid
  for fn in "$VOC_2007_ARCHIVE" "$VOC_2012_ARCHIVE"; do
    [[ -z "$fn" ]] && continue
    if [[ -s "$fn" ]] && tar -tf "$fn" >/dev/null 2>&1; then
      log "[ok] Already valid: $fn"
      continue
    fi
    rm -f "$fn"
    log "[*] Fetching from HF (dataset): $VOC_HF_REPO@$VOC_HF_REVISION :: $fn"
    hf_download "$VOC_HF_REPO" "$fn" "$VOC_HF_REVISION" "$fn" || die "HF download failed for $fn (repo: $VOC_HF_REPO)"
    tar -tf "$fn" >/dev/null 2>&1 || die "Corrupt archive after download: $fn"
  done

  # Extract (don't assume exact paths inside the tars)
  log "[*] Extracting VOC tarballs (HF)"
  case "$VOC_2007_ARCHIVE" in *.tar.gz|*.tgz) tar -xzf "$VOC_2007_ARCHIVE";; *.tar) tar -xf "$VOC_2007_ARCHIVE";; *) tar -xf "$VOC_2007_ARCHIVE";; esac
  case "$VOC_2012_ARCHIVE" in *.tar.gz|*.tgz) tar -xzf "$VOC_2012_ARCHIVE";; *.tar) tar -xf "$VOC_2012_ARCHIVE";; *) tar -xf "$VOC_2012_ARCHIVE";; esac

  # --- Normalize to VOCdevkit/{VOC2007,VOC2012} regardless of how the archives unpacked ---
  normalize_voc_tree() {
    mkdir -p VOCdevkit
    # 1) If a nested VOCdevkit already exists somewhere, merge it in
    while IFS= read -r -d '' nested_voc; do
      for yr in VOC2007 VOC2012; do
        if [[ -d "$nested_voc/$yr" ]]; then
          log "[*] Merging nested $nested_voc/$yr -> VOCdevkit/$yr"
          mkdir -p "VOCdevkit/$yr"
          merge_dir "$nested_voc/$yr/JPEGImages" "VOCdevkit/$yr/JPEGImages"
          merge_dir "$nested_voc/$yr/Annotations" "VOCdevkit/$yr/Annotations"
          merge_dir "$nested_voc/$yr/ImageSets"  "VOCdevkit/$yr/ImageSets"
        fi
      done
    done < <(find . -type d -name VOCdevkit -print0)

    # 2) Any directory that *looks* like a VOC root (has the triplet) — classify by year and merge
    while IFS= read -r -d '' d; do
      # Skip our final destination
      [[ "$d" == "./VOCdevkit/VOC2007" || "$d" == "./VOCdevkit/VOC2012" ]] && continue
      if [[ -d "$d/JPEGImages" && -d "$d/Annotations" && -d "$d/ImageSets" ]]; then
        local year=""
        case "$d" in
          *2007*|*VOC2007*) year=2007 ;;
          *2012*|*VOC2012*) year=2012 ;;
          *) # Heuristic: if it has test split -> likely 2007; else assume 2012
             if [[ -f "$d/ImageSets/Main/test.txt" ]]; then year=2007; else year=2012; fi ;;
        esac
        log "[*] Normalizing $(realpath --relative-to=. "$d") → VOCdevkit/VOC${year}"
        mkdir -p "VOCdevkit/VOC${year}"
        merge_dir "$d/JPEGImages" "VOCdevkit/VOC${year}/JPEGImages"
        merge_dir "$d/Annotations" "VOCdevkit/VOC${year}/Annotations"
        merge_dir "$d/ImageSets"  "VOCdevkit/VOC${year}/ImageSets"
      fi
    done < <(find . -maxdepth 5 -type d -print0)
  }

  normalize_voc_tree

  # Sanity check
  [[ -d VOCdevkit/VOC2007 && -d VOCdevkit/VOC2012 ]] || die "VOCdevkit not properly prepared (expected VOC2007 & VOC2012)"
fi

popd >/dev/null

# ----------------- YOLO conversions (idempotent) -----------------
log "[*] Converting all datasets to YOLO (skip-aware)"

VAL_FRACTION="$VAL_FRACTION" "$PYTHON_BIN" - <<'PY'
import json, yaml, os, shutil, random, sys
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from PIL import Image
    import numpy as np
except Exception as e:
    print("[FATAL] Missing python deps:", e, file=sys.stderr); sys.exit(1)

ROOT = Path(".").resolve()
try:
    VAL_FRACTION = float(os.environ.get("VAL_FRACTION","0.2"))
    if not (0.0 < VAL_FRACTION < 1.0): raise ValueError
except Exception:
    print("[FATAL] VAL_FRACTION must be in (0,1)", file=sys.stderr); sys.exit(1)

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)
def nonempty(p: Path): return p.exists() and any(p.iterdir())

def write_yaml_abs(yaml_path: Path, ds_root: Path, train_images: Path, val_images: Path, names):
    ds_root = ds_root.resolve()
    y = {"path": str(ds_root), "train": str(train_images.resolve()), "val": str(val_images.resolve()), "names": list(names or [])}
    ensure(yaml_path.parent)
    if yaml_path.exists():
        print(f"[ok] {yaml_path} exists, leaving as-is")
    else:
        yaml_path.write_text(yaml.dump(y, sort_keys=False))
        print(f"[ok] wrote {yaml_path}")

def coco_to_yolo(json_path: Path, images_dir: Path, labels_dir: Path, names_out: Path=None, skip_crowd=True):
    if not json_path.exists():
        print(f"[skip] {json_path} missing"); return None
    ensure(labels_dir)
    # If labels already look populated, skip heavy convert
    if any(labels_dir.glob("*.txt")):
        print(f"[ok] labels exist for {json_path.name}, skipping conversion")
        # Still return names for YAML
        try:
            data = json.loads(json_path.read_text())
            categories = sorted(data.get("categories", []), key=lambda c: (c.get("id",0), c.get("name","")))
            names = [c["name"] for c in categories]
            if names_out is not None and not names_out.exists():
                ensure(names_out.parent); names_out.write_text("\n".join(names))
            return names
        except Exception:
            return None

    data = json.loads(json_path.read_text())
    img_by_id = {}
    for im in data.get("images", []):
        fn = im.get("file_name")
        if not fn:
            cu = im.get("coco_url","")
            fn = Path(cu).name if cu else f"{int(im['id']):012d}.jpg"
        img_by_id[im["id"]] = {"file_name": fn, "width": im.get("width"), "height": im.get("height")}
    categories = sorted(data.get("categories", []), key=lambda c: (c.get("id",0), c.get("name","")))
    catid_to_index = {c["id"]: i for i,c in enumerate(categories)}
    names = [c["name"] for c in categories]
    if names_out is not None and not names_out.exists():
        ensure(names_out.parent); names_out.write_text("\n".join(names))
    anns_per_img = {}
    for ann in data.get("annotations", []):
        if skip_crowd and ann.get("iscrowd",0)==1: continue
        anns_per_img.setdefault(ann["image_id"], []).append(ann)

    num_boxes = 0
    for img_id, meta in img_by_id.items():
        fn = meta["file_name"]
        w, h = meta.get("width"), meta.get("height")
        if not w or not h:
            try:
                with Image.open(images_dir/fn) as im: w,h = im.size
            except Exception:
                continue
        out_txt = labels_dir/Path(fn).with_suffix(".txt").name
        if out_txt.exists():  # don't rewrite
            continue
        lines=[]
        for ann in anns_per_img.get(img_id, []):
            cat = catid_to_index.get(ann["category_id"])
            if cat is None: continue
            x,y,bw,bh = ann["bbox"]
            cx = (x + bw/2)/w; cy = (y + bh/2)/h
            nw = bw/w; nh = bh/h
            def clip(v): return max(0.0, min(1.0, float(v)))
            cx,cy,nw,nh = map(clip,(cx,cy,nw,nh))
            lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            num_boxes += 1
        out_txt.write_text("\n".join(lines))
    print(f"[ok] {json_path.name} → {labels_dir} ({num_boxes} boxes)")
    return names

def write_yaml_simple(path_file: Path, ds_root: Path, names: list, img_sub="images", lbl_sub="labels"):
    write_yaml_abs(path_file, ds_root, ds_root/img_sub/"train", ds_root/img_sub/"val", names)

def should_skip_yaml_and_labels(yaml_path: Path, labels_glob: Path):
    return yaml_path.exists() and any(labels_glob.glob("*.txt"))

# ---------------- COCO 2017 ----------------
coco_dir = ROOT/"coco"
coco_img_train = coco_dir/"images/train2017"
coco_img_val   = coco_dir/"images/val2017"
coco_ann_train = coco_dir/"annotations/instances_train2017.json"
coco_ann_val   = coco_dir/"annotations/instances_val2017.json"
coco_yaml = coco_dir/"yolo_coco.yaml"
if coco_img_train.exists() and coco_ann_train.exists():
    if should_skip_yaml_and_labels(coco_yaml, coco_dir/"labels/train2017"):
        print("[ok] COCO YOLO already prepared, skipping")
    else:
        coco_lbl_root = coco_dir/"labels"
        names = coco_to_yolo(coco_ann_train, coco_img_train, coco_lbl_root/"train2017", names_out=coco_dir/"names_coco.txt")
        if coco_ann_val.exists() and coco_img_val.exists():
            _ = coco_to_yolo(coco_ann_val, coco_img_val, coco_lbl_root/"val2017")
        if names:
            write_yaml_abs(coco_yaml, coco_dir, coco_img_train, coco_img_val if coco_img_val.exists() else coco_img_train, names)

# ---------------- LVIS v1 ----------------
lvis_dir = ROOT/"lvis"
lvis_train = lvis_dir/"lvis_v1_train.json"
lvis_val   = lvis_dir/"lvis_v1_val.json"
lvis_yaml = lvis_dir/"yolo_lvis.yaml"
lvis_img_train = lvis_dir/"images/train"
lvis_img_val = lvis_dir/"images/val"

if lvis_train.exists():
    if should_skip_yaml_and_labels(lvis_yaml, lvis_dir/"labels/train2017"):
        print("[ok] LVIS YOLO already prepared, skipping")
    else:
        lvis_lbl_root = lvis_dir/"labels"
        names = coco_to_yolo(lvis_train, lvis_img_train, lvis_lbl_root/"train2017", names_out=lvis_dir/"names_lvis.txt")
        if lvis_val.exists():
            _ = coco_to_yolo(lvis_val, lvis_img_val, lvis_lbl_root/"val2017")
        if names:
            write_yaml_abs(lvis_yaml, lvis_dir, lvis_img_train, lvis_img_val if lvis_img_val.exists() else lvis_img_train, names)

# ---------------- VOC 2007+2012 → YOLO ----------------
voc_root = ROOT/"voc"
voc_yaml = voc_root/"yolo_voc.yaml"
if (voc_root/"VOCdevkit").exists():
    if should_skip_yaml_and_labels(voc_yaml, voc_root/"labels/train"):
        print("[ok] VOC YOLO already prepared, skipping")
    else:
        NAMES_VOC = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        name_to_id = {n:i for i,n in enumerate(NAMES_VOC)}

        def voc_to_yolo(year_dir: Path, split: str, out_images: Path, out_labels: Path):
            jpeg = year_dir/"JPEGImages"; ann = year_dir/"Annotations"
            setfile = year_dir/"ImageSets/Main"/f"{split}.txt"
            if not setfile.exists(): print(f"[skip] {setfile} not found"); return 0,0
            ids = [l.strip() for l in setfile.read_text().splitlines() if l.strip()]
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)
            imgs=boxes=0
            for img_id in ids:
                # image
                img_path = (jpeg/f"{img_id}.jpg")
                if not img_path.exists():
                    alt = jpeg/f"{img_id}.png"
                    if alt.exists(): img_path = alt
                    else: continue
                dst_img = out_images/img_path.name
                if not dst_img.exists():
                    try: os.symlink(img_path.resolve(), dst_img)
                    except Exception:
                        shutil.copy2(img_path, dst_img)
                # label
                out_txt = out_labels/img_path.with_suffix(".txt").name
                if out_txt.exists():  # don't rewrite
                    imgs+=1; continue
                xml_path = ann/f"{img_id}.xml"
                lines=[]
                if xml_path.exists():
                    tree = ET.parse(xml_path)
                    W=int(tree.findtext("size/width", "0")); H=int(tree.findtext("size/height","0"))
                    for obj in tree.findall("object"):
                        cls = obj.findtext("name","")
                        if cls not in name_to_id or obj.findtext("difficult")=="1": continue
                        bb = obj.find("bndbox")
                        xmin=float(bb.findtext("xmin","0")); ymin=float(bb.findtext("ymin","0"))
                        xmax=float(bb.findtext("xmax","0")); ymax=float(bb.findtext("ymax","0"))
                        if W<=0 or H<=0 or xmax<=xmin or ymax<=ymin: continue
                        cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                        bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                        lines.append(f"{name_to_id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        boxes+=1
                out_txt.write_text("\n".join(lines))
                imgs+=1
            return imgs, boxes

        img_train = voc_root/"images/train"; img_val = voc_root/"images/val"
        lbl_train = voc_root/"labels/train"; lbl_val = voc_root/"labels/val"
        img_train.mkdir(parents=True, exist_ok=True)
        img_val.mkdir(parents=True, exist_ok=True)
        lbl_train.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)

        splits = [("VOC2007","trainval"),("VOC2007","test"),("VOC2012","trainval")]
        ti=tb=0
        for year,split in splits:
            ydir = voc_root/"VOCdevkit"/year
            if not ydir.exists(): print(f"[skip] {ydir} missing"); continue
            if split in ("train","trainval"):
                i,b = voc_to_yolo(ydir, split, img_train, lbl_train)
            else:
                i,b = voc_to_yolo(ydir, split, img_val, lbl_val)
            ti+=i; tb+=b

        write_yaml_abs(voc_yaml, voc_root, img_train, img_val, NAMES_VOC)
        print(f"[ok] VOC → YOLO | images:{ti}, boxes:{tb}")

# -------------- Small VOC-like datasets → YOLO --------------
def convert_voc_like(src_root: Path, out_root: Path, train_split: str, val_split: str, names: list):
    jpeg = src_root/"JPEGImages"; ann = src_root/"Annotations"; sets = src_root/"ImageSets"/"Main"
    name_to_id = {n:i for i,n in enumerate(names)}

    def read_ids(split):
        f = sets/f"{split}.txt"
        if f.exists(): return [l.strip() for l in f.read_text().splitlines() if l.strip()]
        return [p.stem for p in jpeg.glob("*.jpg")]

    def do_split(ids, split):
        img_out = out_root/"images"/split; lbl_out = out_root/"labels"/split
        img_out.mkdir(parents=True, exist_ok=True); lbl_out.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            # image
            img = None
            for ext in (".jpg",".png",".jpeg"):
                cand = jpeg/f"{sid}{ext}"
                if cand.exists(): img = cand; break
            if img is None: continue
            dst = img_out/img.name
            if not dst.exists():
                try: dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            # label
            out_txt = lbl_out/f"{sid}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            xml = ann/f"{sid}.xml"
            lines=[]
            if xml.exists():
                root = ET.parse(xml).getroot()
                sz = root.find("size")
                try:
                    W=float(sz.findtext("width","0")); H=float(sz.findtext("height","0"))
                except Exception: W=H=0
                for obj in root.findall("object"):
                    nm = obj.findtext("name","").strip().lower()
                    if nm not in name_to_id: continue
                    bb = obj.find("bndbox")
                    if bb is None or W<=0 or H<=0: continue
                    try:
                        xmin=float(bb.findtext("xmin","0")); ymin=float(bb.findtext("ymin","0"))
                        xmax=float(bb.findtext("xmax","0")); ymax=float(bb.findtext("ymax","0"))
                    except Exception: continue
                    if xmax<=xmin or ymax<=ymin: continue
                    cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                    bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                    lines.append(f"{name_to_id[nm]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))

    train_ids = read_ids(train_split); val_ids = read_ids(val_split)
    do_split(train_ids, "train"); do_split(val_ids, "val")

# Clipart1k
clip_src = ROOT/"clipart1k"
if clip_src.exists() and (clip_src/"JPEGImages").exists():
    clip_out = ROOT/"clipart1k_yolo"
    clip_yaml = ROOT/"configs/datasets/clipart1k.yaml"
    if clip_yaml.exists() and nonempty(clip_out/"labels/train"):
        print("[ok] clipart1k YOLO already prepared, skipping")
    else:
        names20 = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
        convert_voc_like(clip_src, clip_out, "train", "val", names20)
        write_yaml_simple(clip_yaml, clip_out, names20)
else:
    print("[FATAL] clipart1k not present after download")

# Watercolor2k & Comic2k
for name in ("watercolor2k","comic2k"):
    src = ROOT/name
    if src.exists() and (src/"JPEGImages").exists():
        out = ROOT/f"{name}_yolo"
        yml = ROOT/f"configs/datasets/{name}.yaml"
        if yml.exists() and nonempty(out/"labels/train"):
            print(f"[ok] {name} YOLO already prepared, skipping")
        else:
            names6 = ["bicycle","bird","car","cat","dog","person"]
            train_split="train"; val_split=("test" if (src/"ImageSets/Main/test.txt").exists() else "val")
            convert_voc_like(src, out, train_split, val_split, names6)
            write_yaml_simple(yml, out, names6)
    else:
        print(f"[FATAL] {name} not present after download")

# PennFudan (mask → boxes)
src = ROOT/"pennfudan"/"PennFudanPed"
if src.exists():
    out = ROOT/"pennfudan_yolo"
    yml = ROOT/"configs/datasets/pennfudan.yaml"
    if yml.exists() and nonempty(out/"labels/train"):
        print("[ok] PennFudan YOLO already prepared, skipping")
    else:
        ensure(out/"images/train"); ensure(out/"labels/train")
        img_dir = src/"PNGImages"; mask_dir = src/"PedMasks"
        for img in sorted(img_dir.glob("*.png")):
            msk = mask_dir/f"{img.stem}_mask.png"
            if not msk.exists(): continue
            dst = out/"images/train"/img.name
            if not dst.exists():
                try: dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            arr = __import__("numpy").array(Image.open(msk))
            H,W = arr.shape[:2]; lines=[]
            out_txt = out/"labels/train"/f"{img.stem}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            for pid in [p for p in __import__("numpy").unique(arr) if p!=0]:
                ys,xs = (arr==pid).nonzero()
                if xs.size==0 or ys.size==0: continue
                xmin,xmax = xs.min(), xs.max(); ymin,ymax = ys.min(), ys.max()
                if xmax<=xmin or ymax<=ymin: continue
                cx=((xmin+xmax)/2)/W; cy=((ymin+ymax)/2)/H
                bw=(xmax-xmin)/W; bh=(ymax-ymin)/H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))
        write_yaml_abs(yml, out, out/"images/train", out/"images/val", ["person"])
else:
    print("[FATAL] PennFudan dataset not found after download")

# KITTI 2D → YOLO
kitti = ROOT/"kitti"
if (kitti/"image_2").exists() and (kitti/"label_2").exists():
    out = ROOT/"kitti_yolo"
    yml = ROOT/"configs/datasets/kitti.yaml"
    if yml.exists() and nonempty(out/"labels/train"):
        print("[ok] KITTI YOLO already prepared, skipping")
    else:
        ensure(out/"images/train"); ensure(out/"labels/train")
        for img in sorted((kitti/"image_2").glob("*.png")):
            dst = out/"images/train"/img.name
            if not dst.exists():
                try:
                    dst.symlink_to(img.resolve())
                except Exception:
                    shutil.copy2(img, dst)
            try:
                W,H = Image.open(img).size
            except Exception:
                W=H=0
            lblp = kitti/"label_2"/f"{img.stem}.txt"
            out_txt = out/"labels/train"/f"{img.stem}.txt"
            if out_txt.exists():  # don't rewrite
                continue
            lines=[]
            if lblp.exists() and W>0 and H>0:
                for raw in lblp.read_text().splitlines():
                    parts = raw.split()
                    if len(parts)<8: continue
                    cls = parts[0].lower()
                    if cls not in {"car","pedestrian","cyclist"}: continue
                    left,top,right,bottom = map(float, parts[4:8])
                    cx=(left+right)/(2*W); cy=(top+bottom)/(2*H)
                    bw=max(0.0,right-left)/W; bh=max(0.0,bottom-top)/H
                    cid={"car":0,"pedestrian":1,"cyclist":2}[cls]
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            out_txt.write_text("\n".join(lines))
        write_yaml_abs(yml, out, out/"images/train", out/"images/val", ["car","pedestrian","cyclist"])
else:
    print("[FATAL] KITTI expected folders not found after download")

# -------------------- Auto-split when missing val --------------------
def autosplit_if_needed(ds_root: Path):
    img_tr = ds_root/"images/train"; lbl_tr = ds_root/"labels/train"
    img_va = ds_root/"images/val";   lbl_va = ds_root/"labels/val"
    if not img_tr.exists(): return
    ensure(img_va); ensure(lbl_va)
    has_val_imgs = any(img_va.glob("*"))
    if has_val_imgs: return
    imgs = sorted(list(img_tr.glob("*.jpg")) + list(img_tr.glob("*.png")) + list(img_tr.glob("*.jpeg")))
    if not imgs: return
    k = max(1, int(round(len(imgs)*VAL_FRACTION)))
    random.seed(0)
    sample = set(random.sample(imgs, k))
    for p in imgs:
        stem = p.stem
        if p in sample:
            (img_va/p.name).write_bytes(p.read_bytes()); p.unlink(missing_ok=True)
            src_lbl = lbl_tr/f"{stem}.txt"
            dst_lbl = lbl_va/f"{stem}.txt"
            if src_lbl.exists():
                dst_lbl.write_text(src_lbl.read_text()); src_lbl.unlink(missing_ok=True)
            else:
                dst_lbl.write_text("")
    print(f"[ok] autosplit {ds_root.name}: moved {k}/{len(imgs)} images to val")

for ds in ["clipart1k_yolo","watercolor2k_yolo","comic2k_yolo","pennfudan_yolo","kitti_yolo","voc"]:
    root = ROOT/ds
    if root.exists():
        autosplit_if_needed(root)

# ---------------- Final summary ----------------
def count_split(ds_root: Path):
    def cnt_recursive(p: Path):
        return sum(1 for q in p.rglob("*") if q.is_file()) if p.exists() else 0
    img_tr = ds_root/"images"/"train"
    img_va = ds_root/"images"/"val"
    lbl_tr = ds_root/"labels"/"train"
    lbl_va = ds_root/"labels"/"val"
    it = cnt_recursive(img_tr); il = cnt_recursive(lbl_tr)
    vt = cnt_recursive(img_va); vl = cnt_recursive(lbl_va)
    return it, il, vt, vl

print("\n[summary] YOLO datasets (images/labels train|val):")
cands = [ROOT/"coco", ROOT/"lvis", ROOT/"voc", ROOT/"clipart1k_yolo", ROOT/"watercolor2k_yolo", ROOT/"comic2k_yolo", ROOT/"pennfudan_yolo", ROOT/"kitti_yolo"]
ok_any=False
for ds in cands:
    if not ds.exists(): continue
    it,il,vt,vl = count_split(ds)
    print(f"  - {ds.name:<18} train({it:5d}/{il:5d}) | val({vt:5d}/{vl:5d})")
    ok_any=True

if not ok_any:
    print("[FATAL] No YOLO datasets produced"); sys.exit(1)

print("[✓] All conversions complete.")
PY

log "[✓] Dataset preparation finished at $DATA_DIR"
```


---

## `scripts/export_learning_curves.py` <a id="file-scriptsexport_learning_curvespy"></a>

- Size: 3KB

```python
#!/usr/bin/env python
"""Export comparative learning curves into a consolidated CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load a metrics.csv file and ensure required columns are present."""
    frame = pd.read_csv(metrics_path)
    if "seed" not in frame.columns:
        frame["seed"] = metrics_path.parent.name
    frame = frame.sort_values(["seed", "episode"]) if "episode" in frame.columns else frame
    if "num_steps" not in frame.columns:
        raise ValueError(f"{metrics_path} is missing 'num_steps' column")
    frame["step"] = frame.groupby("seed")["num_steps"].cumsum()
    return frame


def _derive_task(metrics_path: Path, frame: pd.DataFrame) -> str:
    if "task" in frame.columns and not frame["task"].isna().all():
        return str(frame["task"].dropna().iloc[0])
    # Fallback: assume directory structure .../<task>/<method>/seed_x/metrics.csv
    if len(metrics_path.parents) >= 3:
        return metrics_path.parents[2].name
    return metrics_path.parent.name


def _derive_method(metrics_path: Path, frame: pd.DataFrame) -> str:
    if "method" in frame.columns and not frame["method"].isna().all():
        return str(frame["method"].dropna().iloc[0])
    # parent of seed_*/ directory
    if len(metrics_path.parents) >= 2:
        return metrics_path.parents[1].name
    return metrics_path.parent.name


def collect_learning_curves(root: Path) -> List[Dict[str, object]]:
    """Collect learning curve rows from all seed metrics under root."""
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(root.rglob("seed_*/metrics.csv")):
        try:
            frame = _load_metrics(metrics_path)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        task = _derive_task(metrics_path, frame)
        method = _derive_method(metrics_path, frame)
        metric_columns = []
        if "macro_accuracy" in frame.columns:
            metric_columns.append(("macro_accuracy", "macro_acc"))
        if "macro_mAP" in frame.columns:
            metric_columns.append(("macro_mAP", "macro_mAP"))
        if not metric_columns:
            continue
        for _, row in frame.iterrows():
            for column, metric_name in metric_columns:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "step": row["step"],
                        "metric": metric_name,
                        "value": row[column],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export comparative learning curves")
    parser.add_argument(
        "--comparative-root",
        default="outputs/comparative_plots",
        help="Root directory containing comparative plot runs",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory where learning_curves.csv will be written",
    )
    args = parser.parse_args()

    root = Path(args.comparative_root)
    if not root.exists():
        return 0

    rows = collect_learning_curves(root)
    if not rows:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "learning_curves.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```


---

## `scripts/generate_ood_grid.py` <a id="file-scriptsgenerate_ood_gridpy"></a>

- Size: 6KB

```python
#!/usr/bin/env python
"""Generate an out-of-distribution grid over noise and imbalance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.ood_grid import evaluate_ood_grid
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.wandb import init_wandb_run, log_metrics, log_checkpoint


def _build_env(
    config: dict,
    task_cfg_path: str,
    seed: int,
    *,
    overrides: dict | None = None,
) -> MaestroEnv:
    datasets = build_from_config(task_cfg_path, seed, overrides=overrides or {})
    env_cfg = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_cfg)


def _parse_list(arg: str) -> List[float]:
    return [float(item) for item in arg.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policy robustness over a grid of noise and imbalance values."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--noise", type=str, default="0.0,0.1,0.3")
    parser.add_argument("--imbalance", type=str, default="0.0,0.4,0.6")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-episodes", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ood_grid"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks: Iterable[str] = config.get("tasks", [])
    tasks = list(tasks)
    if not tasks:
        raise ValueError("No tasks found in config")
    task_cfg = tasks[min(args.task_index, len(tasks) - 1)]

    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )

    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(state.get("policy", state))
    elif args.train_episodes > 0:
        ppo = PPOTeacher(policy, PPOConfig(**config.get("ppo", {})))
        for episode in range(args.train_episodes):
            env = _build_env(config, task_cfg, seed + episode * 37)
            try:
                ppo.train_episode(env, config["horizon"])
            finally:
                env.close()

    noise_vals = _parse_list(args.noise)
    imbalance_vals = _parse_list(args.imbalance)

    run_name = f"ood_grid_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(
        run_name,
        config={
            "config": config,
            "task_cfg": task_cfg,
            "steps": args.steps,
            "noise_values": noise_vals,
            "imbalance_values": imbalance_vals,
        },
    )

    csv_path = args.output_dir / "ood_grid.csv"
    heatmap_path = args.output_dir / "ood_heatmap.png"
    rows = []
    try:
        for i, noise in enumerate(noise_vals):
            for j, imbalance in enumerate(imbalance_vals):
                env = _build_env(
                    config,
                    task_cfg,
                    seed + 1000 + i * 100 + j,
                    overrides={"noise": float(noise), "imbalance": float(imbalance)},
                )
                try:
                    stats = evaluate_ood_grid([env], policy, steps=args.steps)
                finally:
                    env.close()
                rows.append(
                    {
                        "noise": float(noise),
                        "imbalance": float(imbalance),
                        "mean_macro": stats["mean_macro"],
                        "std_macro": stats["std_macro"],
                    }
                )
                log_metrics(
                    {
                        "noise": float(noise),
                        "imbalance": float(imbalance),
                        "mean_macro": stats.get("mean_macro", 0.0),
                        "std_macro": stats.get("std_macro", 0.0),
                    }
                )

        df = pd.DataFrame(rows).sort_values(["noise", "imbalance"]).reset_index(drop=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

        pivot = df.pivot(index="noise", columns="imbalance", values="mean_macro")
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.xticks(
            ticks=range(len(pivot.columns)),
            labels=[f"{col:.2f}" for col in pivot.columns],
        )
        plt.yticks(
            ticks=range(len(pivot.index)),
            labels=[f"{row:.2f}" for row in pivot.index],
        )
        plt.xlabel("imbalance")
        plt.ylabel("noise")
        plt.title("OOD Macro Accuracy")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        log_checkpoint(csv_path, args.output_dir)
        log_checkpoint(heatmap_path, args.output_dir)
    finally:
        wandb_run.finish()

    print(json.dumps({"csv": str(csv_path), "heatmap": str(heatmap_path)}, indent=2))


if __name__ == "__main__":
    main()
```


---

## `scripts/generate_tables.py` <a id="file-scriptsgenerate_tablespy"></a>

- Size: 8KB

```python
#!/usr/bin/env python
"""Generate publication tables with statistical significance estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from maestro.stats import paired_bootstrap, permutation_test


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_ci(mean: float, ci: Tuple[float, float]) -> str:
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _format_with_significance(mean: float, ci: Tuple[float, float], p: float) -> str:
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{mean:.3f}{stars} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _summarise_metric(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    summary = df.groupby(list(group_cols))
    result = summary["value"].agg(["mean", "std", "count"]).reset_index()
    result.rename(columns={"mean": "mean_value", "std": "std_value"}, inplace=True)
    return result


def _bootstrap_deltas(df: pd.DataFrame, pivot_cols: Iterable[str]) -> pd.DataFrame:
    pivot = df.pivot_table(index="seed", columns=list(pivot_cols), values="value")
    if pivot.empty:
        return pd.DataFrame()
    best_method = pivot.mean(axis=0).idxmax()
    results = []
    for column in pivot.columns:
        aligned = pivot[[best_method, column]].dropna()
        if aligned.empty:
            continue
        diff, ci = paired_bootstrap(aligned[best_method].to_numpy(), aligned[column].to_numpy())
        p_value = permutation_test(aligned[best_method].to_numpy(), aligned[column].to_numpy())
        results.append({"comparison": column, "delta": diff, "ci_low": ci[0], "ci_high": ci[1], "p_value": p_value})
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["baseline"] = " vs ".join(str(x) for x in best_method)
    return result_df


def _write_outputs(df: pd.DataFrame, tex_path: Path, csv_path: Path, caption: str, label: str) -> None:
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, float_format="%.3f", caption=caption, label=label, escape=False)
    tex_path.write_text(latex, encoding="utf-8")


def generate_table(
    name: str,
    raw_path: Path,
    outdir: Path,
    group_cols: Iterable[str | None],
    caption: str,
    label: str,
    metric_filter: str | None = None,
) -> Dict[str, object]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw CSV for table '{name}': {raw_path}")
    df = pd.read_csv(raw_path)
    if metric_filter is not None:
        df = df[df["metric"] == metric_filter]
    resolved_groups: list[str] = []
    for idx, col in enumerate(group_cols):
        if col is None:
            continue
        if col in df.columns:
            resolved_groups.append(col)
        elif idx == 0:
            raise KeyError(f"Required column '{col}' missing from {raw_path}")
    if not resolved_groups:
        raise KeyError(f"No valid grouping columns found in {raw_path}")
    summary = _summarise_metric(df, resolved_groups)
    pivot_cols: Tuple[str, ...] = tuple(resolved_groups[:2])
    significance_lookup: Dict[Tuple[object, ...], float] = {}
    if "seed" in df.columns and pivot_cols:
        delta = _bootstrap_deltas(df, pivot_cols)
        if not delta.empty and "comparison" in delta.columns and "p_value" in delta.columns:
            for _, delta_row in delta.iterrows():
                comparison = delta_row["comparison"]
                if isinstance(comparison, tuple):
                    key = comparison
                elif isinstance(comparison, (list, np.ndarray, pd.Index)):
                    key = tuple(comparison)
                else:
                    key = (comparison,)
                significance_lookup[key] = float(delta_row["p_value"])
    else:
        delta = pd.DataFrame()
    def _ci_from_row(row: pd.Series) -> str:
        mean = float(row["mean_value"])
        std_val = float(row["std_value"]) if not pd.isna(row["std_value"]) else 0.0
        count = float(row["count"])
        stderr = std_val / np.sqrt(max(1.0, count))
        ci = (mean - 1.96 * stderr, mean + 1.96 * stderr)
        if significance_lookup:
            key = tuple(row[col] for col in pivot_cols)
            if key in significance_lookup:
                return _format_with_significance(mean, ci, significance_lookup[key])
        return _format_ci(mean, ci)

    summary["ci"] = summary.apply(_ci_from_row, axis=1)
    table_path = outdir / f"table_{name}.csv"
    tex_path = outdir / f"table_{name}.tex"
    display = summary.drop(columns=["std_value", "count"]).rename(columns={"mean_value": "mean"})
    _write_outputs(display, tex_path, table_path, caption, label)
    delta_path = outdir / f"table_{name}_significance.json"
    delta_path.write_text(delta.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "summary": str(table_path),
        "latex": str(tex_path),
        "significance": str(delta_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables for publication")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="Override raw data directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tables_dir = args.out / "tables"
    _ensure_directory(tables_dir)
    raw_dir = args.raw if args.raw else args.out / "raw_data"

    manifest: Dict[str, Dict[str, object]] = {}
    bpath = raw_dir / "baselines.csv"
    if bpath.exists():
        t4 = pd.read_csv(bpath)
        if {"method", "final_macro"}.issubset(t4.columns):
            t4_long = t4.rename(columns={"final_macro": "value"}).copy()
            t4_long["task"] = "aggregate"
            t4_long["metric"] = "macro_acc"
            bpath.write_text(t4_long.to_csv(index=False))

    ablation_path = raw_dir / "ablation.csv"
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path)
        if "component" not in abl.columns and "flags" in abl.columns:
            def _component_from_flags(s: str) -> str:
                stripped = str(s).strip()
                if stripped == "{}":
                    return "full"
                for key in [
                    "drop_grad_cosine",
                    "drop_progress_block",
                    "drop_model_block",
                    "drop_data_block",
                ]:
                    if key in stripped:
                        return key
                return stripped

            abl["component"] = abl["flags"].apply(_component_from_flags)
        if "value" not in abl.columns and "macro_accuracy" in abl.columns:
            abl = abl.rename(columns={"macro_accuracy": "value"})
        ablation_path.write_text(abl.to_csv(index=False))

    configs = [
        ("main", raw_dir / "main_results.csv", ("task",), "Main results (Macro-Acc)", "tab:main_results"),
        ("lofo", raw_dir / "lofo.csv", ("task",), "Cross-task transfer (LOFO)", "tab:lofo"),
        ("baselines", raw_dir / "baselines.csv", ("task", "method"), "Baseline comparisons", "tab:baselines"),
        ("ablations", raw_dir / "ablation.csv", ("component",), "Ablation study", "tab:ablations"),
    ]

    for name, path, group_cols, caption, label in configs:
        try:
            manifest[name] = generate_table(
                name,
                path,
                tables_dir,
                group_cols,
                caption,
                label,
            )
        except FileNotFoundError as exc:
            if not args.dry_run:
                raise
            print(f"[warn] {exc}")

    manifest_path = tables_dir / "tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

```


---

## `scripts/make_publication_figures.py` <a id="file-scriptsmake_publication_figurespy"></a>

- Size: 11KB

```python
#!/usr/bin/env python
"""Generate all publication figures from consolidated CSV artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAPTIONS = {
    "fig1": "Learning curves showing macro-accuracy and macro-mAP versus training steps across tasks.",
    "fig2": "One-step prediction R² demonstrates approximate Markovity of the learned latent features.",
    "fig3": "Macro metrics across train/test population sizes highlight N-invariance generalisation patterns.",
    "fig4": "Heatmap of macro metrics summarising robustness across synthetic noise and imbalance levels.",
    "fig5": "Ablation deltas quantify the contribution of each architectural component to macro performance.",
}


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    return pd.read_csv(path)


def _aggregate_curve(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["task", "method", "step", "metric"])
    stats = grouped["value"].agg(["mean", "std", "count"]).reset_index()
    stats.rename(columns={"mean": "mean_value", "std": "std_value"}, inplace=True)
    return stats


def fig1_learning_curves(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    aggregated = _aggregate_curve(df)
    for task in sorted(aggregated["task"].unique()):
        subset = aggregated[(aggregated["task"] == task) & (aggregated["metric"].isin(["macro_acc", "macro_mAP"]))]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, method_df in subset.groupby("method"):
            ax.plot(method_df["step"], method_df["mean_value"], label=method)
            std = method_df["std_value"].fillna(0.0)
            ax.fill_between(method_df["step"], method_df["mean_value"] - std, method_df["mean_value"] + std, alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Macro metric")
        ax.set_title(f"Learning curves – {task}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = outdir / f"fig1_learning_{task}.{ext}"
            fig.savefig(path, dpi=200)
            outputs.append(path)
        plt.close(fig)
    return outputs


def fig2_markov_diagnostics(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    sorted_df = df.sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sorted_df["feature"], sorted_df["r2"], color="#1f77b4")
    ax.set_ylabel("R²")
    ax.set_title("Markov diagnostics")
    ax.set_xticks(range(len(sorted_df)))
    ax.set_xticklabels(sorted_df["feature"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig2_markov_r2.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig3_n_invariance(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for train_n, group in df.groupby("train_N"):
        ax.plot(group["test_N"], group["macro_metric"], marker="o", label=f"train N={train_n}")
    ax.set_xlabel("Test N")
    ax.set_ylabel("Macro metric")
    ax.set_title("N-invariance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig3_n_invariance.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig4_ood_heatmap(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    pivot = df.pivot_table(index="noise", columns="imbalance", values="macro_metric", aggfunc=np.mean)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Imbalance")
    ax.set_ylabel("Noise")
    ax.set_title("OOD robustness")
    fig.colorbar(cax, ax=ax, label="Macro metric")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig4_ood_heatmap.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def fig5_ablation(df: pd.DataFrame, outdir: Path) -> Iterable[Path]:
    outputs = []
    ordered = df.sort_values("delta_macro")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ordered["component"], ordered["delta_macro"], color="#d62728")
    ax.set_xlabel("Δ macro metric vs full")
    ax.set_title("Ablation study")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = outdir / f"fig5_ablation.{ext}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _write_caption(outdir: Path, fig_key: str) -> Path:
    if fig_key not in CAPTIONS:
        raise KeyError(f"No caption defined for {fig_key}")
    caption_path = outdir / f"{fig_key}_caption.txt"
    caption_path.write_text(CAPTIONS[fig_key], encoding="utf-8")
    return caption_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication figures")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--raw", type=Path, help="Override raw data directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip missing figures for CI")
    args = parser.parse_args()

    outdir = args.out / "figures"
    _ensure_directory(outdir)
    raw_dir = args.raw if args.raw else args.out / "raw_data"

    generated = {}
    try:
        curves_df = _load_csv(raw_dir / "learning_curves.csv")
        generated["fig1"] = [str(p) for p in fig1_learning_curves(curves_df, outdir)]
        if generated["fig1"]:
            generated["fig1_caption"] = str(_write_caption(outdir, "fig1"))
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        markov_df = pd.read_json(raw_dir / "markov_diag.jsonl", lines=True)
        feat_cols = [
            "r2",
            "linear_r2",
            "delta_r2",
            "mlp_r2",
            "gru_history_r2",
            "linear_history_r2",
            "g_data_r2",
            "g_model_r2",
            "g_progress_r2",
        ]
        markov_long = markov_df.melt(
            id_vars=["task"],
            value_vars=[c for c in feat_cols if c in markov_df.columns],
            var_name="feature",
            value_name="_metric",
        ).rename(columns={"_metric": "r2"})
        generated["fig2"] = [str(p) for p in fig2_markov_diagnostics(markov_long, outdir)]
        if generated["fig2"]:
            generated["fig2_caption"] = str(_write_caption(outdir, "fig2"))
    except (FileNotFoundError, ValueError) as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        n_df = _load_csv(raw_dir / "n_invariance.csv")
        if {"train_num_datasets", "eval_num_datasets", "mean_macro"} <= set(n_df.columns):
            n_df = n_df.rename(
                columns={
                    "train_num_datasets": "train_N",
                    "eval_num_datasets": "test_N",
                    "mean_macro": "macro_metric",
                }
            )
        generated["fig3"] = [str(p) for p in fig3_n_invariance(n_df, outdir)]
        if generated["fig3"]:
            generated["fig3_caption"] = str(_write_caption(outdir, "fig3"))
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        ood_df = _load_csv(raw_dir / "ood_grid.csv")
        if "macro_metric" not in ood_df.columns and "mean_macro" in ood_df.columns:
            ood_df = ood_df.rename(columns={"mean_macro": "macro_metric"})
        generated["fig4"] = [str(p) for p in fig4_ood_heatmap(ood_df, outdir)]
        if generated["fig4"]:
            generated["fig4_caption"] = str(_write_caption(outdir, "fig4"))
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    try:
        ablation_df = _load_csv(raw_dir / "ablation.csv")
        macro_col = "macro_accuracy" if "macro_accuracy" in ablation_df.columns else "value"
        if macro_col not in ablation_df.columns:
            raise KeyError("Ablation CSV must contain 'macro_accuracy' or 'value' column")
        if "flags" in ablation_df.columns:
            base_candidates = ablation_df[
                ablation_df["flags"].astype(str).str.strip().isin(
                    [
                        "{}",
                        '{"drop_grad_cosine": false}',
                        '{"drop_progress_block": false}',
                        '{"drop_model_block": false}',
                        '{"drop_data_block": false}',
                    ]
                )
            ][macro_col]
        else:
            base_candidates = pd.Series(dtype=float)
        base_val = (
            float(base_candidates.iloc[0])
            if not base_candidates.empty
            else float(ablation_df[macro_col].max())
        )
        if "component" not in ablation_df.columns:
            if "flags" not in ablation_df.columns:
                raise KeyError("Ablation CSV requires 'component' or 'flags' column")
            ablation_df["component"] = ablation_df["flags"].apply(
                lambda s: "full"
                if str(s).strip() == "{}"
                else next(
                    (
                        k
                        for k in [
                            "drop_grad_cosine",
                            "drop_progress_block",
                            "drop_model_block",
                            "drop_data_block",
                        ]
                        if k in str(s)
                    ),
                    str(s),
                )
            )
        ablation_df["delta_macro"] = ablation_df[macro_col] - base_val
        generated["fig5"] = [
            str(p) for p in fig5_ablation(ablation_df[["component", "delta_macro"]], outdir)
        ]
        if generated["fig5"]:
            generated["fig5_caption"] = str(_write_caption(outdir, "fig5"))
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise
        print(f"[warn] {exc}")

    manifest_path = outdir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps(generated, indent=2))


if __name__ == "__main__":
    main()
```


---

## `scripts/run_all.sh` <a id="file-scriptsrun_allsh"></a>

- Size: 12KB

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Configurable resources
# ---------------------------
GPUS=(${GPUS_OVERRIDE:-0 2 4 5 6 7})   # override via env if needed
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
OUT_ROOT="outputs/publication_${DATE_TAG}"
RAW_DIR="$OUT_ROOT/raw_data"
FIG_DIR="$OUT_ROOT/figures"
TAB_DIR="$OUT_ROOT/tables"
LOG_DIR="$OUT_ROOT/logs"
CKPT_DIR="$OUT_ROOT/checkpoints"
mkdir -p "$RAW_DIR" "$FIG_DIR" "$TAB_DIR" "$LOG_DIR" "$CKPT_DIR"

# ---------------------------
# venv bootstrap
# ---------------------------
VENV_DIR="${VENV_DIR:-.venv}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[run_all] No active venv; creating ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
else
  echo "[run_all] Using venv: ${VIRTUAL_ENV}"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# ---------------------------
# Helpers
# ---------------------------
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift || true
fi

log(){ echo -e "[run_all] $*"; }

USE_TMUX_HELPER="${USE_TMUX_HELPER:-0}"
declare -a TMUX_SIGNALS=()
declare -a BACKGROUND_PIDS=()
if [[ "$USE_TMUX_HELPER" == "1" ]] && ! command -v tmux >/dev/null 2>&1; then
  log "USE_TMUX_HELPER=1 but tmux not found; falling back to background jobs"
  USE_TMUX_HELPER=0
fi

# launch <gpu_id> <cmd...>
launch() {
  local gpu="$1"; shift
  local tag="${1}"; shift
  local outfile="$LOG_DIR/${tag}.log"
  mkdir -p "$LOG_DIR"
  if $DRY_RUN; then
    echo "[DRY] CUDA_VISIBLE_DEVICES=${gpu} $*" | tee "$outfile"
    return 0
  fi
  if [[ "$USE_TMUX_HELPER" == "1" ]]; then
    local session="runall_${tag}"
    local signal="${session}_done"
    local cmd_str
    cmd_str=$(printf '%q ' "$@")
    cmd_str="${cmd_str% }"
    echo "[launch:tmux] gpu=${gpu} tag=${tag} session=${session} -> $outfile"
    tmux new-session -d -s "$session" "set -euo pipefail; export CUDA_VISIBLE_DEVICES='${gpu}'; { ${cmd_str} >>'${outfile}' 2>&1; } || true; tmux wait-for -S '${signal}'"
    TMUX_SIGNALS+=("$signal")
  else
    echo "[launch] gpu=${gpu} tag=${tag} -> $outfile"
    ( export CUDA_VISIBLE_DEVICES="${gpu}"; "$@" 2>&1 | tee "$outfile" ) &
    local pid=$!
    BACKGROUND_PIDS+=("$pid")
  fi
}

# round-robin GPU picker
_next_gpu_i=0
pick_gpu() {
  local idx=${_next_gpu_i}
  _next_gpu_i=$(( (_next_gpu_i + 1) % ${#GPUS[@]} ))
  echo "${GPUS[$idx]}"
}

# ensure single dataset download
DATA_SENTINEL="$OUT_ROOT/.datasets_ready"
if ! $DRY_RUN; then
if [[ ! -f "$DATA_SENTINEL" ]]; then
  log "Downloading/Preparing datasets (one-time)"
  bash scripts/download_datasets.sh ./data | tee "$LOG_DIR/datasets.log"
  touch "$DATA_SENTINEL"
else
  log "Datasets already prepared"
fi
else
  log "Dry run — skipping dataset download"
fi

# ---------------------------
# 0) Supervised baseline benchmarks
# ---------------------------
BASELINE_CONFIGS=(
  "classification:configs/tasks/classification.yaml"
  "ner:configs/tasks/ner.yaml"
  "detection:configs/tasks/detection.yaml"
)
BASELINE_ROOT="$OUT_ROOT/baselines"
mkdir -p "$BASELINE_ROOT"

# Teacher defaults (seed used for baselines too)
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/publication/main_suite.yaml}"
TEACHER_SEED="${TEACHER_SEED:-42}"        # change externally for more seeds
TEACHER_DETERMINISTIC="${TEACHER_DETERMINISTIC:-1}"

log "Launching supervised curriculum baselines across GPUs"
baseline_pid_start=${#BACKGROUND_PIDS[@]}
baseline_signal_start=${#TMUX_SIGNALS[@]}
for entry in "${BASELINE_CONFIGS[@]}"; do
  IFS=":" read -r baseline_name baseline_cfg <<<"$entry"
  gpu="$(pick_gpu)"
  tag="baseline_${baseline_name}"
  base_out="$BASELINE_ROOT/${baseline_name}"
  mkdir -p "$base_out"
  cmd=(python train_baselines.py \
    --tasks "$baseline_cfg" \
    --output-dir "$base_out" \
    --methods standard uniform easy_to_hard greedy linucb \
    --seed "$TEACHER_SEED" \
    --device cuda)
  if $DRY_RUN; then
    cmd+=(--dry-run)
  fi
  launch "$gpu" "$tag" "${cmd[@]}"
done

if [[ ${#BASELINE_CONFIGS[@]} -gt 0 ]]; then
  if [[ "$USE_TMUX_HELPER" == "1" ]]; then
    log "Waiting for supervised baselines to finish (tmux)..."
    for signal in "${TMUX_SIGNALS[@]:baseline_signal_start}"; do
      tmux wait-for "$signal"
    done
  else
    log "Waiting for supervised baselines to finish..."
    for pid in "${BACKGROUND_PIDS[@]:baseline_pid_start}"; do
      wait "$pid"
    done
  fi
  if ! $DRY_RUN; then
    BASELINE_ROOT_ENV="$BASELINE_ROOT" python - <<'PY'
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = Path(os.environ["BASELINE_ROOT_ENV"])
frames = []
for sub in base.glob("*"):
    summary = sub / "baseline_summary.csv"
    if summary.is_file():
        df = pd.read_csv(summary)
        df.insert(0, "task_group", sub.name)
        frames.append(df)
if not frames:
    raise SystemExit(0)
out = pd.concat(frames, ignore_index=True)
summary_all = base / "baseline_summary_all.csv"
out.to_csv(summary_all, index=False)
metric_priority = [
    "final_macro_f1",
    "final_macro_map",
    "final_accuracy",
    "final_map",
    "final_loss",
]
metric_column = next((col for col in metric_priority if col in out.columns), None)
if metric_column is None:
    metric_column = next((col for col in out.columns if col.startswith("final_")), None)
if metric_column:
    pivot = out.pivot_table(index="task", columns="method", values=metric_column, aggfunc="max")
    matrix = pivot.to_numpy(dtype=float)
    if pivot.size and np.isfinite(matrix).any():
        fig_dir = base / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="#f5f5f5")
        fig, ax = plt.subplots(figsize=(1.8 * max(1, len(pivot.columns)), 1.2 * max(1, len(pivot.index)) + 1.5))
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        metric_label = metric_column.replace("final_", "").replace("_", " ").title()
        cbar.set_label(metric_label)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(idx) for idx in pivot.index])
        ax.set_xlabel("Method")
        ax.set_ylabel("Task")
        ax.set_title(f"{metric_label} overview (all baselines)")
        max_val = float(np.nanmax(matrix))
        threshold = max_val * 0.5 if max_val != 0 else 0.0
        for i, task in enumerate(pivot.index):
            for j, method in enumerate(pivot.columns):
                value = pivot.loc[task, method]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < threshold else "black",
                    fontsize=9,
                )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"baseline_summary_overview.{ext}", dpi=200)
        plt.close(fig)
PY
  fi
fi

# ---------------------------
# 1) Train PPO Teacher (publication suite, CPU by default)
# ---------------------------
log "Training PPO teacher: ${TEACHER_CONFIG} (seed=${TEACHER_SEED})"
if $DRY_RUN; then
  echo "[DRY] python train_maestro_teacher.py --config ${TEACHER_CONFIG} --seed ${TEACHER_SEED} --output-dir outputs" | tee "$LOG_DIR/meta_train.log"
else
  python train_maestro_teacher.py \
    --config "${TEACHER_CONFIG}" \
    --seed "${TEACHER_SEED}" \
    --output-dir outputs | tee "$LOG_DIR/meta_train.log"
fi

TEACH_RUN_ID="$(python - <<'PY'
import yaml
conf=yaml.safe_load(open("configs/publication/main_suite.yaml"))
rid=conf.get("run",{}).get("id","publication_main")
print(rid)
PY
)"
TEACH_OUT_DIR="outputs/${TEACH_RUN_ID}"
TEACH_CKPT="${TEACH_OUT_DIR}/policy.pt"

if [[ ! -f "$TEACH_CKPT" ]]; then
  echo "[ERROR] Teacher checkpoint not found at $TEACH_CKPT" >&2
  exit 2
fi
log "Teacher checkpoint: $TEACH_CKPT"

# ---------------------------
# 2) Parallel phase (after teacher)
# ---------------------------
# We now kick off everything that does NOT depend on each other, in parallel.
# Rules:
#  - Each job gets an isolated GPU via CUDA_VISIBLE_DEVICES + '--device 0' for Ultralytics.
#  - Each job writes to a unique subdir/file (use tags).
#  - YOLO gets priority on more GPUs; diagnostics can take leftovers or CPU.

# ----- 2a) YOLO transfer track(s)
# One definitive YOLO “publication” run (adjust segments/budget/batch as needed)
YOLO_TAG="yolo_pub"
YOLO_GPU="$(pick_gpu)"
YOLO_ARGS=(python train_maestro_yolo.py
  --output-root outputs
  --date-tag "$DATE_TAG"
  --no-resume
  --segments 12
  --budget-images 200000
  --batch 16
  --imgsz 896
  --device 0
  --method maestro
  --teacher-ckpt "$TEACH_CKPT"
)
if [[ "$TEACHER_DETERMINISTIC" == "1" ]]; then YOLO_ARGS+=(--teacher-deterministic); else YOLO_ARGS+=(--no-teacher-deterministic); fi
launch "$YOLO_GPU" "$YOLO_TAG" "${YOLO_ARGS[@]}"

# (Optional) Extra YOLO seeds or mixes (uncomment to run more in parallel)
# for S in 43 44; do
#   gpu="$(pick_gpu)"
#   tag="yolo_pub_seed${S}"
#   launch "$gpu" "$tag" "${YOLO_ARGS[@]}"
# done

# ----- 2b) Diagnostics & evals (parallel)
# Markov diagnostics (publication config)
MD_GPU="$(pick_gpu)"
launch "$MD_GPU" "markov_diag" \
  python scripts/run_markov_diag.py --config "$TEACHER_CONFIG" --out "$OUT_ROOT"

# N-invariance (use small config to remain quick—or swap to publication if desired)
NI_GPU="$(pick_gpu)"
launch "$NI_GPU" "n_invariance" \
  python scripts/run_n_invariance.py --config configs/meta_train/small_cpu_debug.yaml --output-dir "$RAW_DIR"

# OOD grid (quick version; adjust flags for heavier sweep)
OOD_GPU="$(pick_gpu)"
launch "$OOD_GPU" "ood_grid" \
  python scripts/generate_ood_grid.py --config configs/meta_train/small_cpu_debug.yaml --output-dir "$RAW_DIR"

# LOFO & main evals (CSV) — these are light and can run on CPU; still put on a GPU slot for consistency
EVAL_GPU="$(pick_gpu)"
launch "$EVAL_GPU" "eval_lofo" \
  python scripts/run_eval.py --config configs/meta_train/lofo_classification.yaml --steps 10 --csv-out "$RAW_DIR/lofo.csv"

EVAL2_GPU="$(pick_gpu)"
launch "$EVAL2_GPU" "eval_main" \
  python scripts/run_eval.py --config "$TEACHER_CONFIG" --steps 10 --csv-out "$RAW_DIR/main_results.csv"

# ----- 2c) (Optional) Comparative baselines at scale (expensive). Uncomment to run.
# for M in ppo uniform easy_to_hard greedy bandit_linucb bandit_thompson pbt bohb; do
#   gpu="$(pick_gpu)"
#   tag="baseline_${M}"
#   launch "$gpu" "$tag" \
#     python scripts/run_comparative.py --config "$TEACHER_CONFIG" --method "$M" --output-dir "$RAW_DIR/baseline_${M}"
# done
# After baselines finish you can consolidate curves:
# python scripts/export_learning_curves.py --out "$RAW_DIR" --comparative-root "$RAW_DIR" || true

# ---------------------------
# 3) Wait for all background jobs
# ---------------------------
if [[ "$USE_TMUX_HELPER" == "1" ]]; then
  log "Waiting for tmux-managed jobs to complete…"
  for signal in "${TMUX_SIGNALS[@]}"; do
    tmux wait-for "$signal"
  done
  log "All tmux-managed jobs signaled completion."
else
  log "Waiting for parallel jobs to complete…"
  wait
  log "All parallel jobs complete."
fi

# ---------------------------
# 4) Figures & tables (post-processing)
# ---------------------------
FIG_CMD=(python scripts/make_publication_figures.py --out "$OUT_ROOT")
TAB_CMD=(python scripts/generate_tables.py        --out "$OUT_ROOT")
if $DRY_RUN; then
  FIG_CMD+=(--dry-run); TAB_CMD+=(--dry-run)
fi
"${FIG_CMD[@]}" | tee "$LOG_DIR/figures.log" || true
"${TAB_CMD[@]}" | tee "$LOG_DIR/tables.log"  || true

log "Publication pipeline complete."
```


---

## `scripts/run_comparative.py` <a id="file-scriptsrun_comparativepy"></a>

- Size: 9KB

```python
"""Run comparative experiments across teacher baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from maestro.baselines import BaselineScheduler, create_scheduler
from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils import RunPaths
from maestro.utils.config import load_config
from maestro.utils.logging import MetricsLogger
from maestro.utils.wandb import init_wandb_run, log_checkpoint, log_metrics


METHOD_CHOICES = [
    "ppo",
    "uniform",
    "easy_to_hard",
    "greedy",
    "bandit_linucb",
    "bandit_thompson",
    "pbt",
    "bohb",
]


def build_env_for_task(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
    datasets = build_from_config(task_cfg, seed)
    env_config = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_config)


def get_output_directory(method: str, args: argparse.Namespace) -> Path:
    base_dir = Path(args.output_dir) if args.output_dir else Path("outputs/comparative")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_paths = RunPaths(base_dir, f"{method}_{timestamp}")
    return run_paths.resolve()


def _baseline_episode(
    env: MaestroEnv,
    scheduler: BaselineScheduler,
    horizon: int,
) -> Dict[str, float]:
    observation, _ = env.reset()
    descriptors = env.last_per_dataset_descriptors
    scheduler.start_episode(
        observation,
        descriptors,
        dataset_metrics={name: {"accuracy": 0.0} for name in scheduler.dataset_names},
    )
    total_reward = 0.0
    macro_accuracy = 0.0
    usages: List[float] = []
    etas: List[float] = []
    usage_fraction: List[float] = []

    for _ in range(horizon):
        action, _, _, action_info = scheduler.act(observation, descriptors)
        next_obs, reward, terminated, truncated, info = env.step(action)
        combined_info = dict(info)
        combined_info.update(action_info)
        scheduler.update(reward, combined_info)
        observation = next_obs
        descriptors = env.last_per_dataset_descriptors
        total_reward += float(reward)
        macro_accuracy = float(info.get("macro_accuracy", macro_accuracy))
        usages.append(float(info.get("usage", 0.0)))
        etas.append(float(action["eta"][0]))
        usage_fraction.append(float(action["u"][0]))
        if terminated or truncated:
            break
    env.close()

    avg_usage = float(np.mean(usages)) if usages else 0.0
    avg_eta = float(np.mean(etas)) if etas else 0.0
    avg_u = float(np.mean(usage_fraction)) if usage_fraction else 0.0
    return {
        "return": total_reward,
        "macro_accuracy": macro_accuracy,
        "avg_usage": avg_usage,
        "avg_eta": avg_eta,
        "avg_u": avg_u,
        "num_steps": float(len(usages)),
    }


def run_baseline(
    method: str,
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Iterable[int],
) -> None:
    tasks: Iterable[str] = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in config")
    baseline_cfg = config.get("baselines", {})
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger(seed_dir)
        torch.manual_seed(seed)
        total_episodes = config.get("run", {}).get("total_episodes", 1)
        for episode in tqdm(range(total_episodes), desc=f"seed={seed}"):
            task_cfg = tasks[episode % len(tasks)]
            env_seed = seed + episode * 31
            env = build_env_for_task(config, task_cfg, env_seed)
            scheduler = create_scheduler(
                method,
                [spec.name for spec in env.config.datasets],
                (config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
                config["horizon"],
                usage=config.get("baseline_usage", 0.1),
                method_kwargs=baseline_cfg.get(method, {}),
            )
            stats = _baseline_episode(env, scheduler, config["horizon"])
            stats.update(
                {
                    "episode": episode,
                    "task": Path(task_cfg).stem,
                    "seed": seed,
                    "method": method,
                }
            )
            logger.log_row(stats)
        logger.flush_json()


def run_ppo(
    config: Dict[str, Any],
    output_dir: Path,
    seeds: Iterable[int],
    deterministic_eval: bool,
) -> None:
    tasks: Iterable[str] = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in config")
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger = MetricsLogger(seed_dir)
        torch.manual_seed(seed)
        policy = TeacherPolicy(
            descriptor_dim=8,
            g_model_dim=6,
            g_progress_dim=11,
            eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
        )
        ppo_cfg = PPOConfig(**config.get("ppo", {}))
        ppo = PPOTeacher(policy, ppo_cfg)
        total_episodes = config.get("run", {}).get("total_episodes", 1)
        checkpoint_interval = config.get("run", {}).get("checkpoint_interval", 50)
        checkpoint_path = seed_dir / "policy.pt"
        best_return: Optional[float] = None
        run_name = f"ppo_comparative_seed{seed}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        wandb_run = init_wandb_run(
            run_name,
            config={"config": config, "seed": seed, "method": "ppo"},
        )

        try:
            for episode in tqdm(range(total_episodes), desc=f"seed={seed}"):
                task_cfg = tasks[episode % len(tasks)]
                env_seed = seed + episode * 31
                env = build_env_for_task(config, task_cfg, env_seed)
                stats = ppo.train_episode(env, config["horizon"])
                env.close()
                stats.update(
                    {
                        "episode": episode,
                        "task": Path(task_cfg).stem,
                        "seed": seed,
                        "method": "ppo",
                    }
                )
                logger.log_row(stats)
                log_metrics(stats)
                current_return = stats["return"]
                is_best = best_return is None or current_return > best_return
                if is_best:
                    best_return = current_return
                should_checkpoint = (episode + 1) % checkpoint_interval == 0 or is_best
                if should_checkpoint:
                    torch.save(
                        {
                            "policy": policy.state_dict(),
                            "optim": ppo.optim.state_dict(),
                            "config": config,
                            "episode": episode + 1,
                            "best_return": best_return,
                            "lambda_cmdp": ppo.lambda_cmdp,
                        },
                        checkpoint_path,
                    )
                    log_checkpoint(checkpoint_path, seed_dir)
        finally:
            logger.flush_json()
            wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparative baselines")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="For PPO, use deterministic policy during eval checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds = args.seeds or [config.get("seed", 0)]
    output_dir = get_output_directory(args.method, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": str(args.config),
        "method": args.method,
        "seeds": list(seeds),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.method == "ppo":
        run_ppo(config, output_dir, seeds, args.deterministic_eval)
    else:
        run_baseline(args.method, config, output_dir, seeds)


if __name__ == "__main__":
    main()
```


---

## `scripts/run_eval.py` <a id="file-scriptsrun_evalpy"></a>

- Size: 4KB

```python
"""Run evaluation for a trained teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.policy.ppo import TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.serialization import load_checkpoint


def build_env_from_task(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
    datasets = build_from_config(task_cfg, seed)
    env_config = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained teacher")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write CSV summary (task,metric,value)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    if args.checkpoint is not None:
        state = load_checkpoint(args.checkpoint)
        policy.load_state_dict(state.get("policy", state))

    task_list: Iterable[str] = config.get("eval_tasks", config["tasks"])
    base_seed = config.get("seed", 0)
    results = {}
    for index, task_cfg in enumerate(task_list):
        env_seed = base_seed + index * 17
        env = build_env_from_task(config, task_cfg, env_seed)
        obs, _ = env.reset()
        descriptors = env.last_per_dataset_descriptors
        total_reward = 0.0
        info = {}
        for _ in range(args.steps):
            action, _, _, _ = policy.act(obs, descriptors)
            obs, reward, done, _, info = env.step(action)
            descriptors = env.last_per_dataset_descriptors
            total_reward += reward
            if done:
                break
        results[Path(task_cfg).stem] = {
            "macro_accuracy": info.get("macro_accuracy", 0.0),
            "return": total_reward,
        }
        env.close()
    if args.csv_out is not None:
        import csv

        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", "metric", "value"])
            writer.writeheader()
            for task, stats in results.items():
                writer.writerow(
                    {
                        "task": task,
                        "metric": "macro_acc",
                        "value": stats.get("macro_accuracy", 0.0),
                    }
                )

    print(results)


if __name__ == "__main__":
    main()
```


---

## `scripts/run_markov_diag.py` <a id="file-scriptsrun_markov_diagpy"></a>

- Size: 4KB

```python
"""Run Markov diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.markov_diag import Transition, compute_markov_diagnostics
from maestro.policy.ppo import TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.wandb import init_wandb_run, log_metrics


def build_env(config: Dict[str, Any], task_cfg: str, seed: int) -> MaestroEnv:
    datasets = build_from_config(task_cfg, seed)
    env_config = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Markov diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional output directory for persisted diagnostics. "
            "Defaults to outputs/publication_<DATE>/raw_data"
        ),
    )
    args = parser.parse_args()

    date_tag = datetime.now().strftime("%Y%m%d")
    default_root = Path("outputs") / f"publication_{date_tag}"
    raw_dir = (args.out or default_root) / "raw_data"

    config = load_config(args.config)
    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )

    diagnostics_per_task: Dict[str, Dict[str, float]] = {}
    task_list: List[str] = list(config.get("eval_tasks", config["tasks"]))
    base_seed = config.get("seed", 0)
    run_name = f"markov_diag_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(
        run_name,
        config={"config": config, "tasks": task_list},
    )

    try:
        for index, task_cfg in enumerate(task_list):
            env_seed = base_seed + index * 19
            env = build_env(config, task_cfg, env_seed)
            transitions: List[Transition] = []
            obs, _ = env.reset()
            descriptors = env.last_per_dataset_descriptors
            for _ in range(config["horizon"]):
                action, _, _, _ = policy.act(obs, descriptors)
                next_obs, reward, done, _, _ = env.step(action)
                transitions.append(
                    Transition(state=obs, action=action, next_state=next_obs)
                )
                obs = next_obs
                descriptors = env.last_per_dataset_descriptors
                if done:
                    break
            task_key = Path(task_cfg).stem
            diagnostics = compute_markov_diagnostics(transitions)
            diagnostics_per_task[task_key] = diagnostics
            log_metrics({f"markov/{task_key}/{k}": v for k, v in diagnostics.items()})
            env.close()
    finally:
        wandb_run.finish()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "markov_diag.jsonl").open("w", encoding="utf-8") as fh:
        import json

        for task, diagnostics in diagnostics_per_task.items():
            fh.write(json.dumps({"task": task, **diagnostics}) + "\n")

    print(diagnostics_per_task)


if __name__ == "__main__":
    main()
```


---

## `scripts/run_n_invariance.py` <a id="file-scriptsrun_n_invariancepy"></a>

- Size: 5KB

```python
#!/usr/bin/env python
"""Evaluate N-invariance of a teacher policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

from maestro.datasets import build_from_config
from maestro.envs.maestro_env import MaestroEnv, MaestroEnvConfig
from maestro.eval.n_invariance import evaluate_permutations
from maestro.policy.ppo import PPOConfig, PPOTeacher, TeacherPolicy
from maestro.utils.config import load_config
from maestro.utils.wandb import init_wandb_run, log_metrics


def _build_env(
    config: dict,
    task_cfg_path: str,
    seed: int,
    num_datasets: int,
) -> MaestroEnv:
    datasets = build_from_config(task_cfg_path, seed, num_datasets=num_datasets)
    env_cfg = MaestroEnvConfig(
        datasets=datasets,
        horizon=config["horizon"],
        batch_size=config["batch_size"],
        initial_budget=config["initial_budget"],
        probe_size=config["probe"]["size"],
        grad_project_dim=config["probe"]["grad_project_dim"],
        grad_ema_beta=config["probe"]["grad_ema_beta"],
        grad_norm_alpha=config["probe"]["grad_norm_alpha"],
        eta_min=config["optimizer"]["eta_min"],
        eta_max=config["optimizer"]["eta_max"],
        weight_decay=config["optimizer"]["weight_decay"],
        momentum=config["optimizer"]["momentum"],
        seed=seed,
        ablations=config.get("ablations"),
    )
    return MaestroEnv(env_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on N=train_num_datasets and evaluate permutation robustness at eval_num_datasets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-num-datasets", type=int, default=3)
    parser.add_argument("--eval-num-datasets", type=int, default=7)
    parser.add_argument("--train-episodes", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=32)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/n_invariance"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks found in config")
    task_cfg = tasks[min(args.task_index, len(tasks) - 1)]

    policy = TeacherPolicy(
        descriptor_dim=8,
        g_model_dim=6,
        g_progress_dim=11,
        eta_bounds=(config["optimizer"]["eta_min"], config["optimizer"]["eta_max"]),
    )
    ppo = PPOTeacher(policy, PPOConfig(**config.get("ppo", {})))

    run_name = f"n_invariance_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    wandb_run = init_wandb_run(
        run_name,
        config={
            "config": config,
            "task_cfg": task_cfg,
            "train_num_datasets": args.train_num_datasets,
            "eval_num_datasets": args.eval_num_datasets,
            "train_episodes": args.train_episodes,
            "permutations": args.permutations,
        },
    )

    try:
        if args.checkpoint and args.checkpoint.exists():
            state = torch.load(args.checkpoint, map_location="cpu")
            policy.load_state_dict(state.get("policy", state))
        else:
            for episode in range(max(0, args.train_episodes)):
                env = _build_env(
                    config,
                    task_cfg,
                    seed + episode * 31,
                    num_datasets=args.train_num_datasets,
                )
                try:
                    train_stats = ppo.train_episode(env, config["horizon"])
                finally:
                    env.close()
                train_stats = {f"train/{k}": v for k, v in train_stats.items()}
                train_stats["train/episode"] = episode
                log_metrics(train_stats)

        eval_env = _build_env(
            config,
            task_cfg,
            seed + 999,
            num_datasets=args.eval_num_datasets,
        )
        try:
            rng = np.random.default_rng(seed + 123)
            n = args.eval_num_datasets
            perms: List[List[int]] = [
                rng.permutation(n).tolist() for _ in range(args.permutations)
            ]
            stats = evaluate_permutations(eval_env, policy, perms)
        finally:
            eval_env.close()
        log_metrics({f"eval/{k}": v for k, v in stats.items()})
    finally:
        wandb_run.finish()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "train_num_datasets": args.train_num_datasets,
        "eval_num_datasets": args.eval_num_datasets,
        "permutations": args.permutations,
        "mean_macro": stats["mean_macro"],
        "sigma_macro": stats["std_macro"],
    }
    output_path = args.output_dir / "n_invariance.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
```


---

## `train_baselines.py` <a id="file-train_baselinespy"></a>

- Size: 30KB

```python
#!/usr/bin/env python
"""Supervised curriculum baselines.

This script benchmarks a suite of supervised curricula across the built-in
MAESTRO tasks (classification, NER, and detection).  Each task/method pair is
trained with the same token budget as a MAESTRO episode, and detailed metrics
are logged so we can reason about sample efficiency before introducing
reinforcement learning.

The baselines implemented are:

* ``standard``  – conventional shuffled mini-batch SGD with early stopping.
* ``uniform``   – uniform sampling without shuffling between episodes.
* ``easy_to_hard`` – deterministic difficulty sweep from easy to hard
  examples.
* ``greedy``    – adaptive sampling that focuses on the highest-loss samples
  observed so far.
* ``linucb``    – a simple contextual bandit sampler that balances exploration
  and exploitation based on per-sample statistics.

The script produces:

1. ``baseline_runs.csv`` containing per-step metrics for every run.
2. ``baseline_summary.csv`` aggregating the final validation metrics per task
   and method (ready for inclusion in reports/tables).
3. Learning-curve plots under ``<output>/figures``.
4. Saved checkpoints for each task/method under ``<output>/checkpoints``.

Example usage::

    python train_baselines.py \\
        --tasks configs/tasks/classification.yaml configs/tasks/ner.yaml \\
        --output-dir outputs/baselines --budget 2048 --batch-size 32

Use ``--dry-run`` to generate the pipeline artefacts without performing actual
training (useful in CI smoke tests).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from maestro.datasets import DatasetSpec, build_from_config, detection_collate
from maestro.students import build_student
from maestro.utils.logging import MetricsLogger
from maestro.utils.seeding import seed_everything


# ---------------------------------------------------------------------------
# Task adapters
# ---------------------------------------------------------------------------


@dataclass
class SampleInfo:
    index: int
    difficulty: float
    context: np.ndarray
    last_loss: float = 1.0


class TaskAdapter:
    """Task-specific helpers for batching and metrics."""

    def __init__(self, spec: DatasetSpec, device: torch.device):
        self.spec = spec
        self.device = device
        self.task = spec.task_type
        self._collate_fn: Optional[Callable] = (
            detection_collate if self.task == "detection" else None
        )

    # ------------------------------------------------------------------
    def collate(self, samples: Sequence) -> Tuple:
        if self.task == "detection":
            images = torch.stack([sample[0] for sample in samples])
            boxes = [sample[1] for sample in samples]
            return images, boxes
        inputs = torch.stack([sample[0] for sample in samples])
        targets = torch.stack([sample[1] for sample in samples])
        return inputs, targets

    # ------------------------------------------------------------------
    def make_loader(self, split: str, batch_size: int) -> DataLoader:
        dataset = getattr(self.spec, split)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    # ------------------------------------------------------------------
    def estimate_tokens(self, batch: Tuple) -> int:
        if self.task == "ner":
            seq = batch[0]
            return int(seq.numel())
        if self.task == "detection":
            images = batch[0]
            return int(images.size(0) * images.size(-1) * images.size(-2))
        inputs = batch[0]
        return int(inputs.shape[0])

    # ------------------------------------------------------------------
    def evaluate(self, student: torch.nn.Module, loader: DataLoader) -> Dict[str, float]:
        if loader.dataset is None:
            return {}
        student.eval()
        if self.task == "detection":
            metrics = student.eval_on_loader(loader)
            return {"loss": float(metrics.get("loss", 0.0)), "macro_map": float(metrics.get("accuracy", 0.0))}
        preds: List[int] = []
        targets: List[int] = []
        losses: List[float] = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits = student(inputs)
                batch_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                losses.append(batch_loss.mean().item())
                preds.extend(logits.argmax(dim=-1).view(-1).cpu().tolist())
                targets.extend(labels.view(-1).cpu().tolist())
        result: Dict[str, float] = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean(np.array(preds) == np.array(targets))) if targets else 0.0,
        }
        if targets:
            result["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
        return result

    # ------------------------------------------------------------------
    def confusion(self, student: torch.nn.Module, loader: DataLoader) -> Optional[np.ndarray]:
        if self.task == "detection":
            return None
        preds: List[int] = []
        targets: List[int] = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logits = student(inputs)
                preds.extend(logits.argmax(dim=-1).view(-1).cpu().tolist())
                targets.extend(labels.view(-1).cpu().tolist())
        if not targets:
            return None
        return confusion_matrix(targets, preds)


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def build_sample_pool(dataset, adapter: TaskAdapter) -> List[SampleInfo]:
    pool: List[SampleInfo] = []
    if adapter.task == "classification":
        class_counts = Counter(int(dataset[i][1].item()) for i in range(len(dataset)))
        total = sum(class_counts.values())
        for idx in range(len(dataset)):
            label = int(dataset[idx][1].item())
            rarity = 1.0 - class_counts[label] / max(1, total)
            pool.append(SampleInfo(index=idx, difficulty=float(rarity), context=np.array([rarity], dtype=np.float32)))
    elif adapter.task == "ner":
        for idx in range(len(dataset)):
            tokens, tags = dataset[idx]
            entity_tokens = float((tags > 0).sum().item())
            difficulty = entity_tokens / max(1.0, float(tags.numel()))
            pool.append(SampleInfo(index=idx, difficulty=difficulty, context=np.array([difficulty], dtype=np.float32)))
    elif adapter.task == "detection":
        for idx in range(len(dataset)):
            _, boxes = dataset[idx]
            count = float(boxes.size(0)) if isinstance(boxes, torch.Tensor) else float(len(boxes))
            pool.append(SampleInfo(index=idx, difficulty=count, context=np.array([count], dtype=np.float32)))
    else:
        for idx in range(len(dataset)):
            pool.append(SampleInfo(index=idx, difficulty=0.5, context=np.zeros(1, dtype=np.float32)))
    return pool


# ---------------------------------------------------------------------------
# Trainer base class
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    budget: int
    batch_size: int
    eval_interval: int
    patience: int
    device: torch.device
    output_dir: Path
    log_fn: Optional[Callable[[str], None]] = None


@dataclass
class TrainerOutput:
    task: str
    method: str
    tokens_used: int
    wall_time: float
    history: List[Dict[str, float]]
    final_metrics: Dict[str, float]
    sample_log: List[int]
    checkpoint: Path


class BaseTrainer:
    def __init__(
        self,
        method: str,
        student: torch.nn.Module,
        spec: DatasetSpec,
        adapter: TaskAdapter,
        train_pool: List[SampleInfo],
        val_loader: DataLoader,
        cfg: TrainerConfig,
    ) -> None:
        self.method = method
        self.student = student
        self.spec = spec
        self.adapter = adapter
        self.sample_infos = train_pool
        self.sample_lookup = {info.index: info for info in train_pool}
        self.val_loader = val_loader
        self.cfg = cfg
        self.tokens_used = 0
        self.history: List[Dict[str, float]] = []
        self.sample_log: List[int] = []
        self.best_loss = float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.epochs_without_improve = 0
        self.global_step = 0
        self._last_indices: List[int] = []
        self.log_fn = cfg.log_fn

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.log_fn is None:
            return
        try:
            self.log_fn(message)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def run(self) -> TrainerOutput:
        start = time.perf_counter()
        optimizer = torch.optim.Adam(self.student.parameters(), lr=3e-4)
        loss_meter: List[float] = []
        self._log(f"Starting {self.method} on {self.spec.name} (budget={self.cfg.budget}, eval_every={self.cfg.eval_interval})")
        while self.tokens_used < self.cfg.budget:
            batch_indices = self.sample_indices(self.cfg.batch_size)
            batch = self._build_batch(batch_indices)
            loss = self._step(batch, optimizer)
            loss_meter.append(loss)
            if self.global_step % self.cfg.eval_interval == 0:
                metrics = self.adapter.evaluate(self.student, self.val_loader)
                metrics.update({
                    "step": self.global_step,
                    "tokens": self.tokens_used,
                    "train_loss": float(np.mean(loss_meter)) if loss_meter else loss,
                    "method": self.method,
                    "task": self.spec.name,
                })
                self.history.append(metrics)
                tracked_metrics = {k: v for k, v in metrics.items() if k in ("loss", "macro_f1", "macro_map", "accuracy", "macro_metric")}
                summary_bits = [f"{k}={tracked_metrics[k]:.4f}" for k in sorted(tracked_metrics)]
                if not summary_bits and "train_loss" in metrics:
                    summary_bits.append(f"train_loss={metrics['train_loss']:.4f}")
                summary_str = ", ".join(summary_bits) if summary_bits else "no_metrics"
                self._log(f"step={self.global_step} tokens={self.tokens_used} {summary_str}")
                loss_meter.clear()
                val_loss = float(metrics.get("loss", 0.0))
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.student.state_dict().items()}
                    self.epochs_without_improve = 0
                else:
                    self.epochs_without_improve += 1
                if self.epochs_without_improve >= self.cfg.patience:
                    self._log(f"Early stopping triggered after {self.epochs_without_improve} evals without improvement.")
                    break
        if self.best_state is not None:
            self.student.load_state_dict(self.best_state)
        wall = time.perf_counter() - start
        final_metrics = self.adapter.evaluate(self.student, self.val_loader)
        if final_metrics:
            final_summary = ", ".join(f"{k}={v:.4f}" for k, v in final_metrics.items())
            self._log(f"Finished {self.method} on {self.spec.name} in {wall:.1f}s – {final_summary}")
        ckpt_path = self._save_checkpoint()
        return TrainerOutput(
            task=self.spec.name,
            method=self.method,
            tokens_used=self.tokens_used,
            wall_time=wall,
            history=self.history,
            final_metrics=final_metrics,
            sample_log=self.sample_log,
            checkpoint=ckpt_path,
        )

    # ------------------------------------------------------------------
    def sample_indices(self, batch_size: int) -> Sequence[int]:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _build_batch(self, indices: Sequence[int]):
        samples = [self.spec.train[idx] for idx in indices]
        self.sample_log.extend(int(idx) for idx in indices[:5])
        self._last_indices = list(indices)
        return self.adapter.collate(samples)

    # ------------------------------------------------------------------
    def _step(self, batch, optimizer) -> float:
        self.student.train()
        optimizer.zero_grad()
        metrics = self.student.step_on_minibatch(batch)
        loss = float(metrics.get("loss", 0.0))
        optimizer.step()
        tokens = self.adapter.estimate_tokens(batch)
        self.tokens_used += tokens
        self.global_step += 1
        self.post_step_hook(loss)
        return loss

    # ------------------------------------------------------------------
    def post_step_hook(self, loss: float) -> None:
        for idx in self._last_indices:
            info = self.sample_lookup.get(idx)
            if info is not None:
                info.last_loss = loss

    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> Path:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.cfg.output_dir / f"{self.spec.name}_{self.method}.pt"
        torch.save(self.student.state_dict(), ckpt_path)
        return ckpt_path


# ---------------------------------------------------------------------------
# Trainer specialisations
# ---------------------------------------------------------------------------


class StandardShuffledTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._order = list(range(len(self.spec.train)))
        self._position = 0

    def sample_indices(self, batch_size: int) -> Sequence[int]:
        if self._position + batch_size > len(self._order):
            rng = np.random.default_rng(self.global_step)
            rng.shuffle(self._order)
            self._position = 0
        indices = self._order[self._position : self._position + batch_size]
        self._position += batch_size
        return indices


class UniformSamplingTrainer(BaseTrainer):
    def sample_indices(self, batch_size: int) -> Sequence[int]:
        rng = np.random.default_rng(self.global_step)
        return rng.choice(len(self.spec.train), size=batch_size, replace=True).tolist()


class EasyToHardTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sorted_pool = sorted(self.sample_infos, key=lambda s: s.difficulty)
        self._order = [item.index for item in sorted_pool]
        self._position = 0

    def sample_indices(self, batch_size: int) -> Sequence[int]:
        if self._position >= len(self._order):
            self._position = 0
        end = min(len(self._order), self._position + batch_size)
        chunk = self._order[self._position : end]
        self._position = end
        return chunk


class GreedyTrainer(BaseTrainer):
    def sample_indices(self, batch_size: int) -> Sequence[int]:
        ranked = sorted(self.sample_infos, key=lambda s: s.last_loss, reverse=True)
        indices = [entry.index for entry in ranked[:batch_size]]
        return indices


class LinUCBTrainer(BaseTrainer):
    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        dim = len(self.sample_infos[0].context)
        self.alpha = alpha
        self.A = np.eye(dim)
        self.b = np.zeros(dim)

    def sample_indices(self, batch_size: int) -> Sequence[int]:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        scores = []
        for sample in self.sample_infos:
            mean = sample.context @ theta
            var = math.sqrt(sample.context @ A_inv @ sample.context)
            scores.append(mean + self.alpha * var)
        ranked = np.argsort(scores)[::-1][:batch_size]
        return [self.sample_infos[idx].index for idx in ranked]

    def post_step_hook(self, loss: float) -> None:
        super().post_step_hook(loss)
        reward = -loss
        for idx in self._last_indices:
            info = self.sample_lookup.get(idx)
            if info is None:
                continue
            context = info.context
            self.A += np.outer(context, context)
            self.b += context * reward


TRAINER_REGISTRY = {
    "standard": StandardShuffledTrainer,
    "uniform": UniformSamplingTrainer,
    "easy_to_hard": EasyToHardTrainer,
    "greedy": GreedyTrainer,
    "linucb": LinUCBTrainer,
}


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_task(
    method: str,
    spec: DatasetSpec,
    task_id: str,
    cfg: TrainerConfig,
    device: torch.device,
    output_root: Path,
    log_dir: Optional[Path] = None,
) -> TrainerOutput:
    log_handle: Optional[Any] = None
    log_fn: Optional[Callable[[str], None]] = None

    def _sanitize(text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
        return safe or "run"

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{_sanitize(task_id)}_{_sanitize(method)}.log"

        log_handle = log_path.open("w", encoding="utf-8")

        def _log(message: str, *, _handle=log_handle, _task=task_id, _method=method) -> None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] [{_task}::{_method}] {message}"
            print(line)
            try:
                _handle.write(line + "\n")
                _handle.flush()
            except Exception:
                pass

        log_fn = _log

    adapter = TaskAdapter(spec, device)
    student = build_student([spec]).to(device)
    train_pool = build_sample_pool(spec.train, adapter)
    val_loader = adapter.make_loader("val", batch_size=max(1, cfg.batch_size))
    trainer_cls = TRAINER_REGISTRY[method]
    trainer = trainer_cls(
        method,
        student,
        spec,
        adapter,
        train_pool,
        val_loader,
        TrainerConfig(
            budget=cfg.budget,
            batch_size=cfg.batch_size,
            eval_interval=cfg.eval_interval,
            patience=cfg.patience,
            device=device,
            output_dir=output_root / "checkpoints",
            log_fn=log_fn,
        ),
    )
    try:
        return trainer.run()
    finally:
        if log_handle is not None:
            log_handle.close()


def aggregate_results(outputs: List[TrainerOutput], output_dir: Path) -> None:
    records: List[Dict[str, float]] = []
    for out in outputs:
        record = {
            "task": out.task,
            "method": out.method,
            "tokens": out.tokens_used,
            "wall_time": out.wall_time,
        }
        record.update({f"final_{k}": v for k, v in out.final_metrics.items()})
        records.append(record)
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    summary_path = output_dir / "baseline_summary.csv"
    df.to_csv(summary_path, index=False)

    table_path = output_dir / "comparison_table.txt"
    if df.empty:
        table_path.write_text("No runs completed – comparison table unavailable.\n", encoding="utf-8")
        return
    metric_priority = [
        "final_macro_f1",
        "final_macro_map",
        "final_accuracy",
        "final_map",
        "final_loss",
    ]
    metric_column = next((col for col in metric_priority if col in df.columns), None)
    if metric_column is None:
        metric_column = next((col for col in df.columns if col.startswith("final_")), None)
    if metric_column is None:
        table_path.write_text("No metric columns detected – comparison table unavailable.\n", encoding="utf-8")
        return
    pivot = df.pivot_table(index="task", columns="method", values=metric_column, aggfunc="max")
    if pivot.empty:
        table_path.write_text(f"No values available for {metric_column}.\n", encoding="utf-8")
        return
    metric_label = metric_column.replace("final_", "").replace("_", " ").title()
    table_path = output_dir / "comparison_table.txt"
    table_lines = [
        f"Metric: {metric_label}",
        "Task              | " + " | ".join(f"{col:>10}" for col in pivot.columns),
    ]
    table_lines.append("-" * len(table_lines[0]))
    for task, row in pivot.iterrows():
        values = ["{:.3f}".format(row[col]) if not math.isnan(row[col]) else "-" for col in pivot.columns]
        table_lines.append(f"{task:<16}| " + " | ".join(f"{val:>10}" for val in values))
    table_path.write_text("\n".join(table_lines), encoding="utf-8")
    matrix = pivot.to_numpy(dtype=float)
    if not np.isfinite(matrix).any():
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(matrix)
    max_val = float(np.nanmax(matrix))
    contrast = max_val * 0.5 if max_val != 0 else 0.0
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#f0f0f0")
    fig, ax = plt.subplots(figsize=(1.6 * max(1, len(pivot.columns)), 1.0 * max(1, len(pivot.index)) + 1.5))
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(idx) for idx in pivot.index])
    ax.set_xlabel("Method")
    ax.set_ylabel("Task")
    ax.set_title(f"{metric_label} overview")
    for i, task in enumerate(pivot.index):
        for j, method in enumerate(pivot.columns):
            value = pivot.loc[task, method]
            if math.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < contrast else "black",
                fontsize=9,
            )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"comparison_heatmap.{ext}", dpi=200)
    plt.close(fig)


def plot_learning_curves(outputs: List[TrainerOutput], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task: Dict[str, List[TrainerOutput]] = defaultdict(list)
    for out in outputs:
        by_task[out.task].append(out)
    metric_priority: Sequence[Tuple[str, str]] = (
        ("macro_f1", "Macro F1"),
        ("macro_map", "Macro mAP"),
        ("accuracy", "Accuracy"),
        ("map", "mAP"),
        ("macro_metric", "Macro metric"),
        ("loss", "Loss"),
    )

    def extract_series(out: TrainerOutput) -> Tuple[List[float], List[float], Optional[str]]:
        steps: List[float] = []
        values: List[float] = []
        chosen_label: Optional[str] = None
        for entry in out.history:
            tokens = entry.get("tokens")
            if tokens is None:
                continue
            for key, label in metric_priority:
                if key in entry and entry[key] is not None:
                    steps.append(float(tokens))
                    values.append(float(entry[key]))
                    if chosen_label is None:
                        chosen_label = label
                    break
        return steps, values, chosen_label

    task_plot_data: Dict[str, Dict[str, Any]] = {}
    for task, runs in by_task.items():
        plt.figure(figsize=(7, 4))
        plotted = False
        for out in runs:
            steps, values, metric_label = extract_series(out)
            if not steps:
                continue
            plt.plot(steps, values, label=out.method)
            task_entry = task_plot_data.setdefault(
                task, {"metric_label": metric_label or "Metric", "series": []}
            )
            if metric_label and task_entry["metric_label"] == "Metric":
                task_entry["metric_label"] = metric_label
            task_entry["series"].append((out.method, steps, values))
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel("Tokens consumed")
        plt.ylabel(task_plot_data.get(task, {}).get("metric_label", "Validation metric"))
        plt.title(f"Learning curves – {task}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"learning_curve_{task}.png", dpi=200)
        plt.close()
    if not task_plot_data:
        return
    tasks_sorted = sorted(task_plot_data.items())
    n_tasks = len(tasks_sorted)
    ncols = 2 if n_tasks > 1 else 1
    nrows = math.ceil(n_tasks / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax in flat_axes[n_tasks:]:
        ax.axis("off")
    for ax, (task, data) in zip(flat_axes, tasks_sorted):
        for method, steps, values in data["series"]:
            ax.plot(steps, values, label=method)
        ax.set_title(task)
        ax.set_xlabel("Tokens consumed")
        ax.set_ylabel(data["metric_label"])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"learning_curves_overview.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised baselines with curriculum strategies")
    parser.add_argument("--tasks", nargs="*", default=[
        "configs/tasks/classification.yaml",
        "configs/tasks/ner.yaml",
        "configs/tasks/detection.yaml",
    ])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--methods", nargs="*", default=list(TRAINER_REGISTRY.keys()))
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", type=Path, help="Directory to store per-run logs (defaults to <output>/logs)")
    parser.add_argument("--no-run-logs", action="store_true", help="Disable per-run log files")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir: Optional[Path]
    if args.no_run_logs:
        log_dir = None
    else:
        log_dir = args.log_dir or (args.output_dir / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainerConfig(
        budget=args.budget,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        patience=args.patience,
        device=device,
        output_dir=args.output_dir / "checkpoints",
    )

    methods = args.methods
    results: List[TrainerOutput] = []
    logger = MetricsLogger(args.output_dir, csv_filename="baseline_runs.csv", json_filename="baseline_runs.json")

    if args.dry_run:
        # Produce placeholder files without heavy computation.
        placeholder = {
            "task": "dry_run",
            "method": "standard",
            "tokens": 0,
            "wall_time": 0.0,
            "final_macro_f1": 0.0,
        }
        logger.log_row(placeholder)
        aggregate_results([], args.output_dir)
        print("[dry-run] Baseline scaffolding generated.")
        return

    for task_cfg in args.tasks:
        specs = build_from_config(task_cfg, seed=args.seed)
        for spec in specs:
            val_adapter = TaskAdapter(spec, device)
            val_loader = val_adapter.make_loader("val", args.batch_size)
            for method in methods:
                output = run_task(
                    method,
                    spec,
                    spec.name,
                    cfg,
                    device,
                    args.output_dir,
                    log_dir,
                )
                results.append(output)
                final_metrics = output.final_metrics
                row = {
                    "task": spec.name,
                    "method": method,
                    "tokens": output.tokens_used,
                    "wall_time": output.wall_time,
                }
                row.update({f"final_{k}": v for k, v in final_metrics.items()})
                logger.log_row(row)
                logger.flush_json()
    aggregate_results(results, args.output_dir)
    plot_learning_curves(results, args.output_dir / "figures")
    print(f"Results saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
```


---

## `train_maestro_teacher.py` <a id="file-train_maestro_teacherpy"></a>

- Size: 3KB

```python
"""Convenience CLI to meta-train a MAESTRO teacher with PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_meta_train import run_meta_training


DEFAULT_CONFIG = Path("configs/meta_train/small_cpu_debug.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-train a MAESTRO teacher and save the checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the meta-training config (default: small CPU debug)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory under which <run_id>/policy.pt will be written (default: outputs)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed override"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume training from a checkpoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config and resolved paths only"
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Run the teacher in eval mode for deterministic rollouts",
    )
    # --- Brief imitation warm-start (pass-through to run_meta_training) ---
    parser.add_argument(
        "--bc-warm-start",
        action="store_true",
        help="Run a short behavior-cloning warm start before PPO (default: off)",
    )
    parser.add_argument(
        "--bc-episodes",
        type=int,
        default=2,
        help="Episodes to collect for BC warm start (default: 2)",
    )
    parser.add_argument(
        "--bc-baseline",
        type=str,
        default="uniform",
        choices=["uniform", "easy_to_hard", "greedy", "linucb"],
        help="Baseline scheduler used to synthesize BC targets (default: uniform)",
    )
    parser.add_argument(
        "--bc-usage",
        type=float,
        default=0.4,
        help="Target usage fraction when creating BC targets; must be in (0, 1) (default: 0.4)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=2,
        help="Supervised epochs to fit policy heads during BC warm start (default: 2)",
    )
    args = parser.parse_args()

    output_dir = run_meta_training(
        args.config,
        dry_run=args.dry_run,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        deterministic_eval=args.deterministic_eval,
        bc_warm_start_flag=args.bc_warm_start,
        bc_episodes=args.bc_episodes,
        bc_baseline=args.bc_baseline,
        bc_usage=args.bc_usage,
        bc_epochs=args.bc_epochs,
    )

    if args.dry_run:
        print(
            f"[dry-run] Would write checkpoints and logs under: {output_dir.resolve()}"
        )
    else:
        checkpoint = output_dir / "policy.pt"
        print(f"[✓] Teacher checkpoint saved to {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
```


---

## `train_maestro_yolo.py` <a id="file-train_maestro_yolopy"></a>

- Size: 36KB

```python
"""Segmented multi-dataset YOLO training orchestrated by MAESTRO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from maestro.baselines import create_scheduler
from maestro.policy import MaestroPolicy, MaestroPolicyConfig, TeacherPolicy
from maestro.probes import DummyYOLO, build_model, estimate_probes_with_val
from maestro.utils.wandb import init_wandb_run, log_checkpoint, log_metrics
from maestro.yolo.mix_builder import (
    SourceDS,
    build_mixed_segment,
    build_mixed_val,
    compute_pools_and_canonical,
)


VOC_CANONICAL_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

SMALL_DATASET_DEFAULTS = ["clipart1k", "watercolor2k", "comic2k", "pennfudan", "kitti"]

DEFAULT_DATASETS: Dict[str, Dict[str, object]] = {
    "clipart1k": {
        "yaml": "configs/datasets/clipart1k.yaml",
        "fallback_names": VOC_CANONICAL_NAMES,
    },
    "watercolor2k": {
        "yaml": "configs/datasets/watercolor2k.yaml",
        "fallback_names": ["bicycle", "bird", "car", "cat", "dog", "person"],
    },
    "comic2k": {
        "yaml": "configs/datasets/comic2k.yaml",
        "fallback_names": ["bicycle", "bird", "car", "cat", "dog", "person"],
    },
    "pennfudan": {
        "yaml": "configs/datasets/pennfudan.yaml",
        "fallback_names": ["person"],
    },
    "kitti": {
        "yaml": "configs/datasets/kitti.yaml",
        "fallback_names": ["car", "pedestrian", "cyclist"],
    },
    "coco": {"yaml": "configs/datasets/coco.yaml", "fallback_names": []},
    "lvis": {"yaml": "configs/datasets/lvis.yaml", "fallback_names": []},
    "voc": {"yaml": "configs/datasets/voc.yaml", "fallback_names": VOC_CANONICAL_NAMES},
    "target": {"yaml": "configs/datasets/target.yaml", "fallback_names": []},
}

REPO_ROOT = Path(__file__).resolve().parent


def _descriptor_from_probe(stats: Dict[str, float], size_log: float) -> np.ndarray:
    """Map YOLO probe statistics to the 8-D teacher descriptor space."""

    return np.array(
        [
            float(stats.get("loss_mean", 0.0)),
            float(stats.get("loss_iqr", 0.0)),
            float(stats.get("entropy_mean", 0.0)),
            0.0,  # expected calibration error unavailable for YOLO probes
            float(stats.get("grad_norm_log", 0.0)),
            0.0,  # gradient cosine placeholder
            0.0,  # log effective rank placeholder
            float(size_log),
        ],
        dtype=np.float32,
    )


def _build_teacher_observation(descriptors: np.ndarray) -> Dict[str, np.ndarray]:
    """Construct the observation dictionary expected by ``TeacherPolicy``."""

    if descriptors.size == 0:
        g_data = np.zeros(8, dtype=np.float32)
    else:
        g_data = descriptors.mean(axis=0).astype(np.float32)
    g_model = np.zeros(6, dtype=np.float32)
    g_progress = np.zeros(11, dtype=np.float32)
    return {"g_data": g_data, "g_model": g_model, "g_progress": g_progress}


def _load_teacher_policy(
    checkpoint_path: Path, eta_bounds: Tuple[float, float]
) -> TeacherPolicy:
    """Load a PPO teacher checkpoint onto CPU for inference."""

    policy = TeacherPolicy(
        descriptor_dim=8, g_model_dim=6, g_progress_dim=11, eta_bounds=eta_bounds
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("policy", state)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def _resolve_yaml_path(rel_yaml: str) -> Optional[Path]:
    """Resolve dataset YAML paths in common MAESTRO locations."""

    rel = Path(rel_yaml)
    here = Path(__file__).resolve().parent
    candidates = [
        rel,
        Path("data") / rel,
        here / rel,
        here / "data" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _pretty_path(path: Path | str) -> str:
    if isinstance(path, Path):
        return str(path.resolve())
    return str(path)


def _ensure_ultralytics_model_override(model: object, fallback_weight: str) -> None:
    """Restore the Ultralytics override entry removed by recent releases."""

    overrides = getattr(model, "overrides", None)
    if not isinstance(overrides, dict):
        return
    if overrides.get("model"):
        return
    candidate = getattr(model, "ckpt_path", None)
    if not candidate:
        inner = getattr(model, "model", None)
        candidate = getattr(inner, "yaml", None) if inner is not None else None
    overrides["model"] = candidate or fallback_weight


@dataclass
class RunConfig:
    """Configuration bundle persisted to ``metadata.json`` for reproducibility."""

    weights: str
    segments: int
    budget_images: int
    base_lr: float
    imgsz: int
    batch_size: int
    device: str
    datasets: List[str]
    dry_run: bool
    resume: bool
    method: str
    label_space: str


def _update_metadata(out_dir: Path, record: Dict[str, object]) -> None:
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        data = {"tracks": {}}
    data.setdefault("tracks", {})["yolo"] = record
    metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_segment_csv(csv_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _resolve_dataset_names(datasets: Iterable[str]) -> Dict[str, Dict[str, object]]:
    resolved: Dict[str, Dict[str, object]] = {}
    for name in datasets:
        if name not in DEFAULT_DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Known options: {sorted(DEFAULT_DATASETS)}")
        cfg = DEFAULT_DATASETS[name]
        rel_yaml = cfg.get("yaml")
        rel_yaml = str(rel_yaml) if not isinstance(rel_yaml, str) else rel_yaml
        resolved_path = _resolve_yaml_path(rel_yaml)
        if resolved_path is None:
            print(f"[!] Could not find YAML for '{name}' at '{rel_yaml}' or 'data/{rel_yaml}'.")
            yaml_str = rel_yaml
        else:
            yaml_str = _pretty_path(resolved_path)
        resolved[name] = {
            "yaml": yaml_str,
            "fallback_names": list(cfg.get("fallback_names", [])),
        }
    return resolved


def _prefer_dataset_yaml(dataset: str, base_yaml: Path) -> Path:
    """Prefer a dataset-specific YAML with explicit class names when available."""

    suffix = f"yolo_{dataset}.yaml"
    env_roots = [
        Path(os.environ[key])
        for key in ("MAESTRO_DATA_ROOT", "DATA_ROOT")
        if os.environ.get(key)
    ]
    search_roots = [
        base_yaml.parent,
        base_yaml.parent / "data",
        REPO_ROOT / "data",
        Path.cwd() / "data",
    ]
    search_roots.extend(env_roots)
    search_roots.extend(root / "data" for root in env_roots)

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in search_roots:
        candidate = (root / dataset / suffix) if root else Path(dataset) / suffix
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(resolved)

    for candidate in candidates:
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        names = data.get("names")
        if isinstance(names, dict) and names:
            return candidate
        if isinstance(names, list) and names:
            return candidate
    return base_yaml


def _load_names_from_yaml(yaml_path: Path, fallback: Sequence[str]) -> List[str]:
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            data = {}
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            try:
                items = sorted(names.items(), key=lambda item: int(item[0]))
            except Exception:
                items = sorted(names.items(), key=lambda item: str(item[0]))
            return [str(value) for _, value in items]
        if isinstance(names, list):
            return [str(value) for value in names]
    return [str(value) for value in fallback]


def _resolve_train_dirs(yaml_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not yaml_path.exists():
        return None, None
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        return None, None

    base_entry = data.get("path")
    base_candidates: list[Path] = []
    seen_bases: set[Path] = set()

    def _push_base(path: Path) -> None:
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved in seen_bases:
            return
        seen_bases.add(resolved)
        base_candidates.append(resolved)

    default_roots = [
        yaml_path.parent,
        yaml_path.parent.parent,
        REPO_ROOT,
        REPO_ROOT / "data",
        Path.cwd(),
        Path.cwd() / "data",
    ]
    for key in ("MAESTRO_DATA_ROOT", "DATA_ROOT"):
        env = os.environ.get(key)
        if not env:
            continue
        env_path = Path(env)
        default_roots.extend([env_path, env_path / "data"])

    if base_entry:
        raw = Path(str(base_entry))
        if raw.is_absolute():
            _push_base(raw)
        else:
            base_str = str(raw)
            variants = [raw]
            stripped = base_str.removeprefix("./")
            if stripped != base_str:
                variants.append(Path(stripped))
            if not base_str.startswith("data/"):
                variants.append(Path("data") / raw)
            for root in default_roots:
                for variant in variants:
                    _push_base(root / variant)
    else:
        for root in default_roots:
            _push_base(root)

    base_path: Optional[Path] = None
    for candidate in base_candidates:
        if candidate.exists() and candidate.is_dir():
            base_path = candidate
            break
    if base_path is None:
        return None, None

    train_entry = Path(str(data.get("train", "images/train")))
    train_candidates: list[Path] = []
    if train_entry.is_absolute():
        train_candidates.append(train_entry)
    else:
        train_candidates.append(base_path / train_entry)
        if not train_entry.parts or train_entry.parts[0] != "images":
            train_candidates.append(base_path / "images" / train_entry)

    train_path: Optional[Path] = None
    for candidate in train_candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_dir():
            train_path = resolved
            break
    if train_path is None:
        return None, None

    labels_dir = _infer_labels_dir(train_path, base_path)
    return train_path, labels_dir


def _infer_labels_dir(train_path: Path, base_path: Path) -> Optional[Path]:
    candidates: list[Path] = []
    try:
        rel = train_path.relative_to(base_path)
    except ValueError:
        rel = None
    if rel is not None:
        parts = rel.parts
        if parts and parts[0] == "images":
            suffix = Path(*parts[1:]) if len(parts) > 1 else Path()
            candidates.append(base_path / "labels" / suffix)
        elif parts:
            candidates.append(base_path / "labels" / Path(*parts))
        candidates.append(base_path / "labels")

    parent = train_path.parent
    grandparent = parent.parent
    candidates.append(grandparent / "labels" / train_path.name)
    candidates.append(grandparent / "labels" / parent.name)
    candidates.append(grandparent / "labels")
    candidates.append(base_path / "labels" / train_path.name)
    candidates.append(base_path / "labels" / parent.name)
    candidates.append(base_path / "labels")
    candidates.append(parent / "labels")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            return resolved
    return None


def _load_resume_checkpoint(run_dir: Path) -> Tuple[int, Path | None]:
    log_path = run_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        return 1, None
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return 1, None
    last_segment = max(line.get("segment", 0) for line in lines)
    checkpoint = run_dir.parent / "checkpoints" / f"yolo_seg{last_segment}.pt"
    return last_segment + 1, checkpoint if checkpoint.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAESTRO-controlled YOLO across datasets")
    parser.add_argument("--weights", default="yolov8n.pt", help="Initial YOLO weights to load")
    parser.add_argument("--segments", type=int, default=12, help="Number of curriculum segments")
    parser.add_argument("--budget-images", type=int, default=200_000, help="Total image budget")
    parser.add_argument("--base-lr", type=float, default=0.003, help="Base learning rate")
    parser.add_argument("--imgsz", type=int, default=896, help="Training/eval image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Computation device for YOLO")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=SMALL_DATASET_DEFAULTS,
        help="Datasets to include in the curriculum",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root output directory")
    parser.add_argument("--date-tag", default=datetime.now(UTC).strftime("%Y%m%d"), help="Date tag for outputs")
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run even if logs exist")
    parser.add_argument("--dry-run", action="store_true", help="Run a fast deterministic CI-friendly simulation")
    parser.add_argument("--min-usage", type=float, default=0.2, help="Lower bound for usage fraction")
    parser.add_argument("--max-usage", type=float, default=0.7, help="Upper bound for usage fraction")
    parser.add_argument("--min-lr-scale", type=float, default=0.5)
    parser.add_argument("--max-lr-scale", type=float, default=1.5)
    parser.add_argument(
        "--label-space",
        choices=["overlap_only", "union_full"],
        default="overlap_only",
        help="Strategy for combining label spaces across datasets",
    )
    parser.add_argument(
        "--method",
        choices=[
            "maestro",
            "uniform",
            "easy_to_hard",
            "greedy",
            "linucb",
            "thompson",
            "pbt",
            "bohb",
        ],
        default="maestro",
        help="Curriculum strategy to use",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=None,
        help="Path to a pretrained PPO teacher checkpoint (policy.pt)",
    )
    parser.add_argument(
        "--teacher-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic teacher actions (default: true; pass --no-teacher-deterministic to sample)",
    )
    args = parser.parse_args()

    teacher_ckpt_path: Optional[Path] = None
    if args.teacher_ckpt is not None:
        teacher_ckpt_path = Path(args.teacher_ckpt)
        if not teacher_ckpt_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found at {teacher_ckpt_path}"
            )

    datasets = _resolve_dataset_names(args.datasets)
    dataset_names = list(datasets.keys())

    source_entries: Dict[str, SourceDS] = {}
    for name in dataset_names:
        cfg = datasets[name]
        yaml_candidate = str(cfg["yaml"])
        yaml_path = Path(yaml_candidate)
        if not yaml_path.exists():
            alt_path = _resolve_yaml_path(yaml_candidate)
            if alt_path is not None:
                yaml_path = alt_path
        yaml_path = _prefer_dataset_yaml(name, yaml_path)
        fallback_raw = cfg.get("fallback_names", [])
        fallback_seq: Sequence[str]
        if isinstance(fallback_raw, Sequence) and not isinstance(fallback_raw, (str, bytes)):
            fallback_seq = [str(item) for item in fallback_raw]
        else:
            fallback_seq = []
        names_list = (
            _load_names_from_yaml(yaml_path, fallback_seq)
            if yaml_path.exists()
            else list(fallback_seq)
        )
        cfg["yaml"] = _pretty_path(yaml_path) if yaml_path.exists() else yaml_candidate
        cfg["names"] = names_list
        images_dir, labels_dir = _resolve_train_dirs(yaml_path)
        if images_dir is None or labels_dir is None:
            print(f"[!] Skipping {name}: could not resolve YOLO train/label directories from {yaml_path}")
            continue
        source_entries[name] = SourceDS(name=name, images_dir=images_dir, labels_dir=labels_dir, names=names_list)

    sources = [source_entries[name] for name in dataset_names if name in source_entries]
    if not sources:
        raise RuntimeError("No datasets with available YOLO folders were found for mixing.")
    schedule_names = [source.name for source in sources]

    donor_lists: List[List[str]] = []
    donor_names_in_mix: List[str] = []
    include_donors = args.label_space in {"overlap_only", "union_full"}
    if include_donors:
        for donor in ("coco", "lvis"):
            cfg = DEFAULT_DATASETS.get(donor)
            if not cfg:
                continue
            donor_yaml_candidate = str(cfg.get("yaml"))
            donor_yaml = Path(donor_yaml_candidate)
            if not donor_yaml.exists():
                alt_path = _resolve_yaml_path(donor_yaml_candidate)
                if alt_path is not None:
                    donor_yaml = alt_path
            donor_yaml = _prefer_dataset_yaml(donor, donor_yaml)
            fallback = cfg.get("fallback_names", [])
            donor_names = (
                _load_names_from_yaml(donor_yaml, fallback)
                if donor_yaml.exists()
                else [str(value) for value in fallback]
            )
            images_dir, labels_dir = _resolve_train_dirs(donor_yaml)
            if images_dir is None or labels_dir is None:
                continue
            sources.append(
                SourceDS(name=donor, images_dir=images_dir, labels_dir=labels_dir, names=donor_names)
            )
            schedule_names.append(donor)
            donor_names_in_mix.append(donor)
            if donor_names:
                donor_lists.append(list(donor_names))

    donor_big_names: Optional[List[List[str]]] = donor_lists or None
    for donor in donor_names_in_mix:
        if donor not in datasets and donor in DEFAULT_DATASETS:
            cfg = DEFAULT_DATASETS[donor]
            donor_yaml_candidate = str(cfg.get("yaml"))
            donor_yaml = Path(donor_yaml_candidate)
            if not donor_yaml.exists():
                alt_path = _resolve_yaml_path(donor_yaml_candidate)
                if alt_path is not None:
                    donor_yaml = alt_path
            donor_yaml = _prefer_dataset_yaml(donor, donor_yaml)
            fallback_raw = cfg.get("fallback_names", [])
            if isinstance(fallback_raw, Sequence) and not isinstance(fallback_raw, (str, bytes)):
                fallback_seq = [str(item) for item in fallback_raw]
            else:
                fallback_seq = []
            datasets[donor] = {
                "yaml": _pretty_path(donor_yaml) if donor_yaml.exists() else donor_yaml_candidate,
                "fallback_names": fallback_seq,
            }
            datasets[donor]["names"] = _load_names_from_yaml(donor_yaml, fallback_seq)

    precomputed_pools, precomputed_canonical = compute_pools_and_canonical(
        sources,
        donor_big_names,
        label_space_mode=args.label_space,
    )
    pool_sizes = {name: len(precomputed_pools.get(name, [])) for name in schedule_names}

    date_dir = args.output_root / f"publication_{args.date_tag}"
    run_dir = date_dir / "yolo_track"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = date_dir / "raw_data"
    ckpt_dir = date_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "logs").mkdir(parents=True, exist_ok=True)
    mix_root = run_dir / "mixes"
    mix_root.mkdir(parents=True, exist_ok=True)

    resume = not args.no_resume
    start_segment, checkpoint_path = _load_resume_checkpoint(run_dir) if resume else (1, None)
    if args.dry_run:
        args.segments = min(args.segments, 2)
        args.budget_images = min(args.budget_images, 2048)
        args.batch = min(args.batch, 4)

    if checkpoint_path is not None:
        model = DummyYOLO.load(str(checkpoint_path)) if args.dry_run else build_model(str(checkpoint_path), dry_run=False)
    else:
        model = build_model(args.weights, dry_run=args.dry_run)

    policy_cfg = MaestroPolicyConfig(
        min_lr_scale=args.min_lr_scale,
        max_lr_scale=args.max_lr_scale,
        min_usage=args.min_usage,
        max_usage=args.max_usage,
    )
    method = args.method.lower()
    eta_bounds = (args.min_lr_scale, args.max_lr_scale)
    teacher_policy: Optional[TeacherPolicy] = None
    if teacher_ckpt_path is not None:
        if method != "maestro":
            raise ValueError(
                "--teacher-ckpt can only be used with --method maestro"
            )
        teacher_policy = _load_teacher_policy(teacher_ckpt_path, eta_bounds)
        teacher_meta = {
            "teacher_ckpt_path": str(teacher_ckpt_path.resolve()),
            "eta_bounds": list(eta_bounds),
            "deterministic": bool(args.teacher_deterministic),
            "descriptor_mapping": "v1:nll_mean,nll_iqr,entropy,ece=0,log_grad,cos=0,log_eff_rank=0,size=log(pool+1)",
        }
        (log_dir / "teacher_meta.json").write_text(
            json.dumps(teacher_meta, indent=2), encoding="utf-8"
        )

    policy: Optional[MaestroPolicy] = None
    scheduler = None
    if method == "maestro" and teacher_policy is None:
        policy = MaestroPolicy(policy_cfg)
    elif method != "maestro":
        default_usage = float(np.clip((args.min_usage + args.max_usage) * 0.5, 0.0, 1.0))
        scheduler = create_scheduler(
            method,
            schedule_names,
            eta_bounds,
            args.segments,
            usage=default_usage,
        )
        scheduler.start_episode({}, np.zeros((len(schedule_names), 1), dtype=np.float32))

    log_path = log_dir / "train.jsonl"
    if start_segment == 1 and log_path.exists() and not resume:
        log_path.unlink()

    budget_remaining = args.budget_images
    if start_segment > 1 and log_path.exists():
        lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            budget_remaining = int(last.get("budget_remaining", budget_remaining))

    run_config = RunConfig(
        weights=args.weights,
        segments=args.segments,
        budget_images=args.budget_images,
        base_lr=args.base_lr,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=str(args.device),
        datasets=list(datasets.keys()),
        dry_run=bool(args.dry_run),
        resume=bool(resume),
        method=args.method,
        label_space=args.label_space,
    )
    _update_metadata(date_dir, {"config": asdict(run_config)})

    wandb_run = init_wandb_run(
        f"yolo_track_{args.date_tag}_{datetime.now(UTC).strftime('%H%M%S')}",
        config={"config": asdict(run_config)},
    )

    csv_path = raw_dir / "yolo_segments.csv"
    segments_to_run = range(start_segment, args.segments + 1)
    print(f"Starting MAESTRO YOLO training from segment {start_segment} to {args.segments}")

    try:
        for segment in segments_to_run:
            probes: Dict[str, Dict[str, float]] = {}
            for name, cfg in datasets.items():
                if name not in schedule_names:
                    continue
                probes[name] = estimate_probes_with_val(
                    model, cfg["yaml"], imgsz=args.imgsz, dry_run=args.dry_run
                )
            if donor_names_in_mix:
                if probes:
                    keys = set().union(*(stats.keys() for stats in probes.values()))
                else:
                    keys = set()
                default_stats = {
                    key: float(np.mean([float(stats.get(key, 0.0)) for stats in probes.values()]))
                    for key in keys
                }
                default_stats.setdefault("loss_mean", 1.0)
                default_stats.setdefault("entropy_mean", 0.0)
                default_stats.setdefault("loss_iqr", 0.0)
                default_stats.setdefault("grad_norm_log", 0.0)
                for donor_name in donor_names_in_mix:
                    if donor_name not in probes:
                        probes[donor_name] = dict(default_stats)
            if teacher_policy is not None:
                descriptor_rows = []
                for dataset_name in schedule_names:
                    stats = probes.get(dataset_name, {})
                    pool_size = pool_sizes.get(dataset_name, 0)
                    size_log = float(np.log(float(pool_size) + 1.0))
                    descriptor_rows.append(
                        _descriptor_from_probe(stats, size_log)
                    )
                descriptor_matrix = np.stack(descriptor_rows, axis=0)
                observation = _build_teacher_observation(descriptor_matrix)
                action_np, _, _, _ = teacher_policy.act(
                    observation,
                    descriptor_matrix,
                    deterministic=bool(args.teacher_deterministic),
                )
                mixture = action_np.get("w")
                if mixture is None or np.sum(mixture) <= 0:
                    mixture = np.ones(len(schedule_names), dtype=np.float32)
                mixture = mixture.astype(np.float32)
                mixture_sum = float(np.sum(mixture))
                if mixture_sum <= 0:
                    mixture = np.ones(len(schedule_names), dtype=np.float32)
                    mixture_sum = float(len(schedule_names))
                mixture = mixture / mixture_sum
                eta_array = action_np.get("eta")
                usage_array = action_np.get("u")
                eta_scale = (
                    float(eta_array[0])
                    if eta_array is not None and len(eta_array) > 0
                    else float(np.mean(eta_bounds))
                )
                usage_raw = (
                    float(usage_array[0])
                    if usage_array is not None and len(usage_array) > 0
                    else float(np.clip((args.min_usage + args.max_usage) * 0.5, 0.0, 1.0))
                )
                usage = float(np.clip(usage_raw, args.min_usage, args.max_usage))
                usage = float(np.clip(usage, 0.0, 1.0))
                weights_schedule = {
                    name: float(mixture[i]) for i, name in enumerate(schedule_names)
                }
            elif method == "maestro":
                if policy is None:
                    raise RuntimeError("Maestro policy is not initialised")
                weights_schedule, eta_scale, usage = policy.get_action(
                    probes, budget_remaining, segment, args.segments
                )
            else:
                if scheduler is None:
                    raise RuntimeError("Baseline scheduler not initialised")
                observation = {name: np.zeros(1, dtype=np.float32) for name in schedule_names}
                descriptors = np.zeros((len(schedule_names), 1), dtype=np.float32)
                action, _, _, _ = scheduler.act(observation, descriptors)
                mixture = action.get("w")
                eta_array = action.get("eta")
                usage_array = action.get("u")
                mixture = mixture if mixture is not None else np.ones(len(schedule_names), dtype=np.float32)
                weights_schedule = {
                    name: float(mixture[i]) for i, name in enumerate(schedule_names)
                }
                eta_scale = float(eta_array[0]) if eta_array is not None else 1.0
                usage = float(usage_array[0]) if usage_array is not None else float(np.clip(args.min_usage, 0.0, 1.0))
            weights_full = {name: float(weights_schedule.get(name, 0.0)) for name in schedule_names}
            if budget_remaining <= 0:
                print("No budget remaining; stopping early.")
                break
            steps_total = max(1, int(round(usage * max(1, budget_remaining) / max(1, args.batch))))
            desired_images = max(1, steps_total * args.batch)
            num_images = min(budget_remaining, desired_images)

            mix_dir, _ = build_mixed_segment(
                out_dir=mix_root,
                segment=segment,
                sources=sources,
                donor_big_names=donor_big_names,
                weights={s.name: float(weights_schedule.get(s.name, 0.0)) for s in sources},
                total_images=num_images,
                label_space_mode=args.label_space,
                rng_seed=segment,
                precomputed_pools=precomputed_pools,
                precomputed_canonical=precomputed_canonical,
            )

            # build a small balanced val split for the mix (2% of train; >=25 per source if possible)
            build_mixed_val(mix_dir, fraction=0.02, min_per_source=25)

            manifest_lines = [
                line.strip()
                for line in (mix_dir / "train.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            actual_images = len(manifest_lines)
            if actual_images == 0:
                print(f"[segment {segment}] No samples available after mixing; skipping training")
                budget_remaining = max(0, budget_remaining - actual_images)
                continue

            effective_steps = max(1, math.ceil(actual_images / max(1, args.batch)))
            lr_segment = args.base_lr * eta_scale
            print(
                f"[segment {segment}] weights={weights_schedule} eta_scale={eta_scale:.3f} usage={usage:.3f} images={actual_images} lr={lr_segment:.5f}"
            )

            _ensure_ultralytics_model_override(model, args.weights)
            model.train(
                data=str(mix_dir / "yolo_mix.yaml"),
                imgsz=args.imgsz,
                epochs=1,
                batch=args.batch,
                device=args.device,
                workers=0,
                lr0=lr_segment,
                resume=False,  # Ultralytics resume expects a full checkpoint; we handle segment restarts ourselves.
                project=str(run_dir),
                name="exp",
                save=False,
                val=False,
                verbose=False,
            )

            per_dataset_metrics: Dict[str, Dict[str, float]] = {}
            for dataset_name, cfg in datasets.items():
                metrics = model.val(data=cfg["yaml"], imgsz=args.imgsz, device=args.device, verbose=False, save=False)
                map_value = float(getattr(metrics.box, "map", 0.0)) if hasattr(metrics, "box") else 0.0
                map50_value = float(getattr(metrics.box, "map50", 0.0)) if hasattr(metrics, "box") else 0.0
                per_dataset_metrics[dataset_name] = {"mAP": map_value, "mAP50": map50_value}

            macro_map = float(np.mean([stats["mAP"] for stats in per_dataset_metrics.values()]))
            next_budget = max(0, budget_remaining - actual_images)
            record = {
                "segment": segment,
                "weights": weights_full,
                "eta_scale": eta_scale,
                "usage": usage,
                "budget_remaining": next_budget,
                "per_dataset": per_dataset_metrics,
                "macro_mAP": macro_map,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            rows = []
            for dataset_name, stats in per_dataset_metrics.items():
                rows.append(
                    {
                        "segment": segment,
                        "dataset": dataset_name,
                        "metric": "mAP",
                        "value": stats["mAP"],
                        "macro_mAP": macro_map,
                        "eta_scale": eta_scale,
                        "usage": usage,
                        "weight": float(weights_full.get(dataset_name, 0.0)),
                    }
                )
                rows.append(
                    {
                        "segment": segment,
                        "dataset": dataset_name,
                        "metric": "mAP50",
                        "value": stats["mAP50"],
                        "macro_mAP": macro_map,
                        "eta_scale": eta_scale,
                        "usage": usage,
                        "weight": float(weights_full.get(dataset_name, 0.0)),
                    }
                )
            _append_segment_csv(csv_path, rows)

            weight_metrics = {f"weights/{name}": float(value) for name, value in weights_full.items()}
            dataset_metrics_payload = {
                f"mAP/{name}": stats["mAP"] for name, stats in per_dataset_metrics.items()
            }
            dataset_metrics_payload.update(
                {f"mAP50/{name}": stats["mAP50"] for name, stats in per_dataset_metrics.items()}
            )
            log_metrics(
                {
                    "segment": segment,
                    "steps_total": effective_steps,
                    "images_used": actual_images,
                    "macro_mAP": macro_map,
                    "eta_scale": float(eta_scale),
                    "usage": float(usage),
                    "budget_remaining": next_budget,
                    **weight_metrics,
                    **dataset_metrics_payload,
                }
            )

            ckpt_path = ckpt_dir / f"yolo_seg{segment}.pt"
            if isinstance(model, DummyYOLO):
                model.save(str(ckpt_path))
            else:
                model.save(str(ckpt_path))
            log_checkpoint(ckpt_path, ckpt_dir)

            if scheduler is not None:
                scheduler.update(
                    float(macro_map),
                    {
                        "dataset_metrics": {
                            name: {"accuracy": stats["mAP"]}
                            for name, stats in per_dataset_metrics.items()
                        }
                    },
                )

            budget_remaining = next_budget
            print(
                f"[segment {segment}] macro mAP={macro_map:.4f} budget_remaining={budget_remaining}"
            )
            if budget_remaining <= 0:
                print("Budget exhausted – stopping early.")
                break
    finally:
        wandb_run.finish()

    print(f"[✓] YOLO training complete. Logs written to {log_path}")


if __name__ == "__main__":
    main()
```



---

### Summary

- Files included: **39**
