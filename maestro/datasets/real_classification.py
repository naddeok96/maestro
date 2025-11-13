"""Realistic image classification datasets for MAESTRO."""

from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib import error as urllib_error, request as urllib_request

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


class TinyImageNetDataset(Dataset):
    """Minimal Tiny ImageNet loader with train/val splits."""

    URL = "https://tiny-imagenet-200.s3.amazonaws.com/tiny-imagenet-200.zip"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = False,
        transform=None,
        target_transform=None,
    ) -> None:
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform
        self.target_transform = target_transform
        self.base_dir = self.root / "tiny-imagenet-200"
        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError(
                f"Tiny ImageNet not found under {self.base_dir}. "
                "Run scripts/download_datasets.sh or set MAESTRO_ALLOW_DATASET_DOWNLOAD=1."
            )
        self.classes = self._load_wnids()
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.classes)}
        self.samples = (
            self._load_train_samples()
            if self.train
            else self._load_val_samples()
        )
        if not self.samples:
            raise RuntimeError(
                f"No samples found for Tiny ImageNet split={'train' if self.train else 'val'}"
            )
        self.targets = [label for _, label in self.samples]

    def _check_exists(self) -> bool:
        return (self.base_dir / "wnids.txt").is_file()

    def _download(self) -> None:
        if self._check_exists():
            return
        url = os.environ.get("TINY_IMAGENET_URL", self.URL)
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / "tiny-imagenet-200.zip"
        tmp_path = archive_path.with_suffix(".part")
        try:
            with urllib_request.urlopen(url) as response, open(tmp_path, "wb") as fh:
                shutil.copyfileobj(response, fh)
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Failed to download Tiny ImageNet from {url}") from exc
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(self.root)
        except zipfile.BadZipFile as exc:
            raise RuntimeError("Tiny ImageNet archive is corrupted.") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_wnids(self) -> List[str]:
        wnids_path = self.base_dir / "wnids.txt"
        with wnids_path.open("r", encoding="utf-8") as handle:
            wnids = [line.strip() for line in handle if line.strip()]
        if not wnids:
            raise RuntimeError("Tiny ImageNet wnids.txt is empty.")
        return wnids

    def _load_train_samples(self) -> List[Tuple[Path, int]]:
        train_root = self.base_dir / "train"
        samples: List[Tuple[Path, int]] = []
        for wnid in self.classes:
            images_dir = train_root / wnid / "images"
            if not images_dir.is_dir():
                continue
            for img_path in sorted(images_dir.glob("*.JPEG")):
                samples.append((img_path, self.class_to_idx[wnid]))
        return samples

    def _load_val_samples(self) -> List[Tuple[Path, int]]:
        val_root = self.base_dir / "val"
        ann_path = val_root / "val_annotations.txt"
        images_dir = val_root / "images"
        if not ann_path.is_file():
            raise RuntimeError("Tiny ImageNet val annotations missing.")
        annotations: Dict[str, str] = {}
        with ann_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                annotations[parts[0]] = parts[1]
        samples: List[Tuple[Path, int]] = []
        for filename, wnid in sorted(annotations.items()):
            img_path = images_dir / filename
            if not img_path.is_file():
                continue
            label = self.class_to_idx.get(wnid)
            if label is None:
                continue
            samples.append((img_path, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


_DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "cifar10": {
        "cls": datasets.CIFAR10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "image_size": 32,
    },
    "cifar100": {
        "cls": datasets.CIFAR100,
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "channels": 3,
        "image_size": 32,
    },
    "tiny_imagenet": {
        "cls": TinyImageNetDataset,
        "num_classes": 200,
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2302, 0.2265, 0.2262),
        "channels": 3,
        "image_size": 64,
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
    if dataset_kind in ("cifar10", "cifar100"):
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BILINEAR)
        pad = max(1, image_size // 8)
        padded = TF.pad(image, padding=pad, fill=0)
        max_offset = max(1, pad * 2)
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
        stats_key = dataset_kind
        mean = torch.tensor(_DATASET_REGISTRY[stats_key]["mean"]).view(-1, 1, 1)
        std = torch.tensor(_DATASET_REGISTRY[stats_key]["std"]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    if dataset_kind == "tiny_imagenet":
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.BILINEAR)
        scale = float(rng.uniform(0.9, 1.2) * (1.0 + noise * 0.05))
        scaled = max(image_size, min(int(round(image_size * scale)), image_size * 2))
        if scaled != image_size:
            image = image.resize((scaled, scaled), Image.BILINEAR)
        if scaled > image_size:
            max_offset = scaled - image_size
            top = int(rng.integers(0, max_offset + 1))
            left = int(rng.integers(0, max_offset + 1))
            image = image.crop((left, top, left + image_size, top + image_size))
        if rng.random() < 0.5:
            image = TF.hflip(image)
        brightness = 1.0 + rng.uniform(-0.25, 0.25) * (1.0 + noise)
        contrast = 1.0 + rng.uniform(-0.25, 0.25) * (1.0 + noise)
        saturation = 1.0 + rng.uniform(-0.2, 0.2) * (1.0 + noise)
        hue = rng.uniform(-0.04, 0.04) * (1.0 + noise * 0.5)
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)
        image = TF.adjust_hue(image, hue)
        tensor = TF.to_tensor(image)
        mean = torch.tensor(_DATASET_REGISTRY["tiny_imagenet"]["mean"]).view(-1, 1, 1)
        std = torch.tensor(_DATASET_REGISTRY["tiny_imagenet"]["std"]).view(-1, 1, 1)
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
