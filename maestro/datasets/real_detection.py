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
