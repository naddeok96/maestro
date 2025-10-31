"""Builders for per-segment mixed YOLO datasets."""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .label_space import canonical_from_lists, dump_canonical, id_map
from .yolo_txt import rewrite_label


@dataclass
class SourceDS:
    """Description of a YOLO-format dataset participating in the mix."""

    name: str
    images_dir: Path
    labels_dir: Path
    names: Sequence[str]


def _proportional_counts(weights: Dict[str, float], total: int) -> Dict[str, int]:
    keys = list(weights)
    if total <= 0 or not keys:
        return {k: 0 for k in keys}
    values = [max(0.0, float(weights[k])) for k in keys]
    if not any(values):
        return {k: 0 for k in keys}
    total_value = sum(values)
    values = [v / total_value for v in values]
    counts = [math.floor(v * total) for v in values]
    shortfall = total - sum(counts)
    if shortfall > 0:
        order = sorted(range(len(values)), key=lambda idx: -values[idx])
        for idx in order:
            if shortfall <= 0:
                break
            counts[idx] += 1
            shortfall -= 1
    # guarantee at least one sample for non-zero weight when feasible
    for idx, value in enumerate(values):
        if value > 0 and total >= len(values) and counts[idx] == 0:
            counts[idx] = 1
    while sum(counts) > total:
        for idx in reversed(sorted(range(len(values)), key=lambda i: -values[i])):
            if counts[idx] > 0:
                counts[idx] -= 1
                break
    return {keys[i]: counts[i] for i in range(len(keys))}


def _collect_image_pairs(source: SourceDS, mapping: Dict[int, int]) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not mapping:
        return pairs
    for image_path in source.images_dir.rglob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        rel = image_path.relative_to(source.images_dir)
        label_path = source.labels_dir / rel.with_suffix(".txt")
        if not label_path.exists():
            continue
        keep = False
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            tokens = raw.split()
            if not tokens:
                continue
            try:
                sid = int(tokens[0])
            except ValueError:
                continue
            if sid in mapping:
                keep = True
                break
        if keep:
            pairs.append((image_path, label_path))
    return pairs


def build_mixed_segment(
    out_dir: Path,
    segment: int,
    sources: List[SourceDS],
    donor_big_names: List[List[str]] | None,
    weights: Dict[str, float],
    total_images: int,
    label_space_mode: str = "overlap_only",
    rng_seed: int = 0,
) -> Tuple[Path, List[str]]:
    """Create a YOLO dataset that mixes ``sources`` proportionally to ``weights``."""

    random.seed(rng_seed)
    mix_dir = out_dir / f"mix_seg_{segment:03d}"
    (mix_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (mix_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    small_lists = [list(s.names) for s in sources]
    canonical = canonical_from_lists(small_lists, donor_big_names, mode=label_space_mode)
    dump_canonical(mix_dir / "canonical_names.json", canonical)

    pools: Dict[str, List[Tuple[Path, Path]]] = {}
    for source in sources:
        mapping = id_map(source.names, canonical)
        pools[source.name] = _collect_image_pairs(source, mapping)

    counts = _proportional_counts(weights, total_images)
    chosen: List[Tuple[str, Path, Path]] = []
    for source in sources:
        pool = pools.get(source.name, [])
        k = min(counts.get(source.name, 0), len(pool))
        if k <= 0:
            continue
        chosen.extend((source.name, img, lbl) for img, lbl in random.sample(pool, k))

    random.shuffle(chosen)
    overrides = {source.name: id_map(source.names, canonical) for source in sources}
    manifest: List[str] = []
    for dataset_name, image_path, label_path in chosen:
        rel = Path(dataset_name) / image_path.name
        dst_img = mix_dir / "images" / "train" / rel
        dst_lbl = mix_dir / "labels" / "train" / rel.with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst_img.symlink_to(image_path.resolve())
        except Exception:
            shutil.copy2(image_path, dst_img)
        remapped = rewrite_label(label_path, dst_lbl, overrides[dataset_name])
        if not remapped:
            dst_img.unlink(missing_ok=True)
            dst_lbl.unlink(missing_ok=True)
            continue
        manifest.append(str(dst_img.resolve()))

    random.shuffle(manifest)
    (mix_dir / "train.txt").write_text("\n".join(manifest), encoding="utf-8")

    yaml_lines = [
        f"path: {mix_dir.as_posix()}",
        f"train: {(mix_dir / 'train.txt').as_posix()}",
        f"val: {(mix_dir / 'train.txt').as_posix()}  # reuse train for validation",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(canonical))
    (mix_dir / "yolo_mix.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    return mix_dir, canonical
