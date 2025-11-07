"""Builders for per-segment mixed YOLO datasets."""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def compute_pools_and_canonical(
    sources: List[SourceDS],
    donor_big_names: Optional[List[List[str]]] = None,
    *,
    label_space_mode: str = "overlap_only",
) -> Tuple[Dict[str, List[Tuple[Path, Path]]], List[str]]:
    """Pre-compute descriptor pools and canonical labels for ``sources``."""

    small_lists = [list(s.names) for s in sources]
    canonical = canonical_from_lists(small_lists, donor_big_names, mode=label_space_mode)
    pools: Dict[str, List[Tuple[Path, Path]]] = {}
    for source in sources:
        mapping = id_map(source.names, canonical)
        pools[source.name] = _collect_image_pairs(source, mapping)
    return pools, canonical


def build_mixed_segment(
    out_dir: Path,
    segment: int,
    sources: List[SourceDS],
    donor_big_names: List[List[str]] | None,
    weights: Dict[str, float],
    total_images: int,
    label_space_mode: str = "overlap_only",
    rng_seed: int = 0,
    *,
    precomputed_pools: Optional[Dict[str, List[Tuple[Path, Path]]]] = None,
    precomputed_canonical: Optional[List[str]] = None,
) -> Tuple[Path, List[str]]:
    """Create a YOLO dataset that mixes ``sources`` proportionally to ``weights``."""

    random.seed(rng_seed)
    mix_dir = out_dir / f"mix_seg_{segment:03d}"
    (mix_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (mix_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    if precomputed_pools is None or precomputed_canonical is None:
        pools, canonical = compute_pools_and_canonical(
            sources, donor_big_names, label_space_mode=label_space_mode
        )
    else:
        pools = precomputed_pools
        canonical = list(precomputed_canonical)

    dump_canonical(mix_dir / "canonical_names.json", canonical)

    counts = _proportional_counts(weights, total_images)
    chosen: List[Tuple[str, Path, Path]] = []
    for source in sources:
        pool = pools.get(source.name, [])
        k = min(counts.get(source.name, 0), len(pool))
        if k <= 0:
            continue
        if k < counts.get(source.name, 0):
            print(
                f"[mix] truncating '{source.name}' from {counts.get(source.name, 0)} to {k} due to pool size={len(pool)}"
            )
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


def build_mixed_val(
    mix_dir: Path,
    fraction: float = 0.02,
    min_per_source: int = 25,
) -> Path:
    """
    Create a balanced val.txt for the mixed dataset at `mix_dir`.
    - Reads images from mix_dir/images/train/<source_name>/*.*
    - Proportionally samples ~`fraction` of train, ensuring at least `min_per_source`
      per present source if feasible.
    - Writes mix_dir/val.txt and updates yolo_mix.yaml to point to it.

    Returns:
        Path to the written val.txt
    """

    train_list = (mix_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    train_list = [p.strip() for p in train_list if p.strip()]
    if not train_list:
        # nothing to do; keep val pointing to train.txt
        return mix_dir / "train.txt"

    # group by source name (first directory under images/train/)
    # expected: .../mix_seg_XXX/images/train/<source>/<filename>
    from collections import defaultdict
    import random

    by_source: dict[str, list[str]] = defaultdict(list)
    root_images = (mix_dir / "images" / "train").resolve()
    for abs_path in train_list:
        try:
            rel = Path(abs_path).resolve().relative_to(root_images)
        except Exception:
            rel = Path(abs_path).name
            by_source["_unknown"].append(abs_path)
            continue
        src = rel.parts[0] if len(rel.parts) >= 2 else "_unknown"
        by_source[src].append(abs_path)

    total_val = max(1, int(round(len(train_list) * fraction)))
    sizes = {src: len(lst) for src, lst in by_source.items()}
    total_train = sum(sizes.values()) or 1
    alloc = {src: int((sizes[src] / total_train) * total_val) for src in sizes}
    for src in sizes:
        if sizes[src] > 0:
            alloc[src] = max(alloc[src], min(min_per_source, sizes[src]))
    diff = total_val - sum(alloc.values())
    if diff > 0:
        for src in sorted(sizes, key=lambda s: -sizes[s]):
            take = min(diff, max(0, sizes[src] - alloc[src]))
            alloc[src] += take
            diff -= take
            if diff == 0:
                break
    elif diff < 0:
        for src in sorted(alloc, key=lambda s: -alloc[s]):
            cut = min(-diff, max(0, alloc[src] - 1))
            alloc[src] -= cut
            diff += cut
            if diff == 0:
                break

    random.seed(hash((str(mix_dir), total_val)) & 0xFFFF)
    val_paths: list[str] = []
    for src, lst in by_source.items():
        k = min(alloc.get(src, 0), len(lst))
        if k > 0:
            val_paths.extend(random.sample(lst, k))

    val_txt = mix_dir / "val.txt"
    val_txt.write_text("\n".join(val_paths), encoding="utf-8")

    yml = mix_dir / "yolo_mix.yaml"
    if yml.exists():
        lines = yml.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith("val:"):
                new_lines.append(f"val: {val_txt.as_posix()}")
            else:
                new_lines.append(line)
        yml.write_text("\n".join(new_lines), encoding="utf-8")

    return val_txt
