"""Segmented multi-dataset YOLO training orchestrated by MAESTRO."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from maestro.baselines import create_scheduler
from maestro.policy import MaestroPolicy, MaestroPolicyConfig
from maestro.probes import DummyYOLO, build_model, estimate_probes_with_val
from maestro.utils.wandb import init_wandb_run, log_checkpoint, log_metrics
from maestro.yolo.mix_builder import SourceDS, build_mixed_segment, build_mixed_val


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
    resolved = {}
    for name in datasets:
        if name not in DEFAULT_DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Known options: {sorted(DEFAULT_DATASETS)}")
        cfg = DEFAULT_DATASETS[name]
        yaml_path = cfg.get("yaml")
        yaml_str = str(yaml_path) if isinstance(yaml_path, Path) else str(yaml_path)
        resolved[name] = {
            "yaml": yaml_str,
            "fallback_names": list(cfg.get("fallback_names", [])),
        }
    return resolved


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

    base = data.get("path", "")
    base_path = Path(base) if base else yaml_path.parent
    if not base_path.is_absolute():
        base_path = (yaml_path.parent / base_path).resolve()
    else:
        base_path = base_path.resolve()

    train_entry = data.get("train", "images/train")
    train_path = Path(train_entry)
    if not train_path.is_absolute():
        train_path = (base_path / train_path).resolve()
    else:
        train_path = train_path.resolve()

    if not train_path.exists() or not train_path.is_dir():
        return None, None

    try:
        rel = train_path.relative_to(base_path)
    except ValueError:
        rel = Path()

    if rel.parts and rel.parts[0] == "images":
        labels_rel = Path("labels", *rel.parts[1:])
        labels_dir = (base_path / labels_rel).resolve()
    else:
        labels_dir = train_path.parent / "labels"

    return train_path, labels_dir


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
    args = parser.parse_args()

    datasets = _resolve_dataset_names(args.datasets)
    dataset_names = list(datasets.keys())

    source_entries: Dict[str, SourceDS] = {}
    for name in dataset_names:
        cfg = datasets[name]
        yaml_path = Path(str(cfg["yaml"]))
        fallback_raw = cfg.get("fallback_names", [])
        fallback_seq: Sequence[str]
        if isinstance(fallback_raw, Sequence) and not isinstance(fallback_raw, (str, bytes)):
            fallback_seq = [str(item) for item in fallback_raw]
        else:
            fallback_seq = []
        names_list = _load_names_from_yaml(yaml_path, fallback_seq)
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
            donor_yaml = Path(str(cfg.get("yaml")))
            fallback = cfg.get("fallback_names", [])
            donor_names = _load_names_from_yaml(donor_yaml, fallback)
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
            donor_yaml = Path(str(cfg.get("yaml")))
            fallback_raw = cfg.get("fallback_names", [])
            if isinstance(fallback_raw, Sequence) and not isinstance(fallback_raw, (str, bytes)):
                fallback_seq = [str(item) for item in fallback_raw]
            else:
                fallback_seq = []
            datasets[donor] = {
                "yaml": str(donor_yaml),
                "fallback_names": fallback_seq,
            }
            datasets[donor]["names"] = _load_names_from_yaml(donor_yaml, fallback_seq)
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
    policy: Optional[MaestroPolicy] = None
    scheduler = None
    if method == "maestro":
        policy = MaestroPolicy(policy_cfg)
    else:
        default_usage = float(np.clip((args.min_usage + args.max_usage) * 0.5, 0.0, 1.0))
        scheduler = create_scheduler(
            method,
            schedule_names,
            (args.min_lr_scale, args.max_lr_scale),
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
            if method == "maestro":
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

            model.train(
                data=str(mix_dir / "yolo_mix.yaml"),
                imgsz=args.imgsz,
                epochs=1,
                batch=args.batch,
                device=args.device,
                workers=0,
                lr0=lr_segment,
                resume=True,
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

