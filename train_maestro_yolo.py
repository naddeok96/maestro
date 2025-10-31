"""Segmented multi-dataset YOLO training orchestrated by MAESTRO."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from maestro.policy import MaestroPolicy, MaestroPolicyConfig
from maestro.probes import DummyYOLO, build_model, estimate_probes_with_val
from maestro.utils.wandb import init_wandb_run, log_checkpoint, log_metrics


DEFAULT_DATASETS: Dict[str, Dict[str, str]] = {
    "coco": {"yaml": "configs/datasets/coco.yaml"},
    "lvis": {"yaml": "configs/datasets/lvis.yaml"},
    "voc": {"yaml": "configs/datasets/voc.yaml"},
    "target": {"yaml": "configs/datasets/target.yaml"},
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


def _distribute_steps(weights: Dict[str, float], total_steps: int) -> Dict[str, int]:
    keys = list(weights.keys())
    if total_steps <= 0:
        return {k: 0 for k in keys}
    raw = np.array([max(0.0, weights[k]) for k in keys], dtype=np.float64)
    if raw.sum() <= 0:
        raw[:] = 1.0
    raw /= raw.sum()
    steps = np.floor(raw * total_steps).astype(int)
    shortfall = total_steps - int(steps.sum())
    if shortfall > 0:
        order = np.argsort(-raw)
        for idx in order[:shortfall]:
            steps[idx] += 1
    # Ensure that non-zero weights receive at least one step
    for i, key in enumerate(keys):
        if raw[i] > 0 and steps[i] == 0:
            steps[i] = 1
    # Normalise so we do not exceed the budget
    excess = int(steps.sum()) - total_steps
    if excess > 0:
        order = np.argsort(raw)
        for idx in order:
            if excess <= 0:
                break
            if steps[idx] > 0:
                steps[idx] -= 1
                excess -= 1
    return {k: int(steps[i]) for i, k in enumerate(keys)}


def _resolve_dataset_names(datasets: Iterable[str]) -> Dict[str, Dict[str, str]]:
    resolved = {}
    for name in datasets:
        if name not in DEFAULT_DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Known options: {sorted(DEFAULT_DATASETS)}")
        resolved[name] = DEFAULT_DATASETS[name]
    return resolved


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
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS.keys()), help="Datasets to include")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root output directory")
    parser.add_argument("--date-tag", default=datetime.now(UTC).strftime("%Y%m%d"), help="Date tag for outputs")
    parser.add_argument("--no-resume", action="store_true", help="Start a fresh run even if logs exist")
    parser.add_argument("--dry-run", action="store_true", help="Run a fast deterministic CI-friendly simulation")
    parser.add_argument("--min-usage", type=float, default=0.2, help="Lower bound for usage fraction")
    parser.add_argument("--max-usage", type=float, default=0.7, help="Upper bound for usage fraction")
    parser.add_argument("--min-lr-scale", type=float, default=0.5)
    parser.add_argument("--max-lr-scale", type=float, default=1.5)
    args = parser.parse_args()

    datasets = _resolve_dataset_names(args.datasets)
    date_dir = args.output_root / f"publication_{args.date_tag}"
    run_dir = date_dir / "yolo_track"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = date_dir / "raw_data"
    ckpt_dir = date_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "logs").mkdir(parents=True, exist_ok=True)

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
    policy = MaestroPolicy(policy_cfg)

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
            probes = {
                name: estimate_probes_with_val(model, cfg["yaml"], imgsz=args.imgsz, dry_run=args.dry_run)
                for name, cfg in datasets.items()
            }
            weights, eta_scale, usage = policy.get_action(probes, budget_remaining, segment, args.segments)
            if budget_remaining <= 0:
                print("No budget remaining; stopping early.")
                break
            steps_total = max(1, int(usage * max(1, budget_remaining) / max(args.batch, 1)))
            steps_by_dataset = _distribute_steps(weights, steps_total)
            lr_segment = args.base_lr * eta_scale
            print(
                f"[segment {segment}] weights={weights} eta_scale={eta_scale:.3f} usage={usage:.3f} steps={steps_total} lr={lr_segment:.5f}"
            )

            for dataset_name, steps in steps_by_dataset.items():
                if steps <= 0:
                    continue
                yaml_path = datasets[dataset_name]["yaml"]
                print(f"  -> training {dataset_name} for {steps} micro-epochs")
                for _ in range(steps):
                    overrides = dict(
                        data=yaml_path,
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
                    model.train(**overrides)

            per_dataset_metrics: Dict[str, Dict[str, float]] = {}
            for dataset_name, cfg in datasets.items():
                metrics = model.val(data=cfg["yaml"], imgsz=args.imgsz, device=args.device, verbose=False, save=False)
                map_value = float(getattr(metrics.box, "map", 0.0)) if hasattr(metrics, "box") else 0.0
                map50_value = float(getattr(metrics.box, "map50", 0.0)) if hasattr(metrics, "box") else 0.0
                per_dataset_metrics[dataset_name] = {"mAP": map_value, "mAP50": map50_value}

            macro_map = float(np.mean([stats["mAP"] for stats in per_dataset_metrics.values()]))
            next_budget = max(0, budget_remaining - steps_total * args.batch)
            record = {
                "segment": segment,
                "weights": weights,
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
                    }
                )
                rows.append(
                    {
                        "segment": segment,
                        "dataset": dataset_name,
                        "metric": "mAP50",
                        "value": stats["mAP50"],
                        "macro_mAP": macro_map,
                    }
                )
            _append_segment_csv(csv_path, rows)

            weight_metrics = {f"weights/{name}": float(value) for name, value in weights.items()}
            dataset_metrics_payload = {
                f"mAP/{name}": stats["mAP"] for name, stats in per_dataset_metrics.items()
            }
            dataset_metrics_payload.update(
                {f"mAP50/{name}": stats["mAP50"] for name, stats in per_dataset_metrics.items()}
            )
            log_metrics(
                {
                    "segment": segment,
                    "steps_total": steps_total,
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

            budget_remaining = next_budget
            print(f"[segment {segment}] macro mAP={macro_map:.4f} budget_remaining={budget_remaining}")
            if budget_remaining <= 0:
                print("Budget exhausted – stopping early.")
                break
    finally:
        wandb_run.finish()

    print(f"[✓] YOLO training complete. Logs written to {log_path}")


if __name__ == "__main__":
    main()

