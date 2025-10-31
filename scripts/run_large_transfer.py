#!/usr/bin/env python
"""Large-model transfer experiment controlled by the MAESTRO policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from maestro.policy import MaestroPolicy, MaestroPolicyConfig


DEFAULT_DATASETS: Dict[str, Dict[str, str]] = {
    "coco": {"cfg": "configs/datasets/coco.yaml"},
    "lvis": {"cfg": "configs/datasets/lvis.yaml"},
    "crowdhuman": {"cfg": "configs/datasets/crowdhuman.yaml"},
    "target": {"cfg": "configs/datasets/target.yaml"},
}


@dataclass
class TransferConfig:
    weights: str
    segments: int
    budget_steps: int
    base_lr: float
    datasets: Iterable[str]
    dry_run: bool


class ViTDetector:
    """Toy ViT-style detector used to provide a runnable experiment stub."""

    def __init__(self, weights: str, datasets: Iterable[str]) -> None:
        self.weights = weights
        seed = abs(hash(weights)) % (2**32)
        rng = np.random.default_rng(seed)
        self._scores = {name: rng.uniform(0.2, 0.4) for name in datasets}

    def train_steps(self, dataset: str, steps: int, lr: float) -> None:
        steps = max(0, int(steps))
        if steps == 0:
            return
        current = self._scores.get(dataset, 0.2)
        improvement = np.tanh(steps * lr * 1e-3)
        self._scores[dataset] = float(np.clip(current + improvement * 0.2, 0.0, 0.95))

    def evaluate(self, dataset: str) -> Dict[str, float]:
        score = float(self._scores.get(dataset, 0.2))
        return {"mAP": score, "mAP50": min(1.0, score + 0.05)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"weights": self.weights, "scores": self._scores}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, datasets: Iterable[str]) -> "ViTDetector":
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = cls(payload.get("weights", "pretrained"), datasets)
        model._scores.update(payload.get("scores", {}))
        return model


def _resolve_datasets(names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    resolved = {}
    for name in names:
        if name not in DEFAULT_DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Options: {sorted(DEFAULT_DATASETS)}")
        resolved[name] = DEFAULT_DATASETS[name]
    return resolved


def _update_metadata(out_dir: Path, record: Dict[str, object]) -> None:
    path = out_dir / "metadata.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tracks": {}}
    data.setdefault("tracks", {})["vit_transfer"] = record
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _append_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAESTRO-controlled ViT transfer experiment")
    parser.add_argument("--weights", default="vit_detector_pretrained.pth")
    parser.add_argument("--segments", type=int, default=12)
    parser.add_argument("--budget-steps", type=int, default=60_000)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS.keys()))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--date-tag", default=datetime.utcnow().strftime("%Y%m%d"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = _resolve_datasets(args.datasets)
    out_dir = args.output_root / f"publication_{args.date_tag}"
    run_dir = out_dir / "vit_transfer"
    log_path = run_dir / "logs" / "transfer.jsonl"
    ckpt_path = out_dir / "checkpoints" / "vit_transfer.json"
    raw_csv = out_dir / "raw_data" / "vit_transfer.csv"

    if args.dry_run:
        args.segments = min(args.segments, 3)
        args.budget_steps = min(args.budget_steps, 2_000)

    try:
        model = ViTDetector.load(ckpt_path, datasets.keys())
    except FileNotFoundError:
        model = ViTDetector(args.weights, datasets.keys())

    policy_cfg = MaestroPolicyConfig()
    policy = MaestroPolicy(policy_cfg)

    config = TransferConfig(
        weights=args.weights,
        segments=args.segments,
        budget_steps=args.budget_steps,
        base_lr=args.base_lr,
        datasets=list(datasets.keys()),
        dry_run=bool(args.dry_run),
    )
    _update_metadata(out_dir, {"config": asdict(config)})

    budget = args.budget_steps
    for segment in range(1, args.segments + 1):
        if budget <= 0:
            break
        probes = {name: {"loss_mean": 1.0 - model.evaluate(name)["mAP50"], "entropy_mean": 0.0} for name in datasets}
        weights, eta_scale, usage = policy.get_action(probes, budget, segment, args.segments)
        steps_this_segment = max(1, int(usage * budget))
        budget = max(0, budget - steps_this_segment)
        weights_sum = sum(weights.values()) or 1.0
        norm_weights = {k: v / weights_sum for k, v in weights.items()}
        steps_per_dataset = {k: max(1, int(round(steps_this_segment * norm_weights[k]))) for k in norm_weights}
        lr = args.base_lr * eta_scale

        for name, steps in steps_per_dataset.items():
            model.train_steps(name, steps, lr)

        per_dataset = {name: model.evaluate(name) for name in datasets}
        macro_map = float(np.mean([metrics["mAP"] for metrics in per_dataset.values()]))
        record = {
            "segment": segment,
            "weights": norm_weights,
            "eta_scale": eta_scale,
            "usage": usage,
            "budget_remaining": budget,
            "per_dataset": per_dataset,
            "macro_mAP": macro_map,
        }
        _append_jsonl(log_path, record)
        rows = []
        for name, metrics in per_dataset.items():
            rows.append({
                "segment": segment,
                "dataset": name,
                "metric": "mAP",
                "value": metrics["mAP"],
                "macro_mAP": macro_map,
            })
            rows.append({
                "segment": segment,
                "dataset": name,
                "metric": "mAP50",
                "value": metrics["mAP50"],
                "macro_mAP": macro_map,
            })
        _append_csv(raw_csv, rows)

    model.save(ckpt_path)
    print(f"[✓] Large-model transfer complete. Logs: {log_path}")


if __name__ == "__main__":
    main()

