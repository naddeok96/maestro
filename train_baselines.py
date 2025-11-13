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
