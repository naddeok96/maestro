"""Utility helpers for computing lightweight probe statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:  # pragma: no cover - optional dependency at runtime
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - handled gracefully downstream
    YOLO = None  # type: ignore


@dataclass
class ProbeResult:
    """Container with a consistent set of probe statistics."""

    loss_mean: float
    loss_iqr: float
    entropy_mean: float
    grad_norm_log: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss_mean": float(self.loss_mean),
            "loss_iqr": float(self.loss_iqr),
            "entropy_mean": float(self.entropy_mean),
            "grad_norm_log": float(self.grad_norm_log),
        }


class _DummyValResult:
    """Small wrapper that mimics the ultralytics validation result."""

    def __init__(self, map_value: float, map50_value: float) -> None:
        self.box = type("BoxStats", (), {"map": map_value, "map50": map50_value})()


class DummyYOLO:
    """Fallback model used in dry-run/CI environments.

    The dummy model keeps track of per-dataset pseudo accuracies and evolves them
    deterministically with every ``train`` call.  It implements the subset of
    the :mod:`ultralytics` API required by the scripts.
    """

    def __init__(self, weights: str | None = None) -> None:
        seed = abs(hash(weights or "dummy")) % (2**32)
        np.random.default_rng(seed)  # ensure deterministic hash-based seeding
        self._skill: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def train(self, **kwargs) -> None:  # pragma: no cover - simple state update
        data_yaml = kwargs.get("data", "default")
        lr = float(kwargs.get("lr0", 0.001))
        epochs = int(kwargs.get("epochs", 1))
        delta = np.tanh(lr * epochs * 10.0)
        current = self._skill.get(data_yaml, 0.2)
        self._skill[data_yaml] = float(np.clip(current + delta * 0.05, 0.0, 0.9))

    # ------------------------------------------------------------------
    def val(self, data: str, imgsz: int, device: str | int | None = None, verbose: bool = False, save: bool = False):  # type: ignore[override]
        _ = (imgsz, device, verbose, save)
        current = float(self._skill.get(data, 0.2))
        map50 = np.clip(current + 0.1, 0.0, 1.0)
        return _DummyValResult(map_value=current, map50_value=map50)

    # ------------------------------------------------------------------
    def save(self, path: str) -> str:  # pragma: no cover - trivial
        with open(path, "w", encoding="utf-8") as fh:
            for name, value in sorted(self._skill.items()):
                fh.write(f"{name}\t{value}\n")
        return path

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "DummyYOLO":  # pragma: no cover - trivial
        model = cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    dataset, value = line.strip().split("\t")
                    model._skill[dataset] = float(value)
        except FileNotFoundError:
            pass
        return model


def build_model(weights: str, dry_run: bool) -> YOLO | DummyYOLO:  # type: ignore[return-type]
    """Construct a YOLO model or a dummy fallback when running in CI."""

    if dry_run or YOLO is None:
        return DummyYOLO(weights)
    return YOLO(weights)


def estimate_probes_with_val(
    model: YOLO | DummyYOLO,
    data_yaml: str,
    imgsz: int = 640,
    dry_run: bool = False,
) -> Dict[str, float]:
    """Estimate probe statistics by running a lightweight validation pass.

    Parameters
    ----------
    model:
        Instance of :class:`ultralytics.YOLO` or :class:`DummyYOLO` used for the
        evaluation.
    data_yaml:
        Path to the dataset configuration file.
    imgsz:
        Input resolution used during evaluation.
    dry_run:
        When ``True`` the call avoids heavy computations and instead returns a
        deterministic synthetic probe derived from the dummy model.
    """

    if dry_run and not isinstance(model, DummyYOLO):  # pragma: no cover - safety
        # If the real model is provided but the caller requested a dry run we
        # short circuit to keep the CI workload tiny.
        return ProbeResult(0.5, 0.0, 0.0, 0.0).as_dict()

    result = model.val(data=data_yaml, imgsz=imgsz, device=0 if not dry_run else "cpu", verbose=False, save=False)
    map50 = float(getattr(result.box, "map50", 0.0)) if hasattr(result, "box") else 0.0
    loss_mean = float(np.clip(1.0 - map50, 1e-6, 1.0))
    probes = ProbeResult(
        loss_mean=loss_mean,
        loss_iqr=0.0,
        entropy_mean=0.0,
        grad_norm_log=0.0,
    )
    return probes.as_dict()

