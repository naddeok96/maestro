"""Helper utilities for Weights & Biases logging."""

from __future__ import annotations

import os
import numbers
from pathlib import Path
from typing import Any, Mapping, Optional

try:  # pragma: no cover - import guard for optional dependency
    import wandb
except ImportError:  # pragma: no cover - fallback stub for environments without wandb
    class _DummySettings:
        def __init__(self, **_: Any) -> None:
            pass

    class _DummyRun:
        def __init__(self, mode: str = "disabled") -> None:
            self.mode = mode

        def finish(self) -> None:  # noqa: D401 - interface compatibility
            """End the run (no-op)."""

    class _DummyWandb:
        def __init__(self) -> None:
            self.run: Optional[_DummyRun] = None
            self.Settings = _DummySettings

        def init(self, **kwargs: Any) -> _DummyRun:
            mode = kwargs.get("mode", "disabled")
            self.run = _DummyRun(mode=mode)
            return self.run

        def log(self, *_: Any, **__: Any) -> None:
            pass

        def save(self, *_: Any, **__: Any) -> None:
            pass

    wandb = _DummyWandb()  # type: ignore[assignment]


def _resolve_mode() -> str:
    """Resolve the W&B run mode based on environment variables."""
    if wandb.run is not None:
        return wandb.run.mode
    mode = os.environ.get("WANDB_MODE")
    if mode:
        return mode
    api_key = os.environ.get("WANDB_API_KEY")
    return "online" if api_key else "offline"


def init_wandb_run(
    name: str,
    *,
    project: str = "maestro",
    config: Optional[Mapping[str, Any]] = None,
    tags: Optional[list[str]] = None,
    **kwargs: Any,
) -> wandb.sdk.wandb_run.Run:
    """Initialise a W&B run with sensible defaults."""
    mode = _resolve_mode()
    settings = kwargs.pop("settings", None)
    if settings is None:
        settings = wandb.Settings(start_method="thread")
    run = wandb.init(
        project=project,
        name=name,
        config=dict(config or {}),
        mode=mode,
        tags=tags,
        settings=settings,
        **kwargs,
    )
    return run


def log_metrics(metrics: Mapping[str, Any]) -> None:
    """Log metrics to the active W&B run, if any."""
    if not metrics:
        return
    if wandb.run is None:
        return
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, numbers.Number)
    }
    if numeric_metrics:
        wandb.log(numeric_metrics)


def log_checkpoint(path: Path, base_path: Path | None = None) -> None:
    """Upload a checkpoint to the active W&B run, if possible."""
    if wandb.run is None or not path.exists():
        return
    try:
        if base_path is not None:
            wandb.save(str(path), base_path=str(base_path))
        else:
            wandb.save(str(path))
    except Exception:
        # Ignore failures so that training does not crash in offline mode.
        pass
