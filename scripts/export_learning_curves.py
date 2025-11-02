#!/usr/bin/env python
"""Export comparative learning curves into a consolidated CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load a metrics.csv file and ensure required columns are present."""
    frame = pd.read_csv(metrics_path)
    if "seed" not in frame.columns:
        frame["seed"] = metrics_path.parent.name
    frame = frame.sort_values(["seed", "episode"]) if "episode" in frame.columns else frame
    if "num_steps" not in frame.columns:
        raise ValueError(f"{metrics_path} is missing 'num_steps' column")
    frame["step"] = frame.groupby("seed")["num_steps"].cumsum()
    return frame


def _derive_task(metrics_path: Path, frame: pd.DataFrame) -> str:
    if "task" in frame.columns and not frame["task"].isna().all():
        return str(frame["task"].dropna().iloc[0])
    # Fallback: assume directory structure .../<task>/<method>/seed_x/metrics.csv
    if len(metrics_path.parents) >= 3:
        return metrics_path.parents[2].name
    return metrics_path.parent.name


def _derive_method(metrics_path: Path, frame: pd.DataFrame) -> str:
    if "method" in frame.columns and not frame["method"].isna().all():
        return str(frame["method"].dropna().iloc[0])
    # parent of seed_*/ directory
    if len(metrics_path.parents) >= 2:
        return metrics_path.parents[1].name
    return metrics_path.parent.name


def collect_learning_curves(root: Path) -> List[Dict[str, object]]:
    """Collect learning curve rows from all seed metrics under root."""
    rows: List[Dict[str, object]] = []
    for metrics_path in sorted(root.rglob("seed_*/metrics.csv")):
        try:
            frame = _load_metrics(metrics_path)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        task = _derive_task(metrics_path, frame)
        method = _derive_method(metrics_path, frame)
        metric_columns = []
        if "macro_accuracy" in frame.columns:
            metric_columns.append(("macro_accuracy", "macro_acc"))
        if "macro_mAP" in frame.columns:
            metric_columns.append(("macro_mAP", "macro_mAP"))
        if not metric_columns:
            continue
        for _, row in frame.iterrows():
            for column, metric_name in metric_columns:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "step": row["step"],
                        "metric": metric_name,
                        "value": row[column],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export comparative learning curves")
    parser.add_argument(
        "--comparative-root",
        default="outputs/comparative_plots",
        help="Root directory containing comparative plot runs",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory where learning_curves.csv will be written",
    )
    args = parser.parse_args()

    root = Path(args.comparative_root)
    if not root.exists():
        return 0

    rows = collect_learning_curves(root)
    if not rows:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "learning_curves.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
