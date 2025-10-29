"""Logging helpers for MAESTRO runs."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class MetricsLogger:
    output_dir: Path
    csv_filename: str = "metrics.csv"
    json_filename: str = "metrics.json"
    csv_fieldnames: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / self.csv_filename
        self.json_path = self.output_dir / self.json_filename
        self.rows: list[Dict[str, float]] = []

    def log_row(self, row: Dict[str, float]) -> None:
        if self.csv_fieldnames is None:
            self.csv_fieldnames = list(row.keys())
        self.rows.append(row)
        with self.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fieldnames)
            if self.csv_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    def flush_json(self) -> None:
        with self.json_path.open("w") as handle:
            json.dump(self.rows, handle, indent=2)


@dataclass
class RunPaths:
    base: Path
    run_id: str

    def resolve(self) -> Path:
        path = self.base / self.run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
