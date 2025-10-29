"""Logging helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class JSONLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[Dict[str, Any]] = []

    def log(self, record: Dict[str, Any]) -> None:
        self._records.append(record)
        self.path.write_text("\n".join(json.dumps(r) for r in self._records))

    def flush(self) -> None:
        self.path.write_text("\n".join(json.dumps(r) for r in self._records))
