"""Quick end-to-end verification for the MAESTRO teacher + YOLO pipeline."""

from __future__ import annotations

import csv
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.run_meta_train import run_meta_training


def _ensure_columns(csv_path: Path, required: set[str]) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
        if row is None:
            raise RuntimeError(f"CSV {csv_path} is empty")
        missing = required.difference(row.keys())
        if missing:
            raise RuntimeError(f"CSV {csv_path} missing columns: {sorted(missing)}")


def main() -> None:
    teacher_dir = run_meta_training(Path("configs/meta_train/small_cpu_debug.yaml"))
    checkpoint = teacher_dir / "policy.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected teacher checkpoint at {checkpoint}")

    date_tag = datetime.now(UTC).strftime("%Y%m%d")
    yolo_cmd = [
        sys.executable,
        "train_maestro_yolo.py",
        "--teacher-ckpt",
        str(checkpoint),
        "--method",
        "maestro",
        "--teacher-deterministic",
        "--output-root",
        "outputs",
        "--date-tag",
        date_tag,
        "--segments",
        "2",
        "--budget-images",
        "2048",
        "--batch",
        "4",
        "--dry-run",
        "--no-resume",
    ]
    subprocess.run(yolo_cmd, check=True)

    csv_path = Path("outputs") / f"publication_{date_tag}" / "raw_data" / "yolo_segments.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected YOLO CSV at {csv_path}")
    _ensure_columns(csv_path, {"macro_mAP", "weight", "eta_scale", "usage"})

    ckpt_dir = Path("outputs") / f"publication_{date_tag}" / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Expected checkpoint directory at {ckpt_dir}")

    print("[✓] Teacher checkpoint:", checkpoint)
    print("[✓] YOLO segments CSV:", csv_path)
    print("[✓] Segment checkpoints directory:", ckpt_dir)


if __name__ == "__main__":
    main()
