#!/usr/bin/env bash
set -euo pipefail

# --- venv bootstrap ---
VENV_DIR="${VENV_DIR:-.venv}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[run_all] No active venv found; creating ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  echo "[run_all] Activated venv at ${VENV_DIR}"
else
  echo "[run_all] Using existing venv: ${VIRTUAL_ENV}"
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift || true
fi

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
OUT_ROOT="outputs/publication_${DATE_TAG}"
RAW_DIR="$OUT_ROOT/raw_data"
FIG_DIR="$OUT_ROOT/figures"
TAB_DIR="$OUT_ROOT/tables"
LOG_DIR="$OUT_ROOT/logs"
CKPT_DIR="$OUT_ROOT/checkpoints"

mkdir -p "$RAW_DIR" "$FIG_DIR" "$TAB_DIR" "$LOG_DIR" "$CKPT_DIR"

echo "[run_all] Outputs -> $OUT_ROOT"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ "$DRY_RUN" == "false" ]]; then
  bash scripts/download_datasets.sh ./data
else
  echo "[run_all] Dry run – skipping dataset downloads"
fi

YOLO_CMD=(python train_maestro_yolo.py --output-root outputs --date-tag "$DATE_TAG" --no-resume)
if [[ "$DRY_RUN" == "true" ]]; then
  YOLO_CMD+=(--dry-run)
fi
"${YOLO_CMD[@]}"

BASELINE_CONFIG="configs/meta_train/small_cpu_debug.yaml"
if [[ -f "$BASELINE_CONFIG" ]]; then
  METHODS=(ppo uniform easy_to_hard greedy bandit_linucb bandit_thompson pbt bohb)
  for METHOD in "${METHODS[@]}"; do
    CMD=(python scripts/run_comparative.py --config "$BASELINE_CONFIG" --method "$METHOD" --output-dir "$RAW_DIR/baseline_${METHOD}")
    if [[ "$DRY_RUN" == "true" ]]; then
      CMD+=(--seed 0)
    fi
    "${CMD[@]}" | tee "$LOG_DIR/baseline_${METHOD}.log"
  done
else
  echo "[run_all] Baseline config $BASELINE_CONFIG not found; skipping"
fi

if [[ "$DRY_RUN" == "false" ]]; then
  python scripts/run_ablation.py --config "$BASELINE_CONFIG" | tee "$LOG_DIR/ablation.log"
  python scripts/generate_ood_grid.py --config "$BASELINE_CONFIG" --output-dir "$RAW_DIR" | tee "$LOG_DIR/ood_grid.log"
  python scripts/run_n_invariance.py --config "$BASELINE_CONFIG" --output-dir "$RAW_DIR" | tee "$LOG_DIR/n_invariance.log"
  if [[ -f "$RAW_DIR/n_invariance.json" ]]; then
    RAW_DIR_ENV="$RAW_DIR" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["RAW_DIR_ENV"])
json_path = root / "n_invariance.json"
data = json.loads(json_path.read_text())
csv_path = root / "n_invariance.csv"
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(data.keys()))
    writer.writeheader()
    writer.writerow(data)
PY
  fi
else
  python scripts/run_ablation.py --config "$BASELINE_CONFIG" --dry-run | tee "$LOG_DIR/ablation.log"
  echo "[run_all] Dry run – skipping OOD grid and N-invariance sweeps" | tee "$LOG_DIR/ood_grid.log"
  echo "[run_all] Dry run – skipping N-invariance computation" | tee "$LOG_DIR/n_invariance.log"
fi

FIG_CMD=(python scripts/make_publication_figures.py --out "$OUT_ROOT")
TAB_CMD=(python scripts/generate_tables.py --out "$OUT_ROOT")
if [[ "$DRY_RUN" == "true" ]]; then
  FIG_CMD+=(--dry-run)
  TAB_CMD+=(--dry-run)
fi
"${FIG_CMD[@]}" | tee "$LOG_DIR/figures.log"
"${TAB_CMD[@]}" | tee "$LOG_DIR/tables.log"

echo "[run_all] Pipeline complete"
