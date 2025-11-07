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

# --- Train PPO teacher on synthetic episodes (fast config for pipeline) ---
# Allow overrides via env; fall back to debug config for quick pipeline.
: "${TEACHER_CONFIG:=configs/meta_train/small_cpu_debug.yaml}"
: "${TEACHER_SEED:=0}"
: "${TEACHER_DETERMINISTIC:=1}"   # 1=true, 0=false

echo "[run_all] Training PPO teacher using: ${TEACHER_CONFIG} (seed=${TEACHER_SEED})"
python train_maestro_teacher.py \
  --config "${TEACHER_CONFIG}" \
  --seed "${TEACHER_SEED}" \
  | tee "$LOG_DIR/meta_train.log"

# Resolve checkpoint location from the run id in config; for the debug config this is outputs/debug_run/policy.pt
# If a different config sets logging.output_dir/run.id, the checkpoint is still placed under outputs/<run_id>/policy.pt
TEACH_OUT_DIR="outputs/debug_run"
TEACH_CKPT="${TEACH_OUT_DIR}/policy.pt"
if [[ -f "$TEACH_CKPT" ]]; then
  echo "[run_all] Using teacher checkpoint: $TEACH_CKPT"
else
  echo "[run_all] WARNING: teacher checkpoint not found at $TEACH_CKPT; YOLO will use the deterministic stub/baseline"
fi

# --- YOLO track, controlled by the pre-trained teacher when available ---
YOLO_CMD=(python train_maestro_yolo.py --output-root outputs --date-tag "$DATE_TAG" --no-resume)
if [[ "$DRY_RUN" == "true" ]]; then
  YOLO_CMD+=(--dry-run)
fi
if [[ -f "$TEACH_CKPT" ]]; then
  YOLO_CMD+=(--method maestro --teacher-ckpt "$TEACH_CKPT")
  if [[ "$TEACHER_DETERMINISTIC" == "1" ]]; then
    YOLO_CMD+=(--teacher-deterministic)
  else
    YOLO_CMD+=(--no-teacher-deterministic)
  fi
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

  if [[ "$DRY_RUN" == "false" ]]; then
    PLOT_ARGS=()
    for METHOD in "${METHODS[@]}"; do
      if [[ -d "$RAW_DIR/baseline_${METHOD}" ]]; then
        PLOT_ARGS+=(--run "${METHOD}=$RAW_DIR/baseline_${METHOD}")
      fi
    done
    if (( ${#PLOT_ARGS[@]} > 0 )); then
      COMP_PLOTS_DIR="$OUT_ROOT/comparative_plots/$(date +%Y%m%d-%H%M%S)"
      python scripts/plot_comparative.py "${PLOT_ARGS[@]}" --output-dir "$COMP_PLOTS_DIR" | tee "$LOG_DIR/comparative_plots.log"
    else
      echo "[run_all] No comparative baselines found for plotting"
    fi
    latest_cp_dir="$(ls -td "$OUT_ROOT"/comparative_plots/* 2>/dev/null | head -n1 || true)"
    if [[ -n "$latest_cp_dir" && -f "$latest_cp_dir/table4_metrics.csv" ]]; then
      cp -f "$latest_cp_dir/table4_metrics.csv" "$RAW_DIR/baselines.csv"
    else
      : > "$RAW_DIR/baselines.csv"
    fi
    # learning curves for Fig1
    python scripts/export_learning_curves.py --out "$RAW_DIR" \
      --comparative-root "$RAW_DIR" || true

    # markov diagnostics for Fig2
    python scripts/run_markov_diag.py --config "$BASELINE_CONFIG" --out "$OUT_ROOT" \
      | tee "$LOG_DIR/markov_diag.log"
    if [[ ! -f "$RAW_DIR/baselines.csv" ]]; then
      : > "$RAW_DIR/baselines.csv"
    fi
  else
    : > "$RAW_DIR/baselines.csv"
  fi
else
  echo "[run_all] Baseline config $BASELINE_CONFIG not found; skipping"
fi

if [[ "$DRY_RUN" == "false" ]]; then
  python scripts/run_ablation_suite.py --config "$BASELINE_CONFIG" --output-dir "$RAW_DIR" | tee "$LOG_DIR/ablation.log"
  cp -f "$RAW_DIR/ablation_results.csv" "$RAW_DIR/ablation.csv" 2>/dev/null || true
  cp -f "$RAW_DIR/ablation_results.csv" "$RAW_DIR/ablations.csv" 2>/dev/null || true
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
  : > "$RAW_DIR/ablation.csv"
  : > "$RAW_DIR/ablations.csv"
  python scripts/run_ablation.py --config "$BASELINE_CONFIG" --dry-run | tee "$LOG_DIR/ablation.log"
  echo "[run_all] Dry run – skipping OOD grid and N-invariance sweeps" | tee "$LOG_DIR/ood_grid.log"
  echo "[run_all] Dry run – skipping N-invariance computation" | tee "$LOG_DIR/n_invariance.log"
fi

if [[ -f "$BASELINE_CONFIG" ]]; then
  python scripts/run_eval.py --config configs/meta_train/lofo_classification.yaml --steps 10 --csv-out "$RAW_DIR/lofo.csv" | tee "$LOG_DIR/lofo_eval.log"
  python scripts/run_eval.py --config "$BASELINE_CONFIG" --steps 10 --csv-out "$RAW_DIR/main_results.csv" | tee "$LOG_DIR/main_eval.log"
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
