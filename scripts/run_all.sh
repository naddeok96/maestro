#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Configurable resources
# ---------------------------
GPUS=(${GPUS_OVERRIDE:-0 2 4 5 6 7})   # override via env if needed
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
OUT_ROOT="outputs/publication_${DATE_TAG}"
RAW_DIR="$OUT_ROOT/raw_data"
FIG_DIR="$OUT_ROOT/figures"
TAB_DIR="$OUT_ROOT/tables"
LOG_DIR="$OUT_ROOT/logs"
CKPT_DIR="$OUT_ROOT/checkpoints"
mkdir -p "$RAW_DIR" "$FIG_DIR" "$TAB_DIR" "$LOG_DIR" "$CKPT_DIR"

# ---------------------------
# venv bootstrap
# ---------------------------
VENV_DIR="${VENV_DIR:-.venv}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[run_all] No active venv; creating ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
else
  echo "[run_all] Using venv: ${VIRTUAL_ENV}"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# ---------------------------
# Helpers
# ---------------------------
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift || true
fi

log(){ echo -e "[run_all] $*"; }

USE_TMUX_HELPER="${USE_TMUX_HELPER:-0}"
declare -a TMUX_SIGNALS=()
declare -a BACKGROUND_PIDS=()
if [[ "$USE_TMUX_HELPER" == "1" ]] && ! command -v tmux >/dev/null 2>&1; then
  log "USE_TMUX_HELPER=1 but tmux not found; falling back to background jobs"
  USE_TMUX_HELPER=0
fi

# launch <gpu_id> <cmd...>
launch() {
  local gpu="$1"; shift
  local tag="${1}"; shift
  local outfile="$LOG_DIR/${tag}.log"
  mkdir -p "$LOG_DIR"
  if $DRY_RUN; then
    echo "[DRY] CUDA_VISIBLE_DEVICES=${gpu} $*" | tee "$outfile"
    return 0
  fi
  if [[ "$USE_TMUX_HELPER" == "1" ]]; then
    local session="runall_${tag}"
    local signal="${session}_done"
    local cmd_str
    cmd_str=$(printf '%q ' "$@")
    cmd_str="${cmd_str% }"
    echo "[launch:tmux] gpu=${gpu} tag=${tag} session=${session} -> $outfile"
    tmux new-session -d -s "$session" "set -euo pipefail; export CUDA_VISIBLE_DEVICES='${gpu}'; { ${cmd_str} >>'${outfile}' 2>&1; } || true; tmux wait-for -S '${signal}'"
    TMUX_SIGNALS+=("$signal")
  else
    echo "[launch] gpu=${gpu} tag=${tag} -> $outfile"
    ( export CUDA_VISIBLE_DEVICES="${gpu}"; "$@" 2>&1 | tee "$outfile" ) &
    local pid=$!
    BACKGROUND_PIDS+=("$pid")
  fi
}

# round-robin GPU picker
_next_gpu_i=0
pick_gpu() {
  local idx=${_next_gpu_i}
  _next_gpu_i=$(( (_next_gpu_i + 1) % ${#GPUS[@]} ))
  echo "${GPUS[$idx]}"
}

# ensure single dataset download
DATA_SENTINEL="$OUT_ROOT/.datasets_ready"
if ! $DRY_RUN; then
if [[ ! -f "$DATA_SENTINEL" ]]; then
  log "Downloading/Preparing datasets (one-time)"
  bash scripts/download_datasets.sh ./data | tee "$LOG_DIR/datasets.log"
  touch "$DATA_SENTINEL"
else
  log "Datasets already prepared"
fi
else
  log "Dry run — skipping dataset download"
fi

# ---------------------------
# 0) Supervised baseline benchmarks
# ---------------------------
BASELINE_CONFIGS=(
  "classification:configs/tasks/classification.yaml"
  "ner:configs/tasks/ner.yaml"
  "detection:configs/tasks/detection.yaml"
)
BASELINE_ROOT="$OUT_ROOT/baselines"
mkdir -p "$BASELINE_ROOT"

# Teacher defaults (seed used for baselines too)
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/publication/main_suite.yaml}"
TEACHER_SEED="${TEACHER_SEED:-42}"        # change externally for more seeds
TEACHER_DETERMINISTIC="${TEACHER_DETERMINISTIC:-1}"

log "Launching supervised curriculum baselines across GPUs"
baseline_pid_start=${#BACKGROUND_PIDS[@]}
baseline_signal_start=${#TMUX_SIGNALS[@]}
for entry in "${BASELINE_CONFIGS[@]}"; do
  IFS=":" read -r baseline_name baseline_cfg <<<"$entry"
  gpu="$(pick_gpu)"
  tag="baseline_${baseline_name}"
  base_out="$BASELINE_ROOT/${baseline_name}"
  mkdir -p "$base_out"
  cmd=(python train_baselines.py \
    --tasks "$baseline_cfg" \
    --output-dir "$base_out" \
    --methods standard uniform easy_to_hard greedy linucb \
    --seed "$TEACHER_SEED" \
    --device cuda)
  if $DRY_RUN; then
    cmd+=(--dry-run)
  fi
  launch "$gpu" "$tag" "${cmd[@]}"
done

if [[ ${#BASELINE_CONFIGS[@]} -gt 0 ]]; then
  if [[ "$USE_TMUX_HELPER" == "1" ]]; then
    log "Waiting for supervised baselines to finish (tmux)..."
    for signal in "${TMUX_SIGNALS[@]:baseline_signal_start}"; do
      tmux wait-for "$signal"
    done
  else
    log "Waiting for supervised baselines to finish..."
    for pid in "${BACKGROUND_PIDS[@]:baseline_pid_start}"; do
      wait "$pid"
    done
  fi
  if ! $DRY_RUN; then
    BASELINE_ROOT_ENV="$BASELINE_ROOT" python - <<'PY'
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base = Path(os.environ["BASELINE_ROOT_ENV"])
frames = []
for sub in base.glob("*"):
    summary = sub / "baseline_summary.csv"
    if summary.is_file():
        df = pd.read_csv(summary)
        df.insert(0, "task_group", sub.name)
        frames.append(df)
if not frames:
    raise SystemExit(0)
out = pd.concat(frames, ignore_index=True)
summary_all = base / "baseline_summary_all.csv"
out.to_csv(summary_all, index=False)
metric_priority = [
    "final_macro_f1",
    "final_macro_map",
    "final_accuracy",
    "final_map",
    "final_loss",
]
metric_column = next((col for col in metric_priority if col in out.columns), None)
if metric_column is None:
    metric_column = next((col for col in out.columns if col.startswith("final_")), None)
if metric_column:
    pivot = out.pivot_table(index="task", columns="method", values=metric_column, aggfunc="max")
    matrix = pivot.to_numpy(dtype=float)
    if pivot.size and np.isfinite(matrix).any():
        fig_dir = base / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.cm.magma.copy()
        cmap.set_bad(color="#f5f5f5")
        fig, ax = plt.subplots(figsize=(1.8 * max(1, len(pivot.columns)), 1.2 * max(1, len(pivot.index)) + 1.5))
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        metric_label = metric_column.replace("final_", "").replace("_", " ").title()
        cbar.set_label(metric_label)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(idx) for idx in pivot.index])
        ax.set_xlabel("Method")
        ax.set_ylabel("Task")
        ax.set_title(f"{metric_label} overview (all baselines)")
        max_val = float(np.nanmax(matrix))
        threshold = max_val * 0.5 if max_val != 0 else 0.0
        for i, task in enumerate(pivot.index):
            for j, method in enumerate(pivot.columns):
                value = pivot.loc[task, method]
                if not np.isfinite(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < threshold else "black",
                    fontsize=9,
                )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"baseline_summary_overview.{ext}", dpi=200)
        plt.close(fig)
PY
  fi
fi

# ---------------------------
# 1) Train PPO Teacher (publication suite, CPU by default)
# ---------------------------
log "Training PPO teacher: ${TEACHER_CONFIG} (seed=${TEACHER_SEED})"
if $DRY_RUN; then
  echo "[DRY] python train_maestro_teacher.py --config ${TEACHER_CONFIG} --seed ${TEACHER_SEED} --output-dir outputs" | tee "$LOG_DIR/meta_train.log"
else
  python train_maestro_teacher.py \
    --config "${TEACHER_CONFIG}" \
    --seed "${TEACHER_SEED}" \
    --output-dir outputs | tee "$LOG_DIR/meta_train.log"
fi

TEACH_RUN_ID="$(python - <<'PY'
import yaml
conf=yaml.safe_load(open("configs/publication/main_suite.yaml"))
rid=conf.get("run",{}).get("id","publication_main")
print(rid)
PY
)"
TEACH_OUT_DIR="outputs/${TEACH_RUN_ID}"
TEACH_CKPT="${TEACH_OUT_DIR}/policy.pt"

if [[ ! -f "$TEACH_CKPT" ]]; then
  echo "[ERROR] Teacher checkpoint not found at $TEACH_CKPT" >&2
  exit 2
fi
log "Teacher checkpoint: $TEACH_CKPT"

# ---------------------------
# 2) Parallel phase (after teacher)
# ---------------------------
# We now kick off everything that does NOT depend on each other, in parallel.
# Rules:
#  - Each job gets an isolated GPU via CUDA_VISIBLE_DEVICES + '--device 0' for Ultralytics.
#  - Each job writes to a unique subdir/file (use tags).
#  - YOLO gets priority on more GPUs; diagnostics can take leftovers or CPU.

# ----- 2a) YOLO transfer track(s)
# One definitive YOLO “publication” run (adjust segments/budget/batch as needed)
YOLO_TAG="yolo_pub"
YOLO_GPU="$(pick_gpu)"
YOLO_ARGS=(python train_maestro_yolo.py
  --output-root outputs
  --date-tag "$DATE_TAG"
  --no-resume
  --segments 12
  --budget-images 200000
  --batch 16
  --imgsz 896
  --device 0
  --method maestro
  --teacher-ckpt "$TEACH_CKPT"
)
if [[ "$TEACHER_DETERMINISTIC" == "1" ]]; then YOLO_ARGS+=(--teacher-deterministic); else YOLO_ARGS+=(--no-teacher-deterministic); fi
launch "$YOLO_GPU" "$YOLO_TAG" "${YOLO_ARGS[@]}"

# (Optional) Extra YOLO seeds or mixes (uncomment to run more in parallel)
# for S in 43 44; do
#   gpu="$(pick_gpu)"
#   tag="yolo_pub_seed${S}"
#   launch "$gpu" "$tag" "${YOLO_ARGS[@]}"
# done

# ----- 2b) Diagnostics & evals (parallel)
# Markov diagnostics (publication config)
MD_GPU="$(pick_gpu)"
launch "$MD_GPU" "markov_diag" \
  python scripts/run_markov_diag.py --config "$TEACHER_CONFIG" --out "$OUT_ROOT"

# N-invariance (use small config to remain quick—or swap to publication if desired)
NI_GPU="$(pick_gpu)"
launch "$NI_GPU" "n_invariance" \
  python scripts/run_n_invariance.py --config configs/meta_train/small_cpu_debug.yaml --output-dir "$RAW_DIR"

# OOD grid (quick version; adjust flags for heavier sweep)
OOD_GPU="$(pick_gpu)"
launch "$OOD_GPU" "ood_grid" \
  python scripts/generate_ood_grid.py --config configs/meta_train/small_cpu_debug.yaml --output-dir "$RAW_DIR"

# LOFO & main evals (CSV) — these are light and can run on CPU; still put on a GPU slot for consistency
EVAL_GPU="$(pick_gpu)"
launch "$EVAL_GPU" "eval_lofo" \
  python scripts/run_eval.py --config configs/meta_train/lofo_classification.yaml --steps 10 --csv-out "$RAW_DIR/lofo.csv"

EVAL2_GPU="$(pick_gpu)"
launch "$EVAL2_GPU" "eval_main" \
  python scripts/run_eval.py --config "$TEACHER_CONFIG" --steps 10 --csv-out "$RAW_DIR/main_results.csv"

# ----- 2c) (Optional) Comparative baselines at scale (expensive). Uncomment to run.
# for M in ppo uniform easy_to_hard greedy bandit_linucb bandit_thompson pbt bohb; do
#   gpu="$(pick_gpu)"
#   tag="baseline_${M}"
#   launch "$gpu" "$tag" \
#     python scripts/run_comparative.py --config "$TEACHER_CONFIG" --method "$M" --output-dir "$RAW_DIR/baseline_${M}"
# done
# After baselines finish you can consolidate curves:
# python scripts/export_learning_curves.py --out "$RAW_DIR" --comparative-root "$RAW_DIR" || true

# ---------------------------
# 3) Wait for all background jobs
# ---------------------------
if [[ "$USE_TMUX_HELPER" == "1" ]]; then
  log "Waiting for tmux-managed jobs to complete…"
  for signal in "${TMUX_SIGNALS[@]}"; do
    tmux wait-for "$signal"
  done
  log "All tmux-managed jobs signaled completion."
else
  log "Waiting for parallel jobs to complete…"
  wait
  log "All parallel jobs complete."
fi

# ---------------------------
# 4) Figures & tables (post-processing)
# ---------------------------
FIG_CMD=(python scripts/make_publication_figures.py --out "$OUT_ROOT")
TAB_CMD=(python scripts/generate_tables.py        --out "$OUT_ROOT")
if $DRY_RUN; then
  FIG_CMD+=(--dry-run); TAB_CMD+=(--dry-run)
fi
"${FIG_CMD[@]}" | tee "$LOG_DIR/figures.log" || true
"${TAB_CMD[@]}" | tee "$LOG_DIR/tables.log"  || true

log "Publication pipeline complete."
