#!/usr/bin/env bash
set -euo pipefail

# -------- settings you may tweak --------
RUN_DIR="outputs/exp_suite_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_DIR/logs"
PY=python                 # or path to your venv python

# Which GPUs to use per job (indices or empty for CPU)
GPU_DEBUG=0
GPU_LOFO_DET=1
GPU_LOFO_CLS=2
# ----------------------------------------

mkdir -p "$LOG_DIR" "configs/overrides" "$RUN_DIR"

# Create override configs that inherit the originals but write to subdirs.
cat > configs/overrides/debug.yaml <<'YAML'
defaults: [../meta_train/small_cpu_debug.yaml]
logging:
  output_dir: __RUN_DIR__/debug
YAML

cat > configs/overrides/lofo_detection.yaml <<'YAML'
defaults: [../meta_train/lofo_detection.yaml]
logging:
  output_dir: __RUN_DIR__/lofo_detection
YAML

cat > configs/overrides/lofo_classification.yaml <<'YAML'
defaults: [../meta_train/lofo_classification.yaml]
logging:
  output_dir: __RUN_DIR__/lofo_classification
YAML

# Patch in the actual RUN_DIR path
for f in configs/overrides/*.yaml; do
  sed -i.bak "s|__RUN_DIR__|$RUN_DIR|g" "$f" && rm -f "$f.bak"
done

# Helper to launch a tmux session bound to one GPU (or CPU if empty)
# Usage: launch_tmux <session_name> <gpu_id_or_empty> "<command>" "<log_file>"
launch_tmux () {
  local sess="$1"
  local gpu="$2"
  local cmd="$3"
  local log="$4"

  if [[ -n "${gpu}" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
  else
    cmd="CUDA_VISIBLE_DEVICES= ${cmd}"
  fi

  # Pipe both stdout and stderr to log (and keep interactive output in tmux).
  cmd="${cmd} 2>&1 | tee -a '${log}'"

  tmux new-session -d -s "${sess}" "echo '>>> $(date) :: ${sess} starting'; ${cmd}; \
    echo '>>> $(date) :: ${sess} finished' | tee -a '${log}'; sleep 2"
}

echo "RUN_DIR: $RUN_DIR"
echo "Logs:    $LOG_DIR"
echo

# Helper to compute the checkpoint path:
# run_meta_train writes to: <output_dir>/<run.id>/policy.pt
RUN_ID_DEBUG="debug_run"
RUN_ID_LOFO_DET="lofo_detection"
RUN_ID_LOFO_CLS="lofo_classification"
CKPT_DEBUG="$RUN_DIR/debug/$RUN_ID_DEBUG/policy.pt"
CKPT_LOFO_DET="$RUN_DIR/lofo_detection/$RUN_ID_LOFO_DET/policy.pt"
CKPT_LOFO_CLS="$RUN_DIR/lofo_classification/$RUN_ID_LOFO_CLS/policy.pt"

# 1) Meta-train (quick debug) -> then plot its learning curve
launch_tmux \
  "maestro_debug" "$GPU_DEBUG" \
  "$PY scripts/run_meta_train.py --config configs/overrides/debug.yaml && \
   $PY scripts/plot_make_figures.py --run-dir '$RUN_DIR/debug/$RUN_ID_DEBUG'" \
  "$LOG_DIR/meta_train_debug.log"

# 2) LOFO: train on classification+NER, eval held-out detection
launch_tmux \
  "maestro_lofo_det" "$GPU_LOFO_DET" \
  "$PY scripts/run_meta_train.py --config configs/overrides/lofo_detection.yaml && \
   $PY scripts/run_eval.py --config configs/overrides/lofo_detection.yaml --steps 10 --checkpoint '$CKPT_LOFO_DET'" \
  "$LOG_DIR/lofo_detection.log"

# 3) LOFO: classification variant
launch_tmux \
  "maestro_lofo_cls" "$GPU_LOFO_CLS" \
  "$PY scripts/run_meta_train.py --config configs/overrides/lofo_classification.yaml && \
   $PY scripts/run_eval.py --config configs/overrides/lofo_classification.yaml --steps 10 --checkpoint '$CKPT_LOFO_CLS'" \
  "$LOG_DIR/lofo_classification.log"

# 4) Markov diagnostics (uses the debug config; runs fast)
launch_tmux \
  "maestro_markov" "$GPU_DEBUG" \
  "$PY scripts/run_markov_diag.py --config configs/overrides/debug.yaml" \
  "$LOG_DIR/markov_diag.log"

echo "Launched sessions:"
tmux ls | sed 's/^/  /' || true
echo
echo "Attach with:   tmux attach -t maestro_debug   (or *_lofo_det / *_lofo_cls / maestro_markov)"
echo "Artifacts in:  $RUN_DIR"
echo "Logs in:       $LOG_DIR"
