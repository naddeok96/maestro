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

mkdir -p "$LOG_DIR" "configs/overrides"

# Create tiny override configs that just point logging.output_dir to RUN_DIR.
# They inherit the real configs via the built-in "defaults" merge in your loader.

cat > configs/overrides/debug.yaml <<'YAML'
defaults: [../meta_train/small_cpu_debug.yaml]
logging:
  output_dir: __RUN_DIR__
YAML

cat > configs/overrides/lofo_detection.yaml <<'YAML'
defaults: [../meta_train/lofo_detection.yaml]
logging:
  output_dir: __RUN_DIR__
YAML

cat > configs/overrides/lofo_classification.yaml <<'YAML'
defaults: [../meta_train/lofo_classification.yaml]
logging:
  output_dir: __RUN_DIR__
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

  # Prefix command with CUDA_VISIBLE_DEVICES if gpu is set, else clear it for CPU.
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

# 1) Meta-train (quick debug) -> then plot
launch_tmux \
  "maestro_debug" "$GPU_DEBUG" \
  "$PY scripts/run_meta_train.py --config configs/overrides/debug.yaml && \
   $PY scripts/plot_make_figures.py --run-dir '$RUN_DIR'" \
  "$LOG_DIR/meta_train_debug.log"

# 2) LOFO: train on classification+NER, eval held-out detection
# (Meta-train first; you can add --checkpoint to run_eval to use a saved policy if you add saving)
launch_tmux \
  "maestro_lofo_det" "$GPU_LOFO_DET" \
  "$PY scripts/run_meta_train.py --config configs/overrides/lofo_detection.yaml && \
   $PY scripts/run_eval.py --config configs/overrides/lofo_detection.yaml --steps 10" \
  "$LOG_DIR/lofo_detection.log"

# 3) LOFO: classification variant
launch_tmux \
  "maestro_lofo_cls" "$GPU_LOFO_CLS" \
  "$PY scripts/run_meta_train.py --config configs/overrides/lofo_classification.yaml && \
   $PY scripts/run_eval.py --config configs/overrides/lofo_classification.yaml --steps 10" \
  "$LOG_DIR/lofo_classification.log"

# 4) Markov diagnostics (uses the debug config; runs very fast)
launch_tmux \
  "maestro_markov" "$GPU_DEBUG" \
  "$PY scripts/run_markov_diag.py --config configs/overrides/debug.yaml" \
  "$LOG_DIR/markov_diag.log"

echo
echo "Launched sessions:"
tmux ls | sed 's/^/  /'
echo
echo "Attach to a session, e.g.: tmux attach -t maestro_debug"
echo "All artifacts go to: $RUN_DIR"
echo "All terminal logs in: $LOG_DIR"
