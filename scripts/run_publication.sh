#!/usr/bin/env bash
set -euo pipefail

PY=python
DATE=$(date +%Y%m%d)
ROOT="outputs/publication_${DATE}"
LOG="$ROOT/logs"
mkdir -p "$LOG" "configs/overrides"

# override to inject ROOT while keeping run.id appended
cat > configs/overrides/publication_main.yaml <<'YAML'
defaults: [../publication/main_suite.yaml]
logging:
  output_dir: __ROOT__
YAML
sed -i.bak "s|__ROOT__|$ROOT|g" configs/overrides/publication_main.yaml && rm -f configs/overrides/publication_main.yaml.bak

# Small GPU pool awareness (edit if needed)
GPUS=(0 1 2 3 4 5)
session() { echo "pub_$1"; }

launch() {
  local name="$1"; shift
  local gpu="${1:-}"; shift || true
  local cmd="$*"
  if [[ -n "$gpu" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} $cmd"
  fi
  tmux new-session -d -s "$(session "$name")" "echo '>>> $(date) start $name'; $cmd 2>&1 | tee -a '$LOG/$name.log'; echo '>>> $(date) done $name'; sleep 2"
}

# 1) Main suite across 5 seeds (loop externally for seeds)
for SEED in 42 43 44 45 46; do
  launch "main_${SEED}" "${GPUS[$((SEED%${#GPUS[@]}))]}" \
    "$PY scripts/run_meta_train.py --config configs/overrides/publication_main.yaml --seed $SEED"
done

echo "Launched:"
tmux ls | sed 's/^/  /' || true
echo "Outputs -> $ROOT"
