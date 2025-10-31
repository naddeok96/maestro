#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--dry-run] <config.yaml> [extra args...]" >&2
  exit 1
fi

CONFIG=$1
shift || true
METHODS=(ppo uniform easy_to_hard greedy bandit_linucb bandit_thompson pbt bohb)

for METHOD in "${METHODS[@]}"; do
  echo "Running method: ${METHOD}"
  EXTRA_ARGS=("$@")
  if [[ "$DRY_RUN" == "true" ]]; then
    EXTRA_ARGS+=(--seed 0)
  fi
  python scripts/run_comparative.py --config "${CONFIG}" --method "${METHOD}" "${EXTRA_ARGS[@]}"
done
