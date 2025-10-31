#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra args...]" >&2
  exit 1
fi

CONFIG=$1
shift || true
METHODS=(ppo uniform easy_to_hard greedy bandit_linucb bandit_thompson)

for METHOD in "${METHODS[@]}"; do
  echo "Running method: ${METHOD}"
  python scripts/run_comparative.py --config "${CONFIG}" --method "${METHOD}" "$@"
done
