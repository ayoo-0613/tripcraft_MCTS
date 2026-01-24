#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/Solver_result/outputs/ablation_mcts}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
EVAL_PY="${ROOT_DIR}/evaluation/eval.py"

export MPLCONFIGDIR

ablations=(risk_penalty slack_modulation feasibility_gate)
days=(5 7)

if [[ ! -f "$EVAL_PY" ]]; then
  echo "eval.py not found at: $EVAL_PY" >&2
  exit 1
fi

for ablation in "${ablations[@]}"; do
  for day in "${days[@]}"; do
    file="${OUTPUT_ROOT}/${ablation}/mcts_baseline_${day}day.jsonl"
    if [[ ! -f "$file" ]]; then
      echo "SKIP (missing): ${file}"
      continue
    fi
    if [[ ! -s "$file" ]]; then
      echo "SKIP (empty): ${file}"
      continue
    fi
    echo "==> ${ablation} ${day}-day"
    (cd "${ROOT_DIR}/evaluation" && python eval.py --set_type "${day}d" --evaluation_file_path "${file}")
  done
done
