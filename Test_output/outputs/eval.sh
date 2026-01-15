#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/Test_output/outputs"
EVAL_SCRIPT="${ROOT_DIR}/evaluation/eval.py"

export PYTHONPATH="$ROOT_DIR"

ALL_OUT_FILE="${OUTPUT_DIR}/mcts_deepseek_all.txt"
> "$ALL_OUT_FILE"  # Truncate file at start
for day in 3 5 7; do
  PLAN_FILE="${OUTPUT_DIR}/mcts_baseline_${day}day.jsonl"
  echo "==> Evaluating ${day}-day plan: ${PLAN_FILE}" | tee -a "$ALL_OUT_FILE"
  python "$EVAL_SCRIPT" --set_type validation --evaluation_file_path "$PLAN_FILE" >> "$ALL_OUT_FILE"
  echo -e "\n==============================\n" >> "$ALL_OUT_FILE"
done