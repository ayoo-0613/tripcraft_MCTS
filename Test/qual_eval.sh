#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/Test/eval_outputs"
QUAL_SCRIPT="${ROOT_DIR}/evaluation/qualitative_metrics.py"

ALL_OUT_FILE="${OUTPUT_DIR}/qual_metrics_all.txt"
> "$ALL_OUT_FILE"  # Truncate file at start
for day in 3 5 7; do
  GEN_FILE="${ROOT_DIR}/Test/mcts_baseline_${day}day.jsonl"
  ANNO_FILE="${ROOT_DIR}/TripCraft/Tripcraftzip/tripcraft_${day}day.csv"
  echo "==> Qualitative metrics for ${day}-day: ${GEN_FILE}" | tee -a "$ALL_OUT_FILE"
  python "$QUAL_SCRIPT" --gen_file "$GEN_FILE" --anno_file "$ANNO_FILE" >> "$ALL_OUT_FILE"
  echo -e "\n==============================\n" >> "$ALL_OUT_FILE"
done
