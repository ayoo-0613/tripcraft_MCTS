#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/TripCraft/outputs}"
ROLL_OUTS="${ROLL_OUTS:-5}"
TOPK="${TOPK:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
export MPLCONFIGDIR

run_one() {
  local day="$1"
  local csv="${ROOT_DIR}/TripCraft/Tripcraftzip/tripcraft_${day}day.csv"
  local out="${OUTPUT_DIR}/mcts_baseline_${day}day.jsonl"

  echo "==> Generate ${day}-day: ${out}"
  python -m TripCraft.mcts_baseline.cli \
    --input_csv "$csv" \
    --output_jsonl "$out" \
    --rollouts "$ROLL_OUTS" \
    --topk "$TOPK"

  echo "==> Eval ${day}-day"
  (cd "${ROOT_DIR}/evaluation" && python eval.py --set_type "${day}d" --evaluation_file_path "$out")
}

run_one 3
run_one 5
run_one 7
