#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Test_output/outputs}"
ROLL_OUTS="${ROLL_OUTS:-5}"
TOPK="${TOPK:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
GUIDANCE_MODE="${GUIDANCE_MODE:-ollama}"
GUIDANCE_MODEL="${GUIDANCE_MODEL:-deepseek-r1:14b}"
GUIDANCE_BASE_URL="${GUIDANCE_BASE_URL:-http://localhost:11434}"
GUIDANCE_TIMEOUT="${GUIDANCE_TIMEOUT:-999}"
GUIDANCE_VALUE_WEIGHT="${GUIDANCE_VALUE_WEIGHT:-0}"
GUIDANCE_PRIOR_C="${GUIDANCE_PRIOR_C:-1.4}"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
export MPLCONFIGDIR

run_one() {
  local day="$1"
  local csv="${ROOT_DIR}/Tripcraft/Tripcraftzip/tripcraft_${day}day.csv"
  local out="${OUTPUT_DIR}/mcts_baseline_${day}day.jsonl"

  echo "==> Generate ${day}-day: ${out}"
  python -m mcts_baseline.cli \
    --input_csv "$csv" \
    --output_jsonl "$out" \
    --rollouts "$ROLL_OUTS" \
    --topk "$TOPK" \
    --guidance "$GUIDANCE_MODE" \
    --guidance_model "$GUIDANCE_MODEL" \
    --guidance_base_url "$GUIDANCE_BASE_URL" \
    --guidance_timeout "$GUIDANCE_TIMEOUT" \
    --guidance_value_weight "$GUIDANCE_VALUE_WEIGHT" \
    --guidance_prior_c "$GUIDANCE_PRIOR_C"

}

run_one 3
run_one 5
run_one 7
