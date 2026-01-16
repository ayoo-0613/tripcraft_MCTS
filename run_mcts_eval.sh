#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/Test_output/outputs}"
ROLL_OUTS="${ROLL_OUTS:-20}"
TOPK="${TOPK:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
GUIDANCE_MODE="${GUIDANCE_MODE:-persona}"
GUIDANCE_MODEL="${GUIDANCE_MODEL:-deepseek-r1:14b}"
GUIDANCE_BASE_URL="${GUIDANCE_BASE_URL:-http://localhost:11434}"
GUIDANCE_TIMEOUT="${GUIDANCE_TIMEOUT:-999}"
GUIDANCE_VALUE_WEIGHT="${GUIDANCE_VALUE_WEIGHT:-0}"
GUIDANCE_PRIOR_C="${GUIDANCE_PRIOR_C:-1.7}"
LLM_MODEL="${LLM_MODEL:-$GUIDANCE_MODEL}"
LLM_BASE_URL="${LLM_BASE_URL:-$GUIDANCE_BASE_URL}"
TEMPORAL_GUIDANCE="${TEMPORAL_GUIDANCE:-ollama}"
TEMPORAL_PROMPT="${TEMPORAL_PROMPT:-}"
TEMPORAL_TIMEOUT="${TEMPORAL_TIMEOUT:-}"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
export MPLCONFIGDIR

run_one() {
  local day="$1"
  local csv="${ROOT_DIR}/Tripcraft/Tripcraftzip/tripcraft_${day}day.csv"
  local out="${OUTPUT_DIR}/mcts_baseline_${day}gitday.jsonl"

  echo "==> Generate ${day}-day: ${out}"
  python -m mcts_baseline.cli \
    --input_csv "$csv" \
    --output_jsonl "$out" \
    --rollouts "$ROLL_OUTS" \
    --topk "$TOPK" \
    --guidance "$GUIDANCE_MODE" \
    --llm_model "$LLM_MODEL" \
    --llm_base_url "$LLM_BASE_URL" \
    --guidance_timeout "$GUIDANCE_TIMEOUT" \
    --guidance_value_weight "$GUIDANCE_VALUE_WEIGHT" \
    --guidance_prior_c "$GUIDANCE_PRIOR_C" \
    --temporal_guidance "$TEMPORAL_GUIDANCE" \
    ${TEMPORAL_PROMPT:+--temporal_prompt "$TEMPORAL_PROMPT"} \
    ${TEMPORAL_TIMEOUT:+--temporal_timeout "$TEMPORAL_TIMEOUT"}

}

run_one 3
run_one 5
run_one 7
