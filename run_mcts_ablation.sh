#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_DIR="${CSV_DIR:-${ROOT_DIR}/Tripcraft/Tripcraftzip}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/Solver_result/outputs/ablation_mcts}"
ROLL_OUTS="${ROLL_OUTS:-100}"
TOPK="${TOPK:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
GUIDANCE_MODE="${GUIDANCE_MODE:-persona}"
GUIDANCE_MODEL="${GUIDANCE_MODEL:-deepseek-r1:14b}"
GUIDANCE_BASE_URL="${GUIDANCE_BASE_URL:-http://localhost:11434}"
GUIDANCE_TIMEOUT="${GUIDANCE_TIMEOUT:-999}"
GUIDANCE_VALUE_WEIGHT="${GUIDANCE_VALUE_WEIGHT:-0.8}"
GUIDANCE_PRIOR_C="${GUIDANCE_PRIOR_C:-1.7}"
LLM_MODEL="${LLM_MODEL:-$GUIDANCE_MODEL}"
LLM_BASE_URL="${LLM_BASE_URL:-$GUIDANCE_BASE_URL}"
TEMPORAL_GUIDANCE="${TEMPORAL_GUIDANCE:-ollama}"
TEMPORAL_PROMPT="${TEMPORAL_PROMPT:-}"
TEMPORAL_TIMEOUT="${TEMPORAL_TIMEOUT:-}"

MCTS_RISK_LAMBDA="${MCTS_RISK_LAMBDA:-0.5}"
MCTS_RISK_HORIZON="${MCTS_RISK_HORIZON:-2}"
MCTS_RISK_SATURATE_K="${MCTS_RISK_SATURATE_K:-}"
MCTS_SLACK_FLOOR="${MCTS_SLACK_FLOOR:-0.0}"
MCTS_FEASIBILITY_WEIGHT="${MCTS_FEASIBILITY_WEIGHT:-1.0}"

mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"
export MPLCONFIGDIR

build_base_args() {
  local args=(
    --rollouts "$ROLL_OUTS"
    --topk "$TOPK"
    --guidance "$GUIDANCE_MODE"
    --llm_model "$LLM_MODEL"
    --llm_base_url "$LLM_BASE_URL"
    --guidance_timeout "$GUIDANCE_TIMEOUT"
    --guidance_value_weight "$GUIDANCE_VALUE_WEIGHT"
    --guidance_prior_c "$GUIDANCE_PRIOR_C"
    --temporal_guidance "$TEMPORAL_GUIDANCE"
  )
  if [[ -n "$TEMPORAL_PROMPT" ]]; then
    args+=(--temporal_prompt "$TEMPORAL_PROMPT")
  fi
  if [[ -n "$TEMPORAL_TIMEOUT" ]]; then
    args+=(--temporal_timeout "$TEMPORAL_TIMEOUT")
  fi
  echo "${args[@]}"
}

run_setting() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "$out_dir"
  local base_args
  base_args=$(build_base_args)

  for day in 3 5 7; do
    local csv="${CSV_DIR}/tripcraft_${day}day.csv"
    local out="${out_dir}/mcts_baseline_${day}day.jsonl"
    echo "==> ${name}: ${day}-day -> ${out}"
    python -m mcts_baseline.cli \
      --input_csv "$csv" \
      --output_jsonl "$out" \
      $base_args \
      "$@"
  done
}

run_setting "risk_penalty" \
  --mcts_risk_penalty \
  --mcts_risk_lambda "$MCTS_RISK_LAMBDA" \
  --mcts_risk_horizon "$MCTS_RISK_HORIZON" \
  ${MCTS_RISK_SATURATE_K:+--mcts_risk_saturate_k "$MCTS_RISK_SATURATE_K"}

run_setting "slack_modulation" \
  --mcts_slack_modulation \
  --mcts_slack_floor "$MCTS_SLACK_FLOOR"

run_setting "feasibility_gate" \
  --mcts_feasibility_gate \
  --mcts_feasibility_weight "$MCTS_FEASIBILITY_WEIGHT"
