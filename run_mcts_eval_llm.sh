#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/TripCraft/outputs}"
ROLL_OUTS="${ROLL_OUTS:-5}"
TOPK="${TOPK:-10}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.mplconfig}"
RUN_EVAL="${RUN_EVAL:-1}"

GUIDANCE_MODE="${GUIDANCE_MODE:-ollama}"
GUIDANCE_VALUE_WEIGHT="${GUIDANCE_VALUE_WEIGHT:-0.2}"
GUIDANCE_PRIOR_C="${GUIDANCE_PRIOR_C:-1.4}"
GUIDANCE_TIMEOUT="${GUIDANCE_TIMEOUT:-999}"
TEMPORAL_GUIDANCE="${TEMPORAL_GUIDANCE:-ollama}"
TEMPORAL_PROMPT="${TEMPORAL_PROMPT:-}"
TEMPORAL_TIMEOUT="${TEMPORAL_TIMEOUT:-}"
LLM_CONFIG="${LLM_CONFIG:-}"
LLM_MODEL="${LLM_MODEL:-}"
LLM_BASE_URL="${LLM_BASE_URL:-}"

QUERY_TEXT="${QUERY_TEXT:-}"
QUERY_FILE="${QUERY_FILE:-}"
QUERY_CONTEXT_JSON="${QUERY_CONTEXT_JSON:-}"
QUERY_OUTPUT_JSON="${QUERY_OUTPUT_JSON:-}"
QUERY_OUTPUT_JSONL="${QUERY_OUTPUT_JSONL:-${OUTPUT_DIR}/mcts_baseline_query.jsonl}"

mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"
export MPLCONFIGDIR

if [[ ( "$GUIDANCE_MODE" != "none" || "$TEMPORAL_GUIDANCE" != "none" ) && -z "$LLM_CONFIG" && -z "$LLM_MODEL" ]]; then
  echo "LLM_CONFIG or LLM_MODEL is required when guidance is enabled." >&2
  exit 1
fi

run_query() {
  if [[ -z "$QUERY_TEXT" && -z "$QUERY_FILE" ]]; then
    return
  fi
  local args=(
    python -m TripCraft.mcts_baseline.cli
    --output_jsonl "$QUERY_OUTPUT_JSONL"
    --rollouts "$ROLL_OUTS"
    --topk "$TOPK"
  )
  if [[ -n "$LLM_CONFIG" ]]; then
    args+=(--llm_config "$LLM_CONFIG")
  fi
  if [[ -n "$LLM_MODEL" ]]; then
    args+=(--llm_model "$LLM_MODEL")
  fi
  if [[ -n "$LLM_BASE_URL" ]]; then
    args+=(--llm_base_url "$LLM_BASE_URL")
  fi
  if [[ "$GUIDANCE_MODE" != "none" ]]; then
    args+=(--guidance "$GUIDANCE_MODE")
    args+=(--guidance_value_weight "$GUIDANCE_VALUE_WEIGHT")
    args+=(--guidance_prior_c "$GUIDANCE_PRIOR_C")
    args+=(--guidance_timeout "$GUIDANCE_TIMEOUT")
  fi
  if [[ "$TEMPORAL_GUIDANCE" != "none" ]]; then
    args+=(--temporal_guidance "$TEMPORAL_GUIDANCE")
    if [[ -n "$TEMPORAL_PROMPT" ]]; then
      args+=(--temporal_prompt "$TEMPORAL_PROMPT")
    fi
    if [[ -n "$TEMPORAL_TIMEOUT" ]]; then
      args+=(--temporal_timeout "$TEMPORAL_TIMEOUT")
    fi
  fi
  if [[ -n "$QUERY_TEXT" ]]; then
    args+=(--input_query "$QUERY_TEXT")
  else
    args+=(--input_query_file "$QUERY_FILE")
  fi
  if [[ -n "$QUERY_CONTEXT_JSON" ]]; then
    args+=(--query_context_json "$QUERY_CONTEXT_JSON")
  fi
  if [[ -n "$QUERY_OUTPUT_JSON" ]]; then
    args+=(--query_output_json "$QUERY_OUTPUT_JSON")
  fi

  echo "==> Query generate: ${QUERY_OUTPUT_JSONL}"
  "${args[@]}"
}

run_one() {
  local day="$1"
  local csv="${ROOT_DIR}/TripCraft/Tripcraftzip/tripcraft_${day}day.csv"
  local out="${OUTPUT_DIR}/mcts_baseline_${day}day.jsonl"

  echo "==> Generate ${day}-day: ${out}"
  local args=(
    python -m TripCraft.mcts_baseline.cli
    --input_csv "$csv"
    --output_jsonl "$out"
    --rollouts "$ROLL_OUTS"
    --topk "$TOPK"
  )
  if [[ -n "$LLM_CONFIG" ]]; then
    args+=(--llm_config "$LLM_CONFIG")
  fi
  if [[ -n "$LLM_MODEL" ]]; then
    args+=(--llm_model "$LLM_MODEL")
  fi
  if [[ -n "$LLM_BASE_URL" ]]; then
    args+=(--llm_base_url "$LLM_BASE_URL")
  fi
  if [[ "$GUIDANCE_MODE" != "none" ]]; then
    args+=(--guidance "$GUIDANCE_MODE")
    args+=(--guidance_value_weight "$GUIDANCE_VALUE_WEIGHT")
    args+=(--guidance_prior_c "$GUIDANCE_PRIOR_C")
    args+=(--guidance_timeout "$GUIDANCE_TIMEOUT")
  fi
  if [[ "$TEMPORAL_GUIDANCE" != "none" ]]; then
    args+=(--temporal_guidance "$TEMPORAL_GUIDANCE")
    if [[ -n "$TEMPORAL_PROMPT" ]]; then
      args+=(--temporal_prompt "$TEMPORAL_PROMPT")
    fi
    if [[ -n "$TEMPORAL_TIMEOUT" ]]; then
      args+=(--temporal_timeout "$TEMPORAL_TIMEOUT")
    fi
  fi
  "${args[@]}"

  if [[ "$RUN_EVAL" == "1" ]]; then
    echo "==> Eval ${day}-day"
    (cd "${ROOT_DIR}/evaluation" && python eval.py --set_type "${day}d" --evaluation_file_path "$out")
  fi
}

run_query
run_one 3
run_one 5
run_one 7
