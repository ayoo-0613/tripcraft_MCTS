#!/usr/bin/env bash
set -euo pipefail

# Run MCTS and Solver baselines for 3/5/7-day datasets.
# Outputs are written under OUTPUT_ROOT/{mcts,solver}.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CSV location (supports either Tripcraft/ or TripCraft/).
CSV_DIR="${ROOT_DIR}/Tripcraft/Tripcraftzip"
if [[ ! -d "$CSV_DIR" ]]; then
  CSV_DIR="${ROOT_DIR}/TripCraft/Tripcraftzip"
fi

# Configurable knobs via env vars.
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/baseline_outputs}"
DAYS="${DAYS:-3 5 7}"

# LLM settings (only required when enabling parse/render).
LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434}"
LLM_TIMEOUT="${LLM_TIMEOUT:-999}"

# MCTS settings.
MCTS_OUT="${MCTS_OUT:-${OUTPUT_ROOT}/mcts}"
MCTS_ROLLOUTS="${MCTS_ROLLOUTS:-200}"
MCTS_TOPK="${MCTS_TOPK:-20}"
MCTS_LLM_PARSE_QUERY="${MCTS_LLM_PARSE_QUERY:-1}"   # 1 to enable
MCTS_LLM_RENDER="${MCTS_LLM_RENDER:-0}"             # 1 to enable

# Solver settings.
SOLVER_OUT="${SOLVER_OUT:-${OUTPUT_ROOT}/solver}"
SOLVER_TIMEOUT_MS="${SOLVER_TIMEOUT_MS:-60000}"
SOLVER_LLM_PARSE_QUERY="${SOLVER_LLM_PARSE_QUERY:-1}"   # 1 to enable
SOLVER_LLM_RENDER="${SOLVER_LLM_RENDER:-0}"             # 1 to enable

mkdir -p "$MCTS_OUT" "$SOLVER_OUT"

run_mcts() {
  local day="$1"
  local csv_file="${CSV_DIR}/tripcraft_${day}day.csv"
  local out_jsonl="${MCTS_OUT}/mcts_${day}day.jsonl"
  local debug_dir="${MCTS_OUT}/debug_${day}day"

  local args=(
    --input_csv "$csv_file"
    --output_jsonl "$out_jsonl"
    --rollouts "$MCTS_ROLLOUTS"
    --topk "$MCTS_TOPK"
    --debug_dir "$debug_dir"
  )

  if [[ "$MCTS_LLM_PARSE_QUERY" == "1" || "$MCTS_LLM_RENDER" == "1" ]]; then
    args+=(--llm_model "$LLM_MODEL" --llm_base_url "$LLM_BASE_URL" --query_timeout "$LLM_TIMEOUT")
  fi
  if [[ "$MCTS_LLM_PARSE_QUERY" == "1" ]]; then
    args+=(--llm_parse_query)
  fi
  if [[ "$MCTS_LLM_RENDER" == "1" ]]; then
    args+=(--llm_render_plan)
  fi

  echo "==> MCTS ${day}day -> ${out_jsonl}"
  python -m mcts_baseline.cli "${args[@]}"
}

run_solver() {
  local day="$1"
  local csv_file="${CSV_DIR}/tripcraft_${day}day.csv"
  local out_jsonl="${SOLVER_OUT}/z3_${day}day.jsonl"

  local args=(
    --input_csv "$csv_file"
    --output_jsonl "$out_jsonl"
    --solver_timeout_ms "$SOLVER_TIMEOUT_MS"
  )

  if [[ "$SOLVER_LLM_PARSE_QUERY" == "1" || "$SOLVER_LLM_RENDER" == "1" ]]; then
    args+=(--llm_model "$LLM_MODEL" --llm_base_url "$LLM_BASE_URL" --llm_timeout "$LLM_TIMEOUT")
  fi
  if [[ "$SOLVER_LLM_PARSE_QUERY" == "1" ]]; then
    args+=(--llm_parse_query)
  fi
  if [[ "$SOLVER_LLM_RENDER" == "1" ]]; then
    args+=(--llm_render_plan)
  fi

  echo "==> Solver ${day}day -> ${out_jsonl}"
  python Solver_baseline/JSON_to_CODE.py "${args[@]}"
}

for day in $DAYS; do
  if [[ ! -f "${CSV_DIR}/tripcraft_${day}day.csv" ]]; then
    echo "Missing CSV for ${day}day: ${CSV_DIR}/tripcraft_${day}day.csv" >&2
    exit 1
  fi
  run_mcts "$day"
  run_solver "$day"
done
