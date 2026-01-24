#!/bin/bash
set -euo pipefail

# Resolve repo root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/llm_output/outputs}"  # Path to your output directory
export MODEL_NAME="${MODEL_NAME:-ollama}"                           # ollama / gpt-4o / qwen / phi4
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"                         # Required only for OpenAI models
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                            # Your Google API key
export DAYS="${DAYS:-3 5 7}"                                        # 3/5/7
export STRATEGIES="${STRATEGIES:-reflexion verifier_repair plan_execute react}"     # direct_og / direct_param / react / reflexion / verifier_repair / plan_execute
export POSTPROCESS="${POSTPROCESS:-1}"                             # 1 to generate eval jsonl via Ollama
export SKIP_EXISTING="${SKIP_EXISTING:-0}"                         # 1 to skip generation if jsonl already exists

# Ollama settings (used when MODEL_NAME=ollama or MODEL_NAME=ollama:<model>)
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-999}"

# Align csv layout with run_mcts_eval.sh (server path), with local fallback.
CSV_DIR="${ROOT_DIR}/Tripcraft/Tripcraftzip"
if [[ ! -d "$CSV_DIR" ]]; then
  CSV_DIR="${ROOT_DIR}/TripCraft/Tripcraftzip"
fi

mkdir -p "$OUTPUT_DIR"

# Navigate to the planner directory
cd tools/planner

run_one() {
  local day="$1"
  local strategy="$2"
  local set_type="${strategy}/${day}day"
  local csv_file="${CSV_DIR}/tripcraft_${day}day.csv"
  local output_jsonl="${OUTPUT_DIR}/${strategy}/${day}.jsonl"
  local eval_jsonl="${OUTPUT_DIR}/${strategy}/${day}_eval.jsonl"

  if [[ "$SKIP_EXISTING" == "1" && -s "$output_jsonl" ]]; then
    echo "==> Found existing ${output_jsonl}; skipping generation"
  else
    echo "==> Running ${strategy} for ${day}day with ${MODEL_NAME} -> ${output_jsonl}"
    python sole_planning_mltp.py \
        --day "${day}day" \
        --set_type "$set_type" \
        --output_dir "$OUTPUT_DIR" \
        --output_jsonl "$output_jsonl" \
        --csv_file "$csv_file" \
        --model_name "$MODEL_NAME" \
        --strategy "$strategy"
  fi

  if [[ "$POSTPROCESS" == "1" ]]; then
    echo "==> Converting to eval format -> ${eval_jsonl}"
    if [[ -s "$output_jsonl" ]]; then
      python "${ROOT_DIR}/postprocess/ollama_plan_converter.py" \
        --input_jsonl "$output_jsonl" \
        --csv_file "$csv_file" \
        --output_jsonl "$eval_jsonl" \
        --ollama_model "$OLLAMA_MODEL" \
        --ollama_base_url "$OLLAMA_BASE_URL" \
        --ollama_timeout "$OLLAMA_TIMEOUT"
    else
      echo "==> Skip converting: missing ${output_jsonl}"
    fi
  fi
}

for day in $DAYS; do
  for strategy in $STRATEGIES; do
    run_one "$day" "$strategy"
  done
done
