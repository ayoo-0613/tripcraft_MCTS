#!/bin/bash

# Resolve repo root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${ROOT_DIR}/llm_gen_plan"    # Path to your output directory
export MODEL_NAME="ollama:llama3.1:8b"     # qwen / phi4 / ollama:llama3.1:8b
export OPENAI_API_KEY="YOUR_OPENAI_KEY"             # Your OpenAI API key (required for OpenAI models)
# export OLLAMA_MODEL="llama3.1:8b"                # Used when MODEL_NAME="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"   # Optional Ollama server URL

# Strategies for llm_direct pipeline
STRATEGIES=("llm_direct" "llm_cot" "llm_reflexion")

for STRATEGY in "${STRATEGIES[@]}"; do
    for DAY in 3day 5day 7day; do
        CSV_FILE="${ROOT_DIR}/Tripcraft/Tripcraftzip/tripcraft_${DAY}.csv"
        OUTPUT_SUBDIR="${OUTPUT_DIR}/${STRATEGY}/${DAY}"
        python tools/planner/sole_planning_mltp.py \
            --day $DAY \
            --set_type $OUTPUT_SUBDIR \
            --output_dir "${OUTPUT_DIR}" \
            --csv_file $CSV_FILE \
            --model_name $MODEL_NAME \
            --strategy $STRATEGY
    done
done
