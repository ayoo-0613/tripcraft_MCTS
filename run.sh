#!/bin/bash

# Resolve repo root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${ROOT_DIR}/llm_gen_plan"    # Path to your output directory
export OUTPUT_JSONL_PREFIX="${OUTPUT_DIR}/llm_template_baseline"
export MODEL_NAME="ollama:llama3.1:8b"     # qwen / phi4 / ollama:llama3.1:8b
export OPENAI_API_KEY="YOUR_OPENAI_KEY"             # Your OpenAI API key (required for OpenAI models)
# export MODEL_NAME="ollama:llama3.1:8b"           # Use local Ollama model (example)
# export OLLAMA_MODEL="llama3.1:8b"                # Used when MODEL_NAME="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"   # Optional Ollama server URL
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                 # Your Google API key
export STRATEGY="direct"                                  # direct / cot / react / reflexion
for DAY in 3day 5day 7day; do
    CSV_FILE="${ROOT_DIR}/Tripcraft/Tripcraftzip/tripcraft_${DAY}.csv"
    OUTPUT_JSONL="${OUTPUT_JSONL_PREFIX}_${DAY}.jsonl"
    python tools/planner/sole_planning_template_llm.py \
        --output_jsonl $OUTPUT_JSONL \
        --csv_file $CSV_FILE \
        --model_name $MODEL_NAME \
        --strategy $STRATEGY
done
