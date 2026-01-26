#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/lama_output}"  # Path to your output directory
export MODEL_NAME="ollama"                           # local ollama (or ollama:<model>)
export OLLAMA_MODEL="llama3.1:8b"                    # required when MODEL_NAME=ollama
export STRATEGY="template_action"                    # direct_og / direct_param / react / reflexion / verifier_repair / plan_execute / template_action / fixed_direct / fixed_cot / fixed_react / fixed_reflexion
export ACTION_STRATEGY="${ACTION_STRATEGY:-direct}"  # direct / react / reflexion (only for template_action)
export OLLAMA_BASE_URL="http://localhost:11434"      # ollama server
export OLLAMA_TIMEOUT="999"
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                 # Your Google API key
export DAY="3day"                                   # 3day/5day/7day
if [[ "$STRATEGY" == "template_action" ]]; then
  export SET_TYPE="${SET_TYPE:-${STRATEGY}/${ACTION_STRATEGY}/${DAY}}"
  export OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/${STRATEGY}/${ACTION_STRATEGY}/${DAY}.jsonl}"
else
  export SET_TYPE="${SET_TYPE:-${STRATEGY}/${DAY}}"
  export OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/${STRATEGY}/${DAY}.jsonl}"
fi
export ONE_SHOT="${ONE_SHOT:-1}"                    # 1 to use one-shot action selection
ONE_SHOT_FLAG=""
if [[ "$ONE_SHOT" == "1" ]]; then
  ONE_SHOT_FLAG="--one_shot"
fi
export CSV_DIR="${ROOT_DIR}/Tripcraft/Tripcraftzip"
if [[ ! -d "$CSV_DIR" ]]; then
  CSV_DIR="${ROOT_DIR}/TripCraft/Tripcraftzip"
fi
export CSV_FILE="${CSV_FILE:-${CSV_DIR}/tripcraft_${DAY}.csv}"  # Path to your CSV file

# Navigate to the planner directory
cd tools/planner

# Run the Python script with the environment variables
python sole_planning_mltp.py \
    --day $DAY \
    --set_type $SET_TYPE \
    --output_dir $OUTPUT_DIR \
    --csv_file $CSV_FILE \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY \
    ${ACTION_STRATEGY:+--action_strategy "$ACTION_STRATEGY"} \
    $ONE_SHOT_FLAG \
    --output_jsonl "$OUTPUT_JSONL"
