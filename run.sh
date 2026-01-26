#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/llm_output/outputs}"  # Path to your output directory
export MODEL_NAME="ollama"                           # local ollama (or ollama:<model>)
export OLLAMA_MODEL="llama3.1:8b"                    # required when MODEL_NAME=ollama
export OLLAMA_BASE_URL="http://localhost:11434"      # ollama server
export OLLAMA_TIMEOUT="999"
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                 # Your Google API key
export DAY="3day"                                           # 3day/5day/7day
export SET_TYPE="3day_llama_para"                            # Set type- name of folder in O/P directory where generated outputs get saved 
export STRATEGY="direct_param"                            # use planner_agent_prompt_direct_param
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
    --strategy $STRATEGY
