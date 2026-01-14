#!/bin/bash

# Resolve repo root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variables with paths
export OUTPUT_DIR="${ROOT_DIR}/output"    # Path to your output directory
export MODEL_NAME="gpt-4o"     #qwen    #phi4             # Model name (you can change it as needed)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"             # Your OpenAI API key
# export GOOGLE_API_KEY="YOUR_GOOGLE_KEY"                 # Your Google API key
export DAY="3day"                                           # 3day/5day/7day
export SET_TYPE="3day_gpt4o_orig"                            # Set type- name of folder in O/P directory where generated outputs get saved 
export STRATEGY="direct_og"                                # direct_og / direct_param
export CSV_FILE="${ROOT_DIR}/TripCraft/Tripcraftzip/tripcraft_3day.csv"  # Path to your CSV file

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
