#!/bin/bash
# 自动运行并评价 3/5/7 gitday 结果
set -e

MODEL="llama3.1:8b"
GEN_DIR="result_llama3.1:8b/outputs"
EVAL_SCRIPT="evaluation/llm_persona_score.py"

for DAY in 3 5 7; do
    GEN_FILE="$GEN_DIR/mcts_baseline_${DAY}gitday.jsonl"
    LOG_FILE="$GEN_DIR/llm_persona_score_${DAY}gitday.log"
    if [ -f "$GEN_FILE" ]; then
        echo "Evaluating $GEN_FILE ... (output -> $LOG_FILE)"
        python $EVAL_SCRIPT \
            --model $MODEL \
            --gen_file "$GEN_FILE" \
            --print_per_plan \
            > "$LOG_FILE" 2>&1
    else
        echo "File not found: $GEN_FILE"
    fi
done
