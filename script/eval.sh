#!/bin/bash

set -e
usage="Usage: bash script/eval.sh COUNT EID train|finetune mm|encoding|decoding MASK_RATIO TASK_VAR True|False [--overwrite]"
if [[ "${1:-}" == --help ]]; then echo "$usage"; exit 0; fi
[[ $# -eq 7 || $# -eq 8 ]] || { echo "$usage" >&2; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

num_sessions=$1
eid=$2
train_mode=$3
model_mode=$4
mask_ratio=$5
task_var=$6
search=$7
validate_session "$num_sessions" "$eid"
validate_model "$model_mode" "$mask_ratio" "$task_var"
case "$train_mode" in train|finetune) ;; *) fail "train_mode must be train or finetune" ;; esac
[[ "$train_mode" != finetune || "$eid" != None ]] || fail "Fine-tuning requires an actual EID"
case "$search" in True|False) ;; *) fail "search must be True or False" ;; esac
[[ $# -eq 7 || "$8" == --overwrite ]] || fail "Only --overwrite is accepted as the final argument"
readonly MASK_MODE=temporal
args=(--eid "$eid" --mask_mode "$MASK_MODE" --mask_ratio "$mask_ratio"
      --base_path "$VINED_OUTPUT_DIR" --num_sessions "$num_sessions"
      --model_mode "$model_mode" --enc_task_var "$task_var" --data_path "$VINED_DATA_DIR" --wandb)
[[ "$train_mode" != finetune ]] || args+=(--finetune)
[[ "$search" != True ]] || args+=(--param_search)
[[ "$model_mode" != mm ]] || args+=(--mixed_training --save_plot)
[[ $# -eq 7 ]] || args+=(--overwrite)
"$PYTHON" src/eval.py "${args[@]}"
