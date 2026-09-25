#!/bin/bash

set -e
usage="Usage: bash script/train.sh COUNT EID train|finetune mm|encoding|decoding MASK_RATIO True|False TASK_VAR"
if [[ "${1:-}" == --help ]]; then echo "$usage"; exit 0; fi
[[ $# -eq 7 ]] || { echo "$usage" >&2; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

num_sessions=$1
eid=$2
train_mode=$3
model_mode=$4
mask_ratio=$5
search=$6
task_var=$7
validate_session "$num_sessions" "$eid"
validate_model "$model_mode" "$mask_ratio" "$task_var"
case "$train_mode" in train|finetune) ;; *) fail "train_mode must be train or finetune" ;; esac
[[ "$train_mode" != finetune || "$eid" != None ]] || fail "Fine-tuning requires an actual EID"
case "$search" in True|False) ;; *) fail "search must be True or False" ;; esac
readonly NUM_TUNE_SAMPLES=30
base_path="$VINED_OUTPUT_DIR"
config_dir="$REPO_ROOT/src/configs"
data_path="$VINED_DATA_DIR"
python_file="src/$train_mode.py"
# Python search currently uses ray.init(address="auto"). Keep that contract:
# start a local runtime explicitly with .venv/bin/ray start --head before search.
# RAY_ADDRESS can select an already running runtime. Do not stop user-owned Ray.
if [[ "$search" == True ]]; then
    "$VENV_BIN/ray" status >/dev/null || fail "Start local Ray first: $VENV_BIN/ray start --head"
fi

args=(--eid "$eid" --base_path "$base_path" --mask_ratio "$mask_ratio"
      --num_sessions "$num_sessions" --model_mode "$model_mode"
      --enc_task_var "$task_var" --config_dir "$config_dir" --data_path "$data_path")
[[ "$model_mode" != mm ]] || args+=(--mixed_training)
[[ "$search" != True ]] || args+=(--search --num_tune_sample "$NUM_TUNE_SAMPLES")
"$PYTHON" "$python_file" "${args[@]}"
