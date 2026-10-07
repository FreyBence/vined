#!/bin/bash

set -e
usage="Usage: bash script/train_multi_gpu.sh COUNT EID mm|encoding|decoding MASK_RATIO TASK_VAR [GPU_COUNT] [TRAINING_OPTIONS...]"
if [[ "${1:-}" == --help ]]; then echo "$usage"; exit 0; fi
[[ $# -ge 5 ]] || { echo "$usage" >&2; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

num_sessions=$1
eid=$2
model_mode=$3
mask_ratio=$4
task_var=$5
shift 5
validate_session "$num_sessions" "$eid"
validate_model "$model_mode" "$mask_ratio" "$task_var"
base_path="$VINED_OUTPUT_DIR"
config_dir="$REPO_ROOT/src/configs"
data_path="$VINED_DATA_DIR"
# By default torchrun launches one worker per visible CUDA GPU.
# CUDA_VISIBLE_DEVICES can restrict which GPUs are used on this PC.
gpu_count=gpu
if [[ $# -gt 0 && "$1" != --* ]]; then gpu_count=$1; shift; fi
[[ "$gpu_count" == gpu || "$gpu_count" =~ ^[1-9][0-9]*$ ]] || fail "GPU_COUNT must be a positive integer"

args=(src/train.py --eid "$eid" --base_path "$base_path"
      --mask_ratio "$mask_ratio" --num_sessions "$num_sessions"
      --model_mode "$model_mode" --multi_gpu
      --enc_task_var "$task_var" --data_path "$data_path" --config_dir "$config_dir")
case "$model_mode" in
    mm) args+=(--mixed_training) ;;
    encoding|decoding) ;;
    *) echo "Unsupported model mode: $model_mode" >&2; exit 2 ;;
esac
# Standalone rendezvous is local to this PC; no scheduler or node list is needed.
"$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node "$gpu_count" "${args[@]}" "$@"
