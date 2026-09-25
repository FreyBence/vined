#!/bin/bash

set -e
usage="Usage: bash script/create_dataset.sh COUNT EID"
if [[ "${1:-}" == --help ]]; then echo "$usage"; exit 0; fi
[[ $# -eq 2 ]] || { echo "$usage" >&2; exit 2; }
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

num_sessions=$1
eid=$2
validate_session "$num_sessions" "$eid"
# Cache the active spike/vision modalities; masking happens during training.
readonly MODEL_MODE=mm
"$PYTHON" src/create_dataset.py --eid "$eid" --num_sessions "$num_sessions" \
    --model_mode "$MODEL_MODE" --data_path "$VINED_DATA_DIR"
