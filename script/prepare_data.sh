#!/bin/bash

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

# Optional positional arguments: count, EID, manifest. Defaults select all EIDs.
selection=()
if [[ "${1:-}" == --* ]]; then
    selection=("$@")
else
[[ $# -le 3 ]] || fail "Usage: bash script/prepare_data.sh [COUNT [EID|None [MANIFEST]]] or Python CLI options"
if [[ -n "${1:-}" ]]; then
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || fail "COUNT must be a positive integer"
fi
if [[ -n "${2:-}" && "$2" != None ]]; then
    validate_session "${1:-1}" "$2"
    [[ "${1:-1}" == 1 ]] || fail "An explicit EID requires COUNT=1"
fi
[[ -n "${1:-}" ]] && selection+=(--n-sessions "$1")
[[ -n "${2:-}" && "$2" != "None" ]] && selection+=(--eid "$2")
[[ -n "${3:-}" ]] && selection+=(--eids-file "$3")
fi

"$PYTHON" src/prepare_data.py "${selection[@]}" --base_path "${VINED_DATA_DIR}"
