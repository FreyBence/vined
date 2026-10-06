#!/usr/bin/env bash
# Generate every trial for one EID, or all EIDs when --eid is omitted.
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

output="${VINED_REPLAY_DIR}"
config="data/replay-config.json"
projection=on
reload=()
trials=()
selection=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            echo "Usage: bash script/generate_replay.sh [--eid EID] [--projection on|off]"
            echo "Optional --trial-number N selects an original zero-based trial; otherwise all trials."
            echo "Without --eid, uses all EIDs in data/eids.txt."
            echo "Saves compressed lossless frames and metadata; no MP4 videos."
            echo "Overwrites selected trials under the output directory's EID folders."
            echo "Optional: --force-reload --config FILE --output DIR"
            echo "Reuses cached data and permits remote lookup/download of missing data."
            exit 0 ;;
        --force-reload) reload=(--force-reload); shift ;;
        --eid|--projection|--config|--output|--trial-number|--trial-id)
            [[ $# -ge 2 && -n "$2" ]] || fail "Missing value for $1"
            case "$1" in
                --eid) selection=(--eid "$2") ;;
                --projection) projection="$2" ;;
                --config) config="$2" ;;
                --output) output="$2" ;;
                --trial-number|--trial-id) trials+=(--trial-id "$2") ;;
            esac
            shift 2 ;;
        *) fail "Unknown option: $1. Use --help." ;;
    esac
done
exec "$PYTHON" src/generate_replays.py --config "$config" --output "$output" \
    --projection "$projection" "${selection[@]}" "${reload[@]}" "${trials[@]}"
