#!/usr/bin/env bash
# Source this helper; do not activate a user shell or a Conda environment.
if [[ -n "${VINED_REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$VINED_REPO_ROOT" && pwd)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
if [[ "$VENV_DIR" != /* && "$VENV_DIR" != [A-Za-z]:* ]]; then
    VENV_DIR="$REPO_ROOT/$VENV_DIR"
fi
if [[ -d "$VENV_DIR" ]]; then
    VENV_DIR="$(cd "$VENV_DIR" && pwd)"
fi
if [[ -x "$VENV_DIR/bin/python" ]]; then
    VENV_BIN="$VENV_DIR/bin"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
    VENV_BIN="$VENV_DIR/Scripts"
else
    echo "Missing venv Python in $VENV_DIR. See README.md." >&2
    return 1
fi
export PYTHON="$VENV_BIN/python"
[[ -x "$PYTHON" ]] || export PYTHON="$VENV_BIN/python.exe"
export PATH="$VENV_BIN:$PATH"
export VINED_REPO_ROOT="$REPO_ROOT"
# Local workers import project modules directly from the checkout.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VINED_OUTPUT_DIR="${VINED_OUTPUT_DIR:-$REPO_ROOT/output}"
if [[ "$VINED_OUTPUT_DIR" != /* && "$VINED_OUTPUT_DIR" != [A-Za-z]:* ]]; then
    VINED_OUTPUT_DIR="$REPO_ROOT/$VINED_OUTPUT_DIR"
fi
export VINED_DATA_DIR="${VINED_DATA_DIR:-$VINED_OUTPUT_DIR/datasets}"
if [[ "$VINED_DATA_DIR" != /* && "$VINED_DATA_DIR" != [A-Za-z]:* ]]; then
    VINED_DATA_DIR="$REPO_ROOT/$VINED_DATA_DIR"
fi
export VINED_VISUAL_DIR="${VINED_VISUAL_DIR:-$VINED_DATA_DIR/vis_stim}"
export VINED_REPLAY_DIR="${VINED_REPLAY_DIR:-$VINED_OUTPUT_DIR/visual_replays}"
cd "$REPO_ROOT"

# Shared validation for the positional training/cache launchers.
fail() { echo "Error: $*" >&2; exit 2; }
validate_session() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || fail "COUNT must be a positive integer"
    local uuid='^[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$'
    if [[ "$2" == None && "$1" != 1 ]]; then return; fi
    [[ "$2" =~ $uuid ]] || fail "EID must be a UUID (or None for multiple sessions)"
}
validate_model() {
    case "$1" in mm|encoding|decoding) ;; *) fail "Unsupported model mode: $1" ;; esac
    [[ "$2" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] || fail "MASK_RATIO must be between 0 and 1"
    case "$3" in all|random|vision-clip) ;; *) fail "TASK_VAR must be all, random, or vision-clip" ;; esac
}
