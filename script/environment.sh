#!/usr/bin/env bash
# Source this helper; do not activate a user shell or a Conda environment.
# sbatch copies job scripts, so submit from the checkout or export VINED_REPO_ROOT.
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
    echo "Missing venv Python in $VENV_DIR. See docs/environment.md." >&2
    return 1
fi
export PYTHON="$VENV_BIN/python"
[[ -x "$PYTHON" ]] || export PYTHON="$VENV_BIN/python.exe"
export PATH="$VENV_BIN:$PATH"
export VINED_REPO_ROOT="$REPO_ROOT"
# Ray/Slurm workers import project modules directly from the shared checkout.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VINED_DATA_DIR="${VINED_DATA_DIR:-$REPO_ROOT/datasets}"
export VINED_VISUAL_DIR="${VINED_VISUAL_DIR:-$VINED_DATA_DIR/vis_stim}"
export VINED_REPLAY_DIR="${VINED_REPLAY_DIR:-$REPO_ROOT/ibl_task_replay}"
cd "$REPO_ROOT"
