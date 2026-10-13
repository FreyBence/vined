#!/usr/bin/env bash
# Compare two runs: compare_evaluations.sh SCENARIO_A ATTEMPT_A SCENARIO_B ATTEMPT_B
# Inspect targets: compare_evaluations.sh SCENARIO ATTEMPT --ground-truth
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
"$PYTHON" src/compare_evaluations.py "$@"
