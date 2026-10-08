#!/usr/bin/env bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
"$PYTHON" src/eval.py "$@"
