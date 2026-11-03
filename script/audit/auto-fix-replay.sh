#!/usr/bin/env bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/environment.sh"
exec "$PYTHON" script/audit/auto_fix_replay.py "$@"
