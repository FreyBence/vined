#!/usr/bin/env bash
# Produce inspection videos from saved replay images for one EID.
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
exec "$PYTHON" src/preview_replays.py "$@"
