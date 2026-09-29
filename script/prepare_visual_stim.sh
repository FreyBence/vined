#!/bin/bash

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

# Without --eid, discover all published replays in the configured replay directory.
exec "$PYTHON" src/prepare_visual_stim.py "$@"
