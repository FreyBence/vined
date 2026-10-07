#!/usr/bin/env bash

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

# Forward explicit neural requests and worker settings to the standalone CLI.
exec "$PYTHON" src/prepare_neural_data.py "$@"
