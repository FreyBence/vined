#!/bin/bash

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

exec "$PYTHON" src/create_dataset.py "$@"
