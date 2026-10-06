#!/bin/bash

set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"

# Explicit prepared inputs; alignment does not acquire sources or split datasets.
"$PYTHON" src/prepare_alignment.py "$@"
