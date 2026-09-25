#!/usr/bin/env bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
# Shared --eid / --eids-file / --n-sessions arguments; defaults to data/eids.txt.
"$PYTHON" -m utils.sessions --cache "$@"
