#!/usr/bin/env bash
set -e
# Slurm spools the script: locate the original checkout through the submit directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    _root="${VINED_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the checkout}}"
    [[ -f "$_root/environment.sh" ]] && _root="$_root/.."
    source "$_root/script/environment.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
fi
while IFS= read -r eid || [[ -n "$eid" ]]; do
    eid="${eid%$'\r'}"
    [[ -z "$eid" ]] && continue
    bash "$REPO_ROOT/script/create_dataset.sh" 1 "$eid"
done < "$REPO_ROOT/data/test_eids.txt"
