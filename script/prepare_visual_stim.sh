#!/bin/bash
#SBATCH -A bcxj-delta-cpu 
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 100000
#SBATCH -t 0-01
#SBATCH --export=ALL

set -e
# Slurm spools the script: locate the original checkout through the submit directory.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    _root="${VINED_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the checkout}}"
    [[ -f "$_root/environment.sh" ]] && _root="$_root/.."
    source "$_root/script/environment.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.sh"
fi

# Optional positional arguments: count, EID, manifest. Defaults select all EIDs.
selection=()
if [[ "${1:-}" == --* ]]; then
    selection=("$@")
else
[[ -n "${1:-}" ]] && selection+=(--n-sessions "$1")
[[ -n "${2:-}" && "$2" != "None" ]] && selection+=(--eid "$2")
[[ -n "${3:-}" ]] && selection+=(--eids-file "$3")
fi

"$PYTHON" src/prepare_visual_stim.py "${selection[@]}" --video_dir "${VINED_REPLAY_DIR}" --output_dir "${VINED_VISUAL_DIR}"
