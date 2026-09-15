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

echo "${TMPDIR:-}"

num_sessions=${1:-20}  # Default to 20 if not provided
eid=${2:-"None"}      # Default to "None" if not provided

user_name=$(whoami)
video_path=${VINED_REPLAY_DIR}
base_path=${VINED_VISUAL_DIR}

if ! [[ "$num_sessions" =~ ^[0-9]+$ ]]; then
    echo "Error: num_sessions must be an integer"
    exit 1
fi

if [ "$num_sessions" -eq 1 ]; then
    echo "Download data for single session"
    if [ "$eid" = "None" ]; then
        echo "Error: eid must be provided for single session"
        exit 1
    fi
else
    echo "Download data for multiple sessions"
    eid="None"
fi

"$PYTHON" src/prepare_visual_stim.py --n_sessions $num_sessions \
                           --eid $eid \
                           --video_dir "${video_path}" \
                           --output_dir "${base_path}"
