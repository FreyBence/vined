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

num_sessions=${1}
eid=${2}

echo "${TMPDIR:-}"

user_name=$(whoami)

"$PYTHON" src/create_dataset.py --eid $eid \
                             --num_sessions $num_sessions \
                             --model_mode mm \
                             --mask_ratio 0.1 \
                             --mixed_training \
                             --base_path "${REPO_ROOT}" \
                             --data_path "${VINED_DATA_DIR}"
