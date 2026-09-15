#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --job-name="train-multi-gpu"
#SBATCH --output="train-multi-gpu.%j.out"
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000   
#SBATCH -t 0-04:00:00
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

num_sessions=${1}
eid=${2}
model_mode=${3}
dummy_size=${4}
mask_ratio=${5}
task_var=${6}

user_name="yzhang39"
base_path="${VINED_OUTPUT_DIR:-$REPO_ROOT}" # change to your own path
config_dir=$(pwd)/src/configs
data_path="${VINED_DATA_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Multi-node training requires a Linux Slurm allocation." >&2
    exit 2
fi
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip="${head_node_ip%% *}"
args=(src/train.py --eid "$eid" --base_path "$base_path"
      --mask_ratio "$mask_ratio" --num_sessions "$num_sessions"
      --dummy_size "$dummy_size" --model_mode "$model_mode" --multi_gpu
      --enc_task_var "$task_var" --data_path "$data_path" --config_dir "$config_dir")
case "$model_mode" in
    mm) args+=(--mixed_training) ;;
    encoding|decoding) ;;
    *) echo "Unsupported model mode: $model_mode" >&2; exit 2 ;;
esac
# An argument array preserves paths containing spaces on every worker.
srun "$PYTHON" -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" --nproc_per_node 1 \
    --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" "${args[@]}"
