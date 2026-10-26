#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=nanogpt-track3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96  # 12 per GPU (partition max)
#SBATCH --gpus-per-node=8
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch run.sbatch [num_trials]   (WANDB_RUN_NAME=... sbatch run.sbatch to name the run)

source $HOME/.bashrc
source ${VENV:-$HOME/training-nano-GPT-speed-run/.venv}/bin/activate
cd $HOME/training-nano-GPT-speed-run  # the script globs data/fineweb10B relative to cwd

# Log to the team W&B server: forge.coreweave.com/wandb/cerulean-labs/midterm-nanogpt
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export WANDB_ENTITY=${WANDB_ENTITY:-cerulean-labs}
export WANDB_PROJECT=${WANDB_PROJECT:-midterm-nanogpt}

# Single-node jobs: NCCL bootstrap over loopback (some nodes pick an unroutable interface and hang)
export NCCL_SOCKET_IFNAME=lo

echo "Start: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv

torchrun --standalone --nproc_per_node=8 train_gpt_simple.py "$@"

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
