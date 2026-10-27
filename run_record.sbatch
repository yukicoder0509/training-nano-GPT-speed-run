#!/bin/bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --job-name=nanogpt-rec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96  # 12 per GPU (partition max)
#SBATCH --gpus-per-node=8
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn157
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out

# Usage: sbatch run_record.sbatch rec-46 [script args, e.g. --seed 0]   (env overrides record.env, e.g. STOP_STEP=250)
# Runs train_gpt.py from a snapshot of the given git ref, so several records can be queued at once.
# Logs (train/val + optimization dynamics) go to the personal project; never to cerulean-labs.

REF=${1:?usage: sbatch run_record.sbatch <git ref> [args]}; shift
REPO=$HOME/training-nano-GPT-speed-run
source $HOME/.bashrc
source ${VENV:-$REPO/.venv}/bin/activate
cd $REPO  # the scripts glob data/fineweb10B and write logs/ relative to cwd

SRC=$REPO/logs/src-$SLURM_JOB_ID
mkdir -p $SRC && git archive $REF | tar -x -C $SRC
export PYTHONPATH=$SRC

# per-record settings (e.g. STOP_STEP = the record's claimed step); variables already set by the caller win
if [ -f $SRC/record.env ]; then
    while IFS='=' read -r k v; do
        [[ -z "$k" || "$k" == \#* ]] && continue
        [ -z "${!k+x}" ] && export "$k=$v"
    done < $SRC/record.env
fi

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}
export WANDB_ENTITY=${WANDB_ENTITY:-bendoris0509-national-yang-ming-chiao-tung-university}
export WANDB_PROJECT=${WANDB_PROJECT:-nano-gpt}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$REF}
export RECORD=$REF

# Single-node jobs: NCCL bootstrap over loopback (some nodes pick an unroutable interface and hang)
export NCCL_SOCKET_IFNAME=lo

echo "Start: $(date '+%Y-%m-%d %H:%M:%S')  Node: $(hostname)  ref: $REF ($(git rev-parse --short $REF))  STOP_STEP=$STOP_STEP"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

torchrun --standalone --nproc_per_node=8 $SRC/train_gpt.py "$@"

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
