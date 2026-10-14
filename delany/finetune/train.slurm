#!/bin/bash
# Fine-tune Delineate Anything v2 on the Ghana dataset on DelftBlue.
#   submit:  sbatch finetune/train.slurm            (from the repo root)
#   resume:  RESUME=1 sbatch finetune/train.slurm   (continues runs/$NAME from last.pt)
#   1 m:     NAME=ghana_1m DATA=yolo_dataset_1m sbatch finetune/train.slurm
#   options: EPOCHS (30), PATIENCE (10), NBS (64, effective batch per update), FREEZE (10), LR0 (1e-4), EXTRA (more train.py flags)
#SBATCH --job-name=delany-ft
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=finetune/runs/%x_%j.log

set -euo pipefail

NAME=${NAME:-ghana_v2}
DATA=${DATA:-yolo_dataset}
EPOCHS=${EPOCHS:-30}
PATIENCE=${PATIENCE:-10}
NBS=${NBS:-64}
FREEZE=${FREEZE:-10}
LR0=${LR0:-1e-4}
EXTRA=${EXTRA:-}   # further train.py options, e.g. "--copy-paste 0.3 --select map50 --max-det 1000"

# compute nodes are offline: use the cached DelineateAnythingv2.pt and skip update checks
export HF_HUB_OFFLINE=1
export YOLO_OFFLINE=True
export PATH="$HOME/.conda/envs/planet/bin:$PATH"

# run from the repo root so the AMP check finds ./yolo11n.pt instead of downloading it
cd "$SLURM_SUBMIT_DIR"

nvidia-smi
if [ "${RESUME:-0}" = "1" ]; then
    python finetune/train.py --name "$NAME" --resume
else
    python finetune/train.py --batch 4 --epochs "$EPOCHS" --patience "$PATIENCE" --nbs "$NBS" --freeze "$FREEZE" --lr0 "$LR0" --name "$NAME" --data "$DATA" $EXTRA
fi
