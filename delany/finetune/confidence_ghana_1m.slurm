#!/bin/bash
# Re-run the Ghana-only model (1) on the 32 Ghana test images with save_detections, then per-field confidences.
#   submit:  sbatch finetune/confidence_ghana_1m.slurm      (from the repo root)
#SBATCH --job-name=delany-conf
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=finetune/runs/%x_%j.log

set -euo pipefail
PLANET=$HOME/.conda/envs/planet/bin/python
export HF_HUB_OFFLINE=1
export YOLO_OFFLINE=True
cd "$SLURM_SUBMIT_DIR"

PROJ_DATA=$HOME/.conda/envs/planet/share/proj $PLANET delineate.py -b finetune/ghana_1m_conf_batch.yaml
$PLANET finetune/polygon_confidence.py finetune/results/ghana_1m_conf
