#!/bin/bash
#SBATCH --job-name=ftw-test-set
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/jtoei/TAHMO
#SBATCH --output=/home/jtoei/TAHMO/ftw/output/test_set/logs/%x-%j.out
#SBATCH --error=/home/jtoei/TAHMO/ftw/output/test_set/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH="/home/jtoei/TAHMO/ftw${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
srun .venv/bin/python ftw/run_ftw_coordinates.py \
    --input test_set.json \
    --output ftw/output/test_set \
    --year 2024 --gpu 0 --num-workers 4
