#!/bin/bash
#SBATCH --job-name=ftp-test-set-b7
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/jtoei/TAHMO
#SBATCH --output=/home/jtoei/TAHMO/ftp/output/ftp-test-set-b7-%j.out
#SBATCH --error=/home/jtoei/TAHMO/ftp/output/ftp-test-set-b7-%j.err

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

nvidia-smi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
srun .venv/bin/python ftp/run_ftp_prue.py \
    --input data/planet_test_set \
    --output ftp/output/test_set_b7 \
    --model b7 \
    --device cuda \
    --channel-order b-a \
    --skip-existing