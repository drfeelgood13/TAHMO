#!/bin/bash
#SBATCH --job-name=google-maps-test
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/jtoei/TAHMO
#SBATCH --output=/home/jtoei/TAHMO/downloaders/google-maps-test-%j.out
#SBATCH --error=/home/jtoei/TAHMO/downloaders/google-maps-test-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

.venv/bin/python downloaders/downloader_google_maps.py \
    --input test_set.json \
    --output data/google_maps_test_set \
    --skip-existing