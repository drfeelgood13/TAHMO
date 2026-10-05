#!/bin/bash
#SBATCH --job-name=planet-test-set
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3900M
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/jtoei/TAHMO
#SBATCH --export=ALL
#SBATCH --output=/home/jtoei/TAHMO/downloaders/planet-test-set-%j.out
#SBATCH --error=/home/jtoei/TAHMO/downloaders/planet-test-set-%j.err

set -euo pipefail
if [[ -z "${PL_API_KEY:-}" ]]; then
    printf 'PL_API_KEY must be set in the environment used for sbatch.\n' >&2
    exit 2
fi

.venv/bin/python downloaders/downloader_planet.py \
    --input test_set.json \
    --output data/planet_test_set