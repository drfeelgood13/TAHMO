#!/bin/bash
# Test the fine-tuned model on the held-out test split and compare it with zero-shot Delineate Anything v2.
#   submit:  sbatch finetune/test.slurm            (from the repo root)
#   skip 1): SKIP_MAP=1 sbatch finetune/test.slurm
#   1 m:     NAME=ghana_1m DATA=yolo_dataset_1m sbatch finetune/test.slurm
#   val:     NAME=ghana_1m_val DATA=yolo_dataset_1m SPLIT=val SKIP_MAP=1 sbatch finetune/test.slurm
#   MODEL:   run in finetune/runs whose weights step 1 uses (default $NAME), e.g.
#            NAME=combined_viedia MODEL=combined_1m DATA=yolo_dataset_viedia sbatch finetune/test.slurm
#   1) tile mAP on yolo_dataset/images/test for the fine-tuned and the original model
#   2) full delineation of the 32 test images -> finetune/results/$NAME, scored with evaluate.py
#SBATCH --job-name=delany-test
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=finetune/runs/%x_%j.log

set -euo pipefail

NAME=${NAME:-ghana_v2}
MODEL=${MODEL:-$NAME}
DATA=${DATA:-yolo_dataset}
SPLIT=${SPLIT:-test}   # images scored by evaluate.py; must match the include list of ${NAME}_batch.yaml
PLANET=$HOME/.conda/envs/planet/bin/python   # ultralytics + gdal + geopandas: inference and scoring

export HF_HUB_OFFLINE=1
export YOLO_OFFLINE=True
cd "$SLURM_SUBMIT_DIR"

if [ -z "${SKIP_MAP:-}" ]; then
echo "=== 1) tile-level mAP on the test split ==="
NAME=$NAME MODEL=$MODEL DATA=$DATA $PLANET - <<'EOF'
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

models = {
    "zero-shot v2": hf_hub_download(repo_id="MykolaL/DelineateAnything", filename="DelineateAnythingv2.pt"),
    "fine-tuned": f"finetune/runs/{os.environ['MODEL']}/weights/best.pt",
}
for label, weights in models.items():
    m = YOLO(weights).val(data=f"finetune/{os.environ['DATA']}/data.yaml", split="test", imgsz=512, batch=16,
                          project="finetune/runs", name=f"{os.environ['NAME']}_test_{label.split()[0]}",
                          exist_ok=True, plots=False)
    print(f"{label:13s} mask P {m.seg.mp:.3f} R {m.seg.mr:.3f} mAP50 {m.seg.map50:.3f} mAP50-95 {m.seg.map:.3f} | "
          f"box mAP50 {m.box.map50:.3f} mAP50-95 {m.box.map:.3f}")
EOF
fi

echo "=== 2) full-image delineation of the test images ==="
# the env is not activated, so point GDAL at its PROJ database (otherwise reading the CRS fails)
PROJ_DATA=$HOME/.conda/envs/planet/share/proj $PLANET delineate.py -b finetune/${NAME}_batch.yaml

echo "=== scores: fine-tuned ==="
$PLANET finetune/evaluate.py finetune/results/$NAME --split $SPLIT --data $DATA
echo "=== scores: fine-tuned, simplified ==="
$PLANET finetune/evaluate.py finetune/results/$NAME --split $SPLIT --data $DATA --simp
