#!/bin/bash
# Pick the candidate with the best Ghana validation mask mAP50 (max_det 300) and score it, plus the current
# combined_1m, once on the test tiles.
set -e
export HF_HUB_OFFLINE=1 YOLO_OFFLINE=True
cd /scratch/vladparau/Delineate-Anything
P=$HOME/.conda/envs/planet/bin/python
BEST=$($P -c "
import pandas as pd
s = pd.read_csv('finetune/tile_experiments/scores.csv')
s = s[(s.split == 'val') & (s.dataset == 'ghana') & (s.max_det == 300) & s.model.str.startswith('combined_1m_unfrozen')]
print(s.sort_values('mask_map50').iloc[-1].model)")
echo "best on Ghana validation: $BEST"
$P finetune/val_tiles.py --models combined_1m "$BEST" --max-det 300 --split test
