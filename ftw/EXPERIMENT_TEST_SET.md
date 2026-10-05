# 32-Location Ghana Experiment

## Scope

The test split contains 32 labeled Ghana locations from `test_set.json`. Each location is evaluated on its supplied WGS84 bounding box; bounding boxes are not regenerated from centers. The evaluator uses the matching grid labels from `data/ground_truth/filtered_grids.gpkg`, excludes non-crop labels, clips labels and predictions to each box, reprojects to the location's metric UTM CRS, and excludes ground-truth fields smaller than 20 m².

## FTW Sentinel-2 Run

The completed run used the released `FTW_PRUE_EFNET_B7_CCBY` model and two Sentinel-2 acquisitions per location. The runner selected scenes from Microsoft Planetary Computer using the crop-calendar start and end dates for 2024, initially searching within ±14 days and requiring cloud cover below 20%. If a seasonal search returned no scenes, it retried the same bounding box with larger date ranges and relaxed cloud limits, up to ±180 days and below 100% cloud cover. Small input rasters were padded for inference and predictions were cropped back to their original dimensions before polygonization.

Submit and evaluate with:

```bash
sbatch ftw/slurm/test_set.sbatch
python evaluation.py ftw/output/test_set --test-set test_set.json
```

Per-location predictions and rasters are stored under `ftw/output/test_set/<location_label>/`. The evaluator writes `scores_test.csv` and `scores_test_summary.json` in that output directory.

Evaluation uses greedy one-to-one polygon matching at IoU ≥ 0.5. Split and merged-field counts use a 50% overlap/coverage threshold. Mean IoU is calculated over matched fields only; mAP is not calculated by this evaluator.

| Evaluation | Data | Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Mean IoU | Split fields | Merged fields |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test (full images) | 32 images, 1,534 fields | FTW_PRUE_EFNET_B7_CCBY | 0.074 | 0.012 | 0.020 | – | – | 0.586 | 2.2% | 8.2% |

All 32 locations produced evaluable outputs, with no missing locations or evaluation errors. The model produced no polygons for eight locations. The low recall and F1 should therefore be interpreted as model performance, not incomplete coverage.