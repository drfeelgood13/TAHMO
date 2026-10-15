# TAHMO: field-boundary delineation on smallholder farmland

Code for comparing three field-boundary delineation models on smallholder farms in Ghana (with Vietnam and Cambodia
added for fine-tuning), and for fine-tuning Delineate Anything v2. Each folder has its own README with the details
needed to reproduce its part.

| Folder | What it is | README |
|---|---|---|
| [downloaders/](downloaders/) | Google Maps and PlanetScope imagery downloaders, and the area lists (`inputs/`) used by all experiments | [downloaders/README.md](downloaders/README.md) |
| [ftw/](ftw/) | **FTW** (Fields of The World, `FTW_PRUE_EFNET_B7_CCBY`) on Sentinel-2 | [ftw/README.md](ftw/README.md) |
| [ftp/](ftp/) | **FTP** (FTW-PRUE+, EfficientNet-B7) on PlanetScope 3 m | [ftp/README.md](ftp/README.md) |
| [delany/](delany/) | **Delineate Anything v2**: inference, dataset preparation and fine-tuning | [delany/README.md](delany/README.md) |

## Experiments

All experiments use the same 32 held-out Ghana test areas ([downloaders/inputs/ghana_test.json](downloaders/inputs/ghana_test.json)),
taken from a set of 110 Ghana areas with field labels.

1. **Base-model comparison.** FTW, FTP and Delineate Anything v2 on the Ghana test areas (a Vietnam and Cambodia
   ground-truth set was also used for Delineate Anything). Each model gets the imagery it is built for:
   - FTW: Sentinel-2, fetched automatically from the coordinates → [ftw/README.md](ftw/README.md)
   - FTP: PlanetScope, 3 m → [downloaders/README.md](downloaders/README.md) to get the imagery, [ftp/README.md](ftp/README.md) to run
   - Delineate Anything v2: Google Maps, 1 m → [downloaders/README.md](downloaders/README.md), [delany/README.md](delany/README.md)
2. **Resolution comparison.** Delineate Anything v2 (zero-shot) on Google Maps imagery at 0.3, 1, 2, 5, 10 and 20 m
   → imagery with `--resolutions` ([downloaders/README.md](downloaders/README.md)), inference
   ([delany/README.md](delany/README.md)).
3. **Fine-tuning Delineate Anything v2 at 1 m**, tested on the held-out test set
   ([delany/README.md](delany/README.md#fine-tuning)):
   - fine-tuned on the Ghana split (backbone frozen)
   - fine-tuned on the Ghana + VIEDIA (Vietnam, Cambodia) split (backbone frozen)
   - fine-tuned on the Ghana + VIEDIA split with the backbone unfrozen

The resulting fine-tuned weights (one `.pt` per variant) and a description of the dataset splits are published on
Hugging Face: [huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2](https://huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2).
The exact areas of each split are listed in [downloaders/inputs/](downloaders/inputs/) (`ghana_train/val/test.json`,
`viedia_train/val/test.json`); all area files share one format, described in
[downloaders/README.md](downloaders/README.md#input-the-area-list).

## Evaluation

[evaluate.py](evaluate.py) scores the output folder of any of the three models against the field labels, so every
experiment is evaluated the same way. Run from the repository root:

```bash
python evaluate.py <results folder> [--areas <area list>] [--labels <labels>] [--extended] [--simp]
```

| Experiment | Command |
|---|---|
| FTW | `python evaluate.py ftw/output/test_set` |
| FTP | `python evaluate.py ftp/output/test_set_b7` |
| Delineate Anything, zero-shot or fine-tuned | `python evaluate.py data/delany_1m` |
| Resolution sweep | one call per resolution, e.g. `python evaluate.py data/delany_0p3m` |
| Fine-tuned, on the VIEDIA test tiles | `python evaluate.py data/delany_viedia --areas downloaders/inputs/viedia_test.json` |

**Areas.** `--areas` is the area list the model was run on (default: the 32 Ghana test areas,
[ghana_test.json](downloaders/inputs/ghana_test.json)). Each area's bounding box, in its UTM zone, is the extent in
which labels and predictions are compared, so all models are judged on the same ground whatever imagery they used.
For VIEDIA files, every labelled 512 m tile (`subareas`) is scored on its own.

**Predictions.** Found per area label in whichever layout the model writes:

| Model | File |
|---|---|
| Delineate Anything | `<results>/<label>.gpkg` (`<label>.simp.gpkg` with `--simp`) |
| FTW, FTP | `<results>/<label>/polygons.parquet` (empty file = no fields found, scored as 0 predictions) |

A resolution suffix is accepted (`<label>_1m.gpkg`, `<label>_0p3m/...`); keep one resolution per results folder.
For VIEDIA, a tile without its own prediction is scored on the whole area's prediction clipped to the tile. Areas
without any prediction are listed and left out of the scores.

**Labels.** `--labels` defaults to `data/datasets-original/ground_truth_ghana/filtered_grids.gpkg` (Ghana; the area's
`grid`, non-crop fields excluded) or `data/datasets-original/VIEDIA 1m - ground_truth` (VIEDIA;
`<area>/<area>_r<row>_c<col>.gpkg`). Labels and predictions are clipped to the extent; pieces under 20 m² are dropped.

**Metrics.** A prediction and a label match if IoU ≥ 0.5 (one-to-one, greedy by IoU); precision, recall and F1 count
matched polygons. Also reported: mean IoU of the matches, over-segmented labels (covered by ≥ 2 predictions) and
under-segmenting predictions (covering ≥ 2 labels), where "covered" means ≥ 50 % of the smaller polygon.
`--extended` adds area omission/commission, boundary precision/recall/F1 within 2.5 and 5 m, and PoLiS (defined
in the docstring of [evaluate.py](evaluate.py)). For Ghana these are judged only inside the convex hull of the labelled fields, as the
labels cover only the surveyed fields; VIEDIA is fully labelled and judged over the whole tile.

**Output**, written into the results folder (`<areas>` is the area list's name, e.g. `ghana_test`):

| File | Content |
|---|---|
| `scores_<areas>.csv` | one row per area/tile: counts, precision, recall, F1, status (`ok`, `missing`, `error`) |
| `scores_<areas>_summary.json` | totals and overall metrics, missing areas |
| `scores_<areas>_extended.csv` | per-area extended metrics (with `--extended`) |

## Setup

### Prerequisites

- **Python 3.12.**

```bash
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt          # for downloaders, FTW, FTP, Delineate Anything
```

Delineate Anything inference (`delany/delineate.py`) also needs GDAL; see [delany/README.md](delany/README.md#environment).
Large data (imagery, datasets) is not in git; it goes under `data/`. The fine-tuned weights are on Hugging Face (see above).
