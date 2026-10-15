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

## Setup

### Prerequisites

- **Python 3.12.**

```bash
python -m venv .venv && source .venv/bin/activate 
pip install -r requirements.txt          # for downloaders, FTW, FTP, Delineate Anything
```

Delineate Anything inference (`delany/delineate.py`) also needs GDAL; see [delany/README.md](delany/README.md#environment).
Large data (imagery, datasets) is not in git; it goes under `data/`. The fine-tuned weights are on Hugging Face (see above).
