# FTW: Fields of The World

This folder is a copy of **[ftw-baselines](https://github.com/fieldsoftheworld/ftw-baselines)**, the code of the
**Fields of The World (FTW)** project: a benchmark dataset and baseline models for field-boundary segmentation from
Sentinel-2 imagery ([project documentation and citation](FTW.md), [dataset](https://source.coop/repositories/kerner-lab/fields-of-the-world/description)).
It is MIT licensed ([LICENSE](LICENSE)). The upstream README is kept as [FTW.md](FTW.md); everything under
`ftw_tools/`, `configs/`, `tests/` and `workflows/` is upstream code (we did not modify it for the experiment). Our addition is:

| File | Purpose |
|---|---|
| [run_ftw_coordinates.py](run_ftw_coordinates.py) | runs the released FTW model for a list of bounding boxes |

We use the released model **`FTW_PRUE_EFNET_B7_CCBY`** (FTW v3.1 "PRUE", EfficientNet-B7, three classes: field,
boundary, neither; trained on CC-BY / CC0 data only). It takes **two Sentinel-2 windows** (start and end of the
growing season) as input. Its weights are downloaded automatically from the ftw-baselines GitHub release on first use
(see `ftw_tools/inference/model_registry.py`).

## What `run_ftw_coordinates.py` does

The upstream CLI (`ftw inference all`) takes a bounding box and does scene selection, download, inference and
polygonization in one call. `run_ftw_coordinates.py` wraps the same upstream functions so that it can be run over a
whole list of areas, resumed, and given exact bounding boxes. For each area in the input JSON:

1. **Select two Sentinel-2 scenes** from Microsoft Planetary Computer for the box and the year (`--year`, default
   2024), using the crop-calendar start and end of season (the upstream `scene_selection`), with a ±14 day window
   (`--buffer-days`) and cloud cover below 20 % (`--cloud-cover-max`). If nothing is found it retries the same box with
   progressively looser settings: ±30, ±60 then ±180 days, all at up to 100 % cloud.
2. **Build the model input** `inference_data.tif` (the two windows stacked) from those scenes.
3. **Pad** rasters smaller than 128 px at the bottom/right (areas are small), run inference, and **crop the padding
   off** again so the prediction has the original footprint.
4. **Run the model** (`--model`, default `FTW_PRUE_EFNET_B7_CCBY`; resize factor 2, batch size 2) on the GPU
   (`--gpu N`, `-1` = CPU).
5. **Polygonize** the prediction (upstream `polygonize`) to `polygons.parquet`.

An area is skipped if its `polygons.parquet` exists (use `--overwrite` to redo), and a failed area does not stop the
others; failures are listed at the end and make the exit status 1.

### Input

The area list JSON of the experiments (see [../downloaders/README.md](../downloaders/README.md#input-the-area-list)).
The `bounding_box` of each location (WGS84) is used exactly as given; the `location_label` becomes the output folder
name. The legacy `{"prefix", "width_km", "coordinates": [[lat, lon], ...]}` format (square around each point) also works.

### Output

Per area, in `<output>/<label>/`:

| File | Content |
|---|---|
| `inference_data.tif` | the model input (two Sentinel-2 windows) |
| `inference_output.tif` | predicted classes |
| `polygons.parquet` | field polygons |
| `inference_data_padded.tif`, `inference_output_padded.tif` | only for areas smaller than 128 px |

On our test set the model produced no polygons at all for 8 of the 32 areas, so an empty result is expected for some
small smallholder areas.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `--output` | required | area list, output folder |
| `--year` | 2024 | Sentinel-2 acquisition year |
| `--model` | `FTW_PRUE_EFNET_B7_CCBY` | any model name from the registry that takes two windows |
| `--gpu` | -1 | CUDA device index, -1 = CPU |
| `--batch-size`, `--num-workers` | 2, 4 | inference batch size, data-loader workers |
| `--cloud-cover-max`, `--buffer-days` | 20, 14 | first scene search; see above for the fallbacks |
| `--overwrite`, `--verbose` | off | redo finished areas, print scene selection details |

## How we ran the experiment

32 Ghana test areas, 2024 Sentinel-2, on one A100 GPU (8 CPUs, 4 GB per CPU). From the repository root:

```bash
export PYTHONPATH="$PWD/ftw"          # so that ftw_tools is importable without installing it
python ftw/run_ftw_coordinates.py \
    --input downloaders/inputs/test_set.json \
    --output ftw/output/test_set \
    --year 2024 --gpu 0 --num-workers 4
```

The machine needs internet access (Planetary Computer for Sentinel-2, GitHub for the model weights on first use).
Note that the scene search is live: results can differ if Planetary Computer's catalogue changes.

### Environment

We used a virtual environment at the repository root (`.venv`) created from
[../requirements.txt](../requirements.txt), plus `PYTHONPATH=ftw` as above. Alternatively use the upstream project definition
with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.12):

```bash
cd ftw && uv sync
uv run python run_ftw_coordinates.py --input ../downloaders/inputs/test_set.json --output output/test_set --gpu 0
```

## Tests

The upstream tests (`ftw/tests`) run with `pytest`; `tests/test_run_ftw_coordinates.py` covers our script.
