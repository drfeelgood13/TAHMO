# Delineate Anything v2

This folder is a copy of **Delineate Anything v2**, a resolution-agnostic field-boundary instance-segmentation model
(YOLO11-based, built on [Ultralytics](https://github.com/ultralytics/ultralytics)) trained on FBIS-73M at 0.25–10 m.

- Code: [github.com/Lavreniuk/Delineate-Anything](https://github.com/Lavreniuk/Delineate-Anything) · [project page](https://lavreniuk.github.io/Delineate-Anything/)
- Paper: [arXiv:2607.19069](https://arxiv.org/abs/2607.19069) (v2); earlier: [arXiv:2504.02534](https://arxiv.org/abs/2504.02534), [arXiv:2511.13417](https://arxiv.org/abs/2511.13417)
- Weights: [huggingface.co/MykolaL/DelineateAnything](https://huggingface.co/MykolaL/DelineateAnything) (`DelineateAnythingv2.pt`, 125 MB)
- License: AGPL-3.0 ([LICENSE](LICENSE)). The upstream README, with citations, is kept as [DELANY.md](DELANY.md) and the
  upstream parameter documentation as [delineation_config_guide.md](delineation_config_guide.md).

Upstream code (`delineate.py`, `methods/`, `simplification/`) is used as is. Our additions are in
[finetune/](finetune/):

| File | Purpose |
|---|---|
| [finetune/conf_zero_shot.yaml](finetune/conf_zero_shot.yaml) | inference config used for every run (upstream defaults, worker counts fitted to 8 CPUs) |
| [finetune/prepare_dataset.py](finetune/prepare_dataset.py) | Ghana 1 m mosaics + field labels → YOLO segmentation dataset |
| [finetune/prepare_viedia.py](finetune/prepare_viedia.py) | VIEDIA tiles + labels → YOLO segmentation dataset |
| [finetune/train.py](finetune/train.py) | fine-tunes `DelineateAnythingv2.pt` |

## Environment

```bash
conda create -n delany python=3.11 && conda activate delany
conda install -c conda-forge gdal
pip install -r requirements.txt          # ultralytics 8.3.148, rasterio, shapely, geopandas, ...
```

A GPU is strongly recommended. The first run downloads the weights from Hugging Face; on an offline node download
them beforehand and set `HF_HUB_OFFLINE=1 YOLO_OFFLINE=True`.

## Inference (zero-shot or fine-tuned)

`delineate.py` is driven by a *batch* file that lists image folders, and a *config* file with the model and
post-processing settings. Run from this folder (`delany/`).

**Images.** `data_root` contains one subfolder per area, each with a `mosaic_utm.tif` (3-band 8-bit RGB GeoTIFF in a
metric UTM CRS, nodata = 0), as produced by [../downloaders/](../downloaders/README.md). Which folders to process is
set with `include`.

**Batch file**, e.g. `batch_1m.yaml`:

```yaml
base_config: finetune/conf_zero_shot.yaml
data_root: /abs/path/data/ghana_1m          # <area>/mosaic_utm.tif
output_root: /abs/path/results/delany_1m
temp_root: /abs/path/results/temp
keep_temp: false
mask_root: /abs/path/no_masks               # we use no land-cover masks (the folder need not exist)
include:
  - GHANA_area004_GH0004386_1m
  - GHANA_area006_GH0171173_1m
exclude: null
override: null
```

```bash
python delineate.py -b batch_1m.yaml
```

**Output.** `<output_root>/<area>.gpkg` (layer `fields`: one polygon per field) and a simplified
`<area>.simp.gpkg`. We used the non-simplified `.gpkg`.

**Config** ([finetune/conf_zero_shot.yaml](finetune/conf_zero_shot.yaml)) is the upstream default for the v2 model.
The settings that matter for our experiments:

| Setting | Value | Meaning |
|---|---|---|
| `model` and `passes[0].model_args[0].name` | `large_v2` | `large_v2` downloads `DelineateAnythingv2.pt`; use a path to a `.pt` file for fine-tuned weights. **Both entries must be changed together** |
| `super_resolution` | `null` (auto) | pixel sizes ≥ 4 m are upsampled 2×, finer ones are used as is. This applies to the 5, 10 and 20 m runs of the resolution experiment |
| `passes[0].tile_step` | 0.5 | 512 px tiles with 50 % overlap |
| `passes[0].model_args[0].minimal_confidence` | 0.15 | detection threshold |
| `delineation_config` | `merge_iou: 0.8`, `compose_merge_iou: 0.8`, `pixel_area_threshold: 512` | merging of overlapping detections |
| `filtering_args` | `minimum_area_m2: 2500`, `automatic_area_scale: True` | smallest field kept (scaled with pixel area) |
| `data_loader` | `bands: [3, 2, 1]`, `nodata_value: [0, 0, 0]` | band order and nodata |
| `postprocess_limits.num_workers` | `[2, 3]` | tuned for 8 CPUs; see the config guide for other machines |

The same config was used for the zero-shot model and for all fine-tuned models; only the model path differs.

### PlanetScope imagery

Delineate Anything also runs on the 4-band PlanetScope surface reflectance from
[../downloaders/downloader_planet.py](../downloaders/README.md#planetscope). Two things differ from Google Maps:

- **One scene per area folder.** All `.tif` files in a folder are read as one mosaic, so the two dates the downloader
  fetches for FTP would be run together and their fields merged. The Ghana area files have a `green_season_date`, for
  which the downloader fetches one 2021 scene per area (the year of the labels). For the 2025 windows use
  `--fixed-windows --window a` (or `b`), or copy one of the two FTP files of each area into a folder of its own.
- **Config [finetune/conf_planet.yaml](finetune/conf_planet.yaml).** The same as `conf_zero_shot.yaml` except
  `bands: [1, 2, 3]`: the files are B, G, R, NIR, so `[3, 2, 1]` would swap red and blue. NIR is not used. The
  16-bit values are stretched by the model's own normalization (1st to 99th percentile per band).

```bash
python ../downloaders/downloader_planet.py --input ../downloaders/inputs/ghana_test.json \
    --output /abs/path/data/planet_test_set_2021
```

Then use `base_config: finetune/conf_planet.yaml` and `data_root: /abs/path/data/planet_test_set_2021` in the batch file;
the area folders are named by the plain label (e.g. `GHANA_area004_GH0004386`). At 3 m no super-resolution is applied
(`super_resolution: null` only upsamples from 4 m); set it to 2 to try 1.5 m.

### Experiments using inference only

- **Base-model comparison:** `large_v2` zero-shot on the Ghana test areas at 1 m Google Maps imagery.
- **Resolution sweep:** the same on imagery downloaded at 0.3, 1, 2, 5, 10 and 20 m
  (`downloader_google_maps.py --resolutions 0.3 1 2 5 10 20`). One batch file per resolution with `data_root` pointing
  at that resolution's folder and `include` listing `<area>_<res>m` (e.g. `_0p3m`, `_1m`, `_2m`).

## Fine-tuning

Three variants of `DelineateAnythingv2.pt`, all trained on 1 m imagery:

| Variant | Training data | Backbone |
|---|---|---|
| Ghana | Ghana | frozen (`--freeze 10`) |
| Ghana + VIEDIA | Ghana + VIEDIA | frozen |
| Ghana + VIEDIA, unfrozen | Ghana + VIEDIA | trainable (`--freeze 0`) |

### 1. Build the datasets

```bash
# Ghana: 110 areas at 1 m  ->  YOLO dataset (images/labels x train/val/test) + splits.csv
# (labels: finetune/labels/filtered_grids(1).gpkg by default, override with --labels)
python finetune/prepare_dataset.py --images /abs/path/data/ghana_1m --out /abs/path/data/yolo_ghana_1m

# VIEDIA (Cambodia, Vietnam) 1 m tiles
python finetune/prepare_viedia.py --images /abs/path/data/viedia_1m --labels /abs/path/data/viedia_labels \
    --out /abs/path/data/yolo_viedia
```

- **Ghana.** Imagery is cut into 512 × 512 px tiles (stride 256 for training, 512 for validation/test); tiles that are
  more than half nodata are skipped; label polygons under 20 m² are dropped; non-crop labels are excluded. The split is
  **by area, never by tile**: the 32 test areas are the ones marked `ref` in the labels, 12 of the remaining 78 are drawn
  for validation (seed 0, stratified by region) and the other 66 are training areas. The exact area lists are
  [../downloaders/inputs/ghana_train.json](../downloaders/inputs/ghana_train.json), `ghana_val.json` and
  `ghana_test.json`. Train/val/test = 259/46/130 tiles.
- **VIEDIA.** Every image is already one 512 px tile. Per country, one area is drawn for test and one for validation
  (seed 0), the rest is training. Tiles with an empty label file are dropped. Train/val/test = 483/162/162 tiles
  (test: Cambodia area 13, Vietnam area 44). The areas and their tiles are listed in
  [../downloaders/inputs/viedia_train.json](../downloaders/inputs/viedia_train.json), `viedia_val.json` and
  `viedia_test.json`.
- **Combined dataset.** For the Ghana + VIEDIA models, write a `data.yaml` that lists both datasets under `train:` and
  `val:` (each dataset keeps its own test split):

  ```yaml
  path: /abs/path/data/yolo_combined
  train: [/abs/path/data/yolo_ghana_1m/images/train, /abs/path/data/yolo_viedia/images/train]
  val:   [/abs/path/data/yolo_ghana_1m/images/val,   /abs/path/data/yolo_viedia/images/val]
  names: {0: field}
  ```

### 2. Train

```bash
python finetune/train.py --batch 4 --epochs 30 --patience 10 --nbs 64 --lr0 1e-4 --freeze 10 \
    --name ghana_1m --data /abs/path/data/yolo_ghana_1m
```

Checkpoints go to `finetune/runs/<name>/weights/{best,last}.pt`. `best.pt` is the one used for inference.
What the script sets, starting from `DelineateAnythingv2.pt`:

- AdamW, learning rate 1e-4 with cosine decay, 1 warm-up epoch (warm-up bias learning rate 0 so the pretrained heads are
  not disturbed), early stopping after `--patience` epochs without improvement of the validation fitness;
- 512 px tiles, batch 4 with gradient accumulation to an effective batch of `--nbs` 64, mixed precision;
- augmentation as in the original training: horizontal and vertical flips, HSV jitter, scale 0.5, no mosaic and
  no copy-paste unless asked for (`--mosaic`, `--copy-paste`);
- `--freeze N` freezes the first N layers (10 = the whole backbone, 0 = train everything);
- `--max-det` (default 300) sets the maximum detections per tile during validation; `--select map50` selects `best.pt`
  by validation mask mAP50 instead of the Ultralytics fitness; `--resume` continues a run from `last.pt`.

The three variants differ only in `--data` (Ghana or combined) and `--freeze`:

```bash
python finetune/train.py ... --name ghana_1m              --data /abs/path/data/yolo_ghana_1m
python finetune/train.py ... --name combined_1m           --data /abs/path/data/yolo_combined
python finetune/train.py ... --name combined_1m_unfrozen  --data /abs/path/data/yolo_combined --freeze 0 --mosaic 1.0
```

The exact `--mosaic` setting of the unfrozen run was not recorded and is a best guess (the original run name
contained "mosaic"). Treat the unfrozen variant as the least certain to reproduce. The original fine-tuned weights are
published on Hugging Face at [huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2](https://huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2)
(`best_trained_on_ghana.pt`, `best_trained_on_ghana_viedia.pt`, `best_trained_on_ghana_viedia_unfrozen.pt`).

### 3. Run the fine-tuned model

Same as zero-shot inference: copy `finetune/conf_zero_shot.yaml`, set `model` and `passes[0].model_args[0].name` to
the `best.pt` path (or to a `.pt` downloaded from
[Hugging Face](https://huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2) to use our weights without training), and
point the batch file's `base_config` at that copy.

## Tips

- Trouble reading CRS information from GeoTIFFs: set `PROJ_DATA` to the `share/proj` folder of your conda env.
- RAM, VRAM and CPU-dependent parameters (`execution_planner`, `postprocess_limits`, `batch_size`) are explained in
  [delineation_config_guide.md](delineation_config_guide.md).
