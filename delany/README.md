# Delineate Anything v2

A copy of **Delineate Anything v2**, a YOLO11-based field-boundary instance-segmentation model
([code](https://github.com/Lavreniuk/Delineate-Anything), [paper](https://arxiv.org/abs/2607.19069),
[weights](https://huggingface.co/MykolaL/DelineateAnything), AGPL-3.0, see [LICENSE](LICENSE)). The upstream README is
kept as [DELANY.md](DELANY.md) and its parameter guide as [delineation_config_guide.md](delineation_config_guide.md).

Upstream code is used as is. Our additions are in [finetune/](finetune/):

| File | Purpose |
|---|---|
| [conf_zero_shot.yaml](finetune/conf_zero_shot.yaml) | inference config for Google Maps RGB (upstream defaults, workers fitted to 8 CPUs) |
| [conf_planet.yaml](finetune/conf_planet.yaml) | the same for PlanetScope (`bands: [1, 2, 3]`, since the files are B, G, R, NIR) |
| [prepare_dataset.py](finetune/prepare_dataset.py), [prepare_viedia.py](finetune/prepare_viedia.py) | imagery + labels → YOLO dataset |
| [train.py](finetune/train.py) | fine-tunes `DelineateAnythingv2.pt` |

All commands are run from the repository root.

## Prerequisite

Conda

## Environment

Dataset preparation and training only need the root [requirements.txt](../requirements.txt). Inference
(`delineate.py`) also needs GDAL's Python bindings, which pip cannot install on Windows; use conda (e.g. Miniforge):

```bash
conda create -n delany python=3.12 && conda activate delany
conda install -c conda-forge gdal
pip install -r requirements.txt
```

A GPU is strongly recommended. The weights are downloaded from Hugging Face on the first run.

## Inference

Download the imagery first (see [../downloaders/](../downloaders/README.md)), e.g. Google Maps at 1 m into
`data/ghana_1m`. Then write a batch file, e.g. `delany/batch_1m.yaml`:

```yaml
base_config: delany/finetune/conf_zero_shot.yaml
data_root: data/ghana_1m             # one folder per area with mosaic_utm.tif
output_root: data/delany_1m
temp_root: data/delany_temp
keep_temp: false
mask_root: data/no_masks             # no land-cover masks (folder need not exist)
include: null                        # all areas, or a list of folder names
exclude: null
override: null
```

```bash
python delany/delineate.py -b delany/batch_1m.yaml
```

Output: `data/delany_1m/<area>.gpkg` (layer `fields`, one polygon per field). We used this file, not `<area>.simp.gpkg`.

- **Other resolutions:** download with `--resolutions 0.3 1 2 5 10 20` and write one batch file per resolution. Pixel
  sizes ≥ 4 m are upsampled 2× automatically (`super_resolution: null`).
- **PlanetScope:** every `.tif` in an area folder is read as one mosaic, so use a folder with one scene per area, e.g.
  `data/planet_test_set_2021` (or `--fixed-windows --window a`). Set `base_config: delany/finetune/conf_planet.yaml`.
- **Fine-tuned weights:** copy the config and set both `model` and `passes[0].model_args[0].name` to the `.pt` path
  (yours, or ours from [Hugging Face](https://huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2)).

All runs use the same config otherwise: 512 px tiles with 50 % overlap, confidence ≥ 0.15, fields ≥ 2500 m².

## Fine-tuning

### 1. Build the datasets

```bash
python delany/finetune/prepare_dataset.py --images data/ghana_1m --out yolo_ghana_1m
python delany/finetune/prepare_viedia.py --images data/viedia_1m --labels data/viedia_labels \
    --out delany/finetune/yolo_viedia
```

`--out` of `prepare_dataset.py` (and `--data` of `train.py`) are relative to `delany/finetune/`. The datasets are split
by area: the areas are listed in [../downloaders/inputs/](../downloaders/inputs/) (`ghana_{train,val,test}.json`,
`viedia_{train,val,test}.json`). Train/val/test tiles: Ghana 259/46/130, VIEDIA 483/162/162.

For the combined models, create `delany/finetune/yolo_combined/data.yaml`:

```yaml
path: .
train: [../yolo_ghana_1m/images/train, ../yolo_viedia/images/train]
val:   [../yolo_ghana_1m/images/val,   ../yolo_viedia/images/val]
names: {0: field}
```

### 2. Train

```bash
python delany/finetune/train.py --batch 4 --epochs 30 --patience 10 --nbs 64 --lr0 1e-4 --freeze 10 \
    --name ghana_1m --data yolo_ghana_1m
```

The three variants differ only in data and freezing:

| `--name` | `--data` | Extra |
|---|---|---|
| `ghana_1m` | `yolo_ghana_1m` | |
| `combined_1m` | `yolo_combined` | |
| `combined_1m_unfrozen` | `yolo_combined` | `--freeze 0 --mosaic 1.0` (mosaic setting is a best guess) |

Weights go to `delany/finetune/runs/<name>/weights/best.pt`. Other options: `--max-det`, `--select map50`, `--resume`
(see `--help`). Our trained weights are on
[Hugging Face](https://huggingface.co/jimmyoe/ghana-fine-tuned-del-any-v2).

### 3. Run the fine-tuned model

See [DELANY.md](./DELANY.md) on how to run the model.

## Tips

- CRS errors when reading GeoTIFFs: set `PROJ_DATA` to the `share/proj` folder of the conda env.
- Offline node: download the weights beforehand and set `HF_HUB_OFFLINE=1 YOLO_OFFLINE=True`.
- Memory and CPU settings (`execution_planner`, `postprocess_limits`, `batch_size`) are explained in
  [delineation_config_guide.md](delineation_config_guide.md).
