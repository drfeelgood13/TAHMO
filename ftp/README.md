# FTP: Fields of the World, PRUE+ on PlanetScope

[run_ftp_prue.py](run_ftp_prue.py) runs the released **FTP-PRUE+** model (EfficientNet-B3 or B7, Hugging Face
`taylor-geospatial/ftp-b3` / `ftp-b7`) on PlanetScope imagery and writes field polygons.
The experiment used **B7**.

## What the script does

For every area it:

1. reads two 4-band PlanetScope surface-reflectance GeoTIFFs (window A = earlier, window B = later) and divides by 10000;
2. aligns B to A's grid if the grids differ (bilinear), and stacks the two windows into an 8-channel image;
3. predicts three classes (background, field interior, boundary) in 512 px tiles with 128 px overlap,
   Hann-weighted blending and 8-fold flip/rotate test-time augmentation (D4);
4. turns the prediction into polygons with an h-maxima seeded watershed on the distance from predicted boundaries,
   and drops polygons under 0.05 ha.

**Note on post-processing.** The watershed in step 4 is this script's own approximation, not the paper's official
`postprocess_eval.py`. Scores from this script may therefore understate the model.

## Input

A folder with one subfolder per area. Each subfolder must contain exactly **two** GeoTIFFs with:

- 4 bands in PlanetScope order B, G, R, NIR (no UDM2 band), DN scale 10000, about 3 m pixels;
- filenames that sort chronologically (PlanetScope filenames embed the acquisition time), earlier one first.

```
planet_test_set/
  GHANA_area004_GH0004386/
    <earlier>.tif
    <later>.tif
  GHANA_area006_GH0171173/
  ...
```

How the imagery is obtained is described in [../downloaders/](../downloaders/).

## Output

Per area, under `<output>/<area>/`:

| File | Content |
|---|---|
| `polygons.parquet` | field polygons |
| `<area>_fields.gpkg` | the same polygons for GIS inspection (only if there is at least one) |
| `<area>_classes.tif` | uint8: 0 background, 1 field interior, 2 boundary, 255 invalid |
| `<area>_probabilities.tif` | float32 class probabilities |

## How the experiment was run

32 Ghana test areas, PlanetScope 3 m, **B7**, on one A100 GPU (8 CPUs, 4 GB per CPU). From the repository root:

```bash
python ftp/run_ftp_prue.py \
    --input data/planet_test_set \
    --output ftp/output/test_set_b7 \
    --model b7 \
    --device cuda \
    --channel-order b-a \
    --skip-existing
```

Everything else uses the script defaults: 512 px tiles, 128 px overlap, TTA on, `--min-area-ha 0.05`, `--h-seed 2.0`.
Score the result with `python evaluate.py ftp/output/test_set_b7` (see [../README.md](../README.md#evaluation)).

Two settings matter for reproducing the numbers:

- **`--channel-order b-a`** puts the later window first. The official FTP evaluation does this; the script's default
  (`a-b`) does not, and prints a reminder.
- **`--model b7`**; the default is b3.

The model weights are downloaded from Hugging Face on first use, so the node needs internet access once
(or pre-populate the Hugging Face cache and set `HF_HUB_OFFLINE=1`).

Without a GPU use `--device cpu` (or `--cpu`). It works but is slow because of the 8-fold TTA; `--no-tta` makes it
8× faster at some cost in quality (not used in the experiment).

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--input`, `--output` | required | input folder of areas, output folder |
| `--model` | `b3` | `b3` or `b7` backbone |
| `--channel-order` | `a-b` | `b-a` = later window first (official evaluation) |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` (`--cpu` forces CPU) |
| `--tile-size`, `--overlap` | 512, 128 | tiling; tile size must be a multiple of 32 |
| `--no-tta` | off | disable the 8-fold test-time augmentation |
| `--min-area-ha` | 0.05 | drop smaller polygons |
| `--h-seed` | 2.0 | h of the h-maxima used to seed the watershed |
| `--skip-existing` | off | skip areas whose outputs already exist (resume) |
