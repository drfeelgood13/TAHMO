# Downloaders

| Script | Imagery | Used by | Needs |
|---|---|---|---|
| [downloader_google_maps.py](downloader_google_maps.py) | Google Maps RGB at any pixel size (we used 0.3–20 m) | Delineate Anything v2 | nothing |
| [downloader_planet.py](downloader_planet.py) | PlanetScope 4-band surface reflectance, 3 m | FTP, Delineate Anything v2 | `PL_API_KEY` (Linux/macOS only) |

Sentinel-2 for FTW is fetched by the FTW runner itself ([../ftw/](../ftw/)). Install with
`pip install -r requirements.txt` at the repository root; all commands below are run from there.

## Input: the area list

| File | Areas | Content |
|---|---|---|
| [ghana_test.json](inputs/ghana_test.json) | 32 | Ghana test areas, used in all experiments |
| [ghana_train.json](inputs/ghana_train.json), [ghana_val.json](inputs/ghana_val.json) | 66, 12 | Ghana fine-tuning split |
| [ghana_all_110.json](inputs/ghana_all_110.json) | 110 | all Ghana areas |
| [viedia_train.json](inputs/viedia_train.json), [viedia_val.json](inputs/viedia_val.json), [viedia_test.json](inputs/viedia_test.json) | 6, 2, 2 | VIEDIA (Cambodia, Vietnam) fine-tuning split |
| [viedia_all_10.json](inputs/viedia_all_10.json) | 10 | all VIEDIA areas |

A Ghana area is an OpenCropMaps footprint of about 0.5 × 0.5 km. A VIEDIA area is a grid of up to 9 × 9 tiles of 512 m
(about 4.6 × 4.6 km); the tiles are listed under `subareas`, but the downloaders fetch the whole area box.

All files share one format: top-level `split`, `dataset`, `coordinate_crs` (`EPSG:4326`), `location_count` and a
`locations` list. Each location has:

| Field | Meaning |
|---|---|
| `location_label` | unique name, used as output folder (`GHANA_area004_GH0004386`, `CAMBODIA_area13`) |
| `bounding_box` | `north_lat`, `south_lat`, `west_lon`, `east_lon` in WGS84 degrees, used as given |
| `green_season_date` | target date for PlanetScope (see below) |
| `utm_epsg` | UTM zone of the labels; output is reprojected to it |
| `id`, `country`, `area`, `split`, `label_year`, `n_fields` | metadata |
| `region`, `grid` | Ghana only: `north` or `ej` (Ejura), OpenCropMaps grid cell |
| `tile_count`, `rows`, `cols`, `subareas` | VIEDIA only: the tile grid |

The downloaders only need `location_label` and `bounding_box`. The labels are from 2021, so `green_season_date` is a
2021 date when crops are greenest:

| Areas | Date | Why |
|---|---|---|
| Ghana `north` | 2021-09-15 | one rainy season, greenest August–September |
| Ghana `ej` | 2021-10-15 | mostly minor-season crops planted August–September |
| Cambodia | 2021-10-01 | wet-season rice, greenest September–October |
| Vietnam | 2021-07-01 | summer–autumn rice, greenest June–July |

## Google Maps

```bash
python downloaders/downloader_google_maps.py --input downloaders/inputs/ghana_test.json \
    --output data/ghana_1m --resolutions 1 --skip-existing
```

| Option | Meaning |
|---|---|
| `--resolutions M [M ...]` | pixel sizes in metres, e.g. `0.3 1 2 5 10 20` (default `2`) |
| `--skip-existing` | skip finished areas; rerun with it to retry failures |

Each area and resolution goes to `<output>/<label>_<res>m/` (`0.3` → `0p3m`): `mosaic_utm.tif` (RGB GeoTIFF in UTM, the
file the models read) and `mosaic.json` (metadata). Failures are listed in `<output>/failed_areas.json`.

Tiles come from Google's public tile server without a key; the imagery can change over time, so the imagery we used is
kept in the dataset zips.

## PlanetScope

**This places paid orders on your Planet quota.**

```bash
export PL_API_KEY=...

# one 2021 scene per area, closest to green_season_date
python downloaders/downloader_planet.py --input downloaders/inputs/ghana_test.json --output data/planet_test_set_2021

# FTP experiment: two scenes, 2025 windows A (Apr–May) and B (Aug–Sep)
python downloaders/downloader_planet.py --input downloaders/inputs/ghana_test.json --output data/planet_test_set \
    --fixed-windows
```

| Option | Meaning |
|---|---|
| `--area N` | only the N-th area (1-based, repeatable) |
| `--fixed-windows` | use windows A and B instead of `green_season_date` |
| `--window a\|b` | with windows: order only one of the two scenes (for Delineate Anything); use a separate output folder |

It picks the clearest scene (≤ 10 % cloud, ≥ 99.9 % coverage) closest to the date or in each window, relaxing the limits
and widening the search if nothing qualifies. Scenes are clipped to the box and reprojected to UTM at 3 m. Search
settings are constants at the top of the script.

Output: `<output>/<label>/<label lowercased>-<scene id>.tif` (bands B, G, R, NIR), one file per scene; with two scenes
the names sort chronologically (A before B), which [../ftp/run_ftp_prue.py](../ftp/run_ftp_prue.py) relies on. Rerun the
same command to resume: order IDs are saved in `<output>/.orders/` and never resubmitted.

## Tests

```bash
python -m pytest downloaders/tests     # no network calls
```
