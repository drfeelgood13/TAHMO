# Downloaders

Two scripts fetch the imagery the experiments need for a list of areas:

| Script | Imagery | Used by | Needs |
|---|---|---|---|
| [downloader_google_maps.py](downloader_google_maps.py) | Google Maps satellite RGB at any pixel size (we used 0.3–20 m) | Delineate Anything v2 | nothing (no key) |
| [downloader_planet.py](downloader_planet.py) | PlanetScope 4-band surface reflectance, 3 m: one scene near each area's `green_season_date`, or two in fixed windows | FTP, Delineate Anything v2 | Planet account, `PL_API_KEY` |

(Sentinel-2 for FTW is fetched by the FTW runner itself; see [../ftw/](../ftw/).)

```bash
pip install -r requirements.txt     # repository root
```

`downloader_planet.py` uses `fcntl` and so runs on Linux/macOS only.

## Input: the area list

Both scripts read the same JSON, one entry per area. The files we used are in [downloaders/inputs/](inputs/)
(all commands in this README are run from the repository root):

| File | Areas | Content |
|---|---|---|
| [inputs/test_set.json](inputs/test_set.json) | 32 | the Ghana test areas used in all experiments |
| [inputs/ghana_110.json](inputs/ghana_110.json) | 110 | all Ghana areas (training + validation + test) |
| [inputs/ghana_train.json](inputs/ghana_train.json), [ghana_val.json](inputs/ghana_val.json), [ghana_test.json](inputs/ghana_test.json) | 66, 12, 32 | the fine-tuning split by area (`ghana_test.json` = `test_set.json`) |
| [inputs/viedia_all.json](inputs/viedia_all.json), `viedia_train/val/test.json` | 10; 6, 2, 2 | the VIEDIA (Cambodia, Vietnam) areas, split by area, with the coordinates of their 512 m sub-tiles |

The VIEDIA files list `areas` instead of `locations`. The PlanetScope downloader reads them too: one AOI per area, the
bounding box around all its sub-tiles (about 4.6 km × 4.6 km); `subareas` are not read.

The areas are the footprints (about 0.5 km × 0.5 km) of the Ghana OpenCropMaps dataset. Minimal entry:

```json
{
  "coordinate_crs": "EPSG:4326",
  "locations": [
    {
      "location_label": "GHANA_area004_GH0004386",
      "bounding_box": {"north_lat": 10.980504, "south_lat": 10.973813,
                       "west_lon": -0.321617, "east_lon": -0.315453}
    }
  ]
}
```

Only `location_label` and `bounding_box` (WGS84 degrees) are read by the downloaders, plus the optional
`green_season_date` and `utm_epsg` by the PlanetScope downloader (see below; without `utm_epsg` it uses the UTM zone
of the box centre). Labels must be unique and usable as folder names. The other fields (`grid`, `area`, `region`,
`n_fields`, ...) are used by the scoring script.

All area files also record when the field labels were made and a date in the growing season of that year:

| Field | Value | Why |
|---|---|---|
| `label_year` | `2021` | Ghana: every label in the OpenCropMaps ground truth (`filtered_grids.gpkg`, column `date`) is dated 2021. VIEDIA: labels from 2021 |
| `green_season_date`, Ghana | `2021-09-15` for `north`, `2021-10-15` for `ej` | north Ghana has one rainy season (about May–October) with crops greenest in August–September; the north labels are dated 2021-09-15. Ejura (`ej`) has two seasons; most of its labelled fields are minor-season crops planted August–September (labels dated 2021-12-15), greenest around October |
| `green_season_date`, VIEDIA | `2021-10-01` for Cambodia, `2021-07-01` for Vietnam | Cambodia: wet-season rice, planted June–August, greenest September–October. Vietnam (central and south-central coast): summer–autumn rice, planted April–May, greenest June–July, in the coast's dry season. The VIEDIA labels carry no dates, so these follow the main crop calendar only |

`ghana_110.json` also has `n_fields`, the number of labelled fields in each area (5,883 in total).
The boxes are used exactly as given. The label carries no resolution: the downloader appends it
(`GHANA_area004_GH0004386` → `GHANA_area004_GH0004386_1m`).

A legacy format is also accepted: `{"prefix": "Ghana", "width_km": 2, "coordinates": [[lat, lon], ...]}` makes a square
of `width_km` around each point.

## Google Maps

```bash
python downloaders/downloader_google_maps.py \
    --input downloaders/inputs/test_set.json \
    --output data/ghana_1m \
    --resolutions 1 \
    --skip-existing
```

| Option | Meaning |
|---|---|
| `--input`, `--output` | area list, output folder |
| `--resolutions M [M ...]` | target pixel sizes in metres, e.g. `0.3 1 2 5 10 20` (default `[2]`, the `TARGET_RES_M` constant in the script). Each resolution gets its own folder per area |
| `--skip-existing` | skip areas whose `mosaic_utm.tif` and `mosaic.json` exist (resume) |

How it works: for each area and resolution it picks the lowest Web Mercator zoom level that is at least as detailed as
requested (`zoom = ceil(log2(156543.03 / resolution))`), downloads the 256 px tiles covering the bounding box, stitches
them (in chunks for large areas), crops to the box, and reprojects to the area's UTM zone, resampling bilinearly to
exactly the requested pixel size.

Output per area and resolution, in `<output>/<label>_<res>m/` (`0.3` → `0p3m`, `1` → `1m`, `2` → `2m`, ...):

| File | Content |
|---|---|
| `mosaic_utm.tif` | 8-bit RGB GeoTIFF in UTM, nodata 0. **This is what the models read** |
| `mosaic.json` | metadata: zoom, bounding box, pixel size, size in metres |

`<output>/failed_areas.json` lists failures; the script exits with status 1 if any area failed, and rerunning with
`--skip-existing` retries only those.

Tiles come from Google's public tile server (`mt1.google.com`) with no API key. Check Google's terms of use before
using the imagery beyond this research, and expect the imagery (and its acquisition dates) to change over time. The
imagery we used is preserved in the dataset zips.

## PlanetScope

The downloader has two modes, chosen per area:

- **Near the green-season date** (areas with `green_season_date`, e.g. all Ghana files): one scene, acquired as close as
  possible to that date, so imagery and labels are from the same season (2021). One GeoTIFF per area folder, ready for
  Delineate Anything.
- **Windows A and B** (areas without a date, or every area with `--fixed-windows`): two scenes in fixed 2025 windows,
  as used for FTP.

```bash
export PL_API_KEY=...            # your Planet API key

# 2021 green-season imagery, one scene per area
python downloaders/downloader_planet.py \
    --input downloaders/inputs/test_set.json \
    --output data/planet_test_set_2021

# FTP experiment: 2025 windows A and B
python downloaders/downloader_planet.py \
    --input downloaders/inputs/test_set.json \
    --output data/planet_test_set \
    --fixed-windows
```

| Option | Meaning |
|---|---|
| `--input`, `--output` | area list, output folder. Reuse the same output folder to resume |
| `--area N` | only the N-th area of the input (1-based, repeatable) |
| `--start-index` | only for the legacy coordinate format |
| `--fixed-windows` | ignore `green_season_date` and use windows A and B for every area |
| `--window a\|b` | windows mode only: order only the scene of window A or B, so each area folder holds one GeoTIFF (for Delineate Anything, which reads all `.tif` files in a folder as one mosaic). Default: both windows (for FTP). Use a separate output folder |

**This places paid orders and uses your Planet quota.** Scenes from 2021 are archive imagery; whether your Planet
plan can download them (and at what quota cost) depends on your subscription. Areas without a downloadable scene end
up in `failed_areas.json`. What it does for each area:

1. Searches PSScene items with a 4-band surface-reflectance asset, standard quality, download permission, and at least
   99.9 % coverage of the box.
   - **Date mode:** within 15 days of `green_season_date`, the scene with ≤ 10 % cloud acquired closest to that date.
     If nothing qualifies it first relaxes the limits (cloud ≤ 20 %, then ≤ 35 %; coverage ≥ 99 %, then ≥ 95 %), then
     widens to 45 and 90 days (`DATE_SEARCH_DAYS`) and prints which fallback it used. Expect fallbacks: September and
     October are in the rainy season.
   - **Windows mode:** the least cloudy scene (≤ 10 %) in **window A: 2025-04-01 to 2025-05-31** and
     **window B: 2025-08-01 to 2025-09-30**. If nothing qualifies it relaxes the same limits and widens the date range
     in steps.
2. Orders the scenes (`analytic_sr_udm2` bundle), clipped to the box and reprojected to the area's UTM zone at
   **3 m** (bilinear), as Cloud Optimized GeoTIFF. Two scenes are co-registered to scene A; a single scene is not
   (there is nothing to align it to).
3. Polls until the order is ready (up to 12 h), downloads and checksums the files, checks they have 4 bands and 3 m
   pixels, and keeps the 4-band rasters.

The windows, search days and thresholds are constants at the top of the script (`WINDOW_A`, `WINDOW_B`,
`DATE_SEARCH_DAYS`, `MAX_CLOUD`, `MIN_COVERAGE`). Changing them, the dates in the input, or the mode starts a new
dataset: the script refuses to reuse an output folder planned with other settings.

Output per area, in `<output>/<label>/`: one file (date mode or `--window`) or two files (windows A and B) named
`<label lowercased>-<scene id>.tif` (4 bands: B, G, R, NIR; DN scale 10000). The scene ID starts with the acquisition
date and time. Two files' names sort chronologically, which is what [../ftp/run_ftp_prue.py](../ftp/run_ftp_prue.py)
relies on to tell window A from window B. To run Delineate Anything on single-scene folders, see
[../delany/README.md](../delany/README.md#planetscope-imagery). Failed areas are reported at the end and in
`failed_areas.json`.

An interrupted run can be restarted with the same command: order IDs are saved in `<output>/.orders/` and never
resubmitted blindly. If the log says a previous submission is *uncertain*, check the orders in your Planet account
before doing anything else. The output folder is locked while a download is running.

Note that the field labels are from 2021, the Planet windows A and B are in 2025, and the FTW runner uses Sentinel-2
from 2024. Only the date mode matches the label year.

## Tests

```bash
python -m pytest downloaders/tests     # from the repository root; `cd downloaders && python -m pytest tests` works too
```

The tests only check the area-list parsing and the resolution/zoom/label helpers; they make no network calls.
