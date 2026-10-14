# Downloaders

Two scripts fetch the imagery the experiments need for a list of areas:

| Script | Imagery | Used by | Needs |
|---|---|---|---|
| [downloader_google_maps.py](downloader_google_maps.py) | Google Maps satellite RGB at any pixel size (we used 0.3–20 m) | Delineate Anything v2 | nothing (no key) |
| [downloader_planet.py](downloader_planet.py) | PlanetScope 4-band surface reflectance, 3 m, two dates per area | FTP | Planet account, `PL_API_KEY` |

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

Only `location_label` and `bounding_box` (WGS84 degrees) are read by the downloaders. Labels must be unique and usable
as folder names. The other fields (`grid`, `area`, `region`, `utm_epsg`, ...) are used by the scoring script.
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

```bash
export PL_API_KEY=...            # your Planet API key
python downloaders/downloader_planet.py \
    --input downloaders/inputs/test_set.json \
    --output data/planet_test_set
```

| Option | Meaning |
|---|---|
| `--input`, `--output` | area list, output folder. Reuse the same output folder to resume |
| `--area N` | only the N-th area of the input (1-based, repeatable) |
| `--start-index` | only for the legacy coordinate format |

**This places paid orders and uses your Planet quota.** What it does for each area:

1. Searches PSScene items with a 4-band surface-reflectance asset, standard quality, download permission, and at least
   99.9 % coverage of the box. It picks the least cloudy scene (≤ 10 %) in two windows:
   **window A: 2025-04-01 to 2025-05-31**, **window B: 2025-08-01 to 2025-09-30**. If nothing qualifies it widens
   the date range and relaxes the limits in steps (cloud ≤ 20 %, then ≤ 35 %; coverage ≥ 99 %, then ≥ 95 %) and
   prints which fallback it used.
2. Orders the two scenes (`analytic_sr_udm2` bundle) with co-registration to scene A, clipping to the box, reprojection
   to the area's UTM zone at **3 m** (bilinear), as Cloud Optimized GeoTIFF.
3. Polls until the order is ready (up to 12 h), downloads and checksums the files, checks they have 4 bands and 3 m
   pixels, and keeps the two 4-band rasters.

The windows and thresholds are constants at the top of the script (`WINDOW_A`, `WINDOW_B`, `MAX_CLOUD`,
`MIN_COVERAGE`). Changing them starts a new dataset: the script refuses to reuse an output folder planned with other
settings.

Output per area, in `<output>/<label>/`: two files named `<label lowercased>-<scene id>.tif` (4 bands: B, G, R, NIR;
DN scale 10000). Their names sort chronologically, which is what [../ftp/run_ftp_prue.py](../ftp/run_ftp_prue.py)
relies on to tell window A from window B. Failed areas are reported at the end and in `failed_areas.json`.

An interrupted run can be restarted with the same command: order IDs are saved in `<output>/.orders/` and never
resubmitted blindly. If the log says a previous submission is *uncertain*, check the orders in your Planet account
before doing anything else. The output folder is locked while a download is running.

Note that the Planet windows are in 2025, while the FTW runner uses Sentinel-2 from 2024.

## Tests

```bash
cd downloaders && python -m pytest tests
```

The tests only check the area-list parsing and the resolution/zoom/label helpers; they make no network calls.
