# TAHMO

Small scripts for downloading satellite imagery and running field-boundary inference.

Run commands from the repository root.

## Setup

Create the environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the FTW package, `uv` is also supported:

```bash
cd ftw
uv sync --all-extras
cd ..
```

## Folders

### `data/`

Downloaded input imagery.

- `data/google_maps/`: Google Maps rasters, grouped by area.
- `data/planet/`: Planet imagery, grouped by area.
- `data/sample/`: Small sample data.

Do not edit downloaded data unless you know the workflow expects it.

### `downloaders/`

Scripts that download imagery from external services.

- `downloader_google_maps.py`: downloads and creates Google Maps GeoTIFFs.
- `downloader_planet.py`: orders and downloads PlanetScope imagery.
- `areas_*.json`: input files containing area centers and sizes.

The JSON format is:

```json
{
  "prefix": "Ghana",
  "width_km": 2,
  "coordinates": [[10.7161, -2.5291], [10.3711, -1.2076]]
}
```

Coordinates are `[latitude, longitude]`.

Download Google Maps imagery:

```bash
python downloaders/downloader_google_maps.py \
  --input downloaders/areas_ghana-1-4.json \
  --output data/google_maps
```

Download Planet imagery. Set your Planet API key first:

```bash
export PL_API_KEY="your-key"
python downloaders/downloader_planet.py \
  --input downloaders/areas_ghana-1-4.json \
  --output data/planet
```

### `ftp/`

Inference for two-window PlanetScope imagery.

Each area folder must contain two aligned four-band GeoTIFFs: an earlier image and a later image.

```bash
python ftp/run_ftp_prue.py \
  --input data/planet \
  --output ftp/output
```

Use this workflow for FTP-PRUE models and PlanetScope data.

### `ftw/`

FTW models, the FTW command-line package, and FTW inference scripts.

#### Run FTW from existing rasters

Use `run_ftw.py` when each area folder already contains one raster:

Use `--list-models` to list available models:

```bash
python ftw/run_ftw.py --list-models
```

Use `run_ftw_coordinates.py` to select and download two Sentinel-2 acquisitions automatically, run inference, and create polygons:

```bash
python ftw/run_ftw_coordinates.py \
  --input downloaders/areas_ghana-1-4.json \
  --output ftw/output/ghana \
  --year 2025 \
  --model FTW_PRUE_EFNET_B7_CCBY \
  --gpu 0
```