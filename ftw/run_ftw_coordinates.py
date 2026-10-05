#!/usr/bin/env python3
"""Download Sentinel-2 scenes and run FTW inference for location JSON files.

Test-set JSON files use each location_label and bounding_box in ``locations``.
Bounding boxes are WGS84 degrees and are used exactly as supplied. Outputs
are stored in a directory named after each location_label.

The legacy JSON format in ``downloaders`` is also supported::

    {"prefix": "Ghana", "width_km": 2,
     "coordinates": [[10.7161, -2.5291], [10.3711, -1.2076]]}

Coordinates are ``[latitude, longitude]``. Each coordinate becomes a square
area. The script selects two seasonal acquisitions, downloads them, runs a
two-window FTW model, and writes polygons for each area.

Example, run from the repository root::

        uv run python ftw/run_ftw_coordinates.py \
            --input test_set.json \
            --output ftw/output/test_set

The default acquisition year is 2024; override it with --year.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

LOGGER = logging.getLogger("run_ftw_coordinates")
EARTH_RADIUS_KM = 6371.0088


def load_coordinates(path: Path) -> tuple[str, float, list[tuple[float, float]]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc

    prefix = data.get("prefix")
    width_km = data.get("width_km")
    coordinates = data.get("coordinates")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("'prefix' must be a non-empty string")
    if not isinstance(width_km, (int, float)) or width_km <= 0:
        raise ValueError("'width_km' must be greater than zero")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("'coordinates' must contain at least one [latitude, longitude]")

    centers = []
    for index, coordinate in enumerate(coordinates, start=1):
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or not all(isinstance(value, (int, float)) for value in coordinate)
        ):
            raise ValueError(f"Coordinate {index} must be a numeric [latitude, longitude]")
        latitude, longitude = map(float, coordinate)
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"Coordinate {index} is outside WGS84 bounds")
        centers.append((latitude, longitude))
    return prefix.strip(), float(width_km), centers



def load_areas(path: Path) -> list[tuple[str, list[float]]]:
    """Load labeled bounding boxes, retaining legacy coordinate-file support."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    if "locations" not in data:
        prefix, width_km, centers = load_coordinates(path)
        areas = [(f"{prefix}_{index:02d}", square_bbox(*center, width_km))
                 for index, center in enumerate(centers, start=1)]
    else:
        if data.get("coordinate_crs", "EPSG:4326") != "EPSG:4326":
            raise ValueError("Location bounding boxes must use EPSG:4326")
        locations = data["locations"]
        if not isinstance(locations, list) or not locations:
            raise ValueError("'locations' must be a non-empty list")
        areas = []
        for index, location in enumerate(locations, start=1):
            if not isinstance(location, dict):
                raise ValueError(f"Location {index} must be an object")
            label = location.get("location_label")
            box = location.get("bounding_box")
            keys = ("west_lon", "south_lat", "east_lon", "north_lat")
            if not isinstance(box, dict) or not all(
                isinstance(box.get(key), (int, float))
                and not isinstance(box.get(key), bool)
                and math.isfinite(box[key]) for key in keys
            ):
                raise ValueError(f"Location {index} needs a numeric bounding_box: {keys}")
            bbox = [box[key] for key in keys]
            west, south, east, north = bbox
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(f"Location {index} has an invalid WGS84 bounding box")
            areas.append((label, bbox))
    labels = set()
    for label, bbox in areas:
        if (not isinstance(label, str) or not label.strip()
                or label in (".", "..") or "/" in label or "\\" in label):
            raise ValueError("Location labels must be non-empty directory names")
        if label in labels:
            raise ValueError(f"Duplicate location label: {label}")
        labels.add(label)
    return areas


def square_bbox(latitude: float, longitude: float, width_km: float) -> list[float]:
    """Return [min_lon, min_lat, max_lon, max_lat] around a center."""
    half_latitude = width_km / (2 * EARTH_RADIUS_KM) * 180 / math.pi
    longitude_scale = max(math.cos(math.radians(latitude)), 1e-6)
    half_longitude = half_latitude / longitude_scale
    return [
        longitude - half_longitude,
        latitude - half_latitude,
        longitude + half_longitude,
        latitude + half_latitude,
    ]


def prepare_inference_input(input_path: Path) -> tuple[Path, tuple[int, int]]:
    """Extend small rasters at the bottom/right without changing pixel size."""
    import numpy as np
    import rasterio

    with rasterio.open(input_path) as src:
        shape = (src.height, src.width)
        extra_height = max(0, 128 - src.height)
        extra_width = max(0, 128 - src.width)
        if not extra_height and not extra_width:
            return input_path, shape
        pixels = np.pad(src.read(), ((0, 0), (0, extra_height), (0, extra_width)), mode="edge")
        profile = src.profile.copy()
        profile.update(height=pixels.shape[1], width=pixels.shape[2])
        tags = src.tags()
    padded_path = input_path.with_name("inference_data_padded.tif")
    with rasterio.open(padded_path, "w", **profile) as dst:
        dst.write(pixels)
        dst.update_tags(**tags)
    LOGGER.info("Padded input from %sx%s to %sx%s pixels", *shape, pixels.shape[1], pixels.shape[2])
    return padded_path, shape


def crop_prediction(input_path: Path, output_path: Path, shape: tuple[int, int]) -> None:
    """Remove synthetic padding, retaining the original raster footprint."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(input_path) as src:
        height, width = shape
        pixels = src.read(window=Window(0, 0, width, height))
        profile = src.profile.copy()
        profile.update(height=height, width=width)
        tags = src.tags()
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pixels)
        dst.update_tags(**tags)


def select_scenes(scene_selection, bbox: list[float], args) -> tuple[str, str]:
    """Retry the same box with wider date windows if a seasonal query is empty."""
    attempts = [
        (args.cloud_cover_max, args.buffer_days),
        (100, max(args.buffer_days, 30)),
        (100, max(args.buffer_days, 60)),
        (100, max(args.buffer_days, 180)),
    ]
    attempts = list(dict.fromkeys(attempts))

    for attempt, (cloud_cover_max, buffer_days) in enumerate(attempts, start=1):
        try:
            return scene_selection(
                bbox=bbox,
                year=args.year,
                stac_host="mspc",
                cloud_cover_max=cloud_cover_max,
                buffer_days=buffer_days,
                s2_collection="c1",
                nodata_max=None,
                verbose=args.verbose,
            )
        except ValueError as exc:
            if "No sentinel scenes within area" not in str(exc):
                raise
            if attempt == len(attempts):
                raise
            LOGGER.warning(
                "No scene found at cloud < %d%% and ±%d days; retrying with a wider search",
                cloud_cover_max,
                buffer_days,
            )


def run_area(label: str, bbox: list[float], args) -> Path:
    from ftw_tools.download.download_img import create_input, scene_selection
    from ftw_tools.inference.inference import run
    from ftw_tools.postprocess.polygonize import polygonize

    area_dir = args.output / label
    input_path = area_dir / "inference_data.tif"
    inference_path = area_dir / "inference_output.tif"
    polygons_path = area_dir / "polygons.parquet"
    area_dir.mkdir(parents=True, exist_ok=True)

    if polygons_path.exists() and not args.overwrite:
        LOGGER.info("%s: already complete, skipping", label)
        return polygons_path

    if not input_path.exists() or args.overwrite:
        window_a, window_b = select_scenes(scene_selection, bbox, args)
        create_input(
            win_a=window_a,
            win_b=window_b,
            out=str(input_path),
            overwrite=args.overwrite,
            bbox=bbox,
            stac_host="mspc",
            s2_collection="c1",
            verbose=args.verbose,
        )
    model_input, original_shape = prepare_inference_input(input_path)
    model_output = (area_dir / "inference_output_padded.tif"
                    if model_input != input_path else inference_path)
    run(
        input=str(model_input),
        model=args.model,
        out=str(model_output),
        resize_factor=2,
        gpu=args.gpu,
        patch_size=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        padding=None,
        overwrite=args.overwrite,
        mps_mode=False,
        save_scores=False,
    )
    if model_input != input_path:
        crop_prediction(model_output, inference_path, original_shape)
    polygonize(input=str(inference_path), out=str(polygons_path), overwrite=args.overwrite)
    return polygons_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Test-set locations or legacy coordinate JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--year", type=int, default=2024, help="Acquisition year (default: 2024)")
    parser.add_argument("--model", default="FTW_PRUE_EFNET_B7_CCBY")
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA device index; -1 for CPU")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cloud-cover-max", type=int, default=20)
    parser.add_argument("--buffer-days", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        areas = load_areas(args.input)
        if args.gpu < -1 or args.batch_size < 1 or args.num_workers < 1:
            raise ValueError("gpu must be >= -1; batch-size and num-workers must be >= 1")
        args.output.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    failures = []
    LOGGER.info("Acquisition year: %d; locations: %d", args.year, len(areas))
    for index, (label, bbox) in enumerate(areas, start=1):
        LOGGER.info("\n=== [%d/%d] %s ===", index, len(areas), label)
        try:
            output = run_area(label, bbox, args)
            LOGGER.info("%s: wrote %s", label, output)
        except Exception as exc:
            LOGGER.error("%s: FAILED: %s", label, exc)
            failures.append(label)

    LOGGER.info("\nDone: %d/%d areas succeeded.", len(areas) - len(failures), len(areas))
    if failures:
        LOGGER.error("Failed areas: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
