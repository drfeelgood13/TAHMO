"""
Run a Fields of the World / Delineate Anything model on a local GeoTIFF.

Examples
--------
uv run python run_local_tif_ftw.py --input "/path/to/image.tif" --model DelineateAnythingV2
uv run python run_local_tif_ftw.py --input "/path/to/image.tif" --model DelineateAnything
uv run python run_local_tif_ftw.py --input "/path/to/stacked_8band.tif" --model FTW_PRUE_EFNET_B7

Optional band selection:
uv run python run_local_tif_ftw.py --input image.tif --model DelineateAnythingV2 --bands 1,2,3

Notes
-----
- DelAny: use a georeferenced RGB or RGB+NIR image.
- PRUE/FTW semantic models: require 8 bands ordered
  R_t1,G_t1,B_t1,NIR_t1,R_t2,G_t2,B_t2,NIR_t2.
- The script processes the entire TIFF and writes one GeoJSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from ftw_tools.inference.inference import run, run_instance_segmentation
from ftw_tools.inference.model_registry import MODEL_REGISTRY
from ftw_tools.postprocess.polygonize import polygonize

DEFAULT_MODEL = "DelineateAnythingV2"
OUTPUT_ROOT = Path("local_tif_model_outputs")

INSTANCE_RESIZE_FACTOR = 2
INSTANCE_PATCH_SIZE = 256
INSTANCE_BATCH_SIZE = 4
INSTANCE_NUM_WORKERS = 4
INSTANCE_MAX_DETECTIONS = 100
INSTANCE_IOU_THRESHOLD = 0.30
INSTANCE_CONF_THRESHOLD = 0.05
INSTANCE_SIMPLIFY_M = 2
INSTANCE_MIN_SIZE_M2 = 500
INSTANCE_MAX_SIZE_M2 = 100000
INSTANCE_OVERLAP_IOU = 0.20
INSTANCE_OVERLAP_CONTAIN = 0.80

SEMANTIC_RESIZE_FACTOR = 2
SEMANTIC_PATCH_SIZE = 256
SEMANTIC_BATCH_SIZE = 2
SEMANTIC_NUM_WORKERS = 4
SEMANTIC_SIMPLIFY_M = 15
SEMANTIC_MIN_SIZE_M2 = 500
CLOSE_INTERIORS = True


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def write_empty_geojson(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}, indent=2))
    return path


def parse_bands(value: str | None) -> list[int] | None:
    if not value:
        return None
    bands = [int(v.strip()) for v in value.split(",")]
    if min(bands) < 1:
        raise ValueError("Band numbers are 1-based.")
    return bands


def inspect_raster(path: Path) -> dict:
    with rasterio.open(path) as src:
        arr = src.read(
            out_shape=(src.count, min(src.height, 256), min(src.width, 256)),
            resampling=Resampling.nearest,
        )
        finite = arr[np.isfinite(arr)]
        return {
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "bounds": src.bounds,
            "nodata": src.nodata,
            "descriptions": src.descriptions,
            "sample_min": float(np.nanmin(finite)) if finite.size else None,
            "sample_max": float(np.nanmax(finite)) if finite.size else None,
        }


def print_info(path: Path, info: dict) -> None:
    print("\nInput raster")
    print("------------")
    print(f"Path         : {path}")
    print(f"Size         : {info['width']} x {info['height']} px")
    print(f"Bands        : {info['count']}")
    print(f"Dtype        : {info['dtype']}")
    print(f"CRS          : {info['crs']}")
    print(f"NoData       : {info['nodata']}")
    print(f"Band labels  : {info['descriptions']}")
    print(f"Sample range : {info['sample_min']} .. {info['sample_max']}")


def choose_utm(path: Path) -> str:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError("Input TIFF has no CRS.")
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    lon = (b[0] + b[2]) / 2
    lat = (b[1] + b[3]) / 2
    zone = int(math.floor((lon + 180) / 6) + 1)
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def select_bands(src_path: Path, dst_path: Path, bands: list[int]) -> Path:
    with rasterio.open(src_path) as src:
        if max(bands) > src.count:
            raise ValueError(f"Raster has only {src.count} bands.")
        profile = src.profile.copy()
        profile.update(count=len(bands))
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for out_i, src_i in enumerate(bands, start=1):
                dst.write(src.read(src_i), out_i)
                if src.descriptions[src_i - 1]:
                    dst.set_band_description(out_i, src.descriptions[src_i - 1])
    return dst_path


def reproject_to_utm(src_path: Path, dst_path: Path) -> Path:
    target_crs = choose_utm(src_path)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=target_crs, transform=transform, width=width, height=height)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
                if src.descriptions[i - 1]:
                    dst.set_band_description(i, src.descriptions[i - 1])
    print(f"Working copy reprojected to {target_crs}: {dst_path}")
    return dst_path


def prepare_input(input_path: Path, bands: list[int] | None, work_dir: Path) -> Path:
    current = input_path
    if bands:
        current = select_bands(
            current,
            work_dir / f"{input_path.stem}_selected_bands.tif",
            bands,
        )
        print(f"Using bands: {bands}")

    info = inspect_raster(current)
    if info["crs"] is None:
        raise ValueError(
            "The TIFF has no CRS/georeferencing. Georeference it before producing GeoJSON."
        )
    if info["crs"].is_geographic:
        current = reproject_to_utm(current, work_dir / f"{input_path.stem}_utm.tif")
    return current


def choose_patch_size(path: Path, preferred: int = 256) -> int:
    with rasterio.open(path) as src:
        d = min(src.width, src.height)
    if d >= preferred:
        return preferred
    if d >= 128:
        print("Raster is smaller than 256 px on one side; using patch_size=128.")
        return 128
    raise ValueError(
        f"Raster is only {d}px on its shortest side; use a larger image (>=128 px)."
    )


def validate_input(model_name: str, path: Path) -> None:
    spec = MODEL_REGISTRY[model_name]
    with rasterio.open(path) as src:
        n = src.count

    if spec.instance_segmentation:
        if n < 3:
            raise ValueError("DelAny needs an RGB/RGB+NIR image with at least 3 bands.")
        if n > 4:
            print(
                "\nWARNING: TIFF has >4 bands. Use --bands to explicitly select "
                "RGB (or RGB+NIR) in the correct order.\n"
            )
    else:
        if n != 8:
            raise ValueError(
                f"{model_name} expects 8 bands, but this TIFF has {n}.\n"
                "Required order: R_t1,G_t1,B_t1,NIR_t1,R_t2,G_t2,B_t2,NIR_t2"
            )


def export_wgs84(src_path: Path, dst_path: Path, source_image: str, model: str) -> Path:
    if not src_path.exists():
        return write_empty_geojson(dst_path)
    try:
        gdf = gpd.read_file(src_path)
    except Exception:
        return write_empty_geojson(dst_path)
    if gdf.empty:
        return write_empty_geojson(dst_path)
    if gdf.crs is None:
        raise ValueError("Prediction output has no CRS.")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["source_image"] = source_image
    gdf["model"] = model
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dst_path, driver="GeoJSON")
    return dst_path


def run_instance_local(path: Path, model: str, out_dir: Path, gpu: int, mps: bool, overwrite: bool) -> Path:
    patch = choose_patch_size(path, INSTANCE_PATCH_SIZE)
    raw = out_dir / "predictions_raw.geojson"
    final = out_dir / f"{path.stem}_{safe_name(model)}.geojson"

    try:
        run_instance_segmentation(
            input=str(path),
            model=model,
            out=str(raw),
            gpu=None if gpu < 0 else gpu,
            num_workers=INSTANCE_NUM_WORKERS,
            patch_size=patch,
            resize_factor=INSTANCE_RESIZE_FACTOR,
            batch_size=INSTANCE_BATCH_SIZE,
            max_detections=INSTANCE_MAX_DETECTIONS,
            iou_threshold=INSTANCE_IOU_THRESHOLD,
            conf_threshold=INSTANCE_CONF_THRESHOLD,
            padding=None,
            overwrite=overwrite,
            mps_mode=mps,
            simplify=INSTANCE_SIMPLIFY_M,
            min_size=INSTANCE_MIN_SIZE_M2,
            max_size=INSTANCE_MAX_SIZE_M2,
            close_interiors=CLOSE_INTERIORS,
            overlap_iou_threshold=INSTANCE_OVERLAP_IOU,
            overlap_contain_threshold=INSTANCE_OVERLAP_CONTAIN,
        )
    except ValueError as exc:
        if "No objects to concatenate" in str(exc):
            print("No field polygons detected.")
            return write_empty_geojson(final)
        raise

    return export_wgs84(raw, final, path.name, model)


def run_semantic_local(path: Path, model: str, out_dir: Path, gpu: int, mps: bool, overwrite: bool) -> Path:
    patch = choose_patch_size(path, SEMANTIC_PATCH_SIZE)
    raster_out = out_dir / "inference_output.tif"
    raw = out_dir / "predictions_raw.geojson"
    final = out_dir / f"{path.stem}_{safe_name(model)}.geojson"

    run(
        input=str(path),
        model=model,
        out=str(raster_out),
        resize_factor=SEMANTIC_RESIZE_FACTOR,
        gpu=gpu,
        patch_size=patch,
        batch_size=SEMANTIC_BATCH_SIZE,
        num_workers=SEMANTIC_NUM_WORKERS,
        padding=None,
        overwrite=overwrite,
        mps_mode=mps,
        save_scores=False,
    )

    polygonize(
        input=str(raster_out),
        out=str(raw),
        simplify=SEMANTIC_SIMPLIFY_M,
        min_size=SEMANTIC_MIN_SIZE_M2,
        max_size=None,
        overwrite=overwrite,
        close_interiors=CLOSE_INTERIORS,
    )

    return export_wgs84(raw, final, path.name, model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--bands", default=None, help="1-based bands, e.g. 1,2,3")
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--mps", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")

    print_info(input_path, inspect_raster(input_path))

    out_dir = OUTPUT_ROOT / safe_name(input_path.stem) / safe_name(args.model)
    work_dir = out_dir / "_working"
    work_dir.mkdir(parents=True, exist_ok=True)

    working = prepare_input(input_path, parse_bands(args.bands), work_dir)
    if working != input_path:
        print_info(working, inspect_raster(working))

    validate_input(args.model, working)
    spec = MODEL_REGISTRY[args.model]

    try:
        if spec.instance_segmentation:
            result = run_instance_local(
                working, args.model, out_dir, args.gpu, args.mps, args.overwrite
            )
        else:
            result = run_semantic_local(
                working, args.model, out_dir, args.gpu, args.mps, args.overwrite
            )

        print("\nDONE")
        print(f"QGIS GeoJSON: {result}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
