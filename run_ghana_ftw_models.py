"""
Run FTW / Delineate Anything models on the same five Northern Ghana sites.

FIXES vs the previous script
----------------------------
1) FTW currently has a verbose-output bug for single-window imagery:
   after successfully writing the TIFF it tries to print identifiers[1],
   which does not exist for DelineateAnything. This script disables that
   verbose summary for single-window downloads.

2) A 1 x 1 km Sentinel-2 crop is only about 100 x 100 pixels, but the
   standard Delineate Anything inference patch is 256 x 256 pixels.
   Therefore this script:
       - evaluates the SAME central 1 x 1 km AOI,
       - downloads/runs inference on a 3 x 3 km context,
       - clips final polygons back to the central 1 x 1 km AOI.

This gives the model enough context while keeping the comparison area identical.

Recommended setup:
    git clone https://github.com/fieldsoftheworld/ftw-baselines.git
    cd ftw-baselines
    uv sync --all-extras

Put this file inside the ftw-baselines folder.

Examples:
    uv run python run_ghana_ftw_models_fixed.py --list-models

    uv run python run_ghana_ftw_models_fixed.py \
        --model DelineateAnythingV2 --site site_1

    uv run python run_ghana_ftw_models_fixed.py \
        --model DelineateAnything --site site_1

    uv run python run_ghana_ftw_models_fixed.py \
        --model FTW_PRUE_EFNET_B7 --site site_1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.geometry import box, mapping


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

MODEL = "DelineateAnythingV2"
YEAR = 2025

# This is the area you will actually compare in QGIS.
EVALUATION_AOI_SIZE_KM = 1.0

# Larger context used only so 256 px inference patches fit.
# At Sentinel-2 10 m resolution, 3 km is ~300 pixels.
PROCESSING_CONTEXT_SIZE_KM = 3.0

OUTPUT_ROOT = Path("ghana_model_comparison")

STAC_HOST = "mspc"
S2_COLLECTION = "c1"
CLOUD_COVER_MAX = 20
BUFFER_DAYS = 14
NODATA_MAX = 50
VERBOSE = True

# Hardware:
# -1 = CPU.
#  0 = first NVIDIA GPU.
GPU = -1

# Set True on Apple Silicon if FTW/PyTorch MPS works in your environment.
MPS_MODE = False

OVERWRITE = False

# Semantic / PRUE inference.
SEMANTIC_RESIZE_FACTOR = 2
SEMANTIC_PATCH_SIZE = 256
SEMANTIC_BATCH_SIZE = 2
SEMANTIC_NUM_WORKERS = 4
SEMANTIC_PADDING = None

POLYGON_SIMPLIFY_M = 15
MIN_FIELD_SIZE_M2 = 500
MAX_FIELD_SIZE_M2 = None
CLOSE_INTERIORS = True

# Delineate Anything inference.
INSTANCE_RESIZE_FACTOR = 2
INSTANCE_PATCH_SIZE = 256
INSTANCE_BATCH_SIZE = 4
INSTANCE_NUM_WORKERS = 4
INSTANCE_MAX_DETECTIONS = 100
INSTANCE_IOU_THRESHOLD = 0.30
INSTANCE_CONF_THRESHOLD = 0.05
INSTANCE_PADDING = None

INSTANCE_SIMPLIFY_M = 2
INSTANCE_MIN_SIZE_M2 = 500
INSTANCE_MAX_SIZE_M2 = 100000
INSTANCE_OVERLAP_IOU = 0.20
INSTANCE_OVERLAP_CONTAIN = 0.80

SITES = {
    "site_1": (8.820008333333334, -0.9106861111111111),
    "site_2": (8.881911111111112, -0.6818222222222222),
    "site_3": (10.23958611111111, -0.6730361111111111),
    "site_4": (10.816247222222223, -2.5668888888888888),
    "site_5": (9.112069444444444, -0.8517972222222222),
}


# ---------------------------------------------------------------------
# FTW IMPORTS
# ---------------------------------------------------------------------

try:
    from ftw_tools.download.download_img import create_input, scene_selection
    from ftw_tools.inference.inference import run, run_instance_segmentation
    from ftw_tools.inference.model_registry import MODEL_REGISTRY
    from ftw_tools.postprocess.polygonize import polygonize
except ImportError as exc:
    print(
        "\nERROR: ftw_tools could not be imported.\n\n"
        "Run this from inside the ftw-baselines repository, after:\n"
        "  uv sync --all-extras\n\n"
        "Then use:\n"
        "  uv run python run_ghana_ftw_models_fixed.py --list-models\n"
    )
    raise exc


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def make_bbox(lat: float, lon: float, size_km: float) -> list[float]:
    """Approximate square WGS84 bbox centred on lat/lon."""
    half_km = size_km / 2.0
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def list_models(include_legacy: bool = False) -> None:
    print("\nModels in the installed FTW registry:\n")

    for name, spec in MODEL_REGISTRY.items():
        if spec.legacy and not include_legacy:
            continue

        task = (
            "instance segmentation"
            if spec.instance_segmentation
            else "semantic segmentation"
        )
        windows = "2 windows" if spec.requires_window else "1 window"

        flags = []
        if spec.default:
            flags.append("default")
        if spec.legacy:
            flags.append("legacy")
        flag_text = f" [{', '.join(flags)}]" if flags else ""

        print(f"{name}{flag_text}")
        print(f"  {spec.title}")
        print(
            f"  {task} | {windows} | version {spec.version} | {spec.license}"
        )
        print()


def write_bbox_layer(path: Path, size_km: float, label: str) -> None:
    features = []

    for site_name, (lat, lon) in SITES.items():
        bounds = make_bbox(lat, lon, size_km)
        geom = box(*bounds)

        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "site": site_name,
                    "type": label,
                    "centre_lat": lat,
                    "centre_lon": lon,
                    "size_km": size_km,
                },
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            indent=2,
        ),
        encoding="utf-8",
    )


def scene_cache_path(site_name: str) -> Path:
    context_tag = str(PROCESSING_CONTEXT_SIZE_KM).replace(".", "p")
    return (
        OUTPUT_ROOT
        / "_shared"
        / "scene_selection"
        / f"{site_name}_{YEAR}_{STAC_HOST}_context{context_tag}km.json"
    )


def scene_cache_matches(cache: dict, bbox: list[float]) -> bool:
    expected = {
        "year": YEAR,
        "stac_host": STAC_HOST,
        "s2_collection": S2_COLLECTION,
        "cloud_cover_max": CLOUD_COVER_MAX,
        "buffer_days": BUFFER_DAYS,
        "nodata_max": NODATA_MAX,
        "processing_context_size_km": PROCESSING_CONTEXT_SIZE_KM,
    }

    for key, value in expected.items():
        if cache.get(key) != value:
            return False

    cached_bbox = cache.get("bbox")
    if not cached_bbox or len(cached_bbox) != 4:
        return False

    return all(
        abs(float(a) - float(b)) < 1e-10
        for a, b in zip(cached_bbox, bbox)
    )


def get_scenes(site_name: str, processing_bbox: list[float]) -> tuple[str, str]:
    cache_file = scene_cache_path(site_name)

    if cache_file.exists() and not OVERWRITE:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

        if scene_cache_matches(cache, processing_bbox):
            print(f"  Reusing cached scene selection: {cache_file}")
            return cache["window_a"], cache["window_b"]

    print("  Selecting Sentinel-2 scenes...")

    win_a, win_b = scene_selection(
        bbox=processing_bbox,
        year=YEAR,
        stac_host=STAC_HOST,
        cloud_cover_max=CLOUD_COVER_MAX,
        buffer_days=BUFFER_DAYS,
        s2_collection=S2_COLLECTION,
        nodata_max=NODATA_MAX,
        verbose=VERBOSE,
    )

    cache = {
        "site": site_name,
        "bbox": processing_bbox,
        "processing_context_size_km": PROCESSING_CONTEXT_SIZE_KM,
        "year": YEAR,
        "stac_host": STAC_HOST,
        "s2_collection": S2_COLLECTION,
        "cloud_cover_max": CLOUD_COVER_MAX,
        "buffer_days": BUFFER_DAYS,
        "nodata_max": NODATA_MAX,
        "window_a": win_a,
        "window_b": win_b,
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    return win_a, win_b


def shared_input_path(site_name: str, two_windows: bool) -> Path:
    mode = "two_window" if two_windows else "single_window"
    context_tag = str(PROCESSING_CONTEXT_SIZE_KM).replace(".", "p")

    return (
        OUTPUT_ROOT
        / "_shared"
        / "sentinel2_inputs"
        / site_name
        / f"{mode}_{YEAR}_context{context_tag}km.tif"
    )


def validate_raster_for_patch(image_path: Path, patch_size: int) -> None:
    with rasterio.open(image_path) as src:
        h = src.height
        w = src.width
        bands = src.count

    print(f"  Input raster: {w} x {h} px, {bands} bands")

    if min(h, w) < patch_size:
        raise ValueError(
            f"Input raster is only {w}x{h} pixels but patch_size={patch_size}. "
            f"Increase PROCESSING_CONTEXT_SIZE_KM above "
            f"{PROCESSING_CONTEXT_SIZE_KM} km."
        )


def prepare_input(
    site_name: str,
    processing_bbox: list[float],
    win_a: str,
    win_b: str,
    two_windows: bool,
) -> Path:
    image_path = shared_input_path(site_name, two_windows)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if image_path.exists() and not OVERWRITE:
        print(f"  Reusing imagery: {image_path}")
        return image_path

    print(
        "  Preparing "
        + ("two-window" if two_windows else "single-window")
        + f" Sentinel-2 input over {PROCESSING_CONTEXT_SIZE_KM} x "
          f"{PROCESSING_CONTEXT_SIZE_KM} km context..."
    )

    # IMPORTANT:
    # Current FTW create_input() has a verbose-summary bug for one-window
    # inputs: it writes the TIFF successfully and then tries identifiers[1].
    # Therefore verbose=False is used only for one-window inputs.
    download_verbose = VERBOSE if two_windows else False

    try:
        create_input(
            win_a=win_a,
            win_b=win_b if two_windows else None,
            out=str(image_path),
            overwrite=OVERWRITE,
            bbox=processing_bbox,
            stac_host=STAC_HOST,
            s2_collection=S2_COLLECTION,
            verbose=download_verbose,
        )
    except IndexError:
        # Extra protection for FTW versions that still hit the single-window
        # verbose summary bug. If the TIFF exists, it was already written.
        if not two_windows and image_path.exists():
            print(
                "  FTW hit its known one-window verbose-summary IndexError, "
                "but the TIFF was successfully created. Continuing."
            )
        else:
            raise

    if not image_path.exists():
        raise RuntimeError(f"Expected imagery was not created: {image_path}")

    return image_path


def clip_result_to_evaluation_aoi(
    source_path: Path,
    final_path: Path,
    lat: float,
    lon: float,
) -> Path:
    """Clip model predictions back to the central 1 x 1 km evaluation AOI."""
    gdf = gpd.read_file(source_path)

    # Preserve a valid empty GeoJSON result as well.
    if gdf.empty:
        gdf.to_file(final_path, driver="GeoJSON")
        return final_path

    eval_bbox = make_bbox(lat, lon, EVALUATION_AOI_SIZE_KM)

    aoi = gpd.GeoDataFrame(
        {"site": ["evaluation_aoi"]},
        geometry=[box(*eval_bbox)],
        crs="EPSG:4326",
    )

    aoi = aoi.to_crs(gdf.crs)
    clipped = gpd.clip(gdf, aoi)

    # Remove non-polygon fragments created exactly on the clipping edge.
    clipped = clipped[
        clipped.geometry.notna()
        & ~clipped.geometry.is_empty
        & clipped.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    final_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(final_path, driver="GeoJSON")

    print(
        f"  Clipped context predictions to central "
        f"{EVALUATION_AOI_SIZE_KM} x {EVALUATION_AOI_SIZE_KM} km AOI: "
        f"{len(clipped)} polygons"
    )

    return final_path


def run_instance_model(
    model_name: str,
    site_name: str,
    lat: float,
    lon: float,
    input_tif: Path,
    out_dir: Path,
) -> Path:
    context_output = out_dir / f"{site_name}_{safe_name(model_name)}_context.geojson"
    final_output = out_dir / f"{site_name}_{safe_name(model_name)}.geojson"

    if final_output.exists() and not OVERWRITE:
        print(f"  Final result already exists: {final_output}")
        return final_output

    validate_raster_for_patch(input_tif, INSTANCE_PATCH_SIZE)

    gpu_arg = None if GPU < 0 else GPU

    print(f"  Running instance model: {model_name}")

    run_instance_segmentation(
        input=str(input_tif),
        model=model_name,
        out=str(context_output),
        gpu=gpu_arg,
        num_workers=INSTANCE_NUM_WORKERS,
        patch_size=INSTANCE_PATCH_SIZE,
        resize_factor=INSTANCE_RESIZE_FACTOR,
        batch_size=INSTANCE_BATCH_SIZE,
        max_detections=INSTANCE_MAX_DETECTIONS,
        iou_threshold=INSTANCE_IOU_THRESHOLD,
        conf_threshold=INSTANCE_CONF_THRESHOLD,
        padding=INSTANCE_PADDING,
        overwrite=OVERWRITE,
        mps_mode=MPS_MODE,
        simplify=INSTANCE_SIMPLIFY_M,
        min_size=INSTANCE_MIN_SIZE_M2,
        max_size=INSTANCE_MAX_SIZE_M2,
        close_interiors=CLOSE_INTERIORS,
        overlap_iou_threshold=INSTANCE_OVERLAP_IOU,
        overlap_contain_threshold=INSTANCE_OVERLAP_CONTAIN,
    )

    return clip_result_to_evaluation_aoi(
        context_output,
        final_output,
        lat,
        lon,
    )


def run_semantic_model(
    model_name: str,
    site_name: str,
    lat: float,
    lon: float,
    input_tif: Path,
    out_dir: Path,
) -> Path:
    raster_output = out_dir / "inference_context.tif"
    context_vector = out_dir / f"{site_name}_{safe_name(model_name)}_context.geojson"
    final_output = out_dir / f"{site_name}_{safe_name(model_name)}.geojson"

    if final_output.exists() and not OVERWRITE:
        print(f"  Final result already exists: {final_output}")
        return final_output

    validate_raster_for_patch(input_tif, SEMANTIC_PATCH_SIZE)

    print(f"  Running semantic model: {model_name}")

    run(
        input=str(input_tif),
        model=model_name,
        out=str(raster_output),
        resize_factor=SEMANTIC_RESIZE_FACTOR,
        gpu=GPU,
        patch_size=SEMANTIC_PATCH_SIZE,
        batch_size=SEMANTIC_BATCH_SIZE,
        num_workers=SEMANTIC_NUM_WORKERS,
        padding=SEMANTIC_PADDING,
        overwrite=OVERWRITE,
        mps_mode=MPS_MODE,
        save_scores=False,
    )

    print("  Polygonizing semantic output...")

    polygonize(
        input=str(raster_output),
        out=str(context_vector),
        simplify=POLYGON_SIMPLIFY_M,
        min_size=MIN_FIELD_SIZE_M2,
        max_size=MAX_FIELD_SIZE_M2,
        overwrite=OVERWRITE,
        close_interiors=CLOSE_INTERIORS,
    )

    return clip_result_to_evaluation_aoi(
        context_vector,
        final_output,
        lat,
        lon,
    )


def run_site(
    model_name: str,
    site_name: str,
    lat: float,
    lon: float,
) -> Path:
    spec = MODEL_REGISTRY[model_name]

    evaluation_bbox = make_bbox(lat, lon, EVALUATION_AOI_SIZE_KM)
    processing_bbox = make_bbox(lat, lon, PROCESSING_CONTEXT_SIZE_KM)

    print("\n" + "=" * 78)
    print(site_name)
    print(f"centre (lat, lon): {lat:.6f}, {lon:.6f}")
    print(
        "evaluation bbox: "
        + ",".join(f"{v:.6f}" for v in evaluation_bbox)
    )
    print(
        "processing bbox: "
        + ",".join(f"{v:.6f}" for v in processing_bbox)
    )

    win_a, win_b = get_scenes(site_name, processing_bbox)

    two_windows = bool(spec.requires_window)

    input_tif = prepare_input(
        site_name=site_name,
        processing_bbox=processing_bbox,
        win_a=win_a,
        win_b=win_b,
        two_windows=two_windows,
    )

    out_dir = OUTPUT_ROOT / safe_name(model_name) / site_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.instance_segmentation:
        return run_instance_model(
            model_name,
            site_name,
            lat,
            lon,
            input_tif,
            out_dir,
        )

    return run_semantic_model(
        model_name,
        site_name,
        lat,
        lon,
        input_tif,
        out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an FTW model on the five Ghana test AOIs."
    )

    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Model-registry name. Default: {MODEL}",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List current non-legacy FTW registry models.",
    )

    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also show historical models.",
    )

    parser.add_argument(
        "--site",
        choices=list(SITES.keys()),
        default=None,
        help="Run only one site, e.g. --site site_1.",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models(include_legacy=args.include_legacy)
        return

    model_name = args.model

    if model_name not in MODEL_REGISTRY:
        print(f"\nUnknown model: {model_name}\n")
        list_models(include_legacy=True)
        sys.exit(2)

    spec = MODEL_REGISTRY[model_name]

    print("\nSelected model")
    print("--------------")
    print(f"Registry name : {model_name}")
    print(f"Title         : {spec.title}")
    print(f"Version       : {spec.version}")
    print(f"License       : {spec.license}")
    print(
        "Task          : "
        + (
            "instance segmentation"
            if spec.instance_segmentation
            else "semantic segmentation"
        )
    )
    print(
        "Input         : "
        + (
            "two Sentinel-2 windows"
            if spec.requires_window
            else "one Sentinel-2 window"
        )
    )
    print(
        f"Evaluation AOI: "
        f"{EVALUATION_AOI_SIZE_KM} x {EVALUATION_AOI_SIZE_KM} km"
    )
    print(
        f"Model context : "
        f"{PROCESSING_CONTEXT_SIZE_KM} x {PROCESSING_CONTEXT_SIZE_KM} km"
    )
    print(f"Year          : {YEAR}")

    # QGIS helper layers.
    write_bbox_layer(
        OUTPUT_ROOT / "evaluation_aoi_boxes.geojson",
        EVALUATION_AOI_SIZE_KM,
        "evaluation",
    )

    write_bbox_layer(
        OUTPUT_ROOT / "processing_context_boxes.geojson",
        PROCESSING_CONTEXT_SIZE_KM,
        "processing_context",
    )

    selected_sites = (
        {args.site: SITES[args.site]}
        if args.site is not None
        else SITES
    )

    successes = []
    failures = []

    for site_name, (lat, lon) in selected_sites.items():
        try:
            result = run_site(
                model_name=model_name,
                site_name=site_name,
                lat=lat,
                lon=lon,
            )
            successes.append((site_name, str(result)))
            print(f"  DONE: {result}")

        except Exception as exc:
            failures.append((site_name, f"{type(exc).__name__}: {exc}"))
            print(
                f"\n  FAILED {site_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            print("\nFull traceback:")
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for site_name, result in successes:
        print(f"[OK]   {site_name}: {result}")

    for site_name, error in failures:
        print(f"[FAIL] {site_name}: {error}")

    print(
        "\nEvaluation AOIs: "
        f"{OUTPUT_ROOT / 'evaluation_aoi_boxes.geojson'}"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
