#!/usr/bin/env python3
"""
Run FTP-PRUE+ (EfficientNet-B3 or B7) on coregistered 4-band PlanetScope
surface-reflectance GeoTIFFs.

Usage:
  python run_ftp_prue.py --input DATA_DIR --output OUT_DIR

DATA_DIR must contain one subfolder per area (e.g. Ghana_01_planet3m), each
holding exactly two GeoTIFFs: an earlier acquisition (window A) and a later
one (window B). Per area, expects:
  - 4 bands in PlanetScope order B, G, R, NIR
  - DN scale 10000 -> reflectance
  - ~3 m pixels
  - same geographic area (the script aligns B to A if needed)

For each area <label>, writes to OUT_DIR/<label>/:
  - <label>_classes.tif       uint8: 0 background, 1 field interior, 2 boundary
  - <label>_probabilities.tif float32: 3 class probabilities
  - <label>_fields.gpkg       approximate watershed vectorization for inspection

Important:
  The raster inference uses the released FTP-PRUE+ checkpoint and optional D4 TTA.
  The included watershed/vectorization is a practical approximation for local
  inspection, NOT a claim of bit-for-bit equivalence to the paper's official
  postprocess_eval.py. Use the repository's official post-processing for a
  benchmark-equivalent evaluation.
"""

import argparse
import sys
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from shapely.geometry import shape
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
import torch
import segmentation_models_pytorch as smp

TIF_SUFFIXES = {".tif", ".tiff"}


def align_to_reference(src_path, ref_profile):
    """Read 4 bands and align to the exact grid of ref_profile if needed."""
    with rasterio.open(src_path) as src:
        if src.count != 4:
            raise ValueError(f"{src_path} has {src.count} bands; expected exactly 4 SR bands (exclude UDM2).")

        same = (
            src.crs == ref_profile["crs"]
            and src.transform == ref_profile["transform"]
            and src.width == ref_profile["width"]
            and src.height == ref_profile["height"]
        )
        if same:
            return src.read([1, 2, 3, 4], masked=True).astype(np.float32).filled(np.nan)

        dst = np.zeros(
            (4, ref_profile["height"], ref_profile["width"]), dtype=np.float32
        )
        for i in range(4):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=dst[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        return dst


def d4_forward(x, k, flip):
    y = torch.rot90(x, k, dims=(-2, -1))
    if flip:
        y = torch.flip(y, dims=(-1,))
    return y


def d4_inverse(y, k, flip):
    if flip:
        y = torch.flip(y, dims=(-1,))
    return torch.rot90(y, -k, dims=(-2, -1))


@torch.inference_mode()
def predict_tile(model, tile, device, use_tta):
    x = torch.from_numpy(tile[None]).to(device=device, dtype=torch.float32)

    if not use_tta:
        return torch.softmax(model(x), dim=1)[0].cpu().numpy()

    probs = None
    n = 0
    for k in range(4):
        for flip in (False, True):
            xt = d4_forward(x, k, flip)
            p = torch.softmax(model(xt), dim=1)
            p = d4_inverse(p, k, flip)
            probs = p if probs is None else probs + p
            n += 1
    return (probs / n)[0].cpu().numpy()


def tiled_predict(model, image, device, tile_size=512, overlap=128, use_tta=True):
    _, h, w = image.shape
    stride = tile_size - overlap
    prob_sum = np.zeros((3, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, max(h - tile_size, 0) + 1, stride))
    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))
    if not xs or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))

    # Hann-style centre weighting reduces tile-edge seams.
    wy = np.hanning(tile_size).astype(np.float32)
    wx = np.hanning(tile_size).astype(np.float32)
    blend = np.outer(wy, wx)
    blend = np.maximum(blend, 1e-3)

    total = len(ys) * len(xs)
    done = 0

    for y0 in ys:
        for x0 in xs:
            done += 1
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            chip = image[:, y0:y1, x0:x1]
            ch, cw = chip.shape[-2:]
            if ch < tile_size or cw < tile_size:
                pad_h = tile_size - ch
                pad_w = tile_size - cw
                chip = np.pad(
                    chip,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="reflect",
                )

            p = predict_tile(model, chip, device, use_tta)[:, :ch, :cw]
            b = blend[:ch, :cw]
            prob_sum[:, y0:y1, x0:x1] += p * b
            weight[y0:y1, x0:x1] += b

            print(f"\rInference tile {done}/{total}", end="", flush=True)

    print()
    return prob_sum / np.maximum(weight[None], 1e-6)


def approximate_watershed(probs, h_seed=2.0):
    cls = probs.argmax(axis=0).astype(np.uint8)
    field_mask = cls != 0
    boundary_mask = cls == 2

    # Distance away from predicted boundaries; local maxima form field seeds.
    dist = ndi.distance_transform_edt(~boundary_mask)
    seeds = h_maxima(dist, h=h_seed) & (cls == 1)
    markers, _ = ndi.label(seeds)

    if markers.max() == 0:
        markers, _ = ndi.label(cls == 1)

    labels = watershed(-dist, markers=markers, mask=field_mask)
    return cls, labels.astype(np.int32)


def labels_to_gdf(labels, transform, crs, min_area_ha):
    geoms = []
    ids = []
    for geom, value in shapes(labels, mask=labels > 0, transform=transform):
        value = int(value)
        if value <= 0:
            continue
        geoms.append(shape(geom))
        ids.append(value)

    if not geoms:
        return gpd.GeoDataFrame({"field_id": []}, geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame({"field_id": ids}, geometry=geoms, crs=crs)

    # Filter by area in a metric CRS.
    if gdf.crs and gdf.crs.is_projected:
        metric = gdf
    else:
        metric_crs = gdf.estimate_utm_crs()
        metric = gdf.to_crs(metric_crs)

    area_ha = metric.geometry.area.values / 10000.0
    gdf["area_ha"] = area_ha
    gdf = gdf[gdf["area_ha"] >= min_area_ha].copy()
    gdf.reset_index(drop=True, inplace=True)
    return gdf


def discover_areas(input_dir):
    """Find one (label, window_a, window_b) triple per subfolder of input_dir.

    Each subfolder must hold exactly two GeoTIFFs; sorted by filename, the
    earlier acquisition (window A) sorts before the later one (window B)
    because PlanetScope filenames embed the acquisition timestamp.
    """
    areas = []
    for sub in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        tifs = sorted(p for p in sub.iterdir() if p.suffix.lower() in TIF_SUFFIXES)
        if len(tifs) != 2:
            raise ValueError(f"{sub}: expected exactly 2 GeoTIFFs, found {len(tifs)}")
        areas.append((sub.name, tifs[0], tifs[1]))
    if not areas:
        raise ValueError(f"{input_dir}: no subfolders found")
    return areas


def load_window_pair(window_a, window_b):
    with rasterio.open(window_a) as a:
        if a.count != 4:
            raise ValueError(f"{window_a} has {a.count} bands; expected exactly 4 SR bands (exclude UDM2).")
        arr_a = a.read([1, 2, 3, 4], masked=True).astype(np.float32).filled(np.nan)
        profile = {
            "crs": a.crs,
            "transform": a.transform,
            "width": a.width,
            "height": a.height,
            "nodata": a.nodata,
        }
    arr_b = align_to_reference(window_b, profile)
    return arr_a, arr_b, profile


def run_area(label, window_a, window_b, out_dir, model, device, args):
    """Run inference for a single area and write its outputs under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / label

    arr_a, arr_b, profile = load_window_pair(window_a, window_b)

    # PlanetScope 4-band SR band order is B,G,R,NIR; SR reflectance = DN / 10000.
    arr_a /= 10000.0
    arr_b /= 10000.0
    windows = [arr_a, arr_b] if args.channel_order == "a-b" else [arr_b, arr_a]
    image = np.concatenate(windows, axis=0).astype(np.float32)

    invalid = (~np.isfinite(image).all(axis=0)
               | np.all(arr_a == 0, axis=0) | np.all(arr_b == 0, axis=0))
    image[:, invalid] = 0.0

    probs = tiled_predict(
        model,
        image,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        use_tta=not args.no_tta,
    )
    probs[:, invalid] = 0.0

    classes, labels = approximate_watershed(probs, h_seed=args.h_seed)
    classes[invalid] = 255
    labels[invalid] = 0
    probs[:, invalid] = np.nan

    prob_path = out_prefix.with_name(out_prefix.name + "_probabilities.tif")
    cls_path = out_prefix.with_name(out_prefix.name + "_classes.tif")
    vec_path = out_prefix.with_name(out_prefix.name + "_fields.gpkg")

    base_profile = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "crs": profile["crs"],
        "transform": profile["transform"],
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(
        prob_path, "w", count=3, dtype="float32", nodata=np.nan, **base_profile
    ) as dst:
        dst.write(probs.astype(np.float32))
        dst.set_band_description(1, "background_probability")
        dst.set_band_description(2, "field_interior_probability")
        dst.set_band_description(3, "boundary_probability")

    with rasterio.open(
        cls_path, "w", count=1, dtype="uint8", nodata=255, **base_profile
    ) as dst:
        dst.write(classes, 1)
        dst.set_band_description(1, "FTP_PRUE_class_0bg_1field_2boundary")

    gdf = labels_to_gdf(
        labels,
        transform=profile["transform"],
        crs=profile["crs"],
        min_area_ha=args.min_area_ha,
    )
    if len(gdf):
        gdf.to_file(vec_path, layer="fields", driver="GPKG")

    return prob_path, cls_path, (vec_path if len(gdf) else None), len(gdf)


def resolve_device(args):
    if args.cpu:
        return "cpu"
    if args.device != "auto":
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                     help="Directory with one subfolder per area, each holding 2 GeoTIFFs")
    ap.add_argument("--output", type=Path, required=True,
                     help="Directory to write per-area results under (mirrors --input subfolders)")
    ap.add_argument("--model", choices=["b3", "b7"], default="b3",
                    help="Released FTP-PRUE+ backbone (default: b3).")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--channel-order", choices=["a-b", "b-a"], default="a-b",
                    help="a-b preserves the requested order; official FTP evaluation uses b-a.")
    ap.add_argument("--min-area-ha", type=float, default=0.05)
    ap.add_argument("--h-seed", type=float, default=2.0)
    ap.add_argument("--skip-existing", action="store_true",
                     help="Skip an area if its _classes.tif already exists")
    args = ap.parse_args()

    if args.tile_size <= 0 or args.tile_size % 32 or not 0 <= args.overlap < args.tile_size:
        raise ValueError("--tile-size must be a positive multiple of 32; require 0 <= overlap < tile-size")

    areas = discover_areas(args.input)
    device = resolve_device(args)
    if args.channel_order == "a-b":
        print("NOTE: official FTP evaluation uses B then A; use --channel-order b-a to match it.")

    model_id = f"taylor-geospatial/ftp-{args.model}"
    print(f"Loading FTP-PRUE+ {args.model.upper()} ({model_id}) on {device} ...")
    model = smp.from_pretrained(model_id).eval().to(device)

    failures = []
    for i, (label, window_a, window_b) in enumerate(areas, start=1):
        print(f"\n=== [{i}/{len(areas)}] {label} ===")
        out_dir = args.output / label
        if args.skip_existing and (out_dir / f"{label}_classes.tif").exists():
            print(f"{label}: already done, skipping")
            continue
        try:
            prob_path, cls_path, vec_path, n_fields = run_area(
                label, window_a, window_b, out_dir, model, device, args
            )
        except Exception:
            print(f"{label}: FAILED", file=sys.stderr)
            traceback.print_exc()
            failures.append(label)
            continue
        print(f"{label}: wrote {cls_path.name}, {prob_path.name}"
              + (f", {vec_path.name} ({n_fields} fields)" if vec_path else " (no fields survived filtering)"))

    print(f"\nDone: {len(areas) - len(failures)}/{len(areas)} areas succeeded.")
    if failures:
        print(f"Failed areas: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
