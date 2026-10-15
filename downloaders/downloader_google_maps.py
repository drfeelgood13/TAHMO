#!/usr/bin/env python3
"""Download Google Maps satellite tiles for areas listed in an input JSON file.

Usage: python downloader_google_maps.py --input AREAS.json --output OUTPUT_DIR
For each area and each TARGET_RES_M, tiles are downloaded, stitched, cropped to
the requested width_km bounds, and reprojected to the area's UTM zone.
"""

import argparse
import os
import math
import json
import time
from numbers import Real
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Output dir (--output) and areas JSON (--input) come from the command line.
EXPORT_FORMAT = "GEOTIFF"  # "PNG" or "GEOTIFF"
STITCH_FINAL_IMAGE = True  # Stitch all chunks into one master file

# Desired final pixel sizes in meters; every resolution is processed for
# every coordinate in the input JSON.
TARGET_RES_M = [2]

# Maximum safe chunk dimension per axis (16384px = ~800MB RAM per chunk canvas).
MAX_CHUNK_PX = 16384

# Zoom level controls ground coverage per 256x256 tile: meters_per_pixel =
# 156543.03 / (2 ** zoom) at the equator (finer at higher latitudes).

# Disable Pillow safety limit for massive high-resolution imagery.
Image.MAX_IMAGE_PIXELS = None

EARTH_RADIUS_KM = 6371.0088
EARTH_RADIUS_M = EARTH_RADIUS_KM * 1000.0

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_areas(path):
    """Load labeled boxes from a 'locations' manifest or legacy coordinate JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "locations" in data:
        if data.get("coordinate_crs", "EPSG:4326") != "EPSG:4326":
            raise ValueError("Location bounding boxes must use EPSG:4326")
        locations = data["locations"]
        if not isinstance(locations, list) or not locations:
            raise ValueError("'locations' must be a non-empty list")
        areas = []
        labels = set()
        for index, location in enumerate(locations, start=1):
            if not isinstance(location, dict):
                raise ValueError(f"Location {index} must be an object")
            label = location.get("location_label")
            box_data = location.get("bounding_box")
            keys = ("west_lon", "south_lat", "east_lon", "north_lat")
            if not isinstance(box_data, dict) or not all(
                isinstance(box_data.get(key), Real)
                and not isinstance(box_data.get(key), bool)
                and math.isfinite(box_data[key])
                for key in keys
            ):
                raise ValueError(f"Location {index} needs numeric bounds: {keys}")
            west, south, east, north = (box_data[key] for key in keys)
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError(f"Location {index} has an invalid WGS84 bounding box")
            if (
                not isinstance(label, str)
                or not label.strip()
                or label in (".", "..")
                or "/" in label
                or "\\" in label
            ):
                raise ValueError(f"Location {index} needs a safe location_label")
            if label in labels:
                raise ValueError(f"Duplicate location label: {label}")
            labels.add(label)
            areas.append({
                "label": label,
                "center": ((south + north) / 2, (west + east) / 2),
                "bbox": (west, south, east, north),
                "width_km": None,
            })
        return areas

    prefix = data["prefix"]
    width_km = data["width_km"]
    centers = [(lat, lon) for lat, lon in data["coordinates"]]
    if not centers:
        raise ValueError(f"{path}: coordinates must contain at least one [lat, lon] pair")
    areas = []
    for index, (lat, lon) in enumerate(centers, start=1):
        north, south, west, east = bbox_from_center_and_width(lat, lon, width_km)
        areas.append({
            "label": f"{prefix}_{index:02d}",
            "center": (lat, lon),
            "bbox": (west, south, east, north),
            "width_km": width_km,
        })
    return areas


def destination_point(lat, lon, bearing_deg, distance_km):
    """Given a start point, bearing, and distance (great-circle, spherical Earth),
    returns the resulting lat/lon. Bearing: 0=N, 90=E, 180=S, 270=W."""
    R = EARTH_RADIUS_KM
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_r = distance_km / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r)
        + math.cos(lat1) * math.sin(d_r) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def bbox_from_center_and_width(center_lat, center_lon, width_km):
    """Builds a square bounding box centered on a point, width/height in km."""
    half_width = width_km / 2.0

    north_lat, _ = destination_point(center_lat, center_lon, 0, half_width)
    south_lat, _ = destination_point(center_lat, center_lon, 180, half_width)
    _, east_lon = destination_point(center_lat, center_lon, 90, half_width)
    _, west_lon = destination_point(center_lat, center_lon, 270, half_width)

    return north_lat, south_lat, west_lon, east_lon


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile


def tile_to_latlon(x, y, zoom):
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return math.degrees(lat_rad), lon_deg


def equalize_tile_grid(x_min, x_max, y_min, y_max):
    """Force the tile grid to be square by symmetrically expanding the
    axis that came up short. Fixes non-square output caused by Mercator
    tile-grid quantization."""
    x_count = x_max - x_min + 1
    y_count = y_max - y_min + 1
    target = max(x_count, y_count)

    if x_count < target:
        diff = target - x_count
        expand_left = diff // 2
        expand_right = diff - expand_left
        x_min -= expand_left
        x_max += expand_right

    if y_count < target:
        diff = target - y_count
        expand_top = diff // 2
        expand_bottom = diff - expand_top
        y_min -= expand_top
        y_max += expand_bottom

    return x_min, x_max, y_min, y_max


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_M * c


def compute_real_world_stats(north_lat, south_lat, west_lon, east_lon, pixel_w, pixel_h):
    """Computes ground width/height/area plus GSD (ground sample distance)
    for the RAW scraped mosaic, based on lat/lon bounds and native pixel count.
    NOTE: this describes the pre-reprojection (EPSG:4326) image, not the
    final UTM-resampled file. See compute_real_world_stats_utm for that."""
    mid_lat = (north_lat + south_lat) / 2.0

    width_m = haversine_distance_m(mid_lat, west_lon, mid_lat, east_lon)
    height_m = haversine_distance_m(north_lat, west_lon, south_lat, west_lon)

    area_m2 = width_m * height_m
    area_ha = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    gsd_x_m_per_px = width_m / pixel_w if pixel_w else None
    gsd_y_m_per_px = height_m / pixel_h if pixel_h else None

    return {
        "width_m": round(width_m, 3),
        "height_m": round(height_m, 3),
        "area_m2": round(area_m2, 3),
        "area_hectares": round(area_ha, 6),
        "area_km2": round(area_km2, 8),
        "ground_sample_distance_m_per_px": {
            "x": round(gsd_x_m_per_px, 6) if gsd_x_m_per_px else None,
            "y": round(gsd_y_m_per_px, 6) if gsd_y_m_per_px else None,
        },
    }


def compute_real_world_stats_utm(utm_path):
    """Reads the ACTUAL pixel size/dimensions from the final reprojected
    UTM GeoTIFF and returns real-world stats consistent with that file.
    This is the block that should be trusted for GSD, not the raw-mosaic one."""
    with rasterio.open(utm_path) as src:
        px_w, px_h = src.res  # meters/pixel, from the actual written transform
        width_m = src.width * px_w
        height_m = src.height * px_h
        area_m2 = width_m * height_m

        return {
            "crs": str(src.crs),
            "width_px": src.width,
            "height_px": src.height,
            "width_m": round(width_m, 3),
            "height_m": round(height_m, 3),
            "area_m2": round(area_m2, 3),
            "area_hectares": round(area_m2 / 10_000.0, 6),
            "area_km2": round(area_m2 / 1_000_000.0, 8),
            "ground_sample_distance_m_per_px": {
                "x": round(px_w, 6),
                "y": round(px_h, 6),
            },
        }


def save_utm_as_png(utm_path, png_path):
    """Export the final UTM raster as an RGB PNG with identical dimensions."""
    with rasterio.open(utm_path) as src:
        image_array = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    Image.fromarray(image_array).save(png_path, format="PNG")


def compute_center_coords(north_lat, south_lat, west_lon, east_lon):
    """Computes the midpoint lat/lon of a bounding box."""
    c_lat = (north_lat + south_lat) / 2.0
    c_lon = (west_lon + east_lon) / 2.0
    return {"lat": round(c_lat, 8), "lon": round(c_lon, 8)}


def compute_vertices(north_lat, south_lat, west_lon, east_lon):
    """Returns the four corner coordinates of the bounding box."""
    return {
        "north_west": {"lat": round(north_lat, 8), "lon": round(west_lon, 8)},
        "north_east": {"lat": round(north_lat, 8), "lon": round(east_lon, 8)},
        "south_east": {"lat": round(south_lat, 8), "lon": round(east_lon, 8)},
        "south_west": {"lat": round(south_lat, 8), "lon": round(west_lon, 8)},
    }


def utm_epsg_from_center(center_lat, center_lon):
    """Return the WGS 84 UTM CRS for a center coordinate."""
    if not -80 <= center_lat <= 84:
        raise ValueError(
            "Automatic UTM selection supports latitudes from -80 to 84 degrees."
        )
    if not -180 <= center_lon <= 180:
        raise ValueError("Center longitude must be between -180 and 180 degrees.")

    zone = min(60, math.floor((center_lon + 180) / 6) + 1)
    epsg_code = 32600 + zone if center_lat >= 0 else 32700 + zone
    return f"EPSG:{epsg_code}"


def zoom_from_target_resolution(target_res_m):
    """Choose the lowest Web Mercator zoom that is at least as detailed."""
    if target_res_m <= 0:
        raise ValueError("TARGET_RES_M must be greater than 0.")

    equator_resolution_m = 156543.03
    zoom = math.ceil(math.log2(equator_resolution_m / target_res_m))
    return max(0, zoom)


def format_resolution_label(target_res_m):
    """Create a filesystem-safe label for a target resolution."""
    if float(target_res_m).is_integer():
        value = str(int(target_res_m))
    else:
        value = f"{target_res_m:g}".replace(".", "p")
    return f"{value}m"


def format_location_label(area_label, resolution_label):
    """Use the requested output resolution instead of the source dataset suffix."""
    source_label = area_label.removesuffix("_0p3m")
    return f"{source_label}_{resolution_label}"


def crop_to_requested_bounds(
    image, requested_north, requested_south, requested_west, requested_east,
    tile_x_min, tile_y_min, zoom,
):
    """Crop a tile-aligned image to the requested geographic bounding box."""
    tile_count = 2.0 ** zoom

    def pixel_x(lon):
        return ((lon + 180.0) / 360.0 * tile_count - tile_x_min) * 256

    def pixel_y(lat):
        y_tile = (
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi)
            / 2.0
            * tile_count
        )
        return (y_tile - tile_y_min) * 256

    left = max(0, math.floor(pixel_x(requested_west)))
    top = max(0, math.floor(pixel_y(requested_north)))
    right = min(image.width, math.ceil(pixel_x(requested_east)))
    bottom = min(image.height, math.ceil(pixel_y(requested_south)))

    if left >= right or top >= bottom:
        raise ValueError("Requested bounds do not overlap the downloaded tile canvas.")

    return image.crop((left, top, right, bottom))


def fetch_tile_with_retry(session, url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.content
        except requests.RequestException:
            pass
        time.sleep(0.5 * (2 ** attempt))
    return None


def save_as_geotiff(pil_img, output_path, west_lon, south_lat, east_lon, north_lat):
    """Write an EPSG:4326 GeoTIFF with nodata=0."""
    img_array = np.array(pil_img)
    # (H, W, C) -> (C, H, W), bands in RGB order
    img_array = np.moveaxis(img_array, -1, 0)

    height, width = pil_img.height, pil_img.width
    transform = from_bounds(west_lon, south_lat, east_lon, north_lat, width, height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img_array.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        nodata=0,
    ) as dst:
        dst.write(img_array)


def reproject_to_utm(src_path, dst_path, dst_crs, dst_res_m):
    """Reproject src_path (EPSG:4326) to UTM and optionally resample.

    - If dst_res_m is a float (e.g. 10.0), enforce that pixel size in meters.
    - If dst_res_m is None, keep near-native resolution (no explicit resolution override).
    """
    with rasterio.open(src_path) as src:
        if dst_res_m is None:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=dst_res_m,
            )

        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=0,
            dtype=rasterio.uint8,
            count=src.count,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def save_chunk(pil_img, output_dir, file_prefix, fmt,
               w_lon, s_lat, e_lon, n_lat, meta_dict, target_crs,
               target_res_m):
    """Handles file saving based on the chosen format toggle."""
    if fmt.upper() == "GEOTIFF":
        src_path = os.path.join(output_dir, f"{file_prefix}.tif")
        save_as_geotiff(pil_img, src_path, w_lon, s_lat, e_lon, n_lat)

        # Reproject/resample to UTM at the requested target resolution.
        dst_path = os.path.join(output_dir, f"{file_prefix}_utm.tif")
        reproject_to_utm(
            src_path, dst_path, dst_crs=target_crs, dst_res_m=target_res_m
        )

        # Recompute real-world stats from the ACTUAL final UTM file,
        # so the JSON reflects the true output resolution/dimensions.
        meta_dict["real_world_utm"] = compute_real_world_stats_utm(dst_path)

        file_path = dst_path  # use UTM file as the main output
        stitch_path = src_path
    else:
        file_path = os.path.join(output_dir, f"{file_prefix}.png")
        stitch_path = os.path.join(output_dir, f"{file_prefix}_stitch.tif")
        stitch_utm_path = os.path.join(output_dir, f"{file_prefix}_stitch_utm.tif")

        # Keep the raw GeoTIFF only as a temporary source for stitching, while
        # the visible PNG uses the same target-resolution dimensions as UTM.
        save_as_geotiff(pil_img, stitch_path, w_lon, s_lat, e_lon, n_lat)
        reproject_to_utm(
            stitch_path,
            stitch_utm_path,
            dst_crs=target_crs,
            dst_res_m=target_res_m,
        )
        meta_dict["real_world_utm"] = compute_real_world_stats_utm(stitch_utm_path)
        save_utm_as_png(stitch_utm_path, file_path)
        os.remove(stitch_utm_path)

    meta_path = os.path.join(output_dir, f"{file_prefix}.json")
    with open(meta_path, "w") as stream:
        json.dump(meta_dict, stream, indent=4)

    return file_path, stitch_path


def process_location(
    center_lat,
    center_lon,
    width_km,
    requested_bbox,
    zoom,
    output_dir,
    location_label,
    target_res_m,
):
    """Runs the full download -> chunk -> save -> stitch pipeline for a single center."""
    target_crs = utm_epsg_from_center(center_lat, center_lon)
    req_west_lon, req_south_lat, req_east_lon, req_north_lat = requested_bbox

    os.makedirs(output_dir, exist_ok=True)

    x_min, y_max = latlon_to_tile(req_south_lat, req_west_lon, zoom)
    x_max, y_min = latlon_to_tile(req_north_lat, req_east_lon, zoom)

    x_min, x_max, y_min, y_max = equalize_tile_grid(x_min, x_max, y_min, y_max)

    total_x_tiles = (x_max - x_min) + 1
    total_y_tiles = (y_max - y_min) + 1
    tile_size = 256

    canvas_w = total_x_tiles * tile_size
    canvas_h = total_y_tiles * tile_size

    max_tiles_per_chunk_axis = MAX_CHUNK_PX // tile_size
    splits_x = math.ceil(total_x_tiles / max_tiles_per_chunk_axis)
    splits_y = math.ceil(total_y_tiles / max_tiles_per_chunk_axis)
    total_chunks = splits_x * splits_y

    chunk_x_size = math.ceil(total_x_tiles / splits_x)
    chunk_y_size = math.ceil(total_y_tiles / splits_y)

    print(
        f"\n=== {location_label} | Center: {center_lat}, {center_lon} | "
        f"Bounds: {requested_bbox} | Width: {width_km if width_km else 'manifest box'} ==="
    )
    print(
        f"Format: {EXPORT_FORMAT.upper()} | Stitch Master File: {STITCH_FINAL_IMAGE}"
    )
    print(f"Target resolution: {target_res_m} m/px | Download zoom: {zoom}")
    print(f"UTM CRS: {target_crs}")
    print(
        f"Zoom Level: {zoom} | Total Tiles: {total_x_tiles * total_y_tiles} "
        f"({total_x_tiles}x{total_y_tiles} grid)"
    )
    print(f"Full Canvas Resolution: {canvas_w} x {canvas_h} pixels")
    print(f"Grid: {splits_x}x{splits_y} ({total_chunks} chunk files required)\n")

    headers = {"User-Agent": USER_AGENT}
    saved_chunks_info = []
    temporary_stitch_paths = []

    with requests.Session() as session:
        chunk_counter = 0
        for cy in range(splits_y):
            for cx in range(splits_x):
                chunk_counter += 1

                sub_x_min = x_min + cx * chunk_x_size
                sub_x_max = min(x_min + (cx + 1) * chunk_x_size - 1, x_max)
                sub_y_min = y_min + cy * chunk_y_size
                sub_y_max = min(y_min + (cy + 1) * chunk_y_size - 1, y_max)

                if sub_x_min > x_max or sub_y_min > y_max:
                    continue

                c_w_tiles = (sub_x_max - sub_x_min) + 1
                c_h_tiles = (sub_y_max - sub_y_min) + 1

                c_pixel_w = c_w_tiles * tile_size
                c_pixel_h = c_h_tiles * tile_size

                chunk_img = Image.new("RGB", (c_pixel_w, c_pixel_h))

                tile_tasks = [
                    (x_idx, x, y_idx, y)
                    for x_idx, x in enumerate(range(sub_x_min, sub_x_max + 1))
                    for y_idx, y in enumerate(range(sub_y_min, sub_y_max + 1))
                ]

                file_prefix = "mosaic" if total_chunks == 1 else f"mosaic_chunk_{cy}_{cx}"
                desc = (
                    f"[{location_label}] Chunk {chunk_counter}/{total_chunks} "
                    f"[{c_pixel_w}x{c_pixel_h}px]"
                )

                for x_idx, x, y_idx, y in tqdm(tile_tasks, desc=desc, unit="tile"):
                    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
                    tile_data = fetch_tile_with_retry(session, url, headers)

                    if not tile_data:
                        raise RuntimeError(f"Failed to download Google Maps tile z={zoom}, x={x}, y={y}")
                    tile_img = Image.open(BytesIO(tile_data))
                    chunk_img.paste(tile_img, (x_idx * tile_size, y_idx * tile_size))
                    time.sleep(0.01)

                top_left_lat, top_left_lon = tile_to_latlon(sub_x_min, sub_y_min, zoom)
                bottom_right_lat, bottom_right_lon = tile_to_latlon(
                    sub_x_max + 1, sub_y_max + 1, zoom
                )

                real_world_stats = compute_real_world_stats(
                    top_left_lat,
                    bottom_right_lat,
                    top_left_lon,
                    bottom_right_lon,
                    c_pixel_w,
                    c_pixel_h,
                )
                center_coords = compute_center_coords(
                    top_left_lat,
                    bottom_right_lat,
                    top_left_lon,
                    bottom_right_lon,
                )
                vertices = compute_vertices(
                    top_left_lat,
                    bottom_right_lat,
                    top_left_lon,
                    bottom_right_lon,
                )

                metadata = {
                    "location_label": location_label,
                    "zoom_level": zoom,
                    "target_resolution_m": target_res_m,
                    "chunk_grid": {
                        "row": cy,
                        "col": cx,
                        "total_rows": splits_y,
                        "total_cols": splits_x,
                    },
                    "pixel_width": c_pixel_w,
                    "pixel_height": c_pixel_h,
                    "bounding_box": {
                        "north_lat": top_left_lat,
                        "south_lat": bottom_right_lat,
                        "west_lon": top_left_lon,
                        "east_lon": bottom_right_lon,
                    },
                    "vertices": vertices,
                    "center_coordinates": center_coords,
                    "utm_epsg": target_crs,
                    # NOTE: describes the RAW pre-reprojection mosaic (EPSG:4326).
                    # See "real_world_utm" (added in save_chunk) for the actual
                    # final UTM file's resolution/dimensions.
                    "real_world": real_world_stats,
                }

                chunk_path, stitch_path = save_chunk(
                    chunk_img,
                    output_dir,
                    file_prefix,
                    EXPORT_FORMAT,
                    top_left_lon,
                    bottom_right_lat,
                    bottom_right_lon,
                    top_left_lat,
                    metadata,
                    target_crs,
                    target_res_m,
                )

                # Stitch raw chunks that share the original Web Mercator pixel
                # grid. The UTM files have independently reprojected dimensions
                # and cannot be pasted into this pixel-based canvas reliably.
                offset_x = cx * chunk_x_size * tile_size
                offset_y = cy * chunk_y_size * tile_size
                saved_chunks_info.append((stitch_path, offset_x, offset_y))
                if EXPORT_FORMAT.upper() == "PNG":
                    temporary_stitch_paths.append(stitch_path)

    # The tile canvas can be larger than the requested bounds because tiles
    # are indivisible; crop the final sweep before saving.
    if STITCH_FINAL_IMAGE:
        print(f"\n[{location_label}] Attempting final master sweep stitch...")
        try:
            master_img = Image.new("RGB", (canvas_w, canvas_h))

            for chunk_path, off_x, off_y in tqdm(
                saved_chunks_info,
                desc=f"[{location_label}] Merging Chunks",
                unit="file",
            ):
                with Image.open(chunk_path) as c_img:
                    master_img.paste(c_img, (off_x, off_y))

            master_img = crop_to_requested_bounds(
                master_img,
                req_north_lat,
                req_south_lat,
                req_west_lon,
                req_east_lon,
                x_min,
                y_min,
                zoom,
            )
            master_prefix = "mosaic"
            master_top_left_lat = req_north_lat
            master_bottom_right_lat = req_south_lat
            master_top_left_lon = req_west_lon
            master_bottom_right_lon = req_east_lon

            master_real_world_stats = compute_real_world_stats(
                master_top_left_lat,
                master_bottom_right_lat,
                master_top_left_lon,
                master_bottom_right_lon,
                master_img.width,
                master_img.height,
            )
            master_center_coords = compute_center_coords(
                master_top_left_lat,
                master_bottom_right_lat,
                master_top_left_lon,
                master_bottom_right_lon,
            )
            master_vertices = compute_vertices(
                master_top_left_lat,
                master_bottom_right_lat,
                master_top_left_lon,
                master_bottom_right_lon,
            )

            master_meta = {
                "location_label": location_label,
                "zoom_level": zoom,
                "target_resolution_m": target_res_m,
                "pixel_width": master_img.width,
                "pixel_height": master_img.height,
                "requested": {
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "bounding_box": {
                        "west_lon": req_west_lon,
                        "south_lat": req_south_lat,
                        "east_lon": req_east_lon,
                        "north_lat": req_north_lat,
                    },
                    **({"width_km": width_km} if width_km is not None else {}),
                },
                "utm_epsg": target_crs,
                "bounding_box": {
                    "north_lat": master_top_left_lat,
                    "south_lat": master_bottom_right_lat,
                    "west_lon": master_top_left_lon,
                    "east_lon": master_bottom_right_lon,
                },
                "vertices": master_vertices,
                "center_coordinates": master_center_coords,
                # NOTE: describes the RAW pre-reprojection master mosaic.
                # "real_world_utm" is added below once the UTM file is written.
                "real_world": master_real_world_stats,
            }

            # Save the master temporarily in EPSG:4326 so it can be reprojected.
            # The raw master is removed after the UTM file is written.
            master_src_path = os.path.join(output_dir, f"{master_prefix}.tif")
            save_as_geotiff(
                master_img,
                master_src_path,
                master_top_left_lon,
                master_bottom_right_lat,
                master_bottom_right_lon,
                master_top_left_lat,
            )

            # Reproject/resample to the requested target resolution.
            master_dst_path = os.path.join(output_dir, f"{master_prefix}_utm.tif")
            reproject_to_utm(
                master_src_path,
                master_dst_path,
                dst_crs=target_crs,
                dst_res_m=target_res_m,
            )

            # Recompute real-world stats from the ACTUAL final UTM file
            master_meta["real_world_utm"] = compute_real_world_stats_utm(master_dst_path)

            if EXPORT_FORMAT.upper() == "PNG":
                master_png_path = os.path.join(output_dir, f"{master_prefix}.png")
                save_utm_as_png(master_dst_path, master_png_path)

            master_meta_path = os.path.join(output_dir, f"{master_prefix}.json")
            with open(master_meta_path, "w") as stream:
                json.dump(master_meta, stream, indent=4)

            os.remove(master_src_path)
            print(f"[{location_label}] Master file saved successfully to: {output_dir}")

        except MemoryError:
            print(f"\n[{location_label}] WARNING: MemoryError during final stitch sweep.")
            print("The full canvas is too large for RAM. Individual chunks remain saved safely.")

    for stitch_path in temporary_stitch_paths:
        if os.path.exists(stitch_path):
            os.remove(stitch_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Test-set manifest or legacy coordinate JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--resolutions", type=float, nargs="+", default=None, metavar="M",
                         help=f"Target pixel sizes in metres, e.g. --resolutions 0.3 1 2 5 10 20 (default: {TARGET_RES_M})")
    parser.add_argument("--start-index", type=int, default=1,
                         help="Label number for the first coordinate in --input; use to continue numbering across batches")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip an area if its master UTM mosaic and metadata already exist")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    input_data = json.loads(cli.input.read_text(encoding="utf-8-sig"))
    manifest_input = "locations" in input_data
    areas = load_areas(cli.input)
    if cli.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    if cli.start_index != 1 and manifest_input:
        raise ValueError("--start-index only applies to legacy coordinate JSON files")

    resolutions = cli.resolutions if cli.resolutions is not None else TARGET_RES_M
    if isinstance(resolutions, Real):
        target_resolutions = [resolutions]
    else:
        target_resolutions = list(resolutions)

    if not target_resolutions:
        raise ValueError("TARGET_RES_M must contain at least one resolution.")
    if any(
        isinstance(target_res_m, bool) or not isinstance(target_res_m, Real)
        or target_res_m <= 0
        for target_res_m in target_resolutions
    ):
        raise ValueError("Every TARGET_RES_M value must be a positive number.")

    total_runs = len(areas) * len(target_resolutions)
    run_number = 0
    failures = []
    cli.output.mkdir(parents=True, exist_ok=True)
    for idx, area in enumerate(areas, start=1):
        if not manifest_input:
            area = area.copy()
            area["label"] = f"{area['label'].rsplit('_', 1)[0]}_{idx + cli.start_index - 1:02d}"
        lat, lon = area["center"]
        for target_res_m in target_resolutions:
            run_number += 1
            resolution_label = format_resolution_label(target_res_m)
            location_label = format_location_label(area["label"], resolution_label)
            print(f"\nStarting run {run_number}/{total_runs}")
            zoom = zoom_from_target_resolution(target_res_m)
            location_output_dir = cli.output / location_label
            if cli.skip_existing and all(
                (location_output_dir / filename).is_file()
                for filename in ("mosaic_utm.tif", "mosaic.json")
            ):
                print(f"{location_label}: complete, skipping")
                continue
            try:
                process_location(
                    lat,
                    lon,
                    area["width_km"],
                    area["bbox"],
                    zoom,
                    str(location_output_dir),
                    location_label,
                    target_res_m,
                )
            except Exception as exc:
                failures.append({"location": location_label, "error": str(exc)})
                print(f"{location_label}: FAILED: {exc}", flush=True)

    failure_path = cli.output / "failed_areas.json"
    failure_path.write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        print(f"\n{len(failures)} location/resolution run(s) failed; see {failure_path}")
        raise SystemExit(1)

    print(
        f"\nAll {total_runs} coordinate/resolution runs processed. "
        f"Output saved under: {cli.output}"
    )


if __name__ == "__main__":
    main()