#!/usr/bin/env python3
"""Order and download PlanetScope imagery for areas listed in an input JSON file.

Usage: python downloader_planet.py --input AREAS.json --output OUTPUT
Set PL_API_KEY in the environment before running. Reuse OUTPUT to resume.
New orders consume Planet quota; edit WINDOW_A/WINDOW_B/MAX_CLOUD/MIN_COVERAGE
below only to request an intentional new dataset.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
from datetime import date
from pathlib import Path

import rasterio
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from urllib3.util.retry import Retry

WINDOW_A = ("2025-04-01", "2025-05-31")
WINDOW_B = ("2025-08-01", "2025-09-30")
MAX_CLOUD = 0.10
MIN_COVERAGE = 0.999
SEARCH_WINDOWS_A = (WINDOW_A, ("2025-03-01", "2025-06-30"), ("2025-01-01", "2025-07-15"))
SEARCH_WINDOWS_B = (WINDOW_B, ("2025-07-16", "2025-11-30"), ("2025-07-16", "2025-12-31"))
SEARCH_FILTERS = ((MAX_CLOUD, MIN_COVERAGE), (0.20, 0.99), (0.35, 0.95))
POLL_SECONDS = 30
WAIT_HOURS = 12
API = "https://api.planet.com"
ORDERS = API + "/compute/ops/orders/v2"


class NoEligibleSceneError(RuntimeError):
    """Raised when no acquisition matches one bounded scene-search attempt."""


def load_areas(path):
    """Load labeled AOIs from a test-set manifest or legacy coordinate JSON."""
    data = json.loads(Path(path).read_text())
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
            bounds = location.get("bounding_box")
            keys = ("west_lon", "south_lat", "east_lon", "north_lat")
            if not isinstance(bounds, dict) or not all(
                isinstance(bounds.get(key), (int, float))
                and not isinstance(bounds.get(key), bool)
                and math.isfinite(bounds[key])
                for key in keys
            ):
                raise ValueError(f"Location {index} needs numeric bounds: {keys}")
            west, south, east, north = (bounds[key] for key in keys)
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
            aoi = mapping(box(west, south, east, north))
            areas.append({
                "label": label,
                "folder": label,
                "work_folder": f".orders/{label}",
                "file_prefix": label.lower(),
                "aoi": aoi,
                "utm_epsg": utm_epsg_for_aoi(aoi),
            })
        return areas

    prefix = data["prefix"]
    width_km = data["width_km"]
    centers = [(lat, lon) for lat, lon in data["coordinates"]]
    if not centers:
        raise ValueError(f"{path}: coordinates must contain at least one [lat, lon] pair")
    areas = []
    for index, center in enumerate(centers, start=1):
        aoi = aoi_bbox(*center, width_km)
        folder = f"{prefix}_{index:02d}_planet3m"
        areas.append({
            "label": folder,
            "folder": folder,
            "work_folder": f"{prefix}_{index:02d}",
            "file_prefix": f"{prefix.lower()}-{index:02d}",
            "aoi": aoi,
            "utm_epsg": utm_epsg_for_aoi(aoi),
            "legacy_prefix": prefix,
            "legacy_number": index,
        })
    return areas


def apply_start_index(area, start_index):
    if "legacy_number" not in area:
        return area
    area = area.copy()
    number = area["legacy_number"] + start_index - 1
    area["label"] = area["folder"] = f"{area['legacy_prefix']}_{number:02d}_planet3m"
    area["work_folder"] = f"{area['legacy_prefix']}_{number:02d}"
    area["file_prefix"] = f"{area['legacy_prefix'].lower()}-{number:02d}"
    return area


def utm_epsg_for_aoi(aoi):
    centroid = shape(aoi).centroid
    zone = min(60, max(1, math.floor((centroid.x + 180) / 6) + 1))
    return (32600 if centroid.y >= 0 else 32700) + zone


def aoi_bbox(lat, lon, width_km):
    """Square AOI centered on (lat, lon), computed via spherical cardinal offsets."""
    phi, lam = math.radians(lat), math.radians(lon)
    d = width_km / 2 / 6371.0088

    def destination(bearing):
        b = math.radians(bearing)
        p = math.asin(math.sin(phi) * math.cos(d)
                      + math.cos(phi) * math.sin(d) * math.cos(b))
        l = lam + math.atan2(math.sin(b) * math.sin(d) * math.cos(phi),
                             math.cos(d) - math.sin(phi) * math.sin(p))
        return math.degrees(p), math.degrees(l)

    north, _ = destination(0)
    south, _ = destination(180)
    _, east = destination(90)
    _, west = destination(270)
    return json.loads(json.dumps(mapping(box(west, south, east, north))))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def open_session(key=None):
    http = requests.Session()
    if key:
        http.auth = (key, "")
    # Never automatically retry order creation POSTs.
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods={"GET"})
    http.mount("https://", HTTPAdapter(max_retries=retry))
    return http


def api_json(http, method, url, **kwargs):
    if not url.startswith(API + "/"):
        raise RuntimeError("Refusing to send Planet credentials to another host")
    try:
        r = http.request(method, url, timeout=(30, 120), **kwargs)
        if not r.ok:
            raise RuntimeError(f"Planet API HTTP {r.status_code}; check permissions, quota and request")
        return r.json()
    except requests.RequestException:
        raise RuntimeError("Planet API network failure; rerun with the same output directory") from None


def find_best_scene(http, aoi, start, end, max_cloud, min_coverage):
    filters = [
        {"type": "GeometryFilter", "field_name": "geometry", "config": aoi},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": start + "T00:00:00Z", "lte": end + "T23:59:59.999999Z"}},
        {"type": "StringInFilter", "field_name": "quality_category", "config": ["standard"]},
        {"type": "AssetFilter", "config": ["ortho_analytic_4b_sr"]},
        {"type": "PermissionFilter", "config": ["assets:download"]},
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": max_cloud}},
    ]
    page = api_json(http, "POST", API + "/data/v1/quick-search",
                    json={"item_types": ["PSScene"], "filter": {"type": "AndFilter", "config": filters}})
    project = Transformer.from_crs(4326, utm_epsg_for_aoi(aoi), always_xy=True).transform
    footprint = transform(project, shape(aoi))
    candidates = []
    while True:
        for item in page["features"]:
            coverage = footprint.intersection(transform(project, shape(item["geometry"]))).area / footprint.area
            if coverage + 1e-9 < min_coverage:
                continue
            p = item["properties"]
            candidates.append({"id": item["id"], "acquired": p["acquired"],
                                "cloud_cover": p["cloud_cover"], "coverage": coverage,
                                "instrument": p.get("instrument")})
        url = page.get("_links", {}).get("_next")
        if not url:
            break
        page = api_json(http, "GET", url)
    if not candidates:
        raise NoEligibleSceneError(
            f"No eligible scene for {start} through {end} at cloud <= {max_cloud:.0%} "
            f"and AOI coverage >= {min_coverage:.1%}"
        )
    candidates.sort(key=lambda x: (x["cloud_cover"], -x["coverage"], x["acquired"], x["id"]))
    return candidates[0]


def find_scene_with_fallback(http, aoi, windows, search=find_best_scene):
    attempts = [
        (window, max_cloud, min_coverage)
        for window in windows
        for max_cloud, min_coverage in SEARCH_FILTERS
    ]
    last_error = None
    for index, ((start, end), max_cloud, min_coverage) in enumerate(attempts):
        try:
            scene = search(http, aoi, start, end, max_cloud, min_coverage)
        except NoEligibleSceneError as exc:
            last_error = exc
            continue
        if index:
            print(
                f"Selected fallback scene for {start} through {end} "
                f"(cloud <= {max_cloud:.0%}, AOI coverage >= {min_coverage:.1%})",
                flush=True,
            )
        return scene
    raise last_error


def build_order_request(name, aoi, scene_a, scene_b, utm_epsg):
    return {
        "name": name, "source_type": "scenes", "order_type": "full",
        "products": [{"item_ids": [scene_a["id"], scene_b["id"]], "item_type": "PSScene",
                      "product_bundle": "analytic_sr_udm2"}],
        "tools": [
            {"coregister": {"anchor_item": scene_a["id"]}},
            {"clip": {"aoi": aoi}},
            {"reproject": {"projection": f"EPSG:{utm_epsg}", "resolution": 3, "kernel": "bilinear"}},
            {"file_format": {"format": "COG"}},
        ],
    }


def ensure_order(http, state, state_path):
    if state.get("order_id"):
        return state["order_id"]
    # Recover by unique name even after a lost POST response. A missing match
    # after an uncertain submission never triggers a second automatic POST.
    url = ORDERS
    matches = []
    while url:
        page = api_json(http, "GET", url, params={"name": state["request"]["name"]})
        matches.extend(o for o in page.get("orders", []) if o["name"] == state["request"]["name"])
        url = page.get("_links", {}).get("_next")
    if len(matches) > 1:
        raise RuntimeError("Multiple matching orders; inspect them before proceeding")
    if matches:
        state["order_id"] = matches[0]["id"]
    else:
        if state.get("submission_started"):
            raise RuntimeError("Previous submission outcome is uncertain. Check Planet Orders by the name in state.json; do not delete state or resubmit blindly.")
        state["submission_started"] = True
        save_json(state_path, state)
        state["order_id"] = api_json(http, "POST", ORDERS, json=state["request"])["id"]
    save_json(state_path, state)
    return state["order_id"]


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Unsafe delivery path")
    return path


def file_ok(path, info):
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == info["digests"]["sha256"]


def download_file(http, url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with http.get(url, stream=True, timeout=(30, 120)) as r:
            if not r.ok:
                raise RuntimeError(f"Download HTTP {r.status_code}; rerun to refresh delivery links")
            with tmp.open("wb") as stream:
                for chunk in r.iter_content(1024 * 1024):
                    stream.write(chunk)
        tmp.replace(path)
    except requests.RequestException:
        raise RuntimeError("Download interrupted; rerun to repair the same order") from None


def finalize_order(order, root, state, out_dir, folder, file_prefix, utm_epsg):
    """Download the order's delivery, verify the two SR rasters, and move them to out_dir."""
    # Signed delivery URLs must not receive API credentials or enter state/logs.
    results = {r["name"]: r["location"] for r in order.get("_links", {}).get("results", [])}
    manifests = [n for n in results if Path(n).name == "manifest.json"]
    if len(manifests) != 1:
        raise RuntimeError("Expected one order manifest; order delivery may be incomplete")
    with open_session() as delivery:
        manifest_path = safe_path(root, manifests[0])
        download_file(delivery, results[manifests[0]], manifest_path)
        entries = json.loads(manifest_path.read_text())["files"]
        for info in entries:
            dest = safe_path(manifest_path.parent, info["path"])
            if not file_ok(dest, info):
                name = dest.relative_to(root.resolve()).as_posix()
                if name not in results:
                    raise RuntimeError(f"Missing delivery link for {name}")
                print("Downloading", name, flush=True)
                download_file(delivery, results[name], dest)
                if not file_ok(dest, info):
                    raise RuntimeError(f"Checksum mismatch for {name}; rerun to repair")

    rasters = {}
    for window in ("a", "b"):
        scene_id = state[window]["id"]
        matches = [safe_path(manifest_path.parent, f["path"]) for f in entries
                   if f.get("annotations", {}).get("planet/asset_type") == "ortho_analytic_4b_sr"
                   and f.get("annotations", {}).get("planet/item_id") == scene_id]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one 4-band SR raster for {scene_id}")
        with rasterio.open(matches[0]) as src:
            if src.count != 4 or src.crs is None or src.crs.to_epsg() != utm_epsg or any(abs(r - 3) > 1e-6 for r in src.res):
                raise RuntimeError(f"Unexpected raster layout: {matches[0]}")
            for _, block in src.block_windows(1):
                src.read(window=block)  # Catch truncated/corrupt TIFFs.
        rasters[window] = (matches[0], scene_id)

    # Only move final rasters out before discarding the rest of the order artifacts.
    final_dir = out_dir / folder
    final_dir.mkdir(exist_ok=True)
    finals = {}
    for window, (path, scene_id) in rasters.items():
        dest = final_dir / f"{file_prefix}-{scene_id}.tif"
        path.replace(dest)
        finals[window] = dest
    shutil.rmtree(root)
    return finals


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON file with prefix, width_km, coordinates")
    parser.add_argument("--output", type=Path, required=True, help="Output directory; reuse it to resume")
    parser.add_argument("--area", type=int, action="append", dest="areas",
                         help="Restrict to one area (1-based position in --input); repeatable. Default: all areas")
    parser.add_argument("--start-index", type=int, default=1,
                         help="Label/folder number for the first coordinate in --input; use to continue numbering across batches")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    areas = load_areas(args.input)
    if args.areas and any(not 1 <= i <= len(areas) for i in args.areas):
        raise ValueError(f"--area must be between 1 and {len(areas)}")
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    if args.start_index != 1 and "locations" in json.loads(args.input.read_text()):
        raise ValueError("--start-index only applies to legacy coordinate JSON files")
    selected_areas = sorted(args.areas) if args.areas else range(1, len(areas) + 1)

    if not 0 <= MAX_CLOUD <= 1 or not 0 < MIN_COVERAGE <= 1:
        raise ValueError("Cloud and coverage thresholds must be fractions between 0 and 1")
    if POLL_SECONDS <= 0 or WAIT_HOURS <= 0:
        raise ValueError("Polling interval and wait time must be positive")
    window_a_start, window_a_end = (date.fromisoformat(d) for d in WINDOW_A)
    window_b_start, window_b_end = (date.fromisoformat(d) for d in WINDOW_B)
    if not window_a_start <= window_a_end < window_b_start <= window_b_end:
        raise ValueError("Require non-overlapping windows: A start <= A end < B start <= B end")
    if max(date.fromisoformat(end) for _, end in SEARCH_WINDOWS_A) >= min(
        date.fromisoformat(start) for start, _ in SEARCH_WINDOWS_B
    ):
        raise ValueError("Fallback acquisition windows must keep A earlier than B")

    for i in selected_areas:
        area = apply_start_index(areas[i - 1], args.start_index)
        print(f"{area['label']}: {area['aoi']['coordinates'][0]}, EPSG:{area['utm_epsg']}", flush=True)
    key = os.environ.get("PL_API_KEY")
    if not key:
        raise ValueError("Set PL_API_KEY in your environment")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    # Lock released automatically on process death; state survives wall-time limits.
    with (out_dir / ".lock").open("a") as lock, open_session(key) as http:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another downloader is using this output directory") from None
        for i in selected_areas:
            area = apply_start_index(areas[i - 1], args.start_index)
            label = area["label"]
            folder = area["folder"]
            work_folder = area["work_folder"]
            file_prefix = area["file_prefix"]
            aoi = area["aoi"]
            utm_epsg = area["utm_epsg"]
            finished = sorted((out_dir / folder).glob(f"{file_prefix}-*.tif"))
            if len(finished) == 2:
                print(label, "already finalized:", *(p.name for p in finished), flush=True)
                continue
            root = out_dir / work_folder
            root.mkdir(parents=True, exist_ok=True)
            state_path = root / "state.json"
            config = {"aoi": aoi, "windows": [window_a_start.isoformat(), window_a_end.isoformat(),
                                               window_b_start.isoformat(), window_b_end.isoformat()],
                      "utm_epsg": utm_epsg,
                      "max_cloud": MAX_CLOUD, "min_coverage": MIN_COVERAGE}
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state["config"] != config:
                    raise RuntimeError(f"{label}: settings differ from saved plan; use a different output directory for intentional new orders")
            else:
                try:
                    scene_a = find_scene_with_fallback(http, aoi, SEARCH_WINDOWS_A)
                    scene_b = find_scene_with_fallback(http, aoi, SEARCH_WINDOWS_B)
                except NoEligibleSceneError as exc:
                    failures.append({"location": label, "error": str(exc)})
                    print(f"{label}: no usable scene pair; continuing: {exc}", flush=True)
                    continue
                signature = hashlib.sha256(
                    json.dumps([config, scene_a["id"], scene_b["id"]], sort_keys=True).encode()
                ).hexdigest()[:16]
                state = {"config": config, "a": scene_a, "b": scene_b,
                         "request": build_order_request(f"planet-{label}-{signature}", aoi, scene_a, scene_b, utm_epsg)}
                save_json(state_path, state)
            print(label, "A:", state["a"], "B:", state["b"], flush=True)
            order_id = ensure_order(http, state, state_path)
            print(label, "order:", order_id, flush=True)
            deadline = time.monotonic() + WAIT_HOURS * 3600
            while True:
                order = api_json(http, "GET", ORDERS + "/" + order_id)
                state["status"] = order["state"]
                save_json(state_path, state)
                print(label, state["status"], flush=True)
                if state["status"] in {"success", "partial", "failed", "cancelled"}:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Wait limit reached; rerun the same command to resume")
                time.sleep(POLL_SECONDS)
            if state["status"] not in {"success", "partial"}:
                raise RuntimeError(f"Order {order_id} ended as {state['status']}; inspect it in Planet Orders. No replacement submitted.")
            finals = finalize_order(order, root, state, out_dir, folder, file_prefix, utm_epsg)
            print(label, "final files:", finals["a"], finals["b"], flush=True)
    save_json(out_dir / "failed_areas.json", failures)
    if failures:
        print(f"No eligible scene pair for {len(failures)} location(s):", flush=True)
        for failure in failures:
            print(f"  {failure['location']}: {failure['error']}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
