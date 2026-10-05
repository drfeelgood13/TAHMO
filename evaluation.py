"""Evaluate FTW polygons against Ghana field labels.

Usage: python evaluation.py ftw/output/test_set --test-set test_set.json
Matching uses greedy one-to-one IoU >= 0.5. Missing or unreadable outputs
are reported separately and excluded from metrics, never treated as empty predictions.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent
IOU_THR = 0.5
COVER_THR = 0.5
MIN_FIELD_M2 = 20


def score_image(pred, gt):
    pairs = []
    for j, i in zip(*pred.sindex.query(gt.geometry, predicate="intersects")):
        g, p = gt.geometry.iloc[j], pred.geometry.iloc[i]
        inter = g.intersection(p).area
        pairs.append((j, i, inter / (g.area + p.area - inter), inter / min(g.area, p.area)))

    matched_g, matched_p, ious = set(), set(), []
    for j, i, iou, _ in sorted(pairs, key=lambda x: -x[2]):
        if iou < IOU_THR:
            break
        if j not in matched_g and i not in matched_p:
            matched_g.add(j); matched_p.add(i); ious.append(iou)

    covered = [(j, i) for j, i, _, c in pairs if c >= COVER_THR]
    per_gt = pd.Series([j for j, _ in covered], dtype="int64").value_counts()
    per_pred = pd.Series([i for _, i in covered], dtype="int64").value_counts()
    return {
        "n_gt": len(gt), "n_pred": len(pred), "tp": len(ious), "sum_iou": sum(ious),
        "over_seg": int((per_gt >= 2).sum()), "under_seg": int((per_pred >= 2).sum()),
    }


def clipped_polygons(frame, extent):
    valid = frame.geometry.notna() & frame.geometry.is_valid & ~frame.geometry.is_empty
    frame = frame.loc[valid].copy()
    frame.geometry = frame.geometry.intersection(extent)
    frame = frame.loc[frame.geometry.is_valid & ~frame.geometry.is_empty].copy()
    return frame.loc[frame.geometry.area > 0].reset_index(drop=True)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def resolve_prediction_path(results_dir, location_label):
    candidates = [location_label]
    if location_label.endswith("_0p3m"):
        candidates.append(location_label.removesuffix("_0p3m"))
    else:
        candidates.append(f"{location_label}_0p3m")
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        pred_path = results_dir / candidate / "polygons.parquet"
        if pred_path.exists():
            return pred_path, candidate
    return results_dir / location_label / "polygons.parquet", location_label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--test-set", type=Path, default=ROOT / "test_set.json")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/ground_truth/filtered_grids.gpkg")
    args = ap.parse_args()
    try:
        manifest = json.loads(args.test_set.read_text())
        locations = manifest["locations"]
        if not isinstance(locations, list) or not locations:
            raise ValueError("Test set must contain a non-empty locations list")
        if manifest.get("coordinate_crs", "EPSG:4326") != "EPSG:4326":
            raise ValueError("Test bounding boxes must use EPSG:4326")
        names = [loc["location_label"] for loc in locations]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate location labels")
        if any(not isinstance(name, str) or not name.strip() or name in (".", "..") or "/" in name or "\\" in name for name in names):
            raise ValueError("Location labels must be directory names")
        labels = gpd.read_file(args.labels)
        if labels.crs is None or not {"crop", "grid"}.issubset(labels.columns):
            raise ValueError("Labels require a CRS and crop/grid columns")
        labels = labels.loc[(labels.crop != "noncrop") & labels.geometry.notna() & labels.geometry.is_valid].copy()
        if not args.results.is_dir():
            raise ValueError(f"Results directory does not exist: {args.results}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.error(str(exc))

    rows = []
    for location in locations:
        name = location["location_label"]
        row = {"folder": name, "grid": location["grid"], "region": location.get("region", ""),
               "status": "missing", "error": ""}
        pred_path, resolved_name = resolve_prediction_path(args.results, name)
        if not pred_path.exists():
            print(f"missing: {name}")
            rows.append(row)
            continue
        try:
            row["folder"] = resolved_name
            crs = CRS.from_user_input(location["utm_epsg"])
            if not crs.is_projected or any(axis.unit_name not in ("metre", "meter") for axis in crs.axis_info):
                raise ValueError("Evaluation CRS must be projected in metres")
            bbox = location["bounding_box"]
            west, south, east, north = [bbox[key] for key in ("west_lon", "south_lat", "east_lon", "north_lat")]
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError("Invalid WGS84 bounding box")
            extent = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326").to_crs(crs).iloc[0]
            gt = labels.loc[labels.grid == location["grid"]].to_crs(crs)
            gt = gt.loc[gt.geometry.area >= MIN_FIELD_M2]
            gt = clipped_polygons(gt, extent)
            pred = gpd.read_parquet(pred_path)
            if pred.crs is None:
                raise ValueError("Predictions have no CRS")
            pred = clipped_polygons(pred.to_crs(crs), extent)
            row.update(status="ok", **score_image(pred, gt))
        except Exception as exc:
            row.update(status="error", error=str(exc))
            print(f"error: {name}: {exc}")
        rows.append(row)

    df = pd.DataFrame(rows)
    for column in ("n_gt", "n_pred", "tp", "sum_iou", "over_seg", "under_seg"):
        if column not in df:
            df[column] = float("nan")
    df["precision"] = df.tp / df.n_pred.clip(lower=1)
    df["recall"] = df.tp / df.n_gt.clip(lower=1)
    df["f1"] = 2 * df.tp / (df.n_gt + df.n_pred).clip(lower=1)
    output = args.results / "scores_test.csv"
    df.to_csv(output, index=False)
    scored = df.loc[df.status == "ok"]
    totals = scored[["n_gt", "n_pred", "tp", "sum_iou", "over_seg", "under_seg"]].sum()
    summary = {
        "expected_locations": len(df), "evaluated_locations": len(scored),
        "missing_locations": df.loc[df.status == "missing", "folder"].tolist(),
        "error_locations": df.loc[df.status == "error", "folder"].tolist(),
        "complete": bool((df.status == "ok").all()),
        "metric_scope": "successfully evaluated locations only",
        "iou_threshold": IOU_THR, "coverage_threshold": COVER_THR,
        "minimum_label_area_m2": MIN_FIELD_M2,
        **{key: float(value) for key, value in totals.items()},
        "precision": ratio(totals.tp, totals.n_pred),
        "recall": ratio(totals.tp, totals.n_gt),
        "f1": ratio(2 * totals.tp, totals.n_gt + totals.n_pred),
        "mean_iou_of_matches": ratio(totals.sum_iou, totals.tp),
        "over_seg_fraction": ratio(totals.over_seg, totals.n_gt),
        "under_seg_fraction": ratio(totals.under_seg, totals.n_pred),
    }
    (args.results / "scores_test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Evaluated {len(scored)}/{len(df)} locations; {len(summary['missing_locations'])} missing; {len(summary['error_locations'])} errors")
    print(f"Metrics for evaluated locations: precision {summary['precision']:.3f}, recall {summary['recall']:.3f}, F1 {summary['f1']:.3f}")
    print(f"Saved {output} and scores_test_summary.json")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
