"""Score field-delineation outputs of any model (Delineate Anything, FTW, FTP) against the field labels.

The areas to score come from an area list in downloaders/inputs/ (the same file the model was run on); each area's
bounding box, reprojected to its UTM zone, is the extent in which predictions and labels are compared, so every model
is judged on exactly the same ground. Predictions are found per area label in the results folder, in any of the layouts
the models write:

    <results>/<label>.gpkg                    Delineate Anything (<label>.simp.gpkg with --simp)
    <results>/<label>/polygons.parquet        FTW, FTP
    the same with a resolution suffix         <label>_1m.gpkg, <label>_0p3m/polygons.parquet, ...

Ghana areas are scored against the grid's labels in filtered_grids.gpkg. VIEDIA areas (with `subareas`) are scored per
512 m tile against <labels>/<area>/<area>_r<row>_c<col>.gpkg; a tile without its own prediction is scored on the area's
prediction clipped to the tile.

Instance matching per area: a prediction and a label match if IoU >= IOU_THR (one-to-one, greedy by IoU).
Also counts over-segmentation (label covered by >= 2 predictions) and under-segmentation
(prediction covering >= 2 labels), where "covered" = at least COVER_THR of the smaller polygon.

Usage: python evaluate.py <results> [--areas downloaders/inputs/ghana_test.json] [--labels ...] [--simp] [--extended]

--extended adds matching-free and shape metrics:
- Area omission / commission: share of the reference field area that no prediction covers, and share of the
  predicted field area that lies outside every reference field. No 1-to-1 matching, so a field drawn in
  several pieces still counts as found.
- Boundary precision / recall / F1 with a distance tolerance: boundaries of both sets are rasterised at
  BOUNDARY_PX; a predicted boundary pixel is correct if a reference boundary pixel lies within the tolerance
  (and vice versa for recall). Distance-transform version (no 1-to-1 pixel assignment).
  Image edges are not boundaries: fields clipped at the image border would otherwise match there trivially.
- PoLiS (Avbelj et al. 2015, as used in Persello et al. 2023): for a pair of polygons A, B,
      PoLiS = 1/(2q) * sum_a d(a, boundary B) + 1/(2r) * sum_b d(b, boundary A)
  over the q vertices a of A and r vertices b of B. Outlines are resampled every POLIS_STEP m first, so the
  value does not depend on how many vertices an outline happens to have. Each reference field is paired with
  the prediction that overlaps it most (any overlap); reference fields without overlap get no pair.
"""
import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent
GHANA_LABELS = ROOT / "data/datasets-original/ground_truth_ghana/filtered_grids.gpkg"
VIEDIA_LABELS = ROOT / "data/datasets-original/VIEDIA 1m - ground_truth"

IOU_THR = 0.5
COVER_THR = 0.5
MIN_FIELD_M2 = 20  # same filter as delany/finetune/prepare_dataset.py, applied to labels and clipped predictions
COUNTS = ["n_gt", "n_pred", "tp", "sum_iou", "over_seg", "under_seg"]
BOUNDARY_PX = 0.5          # m, raster size for the boundary metrics
TOLERANCES = (2.5, 5.0)    # m
POLIS_STEP = 1.0           # m, outline resampling for PoLiS


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


def area_errors(pred, gt, region):
    """Areas (m2) needed for omission = 1 - inter/gt_area and commission = 1 - inter/pred_area."""
    p = shapely.intersection(shapely.union_all(pred.geometry.values), region)
    g = shapely.intersection(shapely.union_all(gt.geometry.values), region)
    return {"gt_area": g.area, "pred_area": p.area, "inter_area": shapely.intersection(p, g).area}


def _boundary_raster(geoms, extent, region, transform, shape):
    lines = shapely.union_all(shapely.boundary(geoms)) if len(geoms) else shapely.LineString()
    lines = shapely.difference(lines, shapely.buffer(shapely.boundary(extent), BOUNDARY_PX))  # drop image edges
    lines = shapely.intersection(lines, region)
    if lines.is_empty:
        return np.zeros(shape, bool)
    return rasterize([lines], out_shape=shape, transform=transform, all_touched=True, dtype="uint8").astype(bool)


def boundary_counts(pred, gt, extent, region, tolerances=TOLERANCES):
    """Pixel counts for boundary precision / recall at each tolerance (summed over images, then divided)."""
    x0, y0, x1, y1 = extent.bounds
    shape = (int(np.ceil((y1 - y0) / BOUNDARY_PX)), int(np.ceil((x1 - x0) / BOUNDARY_PX)))
    transform = from_origin(x0, y1, BOUNDARY_PX, BOUNDARY_PX)
    pb = _boundary_raster(pred.geometry.values, extent, region, transform, shape)
    gb = _boundary_raster(gt.geometry.values, extent, region, transform, shape)
    d_to_g = distance_transform_edt(~gb) * BOUNDARY_PX if gb.any() else np.full(shape, np.inf)
    d_to_p = distance_transform_edt(~pb) * BOUNDARY_PX if pb.any() else np.full(shape, np.inf)
    out = {"b_pred_px": int(pb.sum()), "b_gt_px": int(gb.sum())}
    for t in tolerances:
        out[f"b_pred_ok_{t:g}"] = int((pb & (d_to_g <= t)).sum())
        out[f"b_gt_ok_{t:g}"] = int((gb & (d_to_p <= t)).sum())
    return out


def polis(a, b, step=POLIS_STEP):
    """PoLiS distance (m) between two polygons."""
    pa = shapely.points(shapely.get_coordinates(shapely.segmentize(shapely.boundary(a), step)))
    pb = shapely.points(shapely.get_coordinates(shapely.segmentize(shapely.boundary(b), step)))
    return 0.5 * shapely.distance(pa, shapely.boundary(b)).mean() + 0.5 * shapely.distance(pb, shapely.boundary(a)).mean()


def polis_pairs(pred, gt):
    """PoLiS of each reference field and the prediction that overlaps it most."""
    best = {}
    for j, i in zip(*pred.sindex.query(gt.geometry, predicate="intersects")):
        inter = shapely.intersection(gt.geometry.iloc[j], pred.geometry.iloc[i]).area
        if inter > 0 and inter > best.get(j, (None, 0))[1]:
            best[j] = (i, inter)
    values = [polis(gt.geometry.iloc[j], pred.geometry.iloc[i]) for j, (i, _) in best.items()]
    return {"polis_sum": float(np.sum(values)), "polis_n": len(values), "polis_values": values}


def summarize(t, tolerances=TOLERANCES):
    """Totals (summed per-image counts) -> metric values."""
    out = {
        "area_omission": 1 - t["inter_area"] / max(t["gt_area"], 1e-9),
        "area_commission": 1 - t["inter_area"] / max(t["pred_area"], 1e-9),
        "polis_mean_m": t["polis_sum"] / max(t["polis_n"], 1),
        "polis_paired_share": t["polis_n"] / max(t["n_gt"], 1),
    }
    for tol in tolerances:
        p = t[f"b_pred_ok_{tol:g}"] / max(t["b_pred_px"], 1)
        r = t[f"b_gt_ok_{tol:g}"] / max(t["b_gt_px"], 1)
        out.update({f"boundary_P_{tol:g}m": p, f"boundary_R_{tol:g}m": r,
                    f"boundary_F1_{tol:g}m": 2 * p * r / (p + r) if p + r else 0.0})
    return out


def clip(frame, extent):
    """Repair invalid polygons, clip to the extent and drop what is (almost) nothing left."""
    geom = frame.geometry
    geom = geom.where(geom.is_valid, geom.buffer(0)).intersection(extent)
    frame = frame.set_geometry(geom)
    return frame[~frame.is_empty & (frame.area >= MIN_FIELD_M2)].reset_index(drop=True)


def find_prediction(results, label, simp):
    """Path of the prediction for one area label, or None. Raises if several resolutions match."""
    ext = ".simp.gpkg" if simp else ".gpkg"
    name = re.compile(re.escape(label) + r"(_\d+(p\d+)?m)?")
    found = [p for p in results.glob("*" + ext) if name.fullmatch(p.name[:-len(ext)])]
    if not simp:
        found = [p for p in found if not p.name.endswith(".simp.gpkg")]
        found += [d / "polygons.parquet" for d in results.iterdir()
                  if d.is_dir() and name.fullmatch(d.name) and (d / "polygons.parquet").exists()]
    if len(found) > 1:
        raise ValueError(f"several predictions match {label}: {', '.join(p.name for p in found)}")
    return found[0] if found else None


def read_prediction(path):
    if path.suffix == ".parquet":
        if pq.read_metadata(path).num_rows == 0:  # FTW/FTP write an empty file when no field was found
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def utm_box(bb, crs):
    """WGS84 bounding box of the area file -> polygon in the area's UTM zone."""
    lonlat = box(bb["west_lon"], bb["south_lat"], bb["east_lon"], bb["north_lat"])
    return gpd.GeoSeries([lonlat], crs="EPSG:4326").to_crs(crs).iloc[0]


def units(locations, labels_path):
    """One scoring unit per Ghana area or per VIEDIA tile: label, area label, CRS, extent and a label loader."""
    if any("subareas" in loc for loc in locations):
        for loc in locations:
            for t in loc["subareas"]:
                if not t.get("in_label_split", True):
                    continue
                b = t["bounds_utm"]
                path = labels_path / str(loc["area"]) / f"{loc['area']}_r{t['row']}_c{t['col']}.gpkg"
                yield dict(label=t["location_label"], area=loc["location_label"], crs=t["utm_epsg"], full=True,
                           extent=box(b["xmin"], b["ymin"], b["xmax"], b["ymax"]),
                           gt=lambda crs, path=path: gpd.read_file(path).to_crs(crs).explode(index_parts=False))
    else:
        labels = gpd.read_file(labels_path)
        labels = labels[labels.crop != "noncrop"]
        for loc in locations:
            yield dict(label=loc["location_label"], area=loc["location_label"], crs=loc["utm_epsg"], full=False,
                       extent=utm_box(loc["bounding_box"], loc["utm_epsg"]),
                       gt=lambda crs, grid=loc["grid"]: labels[labels.grid == grid].to_crs(crs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="output folder of one model run")
    ap.add_argument("--areas", type=Path, default=ROOT / "downloaders/inputs/ghana_test.json",
                    help="area list the model was run on (default: the 32 Ghana test areas)")
    ap.add_argument("--labels", type=Path,
                    help=f"Ghana: filtered_grids.gpkg (default {GHANA_LABELS.relative_to(ROOT)}); "
                         f"VIEDIA: the per-tile label folder (default {VIEDIA_LABELS.relative_to(ROOT)})")
    ap.add_argument("--simp", action="store_true", help="score the simplified <label>.simp.gpkg outputs (Delineate Anything)")
    ap.add_argument("--extended", action="store_true",
                    help="also compute area omission/commission, boundary F1 with tolerance and PoLiS")
    args = ap.parse_args()
    if not args.results.is_dir():
        ap.error(f"results folder does not exist: {args.results}")
    locations = json.loads(args.areas.read_text())["locations"]
    viedia = any("subareas" in loc for loc in locations)
    labels_path = args.labels or (VIEDIA_LABELS if viedia else GHANA_LABELS)

    rows, ext_rows, polis_values, area_preds = [], [], [], {}
    for u in units(locations, labels_path):
        label, crs, extent = u["label"], u["crs"], u["extent"]
        row = {"folder": label, "prediction": "", "status": "ok"}
        try:
            path = find_prediction(args.results, label, args.simp)
            if path is None and u["area"] != label:  # VIEDIA tile: fall back to the whole area's prediction
                if u["area"] not in area_preds:
                    area_preds[u["area"]] = find_prediction(args.results, u["area"], args.simp)
                path = area_preds[u["area"]]
            if path is None:
                rows.append({**row, "status": "missing"}); continue
            row["prediction"] = str(path.relative_to(args.results))
            gt = u["gt"](crs)
            gt = clip(gt[gt.geometry.notna()], extent)
            pred = clip(read_prediction(path).to_crs(crs), extent)
        except Exception as exc:
            print(f"error: {label}: {exc}")
            rows.append({**row, "status": "error", "error": str(exc)}); continue
        rows.append({**row, **score_image(pred, gt)})
        if args.extended:
            # VIEDIA is fully digitised -> judge the whole tile; Ghana labels cover only the surveyed fields ->
            # judge predictions only inside the outline (convex hull) around the labelled fields of the area
            region = extent if u["full"] else shapely.intersection(
                shapely.convex_hull(shapely.union_all(gt.geometry.values)), extent)
            pp = polis_pairs(pred, gt)
            polis_values += pp["polis_values"]
            ext_rows.append({"folder": label, "n_gt": len(gt), "region_m2": region.area,
                             **area_errors(pred, gt, region),
                             **boundary_counts(pred, gt, extent, region),
                             "polis_sum": pp["polis_sum"], "polis_n": pp["polis_n"]})

    df = pd.DataFrame(rows).reindex(columns=["folder", "prediction", "status", *COUNTS, "error"])
    missing, errors = df.folder[df.status == "missing"].tolist(), df.folder[df.status == "error"].tolist()
    if missing:
        print(f"no prediction for {len(missing)} of {len(df)}: {', '.join(missing)}")
    ok = df[df.status == "ok"].copy()
    if ok.empty:
        print("nothing to score"); return 1

    df["precision"] = df.tp / df.n_pred.clip(lower=1)
    df["recall"] = df.tp / df.n_gt.clip(lower=1)
    df["f1"] = (2 * df.precision * df.recall / (df.precision + df.recall)).fillna(0).where(df.status == "ok")
    stem = f"scores_{args.areas.stem}{'_simp' if args.simp else ''}"
    df.to_csv(args.results / f"{stem}.csv", index=False)

    t = ok[COUNTS].sum()
    p, r = t.tp / max(t.n_pred, 1), t.tp / max(t.n_gt, 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    unit = "tiles" if viedia else "areas"
    print(f"{unit}: {len(ok)}/{len(df)}  labels: {t.n_gt:.0f}  predictions: {t.n_pred:.0f}  matched (IoU>={IOU_THR}): {t.tp:.0f}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}  mean IoU of matches {t.sum_iou / max(t.tp, 1):.3f}")
    print(f"over-segmented labels: {t.over_seg:.0f} ({t.over_seg / max(t.n_gt, 1):.1%})  "
          f"under-segmenting predictions: {t.under_seg:.0f} ({t.under_seg / max(t.n_pred, 1):.1%})")
    print(f"per-{unit[:-1]} F1 median: {df.f1.median():.3f}  {unit} with no match: {(ok.tp == 0).sum()}")
    summary = {"results": str(args.results), "areas": str(args.areas), "scored": len(ok), "expected": len(df),
               "missing": missing, "errors": errors, "iou_threshold": IOU_THR, "cover_threshold": COVER_THR,
               "min_field_m2": MIN_FIELD_M2, **{k: float(v) for k, v in t.items()},
               "precision": p, "recall": r, "f1": f1, "mean_iou_of_matches": t.sum_iou / max(t.tp, 1)}

    if args.extended:
        ed = pd.DataFrame(ext_rows)
        per = ed.apply(lambda r: pd.Series(summarize(r)), axis=1)
        pd.concat([ed, per], axis=1).to_csv(args.results / f"{stem}_extended.csv", index=False)
        s = summarize(ed.drop(columns="folder").sum())
        summary.update(s, polis_median_m=float(np.median(polis_values)) if polis_values else None)
        print(f"[extended, judged in: {'tile extent' if viedia else 'surveyed area (hull of labels)'}]")
        print(f"area omission {s['area_omission']:.3f}  area commission {s['area_commission']:.3f}")
        print("  ".join(f"boundary@{tol:g}m P {s[f'boundary_P_{tol:g}m']:.3f} R {s[f'boundary_R_{tol:g}m']:.3f} "
                        f"F1 {s[f'boundary_F1_{tol:g}m']:.3f}" for tol in TOLERANCES))
        print(f"PoLiS mean {s['polis_mean_m']:.2f} m  median {np.median(polis_values) if polis_values else float('nan'):.2f} m  "
              f"(reference fields with an overlapping prediction: {s['polis_paired_share']:.1%})")

    (args.results / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {stem}.csv{', ' + stem + '_extended.csv' if args.extended else ''} and {stem}_summary.json in {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
