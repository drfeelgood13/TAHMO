"""Per-field, per-polygon and per-image metrics for the model comparison page.

Computes the same tables that index.html embeds for the
Delineate Anything models (polydata-* and imgdata-*), for any model whose
predictions follow the layout read by evaluation.py:

    <results>/<location_label>/polygons.parquet

Matching, clipping and the 20 m2 label filter are shared with evaluation.py, so
precision / recall / F1 equal its scores_test.csv.

Definitions (from the legends on the page):
  Found         greedy one-to-one match at IoU >= 0.5
  Missed        < 20% of the field's area under any prediction
  Split         two or more predictions cover the field
  Merged        the prediction with the largest overlap also covers another field
  Poor outline  covered, but no prediction fits it
  "Covers" uses evaluation.py's rule: intersection / smaller area >= 0.5.
  Outcome priority: Found, Missed, Split, Merged, Poor outline.
  Area omission = 1 - area-weighted coverage of the reference fields.
  Boundary F1 @ 5 m: boundaries sampled every 1 m, matched within 5 m, edges
  on the image extent ignored.
  PoLiS: symmetric mean vertex-to-boundary distance between all reference and
  all predicted boundaries of an image; undefined without predictions.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry import box
from shapely.strtree import STRtree

from evaluation import COVER_THR, IOU_THR, MIN_FIELD_M2, ROOT, clipped_polygons, resolve_prediction_path

BOUNDARY_TOL_M = 5.0
SAMPLE_STEP_M = 1.0
MISSED_COVER = 0.2
CROP_GROUP = {"maize": "Maize", "rice": "Rice", "fallow": "Fallow", "unidentified": "Unidentified"}
CONF_LABEL = {"high": "high", "medium": "medium", "low": "low"}

FIELD_COLS = ["img", "lid", "crop", "crop1", "conf", "area", "outcome", "iou", "cov", "col", "vari", "tex", "edge", "comp"]


def load_locations(test_set=ROOT / "test_set.json"):
    return json.loads(Path(test_set).read_text())["locations"]


def image_key(loc):
    return f"area{loc['area']:03d}_{loc['grid']}"


def location_geometry(loc, labels):
    """Extent, reference fields and non-crop polygons of one image, clipped and in metres."""
    crs = CRS.from_user_input(loc["utm_epsg"])
    bb = loc["bounding_box"]
    extent = gpd.GeoSeries([box(bb["west_lon"], bb["south_lat"], bb["east_lon"], bb["north_lat"])], crs="EPSG:4326").to_crs(crs).iloc[0]
    grid = labels.loc[labels.grid == loc["grid"]].to_crs(crs)
    fields = grid.loc[(grid.crop != "noncrop") & (grid.geometry.area >= MIN_FIELD_M2)]
    fields = clipped_polygons(fields, extent)
    noncrop = clipped_polygons(grid.loc[grid.crop == "noncrop"], extent)
    return crs, extent, fields, noncrop


def pair_table(gt, pred):
    """All intersecting (field, prediction) pairs with IoU and overlap-over-smaller-area."""
    rows = []
    if len(gt) and len(pred):
        for j, i in zip(*pred.sindex.query(gt.geometry, predicate="intersects")):
            g, p = gt.geometry.iloc[j], pred.geometry.iloc[i]
            inter = g.intersection(p).area
            if inter > 0:
                rows.append((j, i, inter, inter / (g.area + p.area - inter), inter / min(g.area, p.area)))
    return pd.DataFrame(rows, columns=["g", "p", "inter", "iou", "over"])


def greedy_match(pairs):
    matched_g, matched_p = {}, {}
    for r in pairs.sort_values("iou", ascending=False, kind="stable").itertuples():
        if r.iou < IOU_THR:
            break
        if r.g not in matched_g and r.p not in matched_p:
            matched_g[r.g] = r.p
            matched_p[r.p] = r.g
    return matched_g, matched_p


def field_outcomes(gt, pred, pairs, matched_g):
    cover_pairs = pairs[pairs.over >= COVER_THR]
    n_cover_g = cover_pairs.groupby("g").size()
    n_cover_p = cover_pairs.groupby("p").size()
    main = pairs.sort_values("inter", ascending=False, kind="stable").drop_duplicates("g").set_index("g").p
    covered_by_main = {(r.g, r.p) for r in cover_pairs.itertuples()}
    union = pred.geometry.union_all() if len(pred) else None
    out = []
    for j in range(len(gt)):
        g = gt.geometry.iloc[j]
        cov = g.intersection(union).area / g.area if union is not None else 0.0
        mine = pairs[pairs.g == j] if len(pairs) else pairs
        iou = float(mine.iou.max()) if len(mine) else 0.0
        split = n_cover_g.get(j, 0) >= 2
        merged = j in main.index and (j, main[j]) in covered_by_main and n_cover_p.get(main[j], 0) >= 2
        if j in matched_g:
            outcome = "Found"
        elif cov < MISSED_COVER:
            outcome = "Missed"
        elif split:
            outcome = "Split"
        elif merged:
            outcome = "Merged"
        else:
            outcome = "Poor outline"
        out.append(dict(outcome=outcome, iou=round(iou, 2), covered=round(cov, 2), split=bool(split), merged=bool(merged)))
    return pd.DataFrame(out)


def pred_categories(gt, pred, noncrop, pairs, matched_p):
    n_fields = pairs[pairs.over >= COVER_THR].groupby("p").size() if len(pairs) else pd.Series(dtype=int)
    fields_union = gt.geometry.union_all() if len(gt) else None
    nc_union = noncrop.geometry.union_all() if len(noncrop) else None
    out = []
    for i in range(len(pred)):
        p = pred.geometry.iloc[i]
        in_f = p.intersection(fields_union).area / p.area if fields_union is not None else 0.0
        in_nc = p.intersection(nc_union).area / p.area if nc_union is not None else 0.0
        k = int(n_fields.get(i, 0))
        if i in matched_p:
            cat = "Matched field"
        elif in_nc >= 0.5:
            cat = "On non-crop background"
        elif k >= 2:
            cat = "Covers several fields"
        elif in_f >= 0.5:
            cat = "Piece of a field"
        else:
            cat = "Mixed / boundary"
        out.append(dict(cat=cat, in_f=round(in_f, 2), in_nc=round(in_nc, 2), k=k,
                        comp=round(4 * np.pi * p.area / p.length ** 2, 2)))
    return pd.DataFrame(out)


def boundary_lines(frame, extent):
    """Outlines of the polygons, without the parts that only lie on the image extent."""
    lines = shapely.union_all(shapely.boundary(frame.geometry.values))
    return lines.difference(extent.boundary.buffer(0.5))


def segments(lines):
    parts = shapely.get_parts(lines)
    segs = []
    for part in parts:
        c = np.asarray(part.coords)
        segs.extend(shapely.linestrings(np.stack([c[:-1], c[1:]], axis=1)))
    return np.array(segs, dtype=object)


def sample_points(lines):
    pts = []
    for part in shapely.get_parts(lines):
        n = max(2, int(np.ceil(part.length / SAMPLE_STEP_M)) + 1)
        pts.append(shapely.points([part.interpolate(d).coords[0] for d in np.linspace(0, part.length, n)]))
    return np.concatenate(pts) if pts else np.array([], dtype=object)


def vertices(lines):
    return shapely.points(np.vstack([np.asarray(p.coords) for p in shapely.get_parts(lines)]))


def nearest_distance(points, segs):
    tree = STRtree(segs)
    _, dist = tree.query_nearest(points, return_distance=True, all_matches=False)
    return dist


def boundary_metrics(gt, pred, extent):
    """(boundary F1 at 5 m, PoLiS in m); either may be None when undefined."""
    if not len(pred):
        return 0.0, None
    g_lines, p_lines = boundary_lines(gt, extent), boundary_lines(pred, extent)
    if g_lines.is_empty or p_lines.is_empty:
        return 0.0, None
    g_seg, p_seg = segments(g_lines), segments(p_lines)
    pts_g, pts_p = sample_points(g_lines), sample_points(p_lines)
    recall = float((nearest_distance(pts_g, p_seg) <= BOUNDARY_TOL_M).mean())
    precision = float((nearest_distance(pts_p, g_seg) <= BOUNDARY_TOL_M).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    polis = 0.5 * nearest_distance(vertices(g_lines), p_seg).mean() + 0.5 * nearest_distance(vertices(p_lines), g_seg).mean()
    return round(f1, 4), round(float(polis), 4)


def compute_model(results_dir, base, labels, locations, id_column):
    """Return (imgdata, polydata, flags) in the shapes the page embeds.

    flags holds one (split, merged) pair per reference field, in polydata order.

    base is the baseline model's embedded data: it supplies the columns that do
    not depend on the model (image colour statistics, crop group, label confidence).
    """
    base_fields = {(r[0], r[1]): r for r in base["poly"]["fields"]}
    base_img = {r[0]: r for r in base["img"]["rows"]}
    cols = [c[0] for c in base["img"]["cols"]]
    ix = {c: i for i, c in enumerate(cols)}
    fields_out, preds_out, img_rows, flags = [], [], [], []
    for loc in locations:
        key = image_key(loc)
        crs, extent, gt, noncrop = location_geometry(loc, labels)
        pred_path, _ = resolve_prediction_path(Path(results_dir), loc["location_label"])
        pred = clipped_polygons(gpd.read_parquet(pred_path).to_crs(crs), extent)
        pairs = pair_table(gt, pred)
        matched_g, matched_p = greedy_match(pairs)
        fo = field_outcomes(gt, pred, pairs, matched_g)
        po = pred_categories(gt, pred, noncrop, pairs, matched_p)

        areas = []
        for j in range(len(gt)):
            ref = base_fields[(key, gt["id"].iloc[j])]
            fields_out.append([key, ref[1], ref[2], ref[3], ref[4], ref[5], fo.outcome[j], fo.iou[j], fo.covered[j], *ref[9:]])
            areas.append(ref[5])
            flags.append((bool(fo.split[j]), bool(fo.merged[j])))
        for i in range(len(pred)):
            pid = int(pred[id_column].iloc[i]) if id_column in pred else i + 1
            preds_out.append([key, pid, po.cat[i], round(pred.geometry.iloc[i].area), po.in_f[i], po.in_nc[i], int(po.k[i]),
                              None, None, None, None, po.comp[i]])

        n_gt, n_pred, tp = len(gt), len(pred), len(matched_g)
        omission = 1 - float((fo["covered"] * np.array(areas)).sum() / sum(areas)) if len(gt) else None
        bf1, polis = boundary_metrics(gt, pred, extent)
        row = list(base_img[key[:7]])
        row[ix["Predicted"]] = float(n_pred)
        row[ix["Precision"]] = round(tp / max(n_pred, 1), 4)
        row[ix["Recall"]] = round(tp / max(n_gt, 1), 4)
        row[ix["F1"]] = round(2 * tp / max(n_gt + n_pred, 1), 4)
        row[ix["Area omission"]] = round(omission, 4)
        row[ix["Boundary F1 @ 5 m"]] = bf1
        row[ix["PoLiS (m)"]] = polis
        img_rows.append(row)
    img_rows.sort(key=lambda r: (-r[ix["F1"]], r[0]))
    return {"cols": base["img"]["cols"], "rows": img_rows}, {"fields": fields_out, "preds": preds_out}, flags
