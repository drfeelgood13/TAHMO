"""Matching-free and shape metrics for field delineation, used by evaluate.py --extended.

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

All inputs: GeoDataFrames in the same metric CRS (metres). `region` limits where predictions are judged
(the image extent, or the surveyed part of it when the labels are incomplete).
"""
import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

BOUNDARY_PX = 0.5          # m, raster size for the boundary metrics
TOLERANCES = (2.5, 5.0)    # m
POLIS_STEP = 1.0           # m, outline resampling for PoLiS


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
