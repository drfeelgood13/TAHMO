"""Sanity checks for metrics.py on synthetic fields with known answers.  Run: python finetune/test_metrics.py"""
import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import box

import metrics

EXTENT = box(0, 0, 200, 200)
GT = gpd.GeoDataFrame(geometry=[box(20, 20, 80, 80), box(80, 20, 140, 80), box(20, 100, 100, 180)], crs=32648)


def run(pred_geoms):
    pred = gpd.GeoDataFrame(geometry=pred_geoms, crs=32648)
    t = {"n_gt": len(GT), **metrics.area_errors(pred, GT, EXTENT),
         **metrics.boundary_counts(pred, GT, EXTENT, EXTENT)}
    pp = metrics.polis_pairs(pred, GT)
    t.update(polis_sum=pp["polis_sum"], polis_n=pp["polis_n"])
    return metrics.summarize(t)


def close(a, b, tol=0.02):
    return abs(a - b) <= tol


# 1) perfect prediction
m = run(list(GT.geometry))
assert close(m["area_omission"], 0) and close(m["area_commission"], 0), m
assert close(m["boundary_F1_2.5m"], 1) and close(m["boundary_F1_5m"], 1), m
assert close(m["polis_mean_m"], 0) and m["polis_paired_share"] == 1, m

# 2) everything shifted 3 m east: PoLiS ~ 3 m or less (only the edges perpendicular to the shift move 3 m),
#    boundaries within 5 m but not within 2.5 m
m = run([translate(g, 3, 0) for g in GT.geometry])
assert 1.0 < m["polis_mean_m"] <= 3.0, m
assert m["boundary_F1_5m"] > 0.95 and m["boundary_F1_2.5m"] < m["boundary_F1_5m"], m

# 3) the big field drawn as two halves: no area lost, boundaries mostly right (one extra line)
m = run([GT.geometry[0], GT.geometry[1], box(20, 100, 60, 180), box(60, 100, 100, 180)])
assert close(m["area_omission"], 0) and close(m["area_commission"], 0), m
assert m["boundary_R_2.5m"] > 0.99 and m["boundary_P_2.5m"] < 1, m

# 4) one field missed completely: 4800 of 3600+3600+6400 m2 omitted, no commission, 2 of 3 fields paired
m = run([GT.geometry[0], GT.geometry[2]])
assert close(m["area_omission"], 3600 / 13600), m
assert close(m["area_commission"], 0) and close(m["polis_paired_share"], 2 / 3), m

# 5) a false field outside all labels: commission = its share of the predicted area
m = run(list(GT.geometry) + [box(150, 150, 190, 190)])
assert close(m["area_commission"], 1600 / (13600 + 1600)), m

# 6) image edges are not boundaries: a field clipped at the border matches without a boundary there
gt_edge = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 50)], crs=32648)
pred_edge = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 50)], crs=32648)
c = metrics.boundary_counts(pred_edge, gt_edge, EXTENT, EXTENT)
assert c["b_gt_px"] < 2 * 50 / metrics.BOUNDARY_PX * 1.2, c  # only the 2 inner sides, not the image edges

print("all metric checks passed")
