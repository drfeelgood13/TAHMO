"""Give each final delineated field a confidence, from the raw detections saved with save_detections: true.

The pipeline merges overlapping detections (neighbouring tiles overlap by half) and joins fields across tile
seams, so a final polygon can come from several detections. Per final polygon:
    confidence      confidence of the raw detection that overlaps it best (highest IoU)
    match_iou       that IoU (close to 1: the polygon is essentially that one detection)
    conf_max/mean   max / mean confidence of all detections lying mostly (>= 50 %) inside the polygon
    n_detections    how many detections that is

Usage: python finetune/polygon_confidence.py finetune/results/combined_unlabelled
Writes <name>.confidence.gpkg (and <name>.simp.confidence.gpkg) + confidence_summary.csv / confidence_polygons.csv
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

INSIDE_THR = 0.5


def assign(final, det):
    rows = []
    gi, di = final.sindex.query(det.geometry, predicate="intersects")[::-1]
    pairs = pd.DataFrame({"f": gi, "d": di})
    inter = shapely.intersection(final.geometry.values[pairs.f], det.geometry.values[pairs.d])
    pairs["inter"] = shapely.area(inter)
    pairs["iou"] = pairs.inter / (final.area.values[pairs.f] + det.area.values[pairs.d] - pairs.inter)
    pairs["inside"] = pairs.inter / det.area.values[pairs.d]
    pairs["conf"] = det.confidence.values[pairs.d]
    for f in range(len(final)):
        p = pairs[pairs.f == f]
        if len(p) == 0:
            rows.append((np.nan, 0.0, np.nan, np.nan, 0)); continue
        best = p.loc[p.iou.idxmax()]
        inside = p[p.inside >= INSIDE_THR]
        rows.append((best.conf, best.iou, inside.conf.max() if len(inside) else np.nan,
                     inside.conf.mean() if len(inside) else np.nan, len(inside)))
    return pd.DataFrame(rows, columns=["confidence", "match_iou", "conf_max", "conf_mean", "n_detections"], index=final.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    args = ap.parse_args()
    summary, polys = [], []
    for det_path in sorted(glob.glob(os.path.join(args.results, "*.detections.gpkg"))):
        name = os.path.basename(det_path)[:-len(".detections.gpkg")]
        det = gpd.read_file(det_path)
        final = gpd.read_file(os.path.join(args.results, name + ".gpkg")).to_crs(det.crs).reset_index(drop=True)
        conf = assign(final, det)
        out = pd.concat([final, conf], axis=1)
        out.to_file(os.path.join(args.results, name + ".confidence.gpkg"), driver="GPKG")
        simp_path = os.path.join(args.results, name + ".simp.gpkg")
        if os.path.exists(simp_path):  # same field ids as the raw output
            simp = gpd.read_file(simp_path)
            simp.merge(out[["id"] + list(conf.columns)], on="id", how="left") \
                .to_file(os.path.join(args.results, name + ".simp.confidence.gpkg"), driver="GPKG")
        c = out.confidence
        summary.append({"image": name, "polygons": len(out), "raw_detections": len(det),
                        "conf_mean": c.mean(), "conf_median": c.median(), "conf_min": c.min(), "conf_max": c.max(),
                        "conf_<0.3": int((c < 0.3).sum()), "conf_0.3-0.5": int(c.between(0.3, 0.5, "left").sum()),
                        "conf_0.5-0.7": int(c.between(0.5, 0.7, "left").sum()), "conf_>=0.7": int((c >= 0.7).sum()),
                        "match_iou_median": out.match_iou.median(), "no_detection": int(c.isna().sum())})
        polys.append(out.drop(columns="geometry").assign(image=name))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(args.results, "confidence_summary.csv"), index=False)
    pd.concat(polys).to_csv(os.path.join(args.results, "confidence_polygons.csv"), index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
