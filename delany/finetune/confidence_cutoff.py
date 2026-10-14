"""Does model confidence separate good predictions from false ones, and what would a higher cut-off do?

Needs, in <results>: <image>.confidence.gpkg (polygon_confidence.py) and analysis/predictions.csv (analyze_predictions.py).
1) confidence per kind of prediction (matched field, piece of a field, mostly on non-crop background, ...)
2) cut-off sweep: keep only final polygons with confidence >= t and score them as evaluate.py does.
   This filters the finished map; it approximates, but is not identical to, re-running with minimal_confidence = t
   (merging would then see fewer detections).

Usage: python finetune/confidence_cutoff.py finetune/results/combined_ghana_conf
Writes <results>/analysis/{confidence_by_kind.csv, cutoff_sweep.csv}
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from evaluate import IMAGES_GLOB, LABELS_PATH, MIN_FIELD_M2, ROOT, score_image

CUTOFFS = np.round(np.arange(0.15, 0.61, 0.05), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--data", default="yolo_dataset_1m")
    args = ap.parse_args()
    out = os.path.join(args.results, "analysis")

    P = pd.read_csv(os.path.join(out, "predictions.csv"))
    C = pd.concat(gpd.read_file(f).drop(columns="geometry").assign(image=os.path.basename(f)[:-len(".confidence.gpkg")])
                  for f in sorted(glob.glob(os.path.join(args.results, "*.confidence.gpkg"))) if ".simp." not in f)
    P = P.merge(C[["image", "id", "confidence", "conf_max", "match_iou"]], on=["image", "id"], how="left")
    P.to_csv(os.path.join(out, "predictions.csv"), index=False)
    by_kind = P.groupby("kind").confidence.describe(percentiles=[.25, .5, .75])[["count", "mean", "25%", "50%", "75%"]]
    for t in (0.2, 0.3, 0.4):
        by_kind[f"share >= {t}"] = P.groupby("kind").confidence.apply(lambda c: (c >= t).mean())
    by_kind.to_csv(os.path.join(out, "confidence_by_kind.csv"))

    labels = gpd.read_file(LABELS_PATH)
    labels = labels[(labels.crop != "noncrop") & labels.is_valid & (labels.area >= MIN_FIELD_M2)]
    splits = pd.read_csv(os.path.join(ROOT, args.data, "splits.csv"))
    tot = {t: np.zeros(4) for t in CUTOFFS}
    for row in splits[splits.split == "test"].itertuples():
        with rasterio.open(glob.glob(IMAGES_GLOB.format(row.folder))[0]) as src:
            crs, ext = src.crs, box(*src.bounds)
        gt = labels[labels.grid == row.grid].to_crs(crs)
        gt = gt.assign(geometry=gt.intersection(ext)).pipe(lambda d: d[~d.is_empty]).reset_index(drop=True)
        pr = gpd.read_file(os.path.join(args.results, row.folder + ".confidence.gpkg")).to_crs(crs)
        pr = pr[~pr.is_empty & pr.is_valid]
        for t in CUTOFFS:
            s = score_image(pr[pr.confidence >= t].reset_index(drop=True), gt)
            tot[t] += [s["n_gt"], s["n_pred"], s["tp"], s["over_seg"]]
    rows = []
    for t, (n_gt, n_pred, tp, over) in tot.items():
        p, r = tp / max(n_pred, 1), tp / n_gt
        rows.append({"cutoff": t, "predictions": int(n_pred), "matched": int(tp), "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0, "split_share": over / n_gt})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out, "cutoff_sweep.csv"), index=False)
    pd.set_option("display.width", 200)
    print("== confidence by kind of prediction\n" + by_kind.round(3).to_string())
    print("\n== cut-off sweep (filtering the final map)\n" + sweep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
