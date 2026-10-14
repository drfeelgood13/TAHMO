"""Score delineation outputs (one .gpkg per image) against the field labels.

Instance matching per image: a prediction and a label match if IoU >= IOU_THR (one-to-one, greedy by IoU).
Also counts over-segmentation (label covered by >= 2 predictions) and under-segmentation
(prediction covering >= 2 labels), where "covered" = at least COVER_THR of the smaller polygon.

Usage: python finetune/evaluate.py finetune/results/zero_shot [--split test]
       python finetune/evaluate.py finetune/results/combined_viedia --data yolo_dataset_viedia   (per-tile labels)
       add --extended for area omission/commission, boundary F1 with tolerance and PoLiS (see metrics.py)
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import box

import metrics

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_GLOB = os.path.join(ROOT, "DATASET*", "DATASET*", "{}", "mosaic_utm.tif")
LABELS_PATH = os.path.join(ROOT, "labels", "filtered_grids(1).gpkg")

IOU_THR = 0.5
COVER_THR = 0.5
MIN_FIELD_M2 = 20  # same filter as prepare_dataset.py


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
    per_gt = pd.Series([j for j, _ in covered]).value_counts()
    per_pred = pd.Series([i for _, i in covered]).value_counts()
    return {
        "n_gt": len(gt), "n_pred": len(pred), "tp": len(ious), "sum_iou": sum(ious),
        "over_seg": int((per_gt >= 2).sum()), "under_seg": int((per_pred >= 2).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--simp", action="store_true", help="score the simplified <name>.simp.gpkg outputs")
    ap.add_argument("--data", default="yolo_dataset", help="dataset folder whose splits.csv names the images (e.g. yolo_dataset_1m)")
    ap.add_argument("--extended", action="store_true",
                    help="also compute area omission/commission, boundary F1 with tolerance and PoLiS (metrics.py)")
    args = ap.parse_args()
    ext = ".simp.gpkg" if args.simp else ".gpkg"

    splits = pd.read_csv(os.path.join(ROOT, args.data, "splits.csv"))
    per_image = "labels" in splits  # VIEDIA (prepare_viedia.py): one label file and image path per tile
    if not per_image:
        labels = gpd.read_file(LABELS_PATH)
        labels = labels[(labels.crop != "noncrop") & labels.is_valid & (labels.area >= MIN_FIELD_M2)]

    rows, ext_rows, polis_values = [], [], []
    for row in splits[splits.split == args.split].itertuples():
        folder = row.folder
        pred_path = os.path.join(args.results, folder + ext)
        if not os.path.exists(pred_path):
            print("missing", folder); continue
        image = os.path.join(ROOT, row.image) if per_image else glob.glob(IMAGES_GLOB.format(folder))[0]
        with rasterio.open(image) as src:
            crs, extent = src.crs, box(*src.bounds)
        if per_image:
            gt = gpd.read_file(os.path.join(ROOT, row.labels)).to_crs(crs).explode(index_parts=False)
            gt = gt[gt.is_valid & (gt.area >= MIN_FIELD_M2)]
        else:
            gt = labels[labels.grid == row.grid].to_crs(crs)
        gt = gt.assign(geometry=gt.geometry.intersection(extent)).pipe(lambda d: d[~d.is_empty]).reset_index(drop=True)
        pred = gpd.read_file(pred_path).to_crs(crs)
        pred = pred[~pred.is_empty & pred.is_valid].reset_index(drop=True)
        rows.append({"folder": folder, **score_image(pred, gt)})
        if args.extended:
            # VIEDIA is fully digitised -> judge the whole tile; Ghana labels cover only the surveyed fields ->
            # judge predictions only inside the outline (convex hull) around the labelled fields of the image
            region = extent if per_image else shapely.intersection(shapely.convex_hull(shapely.union_all(gt.geometry.values)), extent)
            pp = metrics.polis_pairs(pred, gt)
            polis_values += pp["polis_values"]
            ext_rows.append({"folder": folder, "n_gt": len(gt), "region_m2": region.area,
                             **metrics.area_errors(pred, gt, region),
                             **metrics.boundary_counts(pred, gt, extent, region),
                             "polis_sum": pp["polis_sum"], "polis_n": pp["polis_n"]})

    df = pd.DataFrame(rows)
    df["precision"] = df.tp / df.n_pred.clip(lower=1)
    df["recall"] = df.tp / df.n_gt.clip(lower=1)
    df["f1"] = (2 * df.precision * df.recall / (df.precision + df.recall)).fillna(0)
    df.to_csv(os.path.join(args.results, f"scores_{args.split}{'_simp' if args.simp else ''}.csv"), index=False)

    t = df[["n_gt", "n_pred", "tp", "sum_iou", "over_seg", "under_seg"]].sum()
    p, r = t.tp / t.n_pred, t.tp / t.n_gt
    print(f"images: {len(df)}  labels: {t.n_gt}  predictions: {t.n_pred}  matched (IoU>={IOU_THR}): {t.tp}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {2 * p * r / (p + r):.3f}  mean IoU of matches {t.sum_iou / max(t.tp, 1):.3f}")
    print(f"over-segmented labels: {t.over_seg} ({t.over_seg / t.n_gt:.1%})  under-segmenting predictions: {t.under_seg} ({t.under_seg / t.n_pred:.1%})")
    print(f"per-image F1 median: {df.f1.median():.3f}  images with no match: {(df.tp == 0).sum()}")

    if args.extended:
        ed = pd.DataFrame(ext_rows)
        per = ed.apply(lambda r: pd.Series(metrics.summarize(r)), axis=1)
        pd.concat([ed, per], axis=1).to_csv(
            os.path.join(args.results, f"scores_{args.split}{'_simp' if args.simp else ''}_extended.csv"), index=False)
        s = metrics.summarize(ed.drop(columns="folder").sum())
        tols = metrics.TOLERANCES
        print(f"[extended, judged in: {'image extent' if per_image else 'surveyed area (hull of labels)'}]")
        print(f"area omission {s['area_omission']:.3f}  area commission {s['area_commission']:.3f}")
        print("  ".join(f"boundary@{t:g}m P {s[f'boundary_P_{t:g}m']:.3f} R {s[f'boundary_R_{t:g}m']:.3f} "
                        f"F1 {s[f'boundary_F1_{t:g}m']:.3f}" for t in tols))
        print(f"PoLiS mean {s['polis_mean_m']:.2f} m  median {np.median(polis_values) if polis_values else float('nan'):.2f} m  "
              f"(reference fields with an overlapping prediction: {s['polis_paired_share']:.1%})")


if __name__ == "__main__":
    main()
