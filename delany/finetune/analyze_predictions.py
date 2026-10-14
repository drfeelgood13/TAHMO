"""Characteristics of reference fields and predicted fields on a test set, linked to how well they were delineated.

For every reference field (Ghana labels) and every predicted polygon:
  label metadata (crop, crop1, label confidence, region, survey date), size, shape (compactness),
  image appearance from the 1 m RGB mosaic: mean colour, brightness, greenness (VARI), texture (std of
  brightness inside), and edge contrast (|mean brightness inside - in a 3 m ring outside|),
  and the outcome: matched 1-to-1 at IoU >= 0.5 (as evaluate.py), best IoU, share of the field covered
  by predictions, split (>= 2 predictions cover it), merged into a prediction that covers >= 2 fields.

Usage: python finetune/analyze_predictions.py finetune/results/combined_ghana
Writes <results>/analysis/{fields.csv, predictions.csv, images.csv, summary.json}
"""
import argparse
import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import box

from evaluate import IOU_THR, COVER_THR, LABELS_PATH, MIN_FIELD_M2, IMAGES_GLOB, ROOT

RING_M = 3.0


def group_crop(c):
    return c if c in ("maize", "rice", "fallow", "unidentified") else "other crops"


def appearance(src, rgb, geoms):
    """Per polygon: mean R, G, B, brightness, VARI, texture, edge contrast against a ring outside."""
    out = []
    for g in geoms:
        win = from_bounds(*g.buffer(RING_M + 1).bounds, src.transform).round_offsets().round_lengths()
        r0, c0 = max(win.row_off, 0), max(win.col_off, 0)
        r1, c1 = min(win.row_off + win.height, rgb.shape[1]), min(win.col_off + win.width, rgb.shape[2])
        if r1 <= r0 or c1 <= c0:
            out.append([np.nan] * 8); continue
        sub = rgb[:, r0:r1, c0:c1].astype(float)
        tr = rasterio.windows.transform(rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), src.transform)
        valid = ~np.all(sub == 0, axis=0)
        inside = geometry_mask([g], (r1 - r0, c1 - c0), tr, invert=True) & valid
        ring = geometry_mask([g.buffer(RING_M).difference(g)], (r1 - r0, c1 - c0), tr, invert=True) & valid
        if inside.sum() < 3:
            out.append([np.nan] * 8); continue
        R, G, B = (sub[i][inside] for i in range(3))
        bright = (R + G + B) / 3
        vari = np.clip((G - R) / np.maximum(G + R - B, 1), -1, 1)
        ring_b = sub[:, ring].mean() if ring.sum() >= 3 else np.nan
        out.append([R.mean(), G.mean(), B.mean(), bright.mean(), np.nanmean(vari), bright.std(),
                    abs(bright.mean() - ring_b) if ring_b == ring_b else np.nan, inside.sum()])
    return pd.DataFrame(out, columns=["r", "g", "b", "brightness", "vari", "texture", "edge_contrast", "px"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--data", default="yolo_dataset_1m")
    args = ap.parse_args()
    out_dir = os.path.join(args.results, "analysis"); os.makedirs(out_dir, exist_ok=True)

    labels = gpd.read_file(LABELS_PATH)
    labels = labels[labels.is_valid & (labels.area >= MIN_FIELD_M2)]
    splits = pd.read_csv(os.path.join(ROOT, args.data, "splits.csv"))
    fields, preds, images = [], [], []
    for row in splits[splits.split == "test"].itertuples():
        with rasterio.open(glob.glob(IMAGES_GLOB.format(row.folder))[0]) as src:
            crs, ext = src.crs, box(*src.bounds)
            rgb = src.read([1, 2, 3])
            lab = labels[labels.grid == row.grid].to_crs(crs)
            lab = lab.assign(geometry=lab.geometry.intersection(ext)).pipe(lambda d: d[~d.is_empty])
            gt = lab[lab.crop != "noncrop"].reset_index(drop=True)
            noncrop = shapely.union_all(lab[lab.crop == "noncrop"].geometry.values)
            pr = gpd.read_file(os.path.join(args.results, row.folder + ".gpkg")).to_crs(crs)
            pr = pr[~pr.is_empty & pr.is_valid].reset_index(drop=True)

            # pairs and 1-to-1 matching exactly as evaluate.score_image
            pairs = []
            for j, i in zip(*pr.sindex.query(gt.geometry, predicate="intersects")):
                g, p = gt.geometry.iloc[j], pr.geometry.iloc[i]
                inter = g.intersection(p).area
                pairs.append((j, i, inter / (g.area + p.area - inter), inter / min(g.area, p.area), inter))
            pairs = pd.DataFrame(pairs, columns=["j", "i", "iou", "cover", "inter"])
            mg, mp = {}, {}
            for j, i, iou in pairs.sort_values("iou", ascending=False)[["j", "i", "iou"]].itertuples(index=False):
                if iou < IOU_THR: break
                if j not in mg and i not in mp: mg[j] = iou; mp[i] = iou
            cov = pairs[pairs.cover >= COVER_THR]
            n_pred_cov = cov.groupby("j").size(); n_gt_cov = cov.groupby("i").size()
            pred_union = shapely.union_all(pr.geometry.values)

            ga = appearance(src, rgb, gt.geometry.values)
            g = pd.DataFrame({
                "image": row.folder, "label_id": gt["id"].values, "region": row.region, "crop": gt.crop.values, "crop_group": gt.crop1.map(group_crop).values,
                "crop1": gt.crop1.values, "label_confidence": gt.confidence.fillna("none").values, "survey_date": gt.date.astype(str).values,
                "area_m2": gt.area.values, "compactness": (4 * np.pi * gt.area / gt.length ** 2).values,
                "matched": [j in mg for j in range(len(gt))],
                "best_iou": [pairs[pairs.j == j].iou.max() if (pairs.j == j).any() else 0 for j in range(len(gt))],
                "covered_share": shapely.area(shapely.intersection(gt.geometry.values, pred_union)) / gt.area.values,
                "n_pred_covering": [int(n_pred_cov.get(j, 0)) for j in range(len(gt))]})
            best_pred = pairs.loc[pairs.groupby("j").inter.idxmax()].set_index("j").i if len(pairs) else pd.Series(dtype=int)
            g["merged"] = [bool(n_gt_cov.get(best_pred.get(j, -1), 0) >= 2) for j in range(len(gt))]
            g["split"] = g.n_pred_covering >= 2
            fields.append(pd.concat([g, ga], axis=1))

            pa = appearance(src, rgb, pr.geometry.values)
            in_gt = shapely.area(shapely.intersection(pr.geometry.values, shapely.union_all(gt.geometry.values))) / pr.area.values
            in_nc = shapely.area(shapely.intersection(pr.geometry.values, noncrop)) / pr.area.values if not noncrop.is_empty else np.zeros(len(pr))
            p = pd.DataFrame({"image": row.folder, "id": pr["id"].values, "region": row.region, "area_m2": pr.area.values,
                              "compactness": (4 * np.pi * pr.area / pr.length ** 2).values,
                              "matched": [i in mp for i in range(len(pr))], "share_in_fields": in_gt, "share_in_noncrop": in_nc,
                              "n_fields_covered": [int(n_gt_cov.get(i, 0)) for i in range(len(pr))]})
            p["kind"] = np.select(
                [p.matched, p.n_fields_covered >= 2, p.share_in_noncrop >= 0.5, p.share_in_fields >= 0.5],
                ["matched field", "covers several fields", "mostly in non-crop area", "piece of a field / poor outline"],
                "mixed / boundary")
            preds.append(pd.concat([p, pa], axis=1))

            valid = ~np.all(rgb == 0, axis=0)
            R, G, B = (rgb[i][valid].astype(float) for i in range(3))
            images.append({"image": row.folder, "region": row.region,
                           "survey_date": lab.date.astype(str).mode().iloc[0] if len(lab) else "",
                           "area_ha": valid.sum() / 1e4, "n_fields": len(gt), "fields_per_ha": len(gt) / (valid.sum() / 1e4),
                           "median_field_m2": float(gt.area.median()) if len(gt) else np.nan,
                           "noncrop_share": noncrop.area / ext.area if not noncrop.is_empty else 0.0,
                           "brightness": float(((R + G + B) / 3).mean()),
                           "vari": float(np.clip((G - R) / np.maximum(G + R - B, 1), -1, 1).mean()),
                           "field_edge_contrast": float(ga.edge_contrast.median()),
                           "share_unidentified_fallow": float(gt.crop1.isin(["unidentified", "fallow"]).mean()) if len(gt) else np.nan})
        print("done", row.folder, flush=True)

    F, P, I = pd.concat(fields), pd.concat(preds), pd.DataFrame(images)
    sc = pd.read_csv(os.path.join(args.results, "scores_test.csv"))
    ext_sc = pd.read_csv(os.path.join(args.results, "scores_test_extended.csv"))
    I = I.merge(sc[["folder", "n_pred", "tp", "precision", "recall", "f1"]], left_on="image", right_on="folder").drop(columns="folder")
    I = I.merge(ext_sc[["folder", "area_omission", "area_commission", "boundary_F1_5m", "polis_mean_m"]], left_on="image", right_on="folder").drop(columns="folder")
    F.to_csv(os.path.join(out_dir, "fields.csv"), index=False)
    P.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    I.to_csv(os.path.join(out_dir, "images.csv"), index=False)

    def by(df, col, order=None):
        t = df.groupby(col, observed=True).agg(n=("matched", "size"), found=("matched", "mean"), covered=("covered_share", "mean"),
                                               best_iou=("best_iou", "median"), split=("split", "mean"), merged=("merged", "mean"))
        return t.reindex(order).dropna(how="all") if order else t

    q = lambda s, n=4: pd.qcut(s, n, duplicates="drop")
    S = {"fields": int(len(F)), "predictions": int(len(P)),
         "by_crop_group": by(F, "crop_group"), "by_label_confidence": by(F, "label_confidence", ["high", "medium", "low", "none"]),
         "by_region": by(F, "region"), "by_survey_date": by(F, "survey_date"),
         "by_size": by(F.assign(size=q(F.area_m2)), "size"), "by_vari": by(F.assign(v=q(F.vari)), "v"),
         "by_brightness": by(F.assign(v=q(F.brightness)), "v"), "by_texture": by(F.assign(v=q(F.texture)), "v"),
         "by_edge_contrast": by(F.assign(v=q(F.edge_contrast)), "v"), "by_compactness": by(F.assign(v=q(F.compactness)), "v"),
         "appearance_by_crop_group": F.groupby("crop_group")[["r", "g", "b", "brightness", "vari", "texture", "edge_contrast", "area_m2"]].median(),
         "prediction_kinds": P.kind.value_counts().to_frame("n").assign(share=lambda d: d.n / d.n.sum(),
                                                                       median_area=P.groupby("kind").area_m2.median()),
         "prediction_appearance_by_kind": P.groupby("kind")[["brightness", "vari", "texture", "edge_contrast", "compactness"]].median(),
         "image_correlations_with_f1": I[["f1", "fields_per_ha", "median_field_m2", "noncrop_share", "brightness", "vari",
                                         "field_edge_contrast", "share_unidentified_fallow"]].corr(method="spearman")["f1"].drop("f1").to_frame("spearman_rho"),
         "images": I}
    js = {k: (v.reset_index().astype({c: str for c in v.reset_index().columns if str(v.reset_index()[c].dtype) == "category"}).to_dict("records")
              if isinstance(v, pd.DataFrame) else v) for k, v in S.items()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.width", 200)
    for k, v in S.items():
        if isinstance(v, pd.DataFrame) and k != "images":
            print(f"\n== {k}\n{v.round(3).to_string()}")


if __name__ == "__main__":
    main()
