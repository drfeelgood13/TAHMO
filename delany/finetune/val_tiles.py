"""Tile-level scores (Ultralytics validation) of one or more models on the Ghana and VIEDIA validation (or test) tiles.

Appends one row per model x dataset x max_det to finetune/tile_experiments/scores.csv and prints the table,
with the equal-weight average of Ghana and VIEDIA mask mAP50 (the number experiments are compared on).

Usage (from the repo root):
    python finetune/val_tiles.py --models combined_1m combined_1m_unfrozen --max-det 300 1000 [--split val]
"""
import argparse
import os

import pandas as pd
from ultralytics import YOLO

FT = os.path.dirname(os.path.abspath(__file__))
DATASETS = {"ghana": "yolo_dataset_1m", "viedia": "yolo_dataset_viedia"}
OUT = os.path.join(FT, "tile_experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="run names in finetune/runs (best.pt is used)")
    ap.add_argument("--max-det", nargs="+", type=int, default=[1000])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name in args.models:
        weights = os.path.join(FT, "runs", name, "weights", "best.pt")
        for md in args.max_det:
            for ds, folder in DATASETS.items():
                m = YOLO(weights).val(data=os.path.join(FT, folder, "data.yaml"), split=args.split, imgsz=512, batch=16,
                                      max_det=md, project=os.path.join(OUT, "val_runs"), name=f"{name}_{ds}_{args.split}_{md}",
                                      exist_ok=True, plots=False, verbose=False)
                rows.append({"model": name, "split": args.split, "dataset": ds, "max_det": md,
                             "mask_p": m.seg.mp, "mask_r": m.seg.mr, "mask_map50": m.seg.map50, "mask_map50_95": m.seg.map,
                             "box_map50": m.box.map50, "box_map50_95": m.box.map})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "scores.csv")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    piv = df.pivot_table(index=["model", "max_det"], columns="dataset", values="mask_map50")
    piv["equal_weight_mean"] = piv[["ghana", "viedia"]].mean(axis=1)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("\nmask mAP50 (" + args.split + "):\n" + piv.round(4).to_string())


if __name__ == "__main__":
    main()
