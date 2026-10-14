"""Convert the Ghana mosaics + field polygons into a YOLO segmentation dataset.

Tiles match inference in methods/main: 512x512 native pixels (imagery < 4 m), raw 8-bit values.

Split (by image, never by tile):
    usage == "ref"   -> test
    usage == "train" -> train, minus N_VAL images (stratified by region) -> val

Output: finetune/yolo_dataset/{images,labels}/{train,val,test}, data.yaml, splits.csv

Usage (from delany/):
    python finetune/prepare_dataset.py --images /abs/path/data/ghana_1m --out /abs/path/data/yolo_ghana_1m
The split is seeded, so it is the same at every resolution (tiles smaller than 512 px are padded with nodata).
"""
import argparse
import glob
import os
import random

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(ROOT, "labels", "filtered_grids(1).gpkg")
OUT = os.path.join(ROOT, "yolo_dataset")

TILE = 512
STRIDE = {"train": 256, "val": 512, "test": 512}  # 50% overlap for train, like inference; no overlap for evaluation
N_VAL = 12
SEED = 0
MIN_FIELD_M2 = 20          # drop tiny label polygons
MIN_PART_PX = 32           # drop slivers left after clipping to a tile
MAX_NODATA = 0.5           # skip tiles that are mostly nodata


def tile_starts(size, stride):
    starts = list(range(0, max(size - TILE, 0) + 1, stride))
    if starts[-1] + TILE < size:
        starts.append(size - TILE)  # last tile flush with the edge
    return starts


def copy_split(images, path):
    """Reuse the image-level split of another resolution, matched by grid id."""
    old = pd.read_csv(path).set_index("grid")
    rows = [{"folder": os.path.basename(d), "grid": os.path.basename(d).split("_")[2]} for d in images]
    df = pd.DataFrame(rows)
    for col in ["usage", "region", "split"]:
        df[col] = df.grid.map(old[col])
    if df.split.isna().any():
        raise ValueError(f"not in {path}: {df.folder[df.split.isna()].tolist()}")
    return df


def make_split(images, labels):
    usage = labels.groupby("grid")["usage"].first()
    region = labels.groupby("grid")["region"].first()
    rows = [{"folder": os.path.basename(d), "grid": os.path.basename(d).split("_")[2]} for d in images]
    df = pd.DataFrame(rows)
    df["usage"] = df.grid.map(usage)
    df["region"] = df.grid.map(region)
    df["split"] = np.where(df.usage == "ref", "test", "train")

    rng = random.Random(SEED)
    train = df[df.split == "train"]
    for reg, group in train.groupby("region"):
        n = round(N_VAL * len(group) / len(train))
        df.loc[rng.sample(list(group.index), n), "split"] = "val"
    return df


def tile_labels(ids_raster, valid):
    lines = []
    for field_id in np.unique(ids_raster[valid]):
        if field_id == 0:
            continue
        mask = ((ids_raster == field_id) & valid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:  # a field split into pieces by the tile/nodata becomes one instance per piece
            if cv2.contourArea(c) < MIN_PART_PX or len(c) < 3:
                continue
            xy = (c.reshape(-1, 2) / TILE).clip(0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    return lines


def process_image(folder, split, labels, stats):
    name = os.path.basename(folder)
    grid = name.split("_")[2]
    with rasterio.open(os.path.join(folder, "mosaic_utm.tif")) as src:
        # labels are in EPSG:32630; some images are in 32631 -> always reproject to the image
        fields = labels[labels.grid == grid].to_crs(src.crs)
        shapes = [(geom, i + 1) for i, geom in enumerate(fields.geometry)]
        ids = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="int32") \
            if shapes else np.zeros((src.height, src.width), dtype="int32")
        rgb = src.read([1, 2, 3])
        valid_full = ~np.all(rgb == 0, axis=0)

        for y in tile_starts(src.height, STRIDE[split]):
            for x in tile_starts(src.width, STRIDE[split]):
                valid = valid_full[y:y + TILE, x:x + TILE]
                if 1 - valid.mean() > MAX_NODATA:  # measured on image pixels, before padding
                    stats["skipped_nodata"] += 1
                    continue
                # images smaller than a tile (coarser resolutions): pad with nodata to TILE x TILE
                pad = ((0, TILE - valid.shape[0]), (0, TILE - valid.shape[1]))
                valid = np.pad(valid, pad)
                tile = np.pad(rgb[:, y:y + TILE, x:x + TILE], ((0, 0),) + pad).transpose(1, 2, 0)
                lines = tile_labels(np.pad(ids[y:y + TILE, x:x + TILE], pad), valid)
                stem = f"{name}_{y}_{x}"
                cv2.imwrite(os.path.join(OUT, "images", split, stem + ".png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                stats[split + "_tiles"] += 1
                stats[split + "_instances"] += len(lines)
                stats[split + "_empty"] += len(lines) == 0


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder holding the GHANA_*/mosaic_utm.tif areas")
    ap.add_argument("--labels", default=LABELS_PATH, help="Ghana field labels (filtered_grids.gpkg)")
    ap.add_argument("--out", default="yolo_dataset", help="output folder (relative paths are inside finetune/)")
    ap.add_argument("--splits-from", help="splits.csv of another resolution to reuse instead of drawing a new split "
                                          "(not needed: the seeded split is identical at every resolution)")
    args = ap.parse_args()
    OUT = os.path.join(ROOT, args.out)

    images = sorted(glob.glob(os.path.join(args.images, "GHANA_*")))
    labels = gpd.read_file(args.labels)
    labels = labels[(labels.crop != "noncrop") & labels.is_valid & (labels.area >= MIN_FIELD_M2)]

    df = copy_split(images, args.splits_from) if args.splits_from else make_split(images, labels)
    for split in STRIDE:
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    df.to_csv(os.path.join(OUT, "splits.csv"), index=False)
    print(pd.crosstab(df.split, df.region, margins=True))

    stats = {k: 0 for k in ["skipped_nodata"] + [f"{s}_{t}" for s in STRIDE for t in ("tiles", "instances", "empty")]}
    for folder in images:
        split = df.loc[df.folder == os.path.basename(folder), "split"].item()
        process_image(folder, split, labels, stats)
        print(f"{split:5s} {os.path.basename(folder)}", flush=True)

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: field\n")
    print(stats)


if __name__ == "__main__":
    main()
