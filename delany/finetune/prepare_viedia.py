"""Convert the VIEDIA tiles (Cambodia + Vietnam, 1 m) + per-tile field polygons into a YOLO segmentation dataset.

Every image is already one 512x512 tile, so it is used as is. Labels: labels_viedia/<area>/<area>_r<row>_c<col>.gpkg.

Split (by area, never by tile: neighbouring tiles of an area look alike):
    per country, VAL_AREAS / TEST_AREAS areas are drawn at random (SEED), the rest -> train
Tiles whose label file is empty are left out (unlabelled, not "no fields").

Output: finetune/yolo_dataset_viedia/{images,labels}/{train,val,test}, data.yaml, splits.csv
        (splits.csv also lists each tile's image and label path, for evaluate.py)
"""
import glob
import os
import random
import re

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from prepare_dataset import MAX_NODATA, MIN_FIELD_M2, TILE, tile_labels

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(ROOT, "VIEDIA*", "VIEDIA*")
LABELS = os.path.join(ROOT, "labels_viedia")
OUT = os.path.join(ROOT, "yolo_dataset_viedia")

VAL_AREAS = 1   # per country
TEST_AREAS = 1  # per country
SEED = 0
NAME = re.compile(r"(?P<country>[A-Z]+)_area(?P<area>\d+)_tile\d+\(r(?P<r>\d+)c(?P<c>\d+)_1m\)")


def make_table():
    rows = []
    for d in sorted(glob.glob(os.path.join(IMAGES, "*"))):
        m = NAME.fullmatch(os.path.basename(d))
        key = f"{m['area']}_r{m['r']}_c{m['c']}"
        rows.append({"folder": os.path.basename(d), "grid": key, "area": int(m["area"]), "country": m["country"],
                     "image": os.path.relpath(os.path.join(d, "mosaic_utm.tif"), ROOT),
                     "labels": os.path.relpath(os.path.join(LABELS, m["area"], key + ".gpkg"), ROOT)})
    df = pd.DataFrame(rows)
    # UNKNOWN_area45 tiles belong to VIETNAM_area45: the country of an area is its most common one
    df["region"] = df.groupby("area")["country"].transform(lambda c: c.mode()[0]).str.lower()
    return df


def make_split(df):
    rng = random.Random(SEED)
    split = {}
    for _, areas in df.groupby("region")["area"]:
        areas = sorted(areas.unique())
        rng.shuffle(areas)
        for i, a in enumerate(areas):
            split[a] = "test" if i < TEST_AREAS else "val" if i < TEST_AREAS + VAL_AREAS else "train"
    df["split"] = df.area.map(split)
    return df


def process_tile(row, stats):
    labels = gpd.read_file(os.path.join(ROOT, row.labels))
    if len(labels) == 0:
        stats["skipped_unlabelled"] += 1
        return False
    with rasterio.open(os.path.join(ROOT, row.image)) as src:
        fields = labels.to_crs(src.crs).explode(index_parts=False)
        fields = fields[fields.is_valid & (fields.area >= MIN_FIELD_M2)]
        shapes = [(geom, i + 1) for i, geom in enumerate(fields.geometry)]
        ids = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform, fill=0, dtype="int32") \
            if shapes else np.zeros((src.height, src.width), dtype="int32")
        rgb = src.read([1, 2, 3])
    valid = ~np.all(rgb == 0, axis=0)
    if 1 - valid.mean() > MAX_NODATA:
        stats["skipped_nodata"] += 1
        return False
    pad = ((0, TILE - valid.shape[0]), (0, TILE - valid.shape[1]))  # no-op for 512x512 tiles
    lines = tile_labels(np.pad(ids, pad), np.pad(valid, pad))
    tile = np.pad(rgb, ((0, 0),) + pad).transpose(1, 2, 0)
    cv2.imwrite(os.path.join(OUT, "images", row.split, row.folder + ".png"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
    with open(os.path.join(OUT, "labels", row.split, row.folder + ".txt"), "w") as f:
        f.write("\n".join(lines))
    stats[row.split + "_tiles"] += 1
    stats[row.split + "_instances"] += len(lines)
    return True


def main():
    df = make_split(make_table())
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
    stats = {k: 0 for k in ["skipped_unlabelled", "skipped_nodata"] +
             [f"{s}_{t}" for s in ("train", "val", "test") for t in ("tiles", "instances")]}
    df["used"] = [process_tile(row, stats) for row in df.itertuples()]
    df = df[df.used].drop(columns="used")
    df.to_csv(os.path.join(OUT, "splits.csv"), index=False)
    print(df.groupby(["split", "region"]).area.agg(lambda a: sorted(a.unique())).to_string())
    print(pd.crosstab(df.split, df.region, margins=True))
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: field\n")
    print(stats)


if __name__ == "__main__":
    main()
