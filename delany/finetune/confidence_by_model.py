"""Mean field confidence per image and over the whole Ghana test set, for the three fine-tuned models.

Reads <results>/confidence_polygons.csv (polygon_confidence.py): one row per final field, `confidence` = confidence
of the raw detection that overlaps the field best. Fields with no overlapping detection (NaN) are left out.

Per image: mean over its fields. Test set: mean of the 32 image means (every image weighs the same) and,
for comparison, the mean over all fields pooled (images with many fields weigh more).

Usage: python finetune/confidence_by_model.py     -> finetune/results/confidence_by_model_{image,testset}.csv
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {  # numbering as in index.html
    "(1) Ghana": "ghana_1m_conf",
    "(2) Ghana + VIEDIA": "combined_ghana_conf",
    "(3) Ghana + VIEDIA unfrozen": "unfrozen_ghana",
}


def main():
    per_image, overall = [], []
    for model, folder in MODELS.items():
        path = os.path.join(ROOT, "results", folder, "confidence_polygons.csv")
        if not os.path.exists(path):
            print(f"missing {path}, skipped")
            continue
        p = pd.read_csv(path)
        img = p.groupby("image").agg(fields=("confidence", "size"), no_detection=("confidence", lambda c: c.isna().sum()),
                                     conf_mean=("confidence", "mean")).reset_index()
        per_image.append(img.assign(model=model))
        overall.append({"model": model, "images": len(img), "fields": len(p), "no_detection": int(p.confidence.isna().sum()),
                        "mean_of_image_means": img.conf_mean.mean(), "pooled_field_mean": p.confidence.mean()})
    img = pd.concat(per_image)
    wide = img.pivot(index="image", columns="model", values="conf_mean")
    img[["model", "image", "fields", "no_detection", "conf_mean"]].to_csv(
        os.path.join(ROOT, "results", "confidence_by_model_image.csv"), index=False)
    o = pd.DataFrame(overall)
    o.to_csv(os.path.join(ROOT, "results", "confidence_by_model_testset.csv"), index=False)
    print(wide.round(3).to_string(), "\n")
    print(o.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
