"""Tune the delineation (post-processing) settings of a fine-tuned model on validation images.

Each run = the base config with one or more settings changed. For each run: write the config and a batch file,
delineate the validation images, score them (evaluate.py --extended), and append a row to <out>/summary.csv.
Runs whose summary row already exists are skipped, so the job can be resubmitted after a time-out.

Usage (from the repo root): python finetune/tune_delineation.py --stage oaat
       python finetune/tune_delineation.py --stage viedia_check --dataset viedia   (40 VIEDIA validation tiles)
Outputs: finetune/tuning/{configs,batches,results/<run>}/, finetune/tuning/summary.csv (VIEDIA: *_viedia)
"""
import argparse
import copy
import os
import subprocess
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

FT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(FT)
OUT = os.path.join(FT, "tuning")
BASE_CONFIG = os.path.join(FT, "conf_combined_1m.yaml")
VAL_BATCH = os.path.join(FT, "ghana_1m_val_batch.yaml")      # data root + the 12 Ghana validation images
DATA = "yolo_dataset_1m"
VIEDIA_ROOT = "finetune/VIEDIA 1m (TIFFANIES)-20261007T073153Z-1-001/VIEDIA 1m (TIFFANIES)"
VIEDIA_SAMPLE = 20  # validation tiles per area (2 areas), fixed seed

PASS = ("passes", 0, "delineation_config")
CONF = ("passes", 0, "model_args", 0, "minimal_confidence")

# one setting at a time (defaults in conf_combined_1m.yaml: 512 / 0.8 / 0.8 / 0.5 / 0.15)
STAGES = {
    "oaat": {
        "baseline": {},
        "min_px_256": {PASS + ("pixel_area_threshold",): 256},
        "min_px_128": {PASS + ("pixel_area_threshold",): 128},
        "min_px_64": {PASS + ("pixel_area_threshold",): 64},
        "merge_iou_0.65": {PASS + ("merge_iou",): 0.65},
        "merge_iou_0.5": {PASS + ("merge_iou",): 0.5},
        "compose_iou_0.6": {PASS + ("compose_merge_iou",): 0.6},
        "merge_rel_0.3": {PASS + ("merge_relative_area_threshold",): 0.3},
        "conf_0.10": {CONF: 0.10},
    },
    # round 2: stronger joining between tiles, and combinations of what helped in round 1
    "combo": {
        "merge_iou_0.4": {PASS + ("merge_iou",): 0.4},
        "merge_iou_0.3": {PASS + ("merge_iou",): 0.3},
        "edge_iou_0.4": {PASS + ("merge_edge_iou",): 0.4},
        "miou0.5_edge0.4": {PASS + ("merge_iou",): 0.5, PASS + ("merge_edge_iou",): 0.4},
        "miou0.5_compose0.6": {PASS + ("merge_iou",): 0.5, PASS + ("compose_merge_iou",): 0.6},
        "miou0.4_compose0.6": {PASS + ("merge_iou",): 0.4, PASS + ("compose_merge_iou",): 0.6},
        "miou0.5_compose0.6_edge0.4": {PASS + ("merge_iou",): 0.5, PASS + ("compose_merge_iou",): 0.6, PASS + ("merge_edge_iou",): 0.4},
        "miou0.5_conf0.10": {PASS + ("merge_iou",): 0.5, CONF: 0.10},
    },
    # round 3: the two best Ghana settings against the current ones, on VIEDIA validation tiles
    "viedia_check": {
        "baseline": {},
        "merge_iou_0.3": {PASS + ("merge_iou",): 0.3},
        "miou0.4_compose0.6": {PASS + ("merge_iou",): 0.4, PASS + ("compose_merge_iou",): 0.6},
    },
}


def set_path(cfg, path, value):
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value


def score(results_dir):
    s = pd.read_csv(os.path.join(results_dir, "scores_val.csv"))
    e = pd.read_csv(os.path.join(results_dir, "scores_val_extended.csv"))
    t = s[["n_gt", "n_pred", "tp", "sum_iou", "over_seg", "under_seg"]].sum()
    p, r = t.tp / max(t.n_pred, 1), t.tp / t.n_gt
    x = metrics.summarize(e.drop(columns="folder").sum())
    return {"predictions": int(t.n_pred), "matched": int(t.tp), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0,
            "mean_iou": t.sum_iou / max(t.tp, 1), "split_share": t.over_seg / t.n_gt, "merge_share": t.under_seg / max(t.n_pred, 1),
            "area_omission": x["area_omission"], "area_commission": x["area_commission"],
            "boundary_f1_2.5m": x["boundary_F1_2.5m"], "boundary_f1_5m": x["boundary_F1_5m"], "polis_m": x["polis_mean_m"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="oaat", choices=list(STAGES))
    ap.add_argument("--dataset", default="ghana", choices=["ghana", "viedia"])
    args = ap.parse_args()
    sfx = "" if args.dataset == "ghana" else "_viedia"
    data = DATA if args.dataset == "ghana" else "yolo_dataset_viedia"
    for d in ("configs", "batches", "results" + sfx):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    summary_path = os.path.join(OUT, f"summary{sfx}.csv")
    done = set(pd.read_csv(summary_path).run) if os.path.exists(summary_path) else set()

    base = yaml.safe_load(open(BASE_CONFIG))
    base["simplification_args"]["simplify"] = False  # scoring uses the raw outlines; saves time
    val_batch = yaml.safe_load(open(VAL_BATCH))
    if args.dataset == "viedia":
        sp = pd.read_csv(os.path.join(FT, "yolo_dataset_viedia", "splits.csv"))
        sample = pd.concat(d.sample(VIEDIA_SAMPLE, random_state=0) for _, d in sp[sp.split == "val"].groupby("area"))
        val_batch = dict(val_batch, data_root=VIEDIA_ROOT, include=sorted(sample.folder))
    env = dict(os.environ, PROJ_DATA=os.path.join(sys.prefix, "share", "proj"), HF_HUB_OFFLINE="1", YOLO_OFFLINE="True")

    for run, changes in STAGES[args.stage].items():
        if run in done:
            print(f"skip {run} (already in summary)", flush=True); continue
        cfg = copy.deepcopy(base)
        for path, value in changes.items():
            set_path(cfg, path, value)
        cfg_path = os.path.join(OUT, "configs", run + sfx + ".yaml")
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        res = os.path.join(OUT, "results" + sfx, run)
        batch = dict(val_batch, base_config=os.path.relpath(cfg_path, REPO), output_root=os.path.relpath(res, REPO),
                     temp_root=os.path.relpath(os.path.join(OUT, "temp_" + run + sfx), REPO), keep_temp=False)
        batch_path = os.path.join(OUT, "batches", run + sfx + ".yaml")
        yaml.safe_dump(batch, open(batch_path, "w"), sort_keys=False)

        t0 = time.time()
        print(f"=== {run}: {changes or 'defaults'}", flush=True)
        subprocess.run([sys.executable, "delineate.py", "-b", os.path.relpath(batch_path, REPO)], cwd=REPO, env=env, check=True)
        subprocess.run([sys.executable, os.path.join(FT, "evaluate.py"), res, "--split", "val", "--data", data, "--extended"],
                       cwd=REPO, env=env, check=True)
        row = {"run": run, "stage": args.stage, "changes": "; ".join(f"{p[-1]}={v}" for p, v in changes.items()) or "defaults",
               **score(res), "minutes": (time.time() - t0) / 60}
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    s = pd.read_csv(summary_path)
    pd.set_option("display.width", 250)
    print("\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
