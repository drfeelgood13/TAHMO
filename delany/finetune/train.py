"""Fine-tune Delineate Anything v2 on the Ghana YOLO dataset (made by prepare_dataset.py).

Starts from DelineateAnythingv2.pt and keeps its original augmentation (no mosaic, flips on).
Checkpoints: finetune/runs/<name>/weights/{best,last}.pt  (best = highest val mask mAP)

Usage: python finetune/train.py --batch 4 --epochs 40 --freeze 10 --name ghana_v1
"""
import argparse
import os

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))


def localize_data_yaml(data):
    """Point data.yaml at this checkout (it stores an absolute path from the machine that built it)."""
    with open(data) as f:
        lines = f.read().splitlines()
    fixed = [f"path: {os.path.dirname(data)}" if l.startswith("path:") else l for l in lines]
    if fixed != lines:
        with open(data, "w") as f:
            f.write("\n".join(fixed) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val improvement")
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--nbs", type=int, default=64, help="effective batch: gradients are accumulated over this many tiles per update")
    ap.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (10 = whole backbone), 0 = train all")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of train tiles to use (for quick tests)")
    ap.add_argument("--name", default="ghana_v1")
    ap.add_argument("--data", default="yolo_dataset", help="dataset folder inside finetune/ (e.g. yolo_dataset_1m)")
    ap.add_argument("--mosaic", type=float, default=0.0, help="mosaic probability (original training: 0)")
    ap.add_argument("--copy-paste", type=float, default=0.0, help="segment copy-paste probability (0 = off)")
    ap.add_argument("--max-det", type=int, default=300, help="max detections per tile during validation (VIEDIA tiles can exceed 300)")
    ap.add_argument("--select", default="fitness", choices=["fitness", "map50"],
                    help="best.pt / early stopping by Ultralytics fitness (default) or by validation mask mAP50")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from its last.pt (same settings)")
    args = ap.parse_args()

    data = os.path.join(ROOT, args.data, "data.yaml")
    localize_data_yaml(data)

    if args.select == "map50":
        # Ultralytics keeps best.pt and stops early by `fitness` (0.1 mAP50 + 0.9 mAP50-95, box + mask);
        # use the validation mask mAP50 instead
        from ultralytics.utils.metrics import SegmentMetrics
        SegmentMetrics.fitness = property(lambda self: float(self.seg.map50))

    if args.resume:
        YOLO(os.path.join(ROOT, "runs", args.name, "weights", "last.pt")).train(resume=True)
        return

    weights = hf_hub_download(repo_id="MykolaL/DelineateAnything", filename="DelineateAnythingv2.pt")
    model = YOLO(weights)
    model.train(
        data=data,
        imgsz=512,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        nbs=args.nbs,         # gradients are accumulated up to this effective batch
        optimizer="AdamW",
        lr0=args.lr0,
        cos_lr=True,
        warmup_epochs=1,
        warmup_bias_lr=0.0,   # default 0.1 is for training from scratch; it wrecks the pretrained heads
        freeze=args.freeze,
        fraction=args.fraction,
        # augmentation as in the original training (mosaic and copy-paste off unless asked for)
        mosaic=args.mosaic, close_mosaic=10 if args.mosaic else 0, copy_paste=args.copy_paste,
        degrees=0.0, flipud=0.5, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
        overlap_mask=True, mask_ratio=4,
        amp=True,
        max_det=args.max_det,
        workers=4,
        project=os.path.join(ROOT, "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
