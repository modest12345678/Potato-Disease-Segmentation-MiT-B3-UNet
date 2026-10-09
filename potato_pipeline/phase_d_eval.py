import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import sys
from pathlib import Path

_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import argparse, csv, torch
from tqdm import tqdm

try:
    from config import *
    from model import build_model
    from dataset import SegDataset
    from metrics import confusion_from_logits, per_class_iou_dice
except ImportError:
    from potato_pipeline.config import *
    from potato_pipeline.model import build_model
    from potato_pipeline.dataset import SegDataset
    from potato_pipeline.metrics import confusion_from_logits, per_class_iou_dice

def field_of(stem):
    parts = stem.split("_")
    return parts[0] if len(parts) > 1 else "all"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--csv", default="eval_results.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📊 Phase D: Evaluating Checkpoint {args.ckpt} on {device.upper()}")
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.eval()

    ds = SegDataset(EVAL_IMG_DIR, EVAL_MASK_DIR, train=False)
    fields, rows = {}, []
    pbar = tqdm(range(len(ds)), desc="Evaluating Test Set", unit="img")
    with torch.no_grad():
        for i in pbar:
            x, y = ds[i]
            cm = confusion_from_logits(model(x.unsqueeze(0).to(device)),
                                       y.unsqueeze(0).to(device), NUM_CLASSES)
            f = field_of(ds.imgs[i].stem)
            fields[f] = fields.get(f, torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)) + cm
            iou, _ = per_class_iou_dice(cm)
            rows.append((ds.imgs[i].stem, f,
                         iou[1].item(), iou[2].item(), iou[3].item()))
            pbar.set_postfix({"field": f, "img": ds.imgs[i].stem[:14]})

    hdr = f"\n{'field':<12}{'IoU_h':>8}{'IoU_e':>8}{'IoU_b':>8}{'Dice_h':>8}{'Dice_e':>8}{'Dice_b':>8}"
    print(hdr); print("-" * len(hdr))
    allcm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for f, cm in sorted(fields.items()):
        iou, dice = per_class_iou_dice(cm)
        print(f"{f:<12}{iou[1]:>8.3f}{iou[2]:>8.3f}{iou[3]:>8.3f}"
              f"{dice[1]:>8.3f}{dice[2]:>8.3f}{dice[3]:>8.3f}")
        allcm += cm
    iou, dice = per_class_iou_dice(allcm)
    print(f"{'OVERALL':<12}{iou[1]:>8.3f}{iou[2]:>8.3f}{iou[3]:>8.3f}"
          f"{dice[1]:>8.3f}{dice[2]:>8.3f}{dice[3]:>8.3f}")

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "field", "iou_healthy", "iou_early", "iou_late"])
        w.writerows(rows)
    print(f"\n✅ Per-image detail saved -> {args.csv}\n")

if __name__ == "__main__":
    main()