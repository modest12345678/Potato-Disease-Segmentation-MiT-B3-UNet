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

import math, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from config import *
    from dataset import SegDataset, class_weights
    from model import build_model
    from losses import SegLoss
    from metrics import confusion_from_logits, per_class_iou_dice
except ImportError:
    from potato_pipeline.config import *
    from potato_pipeline.dataset import SegDataset, class_weights
    from potato_pipeline.model import build_model
    from potato_pipeline.losses import SegLoss
    from potato_pipeline.metrics import confusion_from_logits, per_class_iou_dice

@torch.no_grad()
def evaluate(model, loader, device, num_classes=4):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        cm += confusion_from_logits(model(x.to(device)), y.to(device), num_classes)
    return per_class_iou_dice(cm)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Initializing Training Pipeline on Device: {device.upper()}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("📦 Loading datasets...")
    train_ds = SegDataset(TRAIN_IMG_DIR, DENSE_MASK_DIR, train=True)
    eval_ds  = SegDataset(EVAL_IMG_DIR, EVAL_MASK_DIR, train=False)
    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
    eval_ld  = DataLoader(eval_ds, 1, num_workers=2, pin_memory=True)
    print(f"   Train samples: {len(train_ds)} ({len(train_ld)} batches/epoch, batch size={BATCH_SIZE})")
    print(f"   Eval samples:  {len(eval_ds)}")

    print("\n🧠 Constructing MiT-B3 + U-Net (6-channel RGB+HSV input)...")
    model = build_model(num_classes=NUM_CLASSES).to(device)
    print("⚖ Computing class weights from training masks...")
    weights = class_weights(DENSE_MASK_DIR, NUM_CLASSES).to(device)
    print("   Class weights:", [round(w.item(), 3) for w in weights])
    crit = SegLoss(class_weights=weights).to(device)

    opt = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": ENCODER_LR},
         {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
          "lr": DECODER_LR}],
        weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e:
        (e + 1) / WARMUP_EPOCHS if e < WARMUP_EPOCHS else
        0.5 * (1 + math.cos(math.pi * (e - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS))))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    print(f"\n🏋 Starting Training for {EPOCHS} Epochs...")
    best = -1.0
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        pbar = tqdm(train_ld, desc=f"Ep [{ep+1:2d}/{EPOCHS}]", unit="batch", leave=False, dynamic_ncols=True)
        step = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{tot/step:.4f}"})
        sched.step()

        iou, _ = evaluate(model, eval_ld, device)
        avg_loss = tot / len(train_ld)
        print(f"Epoch [{ep+1:2d}/{EPOCHS}] Loss: {avg_loss:.4f} | IoU: soil={iou[0]:.3f} "
              f"healthy={iou[1]:.3f} early={iou[2]:.3f} late={iou[3]:.3f} | mIoU={iou.mean():.3f}")
        if iou.mean() > best:
            best = iou.mean()
            torch.save({"model": model.state_dict(), "epoch": ep}, CKPT_DIR / "best.pt")
            print(f"   ⭐ New best mIoU ({best:.3f}) -> saved to {CKPT_DIR/'best.pt'}")
    print(f"\n🏆 Training Finished! Overall Best mIoU: {best:.3f} saved at {CKPT_DIR/'best.pt'}\n")

if __name__ == "__main__":
    main()