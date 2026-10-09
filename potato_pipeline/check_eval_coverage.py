import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

QA = Path("data/qa"); QA.mkdir(parents=True, exist_ok=True)
PAL = {1: (75, 180, 60), 2: (60, 140, 255), 3: (40, 40, 200)}

img_files = sorted(p for p in Path("data/eval/images").iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith("preview_"))
print(f"\n🔍 Verifying Evaluation Dataset Coverage ({len(img_files)} images)...")

paired = 0
unpaired = 0
pbar = tqdm(img_files, desc="Checking Eval Overlays", unit="img")
for ip in pbar:
    img = cv2.imread(str(ip))
    m = cv2.imread(f"data/eval/masks/{ip.stem}.png", 0)
    if img is None or m is None:
        print(f"\n⚠️ MISSING/UNPAIRED: {ip.stem}")
        unpaired += 1
        continue
    ov = img.copy()
    for v, c in PAL.items():
        ov[m == v] = c
    cv2.imwrite(str(QA / f"eval_check_{ip.stem}.jpg"),
                cv2.addWeighted(img, 0.5, ov, 0.5, 0))
    paired += 1
    pbar.set_postfix({
        "paired": f"{paired}/{len(img_files)}",
        "leaf%": f"{(m>0).mean():.1%}"
    })

print(f"\n✅ Coverage Check Complete: {paired}/{len(img_files)} masks paired and validated!")
if unpaired > 0:
    print(f"⚠️ Warning: {unpaired} images lacked matching masks.")
print(f"   Visual overlay previews saved in {QA}/\n")