import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
import sys

_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from config import IMG_SIZE, IGNORE
except ImportError:
    from potato_pipeline.config import IMG_SIZE, IGNORE

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

train_tf = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.HueSaturationValue(15, 25, 20, p=0.7),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=True):
        self.mask_dir = Path(mask_dir)
        img_p = Path(img_dir)
        all_imgs = sorted(p for p in img_p.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                          and not p.name.startswith("preview_"))

        self.imgs = [p for p in all_imgs if (self.mask_dir / f"{p.stem}.png").exists()]

        if len(self.imgs) == 0:
            raise RuntimeError(
                f"\n❌ Error: No matching masks found in '{mask_dir}' for images in '{img_dir}'!\n"
            )
        self.tf = train_tf if train else val_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img_bgr = cv2.imread(str(ip))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {ip}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask_p = self.mask_dir / f"{ip.stem}.png"
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_p}")

        a = self.tf(image=img, mask=mask)
        img, mask = a["image"], a["mask"]
        rgb = (img.astype(np.float32) / 255.0 - MEAN) / STD
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] /= 179.0; hsv[..., 1] /= 255.0; hsv[..., 2] /= 255.0
        x = np.concatenate([rgb, hsv], axis=2).transpose(2, 0, 1).copy()
        return torch.from_numpy(x), torch.from_numpy(mask.astype(np.int64))

def class_weights(mask_dir, num_classes=4, max_files=400):
    """
    Square-root balanced inverse frequency with clamping.
    Prevents gradient explosion and massive false positives on rare blight classes.
    """
    counts = np.zeros(num_classes)
    mask_files = sorted(Path(mask_dir).glob("*.png"))[:max_files]
    if not mask_files:
        return torch.ones(num_classes, dtype=torch.float32)
    for f in mask_files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            flat = m.reshape(-1)
            counts += np.bincount(flat[flat != IGNORE], minlength=num_classes)

    total = counts.sum()
    # Square-root inverse frequency: smoothly balances without 300x multipliers
    w = np.sqrt(total / np.maximum(counts, 100))
    w = w / w.mean()
    # Clamp weights between 0.3 and 5.0 to maintain training stability
    w = np.clip(w, 0.3, 5.0)
    print(f"   ⚖ Balanced Clamped Class Weights: Soil={w[0]:.2f}, Healthy={w[1]:.2f}, Early={w[2]:.2f}, Late={w[3]:.2f}")
    return torch.tensor(w, dtype=torch.float32)