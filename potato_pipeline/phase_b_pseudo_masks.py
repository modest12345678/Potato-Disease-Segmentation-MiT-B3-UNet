import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import sys, math, argparse, cv2, joblib, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

try:
    from config import *
except ImportError:
    from potato_pipeline.config import *

PALETTE = {SOIL: (60, 60, 60), HEALTHY: (75, 180, 60),
           EARLY_BLIGHT: (60, 140, 255), LATE_BLIGHT: (40, 40, 200),
           IGNORE: (60, 220, 250)}
VOTE_IDS = np.array([HEALTHY, EARLY_BLIGHT, LATE_BLIGHT], np.uint8)

class FastGMM:
    """GPU/CPU PyTorch-accelerated Gaussian Mixture Model log-likelihood evaluator"""
    def __init__(self, gmm, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.means = torch.from_numpy(gmm.means_).float().to(device)
        covs = torch.from_numpy(gmm.covariances_).float().to(device)
        self.weights = torch.from_numpy(gmm.weights_).float().to(device)
        self.inv_covs = torch.inverse(covs)
        self.log_dets = torch.logdet(covs)
        self.const = 3.0 * math.log(2.0 * math.pi)

    def score_samples(self, x_feat):
        x_t = torch.from_numpy(x_feat).float().to(self.device)
        diff = x_t.unsqueeze(1) - self.means.unsqueeze(0)
        mahal = torch.einsum('nki,kij,nkj->nk', diff, self.inv_covs, diff)
        log_probs = -0.5 * (self.const + self.log_dets.unsqueeze(0) + mahal) + torch.log(self.weights.unsqueeze(0))
        return torch.logsumexp(log_probs, dim=1).cpu().numpy()

def find_img(img_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(img_dir) / (stem + ext)
        if p.exists():
            return p
    return None

def colorize(mask):
    out = np.zeros((*mask.shape, 3), np.uint8)
    for v, c in PALETTE.items():
        out[mask == v] = c
    return out

def clean(lbl):
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    se5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Clean up single-pixel ignore noise (do NOT dilate/close ignore!)
    ig = (lbl == IGNORE).astype(np.uint8)
    ig = cv2.morphologyEx(ig, cv2.MORPH_OPEN, se3).astype(bool)

    bins = {}
    for c in (HEALTHY, EARLY_BLIGHT, LATE_BLIGHT):
        b = (lbl == c).astype(np.uint8)
        b = cv2.medianBlur(b, 5)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, se3)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, se5)
        bins[c] = b.astype(bool)

    conflict = ((bins[HEALTHY] & bins[EARLY_BLIGHT]) |
                (bins[HEALTHY] & bins[LATE_BLIGHT]) |
                (bins[EARLY_BLIGHT] & bins[LATE_BLIGHT]))
    out = np.full(lbl.shape, SOIL, np.uint8)
    for c in (HEALTHY, EARLY_BLIGHT, LATE_BLIGHT):
        out[bins[c] & ~conflict] = c
    out[conflict | ig] = IGNORE
    return out

def build_one(img, annotated, fg_h, fg_e, fg_l, space):
    conv = cv2.COLOR_BGR2HSV if space == "hsv" else cv2.COLOR_BGR2Lab
    feat = cv2.cvtColor(img, conv).reshape(-1, 3).astype(np.float32)

    # 1. True deep shadows -> IGNORE (V < 35)
    dark = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].reshape(-1) < DARK_V

    # 2. Candidate unannotated pixels
    manual_flat = annotated.reshape(-1) > 0
    candidate_mask = (~manual_flat) & (~dark)

    # Base background is SOIL (0)
    lbl = np.full(feat.shape[0], SOIL, np.uint8)
    lbl[dark] = IGNORE

    if candidate_mask.any():
        cand_feat = feat[candidate_mask]
        ll_stack = np.stack([fg_h.score_samples(cand_feat),
                             fg_e.score_samples(cand_feat),
                             fg_l.score_samples(cand_feat)])
        best_idx = ll_stack.argmax(axis=0)
        best_ll  = ll_stack.max(axis=0)
        second   = np.partition(ll_stack, -2, axis=0)[-2]

        # Leaf thresholding:
        leaf      = best_ll > SOIL_THRESH
        vote      = VOTE_IDS[best_idx]
        confident = (best_ll - second) > MARGIN

        sub_lbl = np.full(cand_feat.shape[0], SOIL, np.uint8)
        # Confident foliage / lesion vote:
        sub_lbl[leaf & confident] = vote[leaf & confident]
        # Ambiguous foliage (e.g. uncertain between healthy and blight): mark as IGNORE
        sub_lbl[leaf & ~confident] = IGNORE

        lbl[candidate_mask] = sub_lbl

    lbl = clean(lbl.reshape(annotated.shape))
    lbl[annotated > 0] = annotated[annotated > 0]  # manual polygons ALWAYS win
    return lbl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="QA mode: first N images only")
    ap.add_argument("--force", action="store_true", help="Re-process all images even if already present")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Phase B: Generating Dense Pseudo-Masks (Accelerated on {device.upper()})")
    cm = joblib.load(GMM_DIR / "color_model.joblib")
    fg_h = FastGMM(cm["gmm_healthy"], device)
    fg_e = FastGMM(cm["gmm_early"], device)
    fg_l = FastGMM(cm["gmm_late"], device)
    space = cm["space"]

    QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_MASK_DIR.mkdir(parents=True, exist_ok=True)

    masks = sorted(Path(TRAIN_MASK_DIR).glob("*.png"))
    if args.limit:
        masks = masks[: args.limit]
    total_imgs = len(masks)
    print(f"   Target: {total_imgs} images | SOIL_THRESH={SOIL_THRESH} | MARGIN={MARGIN} | DARK_V={DARK_V}")

    pbar = tqdm(enumerate(masks), total=total_imgs, desc="Pseudo-Masking", unit="img")
    processed = 0
    skipped = 0
    for i, mp in pbar:
        out_mask_path = DENSE_MASK_DIR / mp.name
        if out_mask_path.exists() and not args.force and not args.limit:
            skipped += 1
            continue

        ip = find_img(TRAIN_IMG_DIR, mp.stem)
        if ip is None:
            continue
        img = cv2.imread(str(ip))
        ann = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        dense = build_one(img, ann, fg_h, fg_e, fg_l, space)
        cv2.imwrite(str(out_mask_path), dense)

        if i < 25 or args.limit:
            panel = np.hstack([img, colorize(ann), colorize(dense)])
            cv2.imwrite(str(QA_OUT_DIR / f"preview_{mp.stem}.jpg"), panel)
        processed += 1
        pbar.set_postfix({"done": f"{processed + skipped}/{total_imgs}", "new": processed, "cached": skipped})

    print(f"\n✅ Phase B Finished! Total {processed + skipped}/{total_imgs} dense masks ready in {DENSE_MASK_DIR}")

if __name__ == "__main__":
    main()