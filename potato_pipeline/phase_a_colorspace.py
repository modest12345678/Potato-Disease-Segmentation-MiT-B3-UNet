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

import cv2, joblib, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

try:
    from config import (TRAIN_IMG_DIR, TRAIN_MASK_DIR, QA_OUT_DIR, GMM_DIR,
                        HEALTHY, EARLY_BLIGHT, LATE_BLIGHT, BIC_MAX_COMPONENTS)
except ImportError:
    from potato_pipeline.config import (TRAIN_IMG_DIR, TRAIN_MASK_DIR, QA_OUT_DIR, GMM_DIR,
                                        HEALTHY, EARLY_BLIGHT, LATE_BLIGHT, BIC_MAX_COMPONENTS)

CLASSES = (HEALTHY, EARLY_BLIGHT, LATE_BLIGHT)
NAMES   = {HEALTHY: "healthy", EARLY_BLIGHT: "early", LATE_BLIGHT: "late"}
rng = np.random.default_rng(0)
CAP = 150_000

def find_img(img_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(img_dir) / (stem + ext)
        if p.exists():
            return p
    return None

def collect(space):
    conv = cv2.COLOR_BGR2HSV if space == "hsv" else cv2.COLOR_BGR2Lab
    px = {cid: [] for cid in CLASSES}
    mask_files = sorted(Path(TRAIN_MASK_DIR).glob("*.png"))
    pbar = tqdm(mask_files, desc=f"  [1/4] Extracting {space.upper()} pixels", unit="mask")
    for mp in pbar:
        ip = find_img(TRAIN_IMG_DIR, mp.stem)
        if ip is None:
            continue
        img = cv2.imread(str(ip)); mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        feat = cv2.cvtColor(img, conv).reshape(-1, 3).astype(np.float32)
        flat = mask.reshape(-1)
        for cid in CLASSES:
            sel = feat[flat == cid]
            if len(sel):
                px[cid].append(sel)
    out = {}
    for cid in CLASSES:
        if not px[cid]:
            raise SystemExit(f"ERROR: no pixels found for class {cid} ({NAMES[cid]}). "
                             f"Check NAMES_TO_ID vs your Roboflow class names.")
        arr = np.concatenate(px[cid])
        if len(arr) > CAP:
            arr = arr[rng.choice(len(arr), CAP, replace=False)]
        out[cid] = arr
        print(f"      ✔ {NAMES[cid]:<7}: {len(arr):,} sampled pixels")
    return out

def bic_fit(data, tag):
    best_g, best_bic, best_k = None, None, None
    pbar = tqdm(range(1, BIC_MAX_COMPONENTS + 1), desc=f"  [2/4] Fitting GMM {tag}", leave=False)
    for k in pbar:
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=0).fit(data)
        b = g.bic(data)
        pbar.set_postfix({"k": k, "BIC": f"{b:.0f}"})
        if best_bic is None or b < best_bic:
            best_g, best_bic, best_k = g, b, k
    print(f"      ✔ Best {tag}: k={best_k} (BIC={best_bic:.0f})")
    return best_g, best_k

def sep_pair(d_a, d_b, g_a, g_b):
    s_a = (g_a.score_samples(d_a) > g_b.score_samples(d_a)).mean()
    s_b = (g_b.score_samples(d_b) > g_a.score_samples(d_b)).mean()
    return 0.5 * (s_a + s_b)

def main():
    GMM_DIR.mkdir(parents=True, exist_ok=True)
    QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    print("🎨 Starting Phase A: Color Space Selection & GMM Fitting")
    for step_idx, space in enumerate(("hsv", "lab"), 1):
        print(f"\n[{step_idx}/2] Evaluating Color Space: {space.upper()}")
        data = collect(space)
        gmms, ks = {}, {}
        for cid in CLASSES:
            gmms[cid], ks[cid] = bic_fit(data[cid], f"{space}/{NAMES[cid]}")
        print("  [3/4] Pairwise separation matrix:")
        mat = {}
        for i, a in enumerate(CLASSES):
            for b in CLASSES[i + 1:]:
                s = sep_pair(data[a], data[b], gmms[a], gmms[b])
                mat[(a, b)] = s
                print(f"        {NAMES[a]:>7} vs {NAMES[b]:<7}: {s:.3f}")
        bottleneck = min(mat.values())
        print(f"      ✔ Bottleneck (weakest pair): {bottleneck:.3f}")
        if space == "hsv":
            for cid in (EARLY_BLIGHT, LATE_BLIGHT):
                h = data[cid][:, 0]
                wrapped = ((h < 10) | (h > 170)).mean()
                print(f"      ℹ NOTE: {wrapped:.1%} of {NAMES[cid]} pixels have wrapped hue")
        results[space] = dict(data=data, gmms=gmms, ks=ks, bottleneck=bottleneck)

    best = max(results, key=lambda s: results[s]["bottleneck"])
    r = results[best]
    print(f"\n>>> Selected Best Color Space: {best.upper()} "
          f"(healthy k={r['ks'][HEALTHY]}, early k={r['ks'][EARLY_BLIGHT]}, "
          f"late k={r['ks'][LATE_BLIGHT]}, bottleneck={r['bottleneck']:.3f})")
    joblib.dump({"space": best,
                 "gmm_healthy": r["gmms"][HEALTHY],
                 "gmm_early":   r["gmms"][EARLY_BLIGHT],
                 "gmm_late":    r["gmms"][LATE_BLIGHT]},
                GMM_DIR / "color_model.joblib")

    print("  [4/4] Generating QA distribution plot...")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    colors = {HEALTHY: "tab:green", EARLY_BLIGHT: "tab:orange", LATE_BLIGHT: "tab:red"}
    for row, cid in enumerate(CLASSES):
        for c in range(3):
            axes[row, c].hist(r["data"][cid][:, c], bins=80, color=colors[cid], alpha=0.75)
            axes[row, c].set_title(f"{best} ch{c} — {NAMES[cid]}")
    fig.tight_layout()
    fig.savefig(QA_OUT_DIR / "phase_a_distributions.png", dpi=150)
    print(f"✅ Phase A Complete! Saved model to {GMM_DIR/'color_model.joblib'}")
    print(f"   Distribution plot saved to {QA_OUT_DIR/'phase_a_distributions.png'}")

if __name__ == "__main__":
    main()