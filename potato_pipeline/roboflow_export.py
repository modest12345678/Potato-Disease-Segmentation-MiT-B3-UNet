import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

NAMES_TO_ID = {"healthy": 1, "early blight": 2, "late blight": 3}  # matched lowercase

def coco_to_masks(ann_json, out_dir):
    coco = json.loads(Path(ann_json).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"].strip().lower() for c in coco["categories"]}
    print(f"\n📂 Loading COCO JSON: {ann_json}")
    print("   Categories found in JSON:", sorted(set(cats.values())))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])
    print(f"   Found {total_images} images with {total_annotations} total annotations.")

    success_count = 0
    pbar = tqdm(coco["images"], desc=f"Converting -> {out_dir.name}", unit="image")
    for im in pbar:
        mask = np.zeros((im["height"], im["width"]), np.uint8)   # 0 = soil/unlabeled
        def cls_of(a): return NAMES_TO_ID.get(cats.get(a["category_id"], ""), 0)
        for a in sorted(anns.get(im["id"], []), key=cls_of):     # 1 first, 2, then 3 on top
            cls = cls_of(a)
            if cls == 0:
                continue
            seg = a.get("segmentation")
            if seg is None:
                continue

            # Case 1: RLE dict (Roboflow smart polygon / bitmap export)
            if isinstance(seg, dict):
                if mask_utils is None:
                    raise ImportError("pycocotools is required. Run: pip install pycocotools")
                if isinstance(seg.get("counts"), str):
                    rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
                else:
                    rle = seg
                m = mask_utils.decode(rle)
                mask[m > 0] = cls

            # Case 2: Polygon coordinate list [[x1, y1, x2, y2, ...]]
            elif isinstance(seg, list):
                if len(seg) > 0:
                    if isinstance(seg[0], (int, float)):
                        pts = np.array(seg, np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], cls)
                    else:
                        for poly in seg:
                            pts = np.array(poly, np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], cls)

        name = Path(im["file_name"]).with_suffix(".png").name
        cv2.imwrite(str(out_dir / name), mask)
        success_count += 1
        pbar.set_postfix({"saved": f"{success_count}/{total_images}"})
    print(f"✅ Finished! Successfully wrote {success_count}/{total_images} masks to {out_dir}\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    coco_to_masks(a.json, a.out)