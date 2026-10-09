"""
Dataset Downloader & Preprocessor
=================================
Automates the retrieval and setup of both Training and Evaluation UAV datasets
from Roboflow for the MiT-B3 + U-Net Potato Disease Segmentation Pipeline.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dataset URLs (Roboflow exports)
URLS = {
    "eval": {
        "url": "https://app.roboflow.com/ds/VGsAVnyfVC?key=dU3gqpLHoj",
        "zip_name": "roboflow_eval.zip",
        "extract_to": "data/eval_raw",
        "target_img": "data/eval/images",
        "target_mask": "data/eval/masks",
    },
    "train": {
        "url": "https://app.roboflow.com/ds/GKJqAZBaee?key=zTTGuI7Hdv",
        "zip_name": "roboflow_train.zip",
        "extract_to": "data/train_raw",
        "target_img": "data/train/images",
        "target_mask": "data/train/masks",
    }
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=out_path.name) as t:
        urllib.request.urlretrieve(url, filename=str(out_path), reporthook=t.update_to)

def extract_and_prepare(split="eval"):
    cfg = URLS[split]
    zip_path = Path("data") / cfg["zip_name"]
    extract_path = Path(cfg["extract_to"])

    print(f"\n[1/3] Downloading {split.upper()} dataset...")
    download_file(cfg["url"], zip_path)

    print(f"\n[2/3] Extracting {zip_path.name} to {extract_path}...")
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    if zip_path.exists():
        zip_path.unlink()

    print(f"\n[3/3] Parsing annotations and generating masks for {split}...")
    # Import mask converter from potato_pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from potato_pipeline.roboflow_export import coco_to_masks

    json_files = list(extract_path.glob("**/_annotations.coco.json"))
    if not json_files:
        print(f"Warning: No _annotations.coco.json found in {extract_path}")
        return

    dest_img = Path(cfg["target_img"])
    dest_mask = Path(cfg["target_mask"])
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_mask.mkdir(parents=True, exist_ok=True)

    # Copy images
    img_files = [f for ext in ("*.jpg", "*.jpeg", "*.png") for f in extract_path.glob(f"**/{ext}") if not f.name.startswith("preview_")]
    for img in img_files:
        shutil.copy2(str(img), str(dest_img / img.name))
    print(f"Copied {len(img_files)} images to {dest_img}")

    # Generate masks
    for j in json_files:
        coco_to_masks(str(j), str(dest_mask))

    print(f"✅ {split.upper()} dataset is fully prepared!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare UAV potato datasets.")
    parser.add_argument("--split", choices=["eval", "train", "all"], default="eval",
                        help="Which dataset split to download: eval (24MB), train (~1GB), or all.")
    args = parser.parse_args()

    if args.split in ("eval", "all"):
        extract_and_prepare("eval")
    if args.split in ("train", "all"):
        extract_and_prepare("train")

if __name__ == "__main__":
    main()
