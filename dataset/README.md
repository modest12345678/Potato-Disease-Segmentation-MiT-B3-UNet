# 🥔 UAV Potato Foliar Disease Dataset

This directory contains the dataset components, ground-truth annotations, and semantic masks for remote sensing potato disease segmentation (Early Blight and Late Blight).

---

## 📊 Dataset Specifications

- **Platform:** DJI Phantom 4 Pro / Mavic 3 Enterprise UAV
- **Sensor:** RGB High-Resolution Optical Sensor (20 MP, 1-inch CMOS)
- **Flight Altitudes:** Multi-altitude regime: **2m, 7m, 10m, and 12m** Above Ground Level (AGL)
- **Ground Sampling Distance (GSD):**
  - **2m:** ~0.05 cm/pixel (ultra-fine leaf lesion diagnostics)
  - **7m:** ~0.19 cm/pixel (standard canopy-level scouting)
  - **10m:** ~0.27 cm/pixel (wide-area canopy scouting)
  - **12m:** ~0.33 cm/pixel (rapid broad-acre reconnaissance)
- **Primary Disease Target:** Potato Late Blight (*Phytophthora infestans*) and Early Blight (*Alternaria solani*)

---

## 🏷️ Class Definitions & Pixel Encoding

Semantic segmentation masks are single-channel 8-bit PNG images where pixel intensities correspond to:

| Class ID | Class Name | Color Representation | Description |
|:---:|:---|:---:|:---|
| **0** | **Soil / Background** | Black `(0, 0, 0)` | Exposed soil, mulch, furrows, shadows, non-vegetative area |
| **1** | **Healthy Foliage** | Green `(0, 255, 0)` | Asymptomatic potato canopy leaves and stems |
| **2** | **Early Blight** | Orange `(255, 140, 0)` | Target-board concentric ring lesions (*Alternaria solani*) |
| **3** | **Late Blight** | Red `(255, 0, 0)` | Water-soaked dark necrotic lesions (*Phytophthora infestans*) |
| **255** | **Ignore / Uncertain** | Cyan `(0, 220, 250)` | Deep shadows, high-uncertainty foliage boundaries |

---

## 📁 Directory Structure

```text
dataset/
├── raw_coco/                      # Offline COCO JSON annotations & high-res drone images
│   ├── _annotations.coco.json    # Standard COCO polygonal annotation file (293 annotations)
│   ├── DJI_0031_JPG...jpg        # Multi-angle field images
│   └── README.roboflow.txt       # Export provenance metadata
├── masks/                         # Converted 8-bit single-channel semantic ground-truth masks
│   ├── DJI_0031_JPG...png
│   └── ...
└── download_full_dataset.py       # Auto-downloader for full ~1GB training set from Roboflow
```

---

## ⚡ How to Download the Complete Dataset

To pull the full training dataset (~1.02 GB) directly from Roboflow into your local environment:

```bash
# Download Evaluation dataset (24 MB)
python dataset/download_full_dataset.py --split eval

# Download Full Training dataset (~1 GB)
python dataset/download_full_dataset.py --split train

# Or download everything at once:
python dataset/download_full_dataset.py --split all
```

The script will automatically unzip the files into `data/train_raw` and `data/eval_raw` and generate pixel masks into `data/train/masks` and `data/eval/masks`.
