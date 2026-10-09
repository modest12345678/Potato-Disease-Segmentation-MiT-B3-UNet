# 🥔 UAV Potato Foliar Disease Dataset

This directory contains the downloaded UAV dataset images, polygonal annotations, and ground-truth semantic masks for the **MiT-B3 + U-Net** potato disease segmentation model.

---

## 📊 Dataset Overview

- **Platform:** DJI UAV (RGB High-Resolution Optical Sensor, 20 MP, 1-inch CMOS)
- **Flight Altitudes:** Multi-altitude regime: **2m, 7m, 10m, and 12m** Above Ground Level (AGL)
- **Ground Sampling Distance (GSD):**
  - **2m:** ~0.05 cm/pixel
  - **7m:** ~0.19 cm/pixel
  - **10m:** ~0.27 cm/pixel
  - **12m:** ~0.33 cm/pixel
- **Target Diseases:** Potato Early Blight (*Alternaria solani*) and Late Blight (*Phytophthora infestans*)

---

## 🏷️ Class Definitions & Pixel Encoding

Semantic segmentation masks are single-channel 8-bit PNG images where pixel values represent:

| Class ID | Class Name | Color Representation | Description |
|:---:|:---|:---:|:---|
| **0** | **Soil / Background** | Black `(0, 0, 0)` | Exposed soil, mulch, furrows, shadows |
| **1** | **Healthy Foliage** | Green `(0, 255, 0)` | Healthy potato canopy leaves |
| **2** | **Early Blight** | Orange `(255, 140, 0)` | Concentric ring spots (*Alternaria solani*) |
| **3** | **Late Blight** | Red `(255, 0, 0)` | Necrotic lesions (*Phytophthora infestans*) |
| **255** | **Ignore / Uncertain** | Cyan `(0, 220, 250)` | Deep shadow boundaries ($V < 35$) |

---

## 📁 Directory Structure

```text
dataset/
├── images/                   # Drone images (.jpg)
├── masks/                    # Ground-truth semantic masks (.png)
├── _annotations.coco.json    # COCO polygonal annotations
└── README.md                 # Dataset overview
```

---

## 📥 Dataset Download Links (from Roboflow)

If you are running the notebook in Google Colab, both splits are downloaded directly inside the notebook via:

```bash
# Evaluation Dataset (24 MB)
curl -L "https://app.roboflow.com/ds/VGsAVnyfVC?key=dU3gqpLHoj" > roboflow_eval.zip

# Full Training Dataset (~1.02 GB)
curl -L "https://app.roboflow.com/ds/GKJqAZBaee?key=zTTGuI7Hdv" > roboflow_train.zip
```
