import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pathlib import Path

# --- Paths: edit to your layout ---
TRAIN_IMG_DIR  = Path("data/train/images")
TRAIN_MASK_DIR = Path("data/train/masks")          # sparse Roboflow export: 0/1/2/3
DENSE_MASK_DIR = Path("data/train/masks_dense")    # Phase B output (training target)
EVAL_IMG_DIR   = Path("data/eval/images")
EVAL_MASK_DIR  = Path("data/eval/masks")
QA_OUT_DIR     = Path("data/qa")
GMM_DIR        = Path("artifacts/gmm")
CKPT_DIR       = Path("artifacts/ckpt")

# --- Classes ---
SOIL, HEALTHY, EARLY_BLIGHT, LATE_BLIGHT, IGNORE = 0, 1, 2, 3, 255
NUM_CLASSES = 4

# --- Phase B thresholds ---
SOIL_THRESH = -15.0   # log-likelihood below this -> SOIL (0)
MARGIN      = 1.0     # log-likelihood gap needed between best and second class (e^1 ≈ 2.7x odds)
DARK_V      = 35      # HSV V below this -> deep shadow -> IGNORE (255)
BIC_MAX_COMPONENTS = 6

# --- Training ---
IMG_SIZE, BATCH_SIZE, EPOCHS = 512, 4, 60
ENCODER_LR, DECODER_LR = 5e-5, 3e-4
WEIGHT_DECAY, WARMUP_EPOCHS = 1e-4, 3