import os
from typing import Optional, Tuple
import torch

ROOT_DIR   = "/scratch/l.peiwang/kari_brainv33_top300"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

# ---- NEW: allow override via environment variables ----
# default name if env not set
RUN_NAME = os.environ.get("RUN_NAME", "roi_recon_patch_lse_mgda_ub_1")

OUT_RUN    = os.path.join(OUT_DIR, RUN_NAME)
CKPT_DIR   = os.path.join(OUT_RUN, "checkpoints")
VOL_DIR    = os.path.join(OUT_RUN, "volumes")
os.makedirs(OUT_RUN, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VOL_DIR, exist_ok=True)

RESIZE_TO: Optional[Tuple[int,int,int]] = (128,128,128)
RESAMPLE_BACK_TO_T1 = True

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
BATCH_SIZE     = 1
NUM_WORKERS    = 4
PIN_MEMORY     = True

EPOCHS      = 150
LR_G        = 1e-4
LR_D        = 4e-4
GAMMA       = 1.0

DATA_RANGE  = 3.5

torch.backends.cudnn.benchmark = True

SPLITS_DIR = os.path.join(ROOT_DIR, "CV5_braak_strat")

# ---- NEW: FOLD_INDEX also from env (0-based) ----
FOLD_INDEX = int(os.environ.get("FOLD_INDEX", "0"))   # "0".."4"

FOLD_CSV   = os.path.join(SPLITS_DIR, f"fold{FOLD_INDEX+1}.csv")

# =========================
# Imbalance / Oversampling
# =========================
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return float(v) if v is not None else float(default)

# Train-only oversampling (used only when fold CSV is used)
OVERSAMPLE_ENABLE = _env_bool("OVERSAMPLE_ENABLE", True)
# target fraction of label==3 in *training draws* (e.g. 0.35 => ~35%)
OVERSAMPLE_LABEL3_TARGET = _env_float("OVERSAMPLE_LABEL3_TARGET", 0.35)
# safety clamp for huge weights (avoid extreme sampling instability)
OVERSAMPLE_MAX_WEIGHT = _env_float("OVERSAMPLE_MAX_WEIGHT", 50.0)

# =========================
# Training Data Augmentation
# =========================
AUG_ENABLE = _env_bool("AUG_ENABLE", True)

# apply augmentation to a training batch with this probability
AUG_PROB = _env_float("AUG_PROB", 0.9)

# paired spatial flips (applied to MRI+PET+brain/cortex masks)
# per-axis probability
AUG_FLIP_PROB = _env_float("AUG_FLIP_PROB", 0.5)

# MRI-only intensity augmentation probability (PET target unchanged)
AUG_INTENSITY_PROB = _env_float("AUG_INTENSITY_PROB", 0.8)

# MRI intensity jitter parameters (applied inside brain mask)
AUG_NOISE_STD = _env_float("AUG_NOISE_STD", 0.05)      # Gaussian noise sigma
AUG_SCALE_MIN = _env_float("AUG_SCALE_MIN", 0.9)       # multiplicative scale
AUG_SCALE_MAX = _env_float("AUG_SCALE_MAX", 1.1)
AUG_SHIFT_MIN = _env_float("AUG_SHIFT_MIN", -0.1)      # additive shift
AUG_SHIFT_MAX = _env_float("AUG_SHIFT_MAX", 0.1)

