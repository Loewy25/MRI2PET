import os
from typing import Optional, Tuple
import torch


ROOT_DIR   = "/scratch/l.peiwang/kari_brainv11"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

RUN_NAME   = "MGDA_UB_dynamic_stable"
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
LAMBDA_GAN  = 0.5

# Keep fixed SUVR range as in your current code
DATA_RANGE  = 3.5

# Torch flags
torch.backends.cudnn.benchmark = True

# ---- Dynamic L1/SSIM grouping (Idea 1) ----
# Turn the controller on/off
DYN_GROUP = True

# Cosine smoothing across steps (B=1 so we smooth per-epoch)
COS_EMA_BETA = 0.85

# Hysteresis thresholds to avoid flip-flop
COS_HIGH = 0.8   # if cos >= COS_HIGH -> merge L1+SSIM
COS_LOW  = 0.70   # if cos <= COS_LOW  -> split L1 vs SSIM

# Don't switch more than once per this many epochs
MIN_HOLD_EPOCHS = 1

# Optional: if you later want Adam-aware whitening before cos/MGDA (leave False for now)
ADAM_AWARE_NORM = False
# ---- Multi-view MGDA-UB (Idea 2) ----
# Turn the multi-view fusion ON/OFF (set to False to revert to single-view)
MVIEWS_ENABLE = True

# Fixed fusion weights for (bottleneck, u3, output). Must sum ≈ 1.0 (we re-normalize in code).
# Good starting guess (can tune later):
MVIEWS_WEIGHTS = (0.20, 0.30, 0.50)
