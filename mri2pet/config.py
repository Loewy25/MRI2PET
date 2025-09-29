import os
from typing import Optional, Tuple
import torch


ROOT_DIR   = "/scratch/l.peiwang/kari_brainv11"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

RUN_NAME   = "MGDA_UB_c_stable_contra_70_batch1_hierachy"
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

EPOCHS      = 140
LR_G        = 1e-4
LR_D        = 4e-4
GAMMA       = 1.0
LAMBDA_GAN  = 0.5

# Keep fixed SUVR range as in your current code
DATA_RANGE  = 3.5

# Torch flags
torch.backends.cudnn.benchmark = True

# ---- Multi-view MGDA-UB (Idea 2) ----
# Turn the multi-view fusion ON/OFF (set to False to revert to single-view)
MVIEWS_ENABLE = True

# Fixed fusion weights for (bottleneck, u3, output). Must sum ≈ 1.0 (we re-normalize in code).
# Good starting guess (can tune later):
MVIEWS_WEIGHTS = (0.20, 0.30, 0.50)

# ===== Contrastive pre-alignment & auxiliary loss =====
# Master switches
USE_CONTRAST: bool = True              # turn on contrastive aux loss in GAN stage
PREALIGNMENT: bool = True              # run Step-1 pretraining (global InfoNCE) before GAN

# Teacher / embedding space
CONTRAST_DIM: int = 128                # projection head output dim
CONTRAST_TAU: float = 0.10             # InfoNCE temperature
FINETUNE_PCT: float = 0.30             # during pretrain: % of encoder to unfreeze (top-most)

# Optim for pretraining
PRETRAIN_EPOCHS: int = 60
LR_CONTRAST: float = 1e-4
CONTRAST_CKPT: str = os.path.join(CKPT_DIR, "contrast_teachers.pt")

# Aux loss weight in GAN stage (if you don't use MGDA for it)
LAMBDA_CONTRAST: float = 0.20

# Patch-level contrast used ONLY in GAN stage
PATCH_CONTRAST: bool = True
PATCH_SIZE: Tuple[int,int,int] = (32, 32, 32)
PATCHES_PER_SUBJ: int = 32             # with B=1, this gives you 16 in-batch negatives


# --- Hierarchical MGDA (Plan-4) ---
HMGDA_ENABLE: bool = True

# Optional floors to prevent a group from being zeroed at Level-2
HMGDA_FLOOR_RECON: float = 0.00
HMGDA_FLOOR_GAN:   float = 0.00
HMGDA_FLOOR_CONTR: float = 0.00   # small floor so Contrast isn't silenced

# Optional floors inside each group at Level-1 (usually 0.0 is fine)
HMGDA_FLOOR_L1:    float = 0.00
HMGDA_FLOOR_SSIM:  float = 0.00
HMGDA_FLOOR_M2PH:  float = 0.00
HMGDA_FLOOR_PH2P:  float = 0.00

# Turn OFF Plan-3 "contrast outside MGDA" in this branch
CONTRAST_OUTSIDE_MGDA: bool = False

# Logging (keep the earlier flag; add group-level switch)
PRINT_GRAD_COSINES: bool = True
PRINT_GROUP_COSINES: bool = True
