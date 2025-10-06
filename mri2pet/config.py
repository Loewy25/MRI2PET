import os
from typing import Optional, Tuple
import torch


ROOT_DIR   = "/scratch/l.peiwang/kari_brainv11"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

RUN_NAME   = "MGDA_UB_mismat"
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
