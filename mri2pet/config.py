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
LAMBDA_GAN  = 0.5


E_NORMAL1  = int(os.environ.get("E_NORMAL1", "0"))   # epochs of normal training before contrast
E_CONTRAST = int(os.environ.get("E_CONTRAST", "0"))  # number of contrast epochs

CONTRAST_EPS  = float(os.environ.get("CONTRAST_EPS", "0.005"))    # eps for cortex-only sign noise
CONTRAST_EPS0 = float(os.environ.get("CONTRAST_EPS0", "1e-6"))    # epsilon inside log

LAMBDA_CONTRAST_OUT = float(os.environ.get("LAMBDA_CONTRAST_OUT", "0.05"))   # lambda_out
LAMBDA_CONTRAST_CTX = float(os.environ.get("LAMBDA_CONTRAST_CTX", "0.005"))  # lambda_ctx

# ---- NEW: FAMO hyperparameters (env-overridable) ----
# β in the paper: learning rate for the task logits ξ update
FAMO_BETA  = float(os.environ.get("FAMO_BETA", "0.01"))

# γ in the paper: decay for ξ (paper default is 0.001)
FAMO_DECAY = float(os.environ.get("FAMO_DECAY", "0.001"))

# ε: keep losses strictly > 0 for log() and division
FAMO_EPS   = float(os.environ.get("FAMO_EPS", "1e-8"))


DATA_RANGE  = 3.5

torch.backends.cudnn.benchmark = True


SPLITS_DIR = os.path.join(ROOT_DIR, "CV5_braak_strat")

# ---- NEW: FOLD_INDEX also from env (0-based) ----
FOLD_INDEX = int(os.environ.get("FOLD_INDEX", "0"))   # "0".."4"

FOLD_CSV   = os.path.join(SPLITS_DIR, f"fold{FOLD_INDEX+1}.csv")
