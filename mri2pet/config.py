import os
from typing import Optional, Tuple
import torch

ROOT_DIR   = os.environ.get("ROOT_DIR", "/scratch/l.peiwang/kari_flair_all")
OUT_DIR    = os.environ.get("OUT_DIR", "/home/l.peiwang/MRI2PET")

# ---- NEW: allow override via environment variables ----
# default name if env not set
RUN_NAME = os.environ.get("RUN_NAME", "roi_recon_patch_lse_mgda_ub_1")

OUT_RUN    = os.path.join(OUT_DIR, RUN_NAME)
CKPT_DIR   = os.path.join(OUT_RUN, "checkpoints")
VOL_DIR    = os.path.join(OUT_RUN, "volumes")
os.makedirs(OUT_RUN, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VOL_DIR, exist_ok=True)

RESIZE_TO: Optional[Tuple[int,int,int]] = (128, 128, 128)
RESAMPLE_BACK_TO_T1 = False

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "1"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
NUM_WORKERS    = int(os.environ.get("NUM_WORKERS", "2"))
PIN_MEMORY     = True

EPOCHS      = 150
LR_G        = 1e-4
LR_D        = 4e-4
GAMMA       = 1.0

DATA_RANGE  = 3.5

torch.backends.cudnn.benchmark = True

SPLITS_DIR = os.environ.get("SPLITS_DIR", os.path.join(ROOT_DIR, "CV5_braak_strat"))

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

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return int(v) if v is not None else int(default)

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

# =========================
# ROI High-Uptake Loss
# =========================
# PET quantile threshold computed inside cortex mask (ground truth PET).
# Voxels above this threshold receive extra L1 emphasis.
ROI_HI_Q = _env_float("ROI_HI_Q", 0.85)
ROI_HI_LAMBDA = _env_float("ROI_HI_LAMBDA", 2.0)
ROI_HI_MIN_VOXELS = _env_int("ROI_HI_MIN_VOXELS", 32)

# =========================
# CSV Data Sources
# =========================
BRAAK_THRESHOLD = _env_float("BRAAK_THRESHOLD", 1.2)

MR_AMY_TAU_CDR_CSV = os.environ.get(
    "MR_AMY_TAU_CDR_CSV",
    "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv",
)
MR_COG_PET_CSV = os.environ.get(
    "MR_COG_PET_CSV",
    "/scratch/l.peiwang/MR_COG_PET_rsfMRI.csv",
)
DEMOGRAPHICS_CSV = os.environ.get(
    "DEMOGRAPHICS_CSV",
    "/scratch/l.peiwang/demographics.csv",
)

# =========================
# Model Variant
# =========================
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "residual_spatial_prior")
BASE_PRETRAIN_CKPT = os.environ.get("BASE_PRETRAIN_CKPT", "")

# =========================
# Residual Diffusion
# =========================
USE_BASELINE_CACHE = _env_bool("USE_BASELINE_CACHE", True)
BASELINE_CACHE_DIR = os.environ.get("BASELINE_CACHE_DIR", "")

DIFF_TIMESTEPS = _env_int("DIFF_TIMESTEPS", 1000)
DIFF_BETA_START = _env_float("DIFF_BETA_START", 1e-4)
DIFF_BETA_END = _env_float("DIFF_BETA_END", 2e-2)

DIFF_UNET_BASE_CH = _env_int("DIFF_UNET_BASE_CH", 32)
DIFF_EMB_DIM = _env_int("DIFF_EMB_DIM", 128)

DIFF_LR = _env_float("DIFF_LR", 2e-4)
DIFF_WEIGHT_DECAY = _env_float("DIFF_WEIGHT_DECAY", 1e-4)

DIFF_RESIDUAL_MEAN = _env_float("DIFF_RESIDUAL_MEAN", 0.0)
DIFF_RESIDUAL_STD = _env_float("DIFF_RESIDUAL_STD", 0.25)
DIFF_X0_CLIP = _env_float("DIFF_X0_CLIP", 8.0)

DIFF_LAMBDA_X0 = _env_float("DIFF_LAMBDA_X0", 1.0)
DIFF_LAMBDA_ROI = _env_float("DIFF_LAMBDA_ROI", 1.0)
DIFF_LAMBDA_BRAAK = _env_float("DIFF_LAMBDA_BRAAK", 0.25)

DIFF_VAL_SAMPLE_STEPS = _env_int("DIFF_VAL_SAMPLE_STEPS", 50)
DIFF_TEST_SAMPLE_STEPS = _env_int("DIFF_TEST_SAMPLE_STEPS", 100)
DIFF_NUM_SAMPLES = _env_int("DIFF_NUM_SAMPLES", 8)

# =========================
# Residual-Spatial-Prior settings
# =========================
FREEZE_BASE_EPOCHS = _env_int("FREEZE_BASE_EPOCHS", 10)
BASE_LR_MULT = _env_float("BASE_LR_MULT", 0.25)
DETACH_BASE_LATENT_FOR_PRIOR = _env_bool(
    "DETACH_BASE_LATENT_FOR_PRIOR",
    _env_bool("DETACH_BASE_LATENT_FOR_AUX", True),
)

CLINICAL_DIM = _env_int("CLINICAL_DIM", 10)
PROMPT_HIDDEN_DIM = _env_int("PROMPT_HIDDEN_DIM", 128)

# =========================
# Residual-side conditioning
# =========================
USE_FLAIR = _env_bool("USE_FLAIR", True)
USE_CLINICAL = _env_bool("USE_CLINICAL", True)
USE_BRAAK_HEAD = _env_bool("USE_BRAAK_HEAD", True)
USE_SPATIAL_PRIOR = _env_bool("USE_SPATIAL_PRIOR", True)

LAMBDA_BRAAK = _env_float("LAMBDA_BRAAK", 0.25)
LAMBDA_DELTA_SUP = _env_float("LAMBDA_DELTA_SUP", 0.5)
MASK_GLOBAL_RECON = _env_bool("MASK_GLOBAL_RECON", True)

SPATIAL_PRIOR_K = _env_int("SPATIAL_PRIOR_K", 4)
SPATIAL_PRIOR_LR_MULT = _env_float("SPATIAL_PRIOR_LR_MULT", 3.0)
PRIOR_GAIN_INIT_B = _env_float("PRIOR_GAIN_INIT_B", 0.10)
PRIOR_GAIN_INIT_X4 = _env_float("PRIOR_GAIN_INIT_X4", 0.10)
PRIOR_GAIN_INIT_X3 = _env_float("PRIOR_GAIN_INIT_X3", 0.05)

# =========================
# Validation Score
# =========================
VAL_ROI_WEIGHT = _env_float("VAL_ROI_WEIGHT", 0.02)

# =========================
# Validation Scheduling
# =========================
LR_PLATEAU_PATIENCE = _env_int("LR_PLATEAU_PATIENCE", 15)
EARLY_STOP_PATIENCE = _env_int("EARLY_STOP_PATIENCE", 40)

# =========================
# Memory Saving
# =========================
USE_CHECKPOINT = _env_bool("USE_CHECKPOINT", True)
AMP_ENABLE     = _env_bool("AMP_ENABLE", True)
