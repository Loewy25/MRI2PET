# MRI2PET

3D MRI-to-PET synthesis using a conditional PatchGAN with MGDA-based multi-objective training.

## Overview

This project trains a 3D generator to synthesize tau PET volumes from T1 MRI volumes.

- Entry point: `main.py`
- Core package: `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet`
- Tracking: Weights & Biases (`wandb`)

### Model

- Generator: pyramid-UNet style 3D network with skip-gates, channel attention, and spatial attention.
- Discriminator: conditional 3D PatchGAN on concatenated `[MRI, PET]`.

### Training objectives

The generator is optimized with three objectives, combined by MGDA-UB (3-way gradient balancing):

1. Global reconstruction:
`gamma * (L1(fake, pet) + (1 - SSIM3D(fake, pet)))`
2. Cortex ROI reconstruction (high-uptake weighted L1):
`L1(cortex) + ROI_HI_LAMBDA * L1(top-quantile uptake voxels in cortex)`
3. Adversarial objective (LSGAN style):
`0.5 * MSE(D([mri, fake]), 1)`

Discriminator uses:
`0.5 * (MSE(D([mri, pet]), 1) + MSE(D([mri, fake]), 0))`

## Data layout

`ROOT_DIR` should contain subject folders matching `*T807*`, `*t807*`, or `*1451*`.

Each subject folder must contain:

- `T1_masked.nii.gz`
- `PET_in_T1_masked.nii.gz`
- `aseg_brainmask.nii.gz`
- `mask_cortex.nii.gz`

The code currently expects masks and will error if brain/cortex masks are missing.

## Splits

Two split modes are supported:

1. Fold CSV mode (preferred): if `FOLD_CSV` exists  
path pattern: `ROOT_DIR/CV5_braak_strat/fold{FOLD_INDEX+1}.csv`
2. Random split fallback if fold CSV is not found

Fold CSV columns:

- `train`
- `validation`
- `test`
- `label` (required when `OVERSAMPLE_ENABLE=1`; interpreted for `train` rows)

## Configuration

Static defaults live in:
`/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/config.py`

Important: update these for your environment before running:

- `ROOT_DIR`
- `OUT_DIR`

Runtime knobs are controlled by environment variables:

- Run control: `RUN_NAME`, `FOLD_INDEX`
- Oversampling: `OVERSAMPLE_ENABLE`, `OVERSAMPLE_LABEL3_TARGET`, `OVERSAMPLE_MAX_WEIGHT`
- Augmentation: `AUG_ENABLE`, `AUG_PROB`, `AUG_FLIP_PROB`, `AUG_INTENSITY_PROB`, `AUG_NOISE_STD`, `AUG_SCALE_MIN`, `AUG_SCALE_MAX`, `AUG_SHIFT_MIN`, `AUG_SHIFT_MAX`
- High-uptake ROI loss: `ROI_HI_Q`, `ROI_HI_LAMBDA`, `ROI_HI_MIN_VOXELS`

`MY_BATCH_SCRIPT` already exports these variables and calls `python main.py`.

## Installation

Use a Python environment with CUDA-enabled PyTorch for training.

```bash
pip install torch numpy scipy nibabel matplotlib wandb
```

## Run

### Local

```bash
export RUN_NAME="exp_fold1"
export FOLD_INDEX=0
export OVERSAMPLE_ENABLE=1
export OVERSAMPLE_LABEL3_TARGET=0.65
export AUG_ENABLE=1
export ROI_HI_Q=0.85
export ROI_HI_LAMBDA=2.0
export ROI_HI_MIN_VOXELS=32
python main.py
```

### SLURM

```bash
sbatch /Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/MY_BATCH_SCRIPT
```

## Outputs

For a run named `RUN_NAME`, outputs are written to:
`OUT_DIR/RUN_NAME`

Expected artifacts:

- `checkpoints/best_G.pth`
- `checkpoints/best_D.pth`
- `loss_curves.png`
- `training_log.csv`
- `test_metrics.txt`
- `per_subject_metrics.csv`
- `test_metrics_summary.json`
- `volumes/<sid>/MRI.nii.gz`
- `volumes/<sid>/PET_gt.nii.gz`
- `volumes/<sid>/PET_fake.nii.gz`
- `volumes/<sid>/PET_abs_error.nii.gz`

## Evaluation metrics

Reported metrics (masked to brain region):

- `SSIM`
- `PSNR`
- `MSE`
- `MMD`

`test_metrics_summary.json` also includes std and 95% CI per metric.

## File map

- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/main.py`: orchestration (train, evaluate, log, save)
- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/config.py`: static defaults + env overrides
- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/data.py`: dataset, fold parsing, oversampling
- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/models.py`: generator and discriminator
- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/train_eval.py`: train loop (MGDA), evaluation, NIfTI saving
- `/Users/liupeiwang/Desktop/ANTIGRAVITY/MRI2PET/mri2pet/losses.py`: SSIM/L1/MSE/PSNR/MMD helpers
