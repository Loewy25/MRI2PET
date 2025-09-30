#!/usr/bin/env python3
import os
import torch

from mri2pet.config import (
    ROOT_DIR, OUT_DIR, RUN_NAME, OUT_RUN, CKPT_DIR, VOL_DIR,
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE
)
from mri2pet.data import build_loaders
from mri2pet.models import Generator, CondPatchDiscriminator3D
from mri2pet.train_eval import train_paggan, evaluate_and_save
from mri2pet.plotting import save_loss_curves, save_history_csv

# One import to read all toggles/hparams
import mri2pet.config as cfg

from mri2pet.pretrain_contrast import pretrain_encoders
from mri2pet.encoders import build_encoders_and_heads, load_teachers, freeze_teachers


if __name__ == "__main__":
    print(f"Data root: {ROOT_DIR}")
    print(f"Output root: {OUT_DIR}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run dir: {OUT_RUN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build loaders
    train_loader, val_loader, test_loader, N, ntr, nva, nte = build_loaders()
    print(f"Subjects: total={N}, train={ntr}, val={nva}, test={nte}")
    with torch.no_grad():
        sample = next(iter(train_loader))
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            mri0, pet0, meta0 = sample
            sid0 = meta0.get("sid", "NA") if isinstance(meta0, dict) else "NA"
        else:
            mri0, pet0 = sample
            sid0 = "NA"
        print(f"Sample tensor shapes: MRI {tuple(mri0.shape)}, PET {tuple(pet0.shape)}, SID {sid0}")

    # ---- Optional Step-1: Pretrain encoders (global InfoNCE) ----
    contrastive_mods = None
    if cfg.USE_CONTRAST:
        contrastive_mods = build_encoders_and_heads(in_ch_mri=1, in_ch_pet=1, proj_dim=cfg.CONTRAST_DIM)

        if cfg.PREALIGNMENT:
            print("[Contrast] Pretraining encoders (global InfoNCE)...")
            # Build a small-batch loader for pretraining
            pretrain_train_loader, _, _, _, _, _, _ = build_loaders(batch_size=4)
            pretrain_encoders(
                pretrain_train_loader, val_loader, device,
                proj_dim=cfg.CONTRAST_DIM, tau=cfg.CONTRAST_TAU, finetune_pct=cfg.FINETUNE_PCT,
                lr=cfg.LR_CONTRAST, epochs=cfg.PRETRAIN_EPOCHS, ckpt_path=cfg.CONTRAST_CKPT
            )
            print(f"[Contrast] Saved teachers -> {cfg.CONTRAST_CKPT}")

        if os.path.exists(cfg.CONTRAST_CKPT):
            load_teachers(contrastive_mods, cfg.CONTRAST_CKPT, map_location=device)
            print(f"[Contrast] Loaded teachers from {cfg.CONTRAST_CKPT}")

        # Move to training device & freeze for GAN stage
        for m in contrastive_mods.values():
            m.to(device)
        freeze_teachers(contrastive_mods)

    # ---- Contrast configuration passed into the trainer ----
    # (Note: ROI-* knobs are read directly from cfg inside train_paggan; included here for clarity/logging)
    contrast_cfg = {
        "use": cfg.USE_CONTRAST,
        "tau": cfg.CONTRAST_TAU,
        "use_patches": cfg.PATCH_CONTRAST,
        "patch_size": cfg.PATCH_SIZE,
        "patches_per_subj": cfg.PATCHES_PER_SUBJ,

        # ROI-aware contrast (for completeness; trainer will read cfg.*)
        "roi_enable": cfg.ROI_CONTRAST_ENABLE,
        "roi_patches_per_roi": cfg.ROI_PATCHES_PER_ROI,
        "roi_patch_size": cfg.ROI_PATCH_SIZE,
        "roi_weights": cfg.ROI_AGG_WEIGHTS,
        "roi_memory_enable": cfg.ROI_MEMORY_ENABLE,
        "roi_memory_len": cfg.ROI_MEMORY_LEN,
    }

    if cfg.ROI_CONTRAST_ENABLE:
        print("[ROI-Contrast] ENABLED")
        print(f"  patches/ROI   : {cfg.ROI_PATCHES_PER_ROI}")
        print(f"  patch size    : {cfg.ROI_PATCH_SIZE}")
        print(f"  ROI weights   : {cfg.ROI_AGG_WEIGHTS}")
        print(f"  memory queue  : {'ON' if cfg.ROI_MEMORY_ENABLE else 'OFF'} (len={cfg.ROI_MEMORY_LEN})")
    else:
        print("[ROI-Contrast] DISABLED (falling back to brain-mask patch contrast if USE_CONTRAST=True)")

    # Instantiate models
    G = Generator(in_ch=1, out_ch=1)
    D = CondPatchDiscriminator3D(in_ch=2)

    # Train
    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=EPOCHS, gamma=GAMMA, lambda_gan=LAMBDA_GAN, data_range=DATA_RANGE, verbose=True,
        contrastive_mods=contrastive_mods, contrast_cfg=contrast_cfg
    )

    # Save curves & CSV
    curves_path = os.path.join(OUT_RUN, "loss_curves.png")
    save_loss_curves(out["history"], curves_path)
    print(f"Saved loss curves to: {curves_path}")

    csv_path = os.path.join(OUT_RUN, "training_log.csv")
    save_history_csv(out["history"], csv_path)
    print(f"Saved training log CSV to: {csv_path}")

    # Evaluate + Save (single pass)
    metrics = evaluate_and_save(
        G, test_loader, device=device,
        out_dir=VOL_DIR, data_range=DATA_RANGE,
        mmd_voxels=2048
    )
    print("Test metrics:", metrics)

    metrics_txt = os.path.join(OUT_RUN, "test_metrics.txt")
    with open(metrics_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics to: {metrics_txt}")

