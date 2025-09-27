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
from mri2pet.config import (
    ROOT_DIR, OUT_DIR, RUN_NAME, OUT_RUN, CKPT_DIR, VOL_DIR,
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    # NEW:
    USE_CONTRAST, PREALIGNMENT, CONTRAST_DIM, CONTRAST_TAU, FINETUNE_PCT,
    PRETRAIN_EPOCHS, LR_CONTRAST, CONTRAST_CKPT,
    LAMBDA_CONTRAST, PATCH_CONTRAST, PATCH_SIZE, PATCHES_PER_SUBJ
)
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


    # ---- Optional Step-1: Pretrain encoders (global CLIP-style) ----
    contrastive_mods = None
    if USE_CONTRAST:
        # Build teacher slots
        contrastive_mods = build_encoders_and_heads(in_ch_mri=1, in_ch_pet=1, proj_dim=CONTRAST_DIM)
        # If requested, run pretraining then save
        if PREALIGNMENT:
            print("[Contrast] Pretraining encoders (global InfoNCE)...")
            # >>> build a *separate* loader with batch_size >= 4 for pretraining
            pretrain_train_loader, _, _, _, _, _, _ = build_loaders(batch_size=4)
            pretrain_encoders(
                 pretrain_train_loader, val_loader, device,
                 proj_dim=CONTRAST_DIM, tau=CONTRAST_TAU, finetune_pct=FINETUNE_PCT,
                 lr=LR_CONTRAST, epochs=PRETRAIN_EPOCHS, ckpt_path=CONTRAST_CKPT
             )
            print(f"[Contrast] Saved teachers -> {CONTRAST_CKPT}")
    
        # Load (either newly saved or an existing ckpt)
        if os.path.exists(CONTRAST_CKPT):
            load_teachers(contrastive_mods, CONTRAST_CKPT, map_location=device)
            print(f"[Contrast] Loaded teachers from {CONTRAST_CKPT}")

        # >>> ensure teachers on device <<<
        for m in contrastive_mods.values():
            m.to(device)

        # Freeze them for GAN stage
        freeze_teachers(contrastive_mods)
    
    contrast_cfg = {
        "use": USE_CONTRAST,
        "tau": CONTRAST_TAU,
        "lambda_contrast": LAMBDA_CONTRAST,
        "use_patches": PATCH_CONTRAST,
        "patch_size": PATCH_SIZE,
        "patches_per_subj": PATCHES_PER_SUBJ,
    }

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
