#!/usr/bin/env python3
import os
import torch
import wandb  # <-- add this

from mri2pet.config import (
    ROOT_DIR, OUT_DIR, RUN_NAME, OUT_RUN, CKPT_DIR, VOL_DIR,
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    OVERSAMPLE_ENABLE, OVERSAMPLE_LABEL3_TARGET,
    AUG_ENABLE, AUG_PROB, AUG_FLIP_PROB,
    AUG_INTENSITY_PROB, AUG_NOISE_STD,
    AUG_SCALE_MIN, AUG_SCALE_MAX,
    AUG_SHIFT_MIN, AUG_SHIFT_MAX,
)

from mri2pet.data import build_loaders
from mri2pet.config import FOLD_CSV
from mri2pet.data import build_loaders_from_fold_csv
from mri2pet.models import Generator, CondPatchDiscriminator3D
from mri2pet.train_eval import train_paggan, evaluate_and_save
from mri2pet.plotting import save_loss_curves, save_history_csv

if __name__ == "__main__":
    print(f"Data root: {ROOT_DIR}")
    print(f"Output root: {OUT_DIR}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run dir: {OUT_RUN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- NEW: wandb init ----
    wandb.init(
        project="mri2pet",      
        name=RUN_NAME,          
        dir=OUT_RUN,            
        config={
            "root_dir": ROOT_DIR,
            "run_name": RUN_NAME,
            "epochs": EPOCHS,
            "gamma": GAMMA,
            "lambda_gan": LAMBDA_GAN,
            "data_range": DATA_RANGE,
            "batch_size": 1,
            "resize_to": (128, 128, 128),
                        "oversample_enable": OVERSAMPLE_ENABLE,
            "oversample_label3_target": OVERSAMPLE_LABEL3_TARGET,
            "aug_enable": AUG_ENABLE,
            "aug_prob": AUG_PROB,
            "aug_flip_prob": AUG_FLIP_PROB,
            "aug_intensity_prob": AUG_INTENSITY_PROB,
            "aug_noise_std": AUG_NOISE_STD,
            "aug_scale_min": AUG_SCALE_MIN,
            "aug_scale_max": AUG_SCALE_MAX,
            "aug_shift_min": AUG_SHIFT_MIN,
            "aug_shift_max": AUG_SHIFT_MAX,

        },
    )

    # Build loaders
    if os.path.isfile(FOLD_CSV):
        print(f"Using fold CSV: {FOLD_CSV}")
        train_loader, val_loader, test_loader, N, ntr, nva, nte = build_loaders_from_fold_csv(FOLD_CSV)
    else:
        print("No fold CSV found; falling back to random split.")
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

    # Instantiate models
    G = Generator(in_ch=1, out_ch=1)
    D = CondPatchDiscriminator3D(in_ch=2)

    wandb.watch(G, log="gradients", log_freq=50)
    wandb.watch(D, log="gradients", log_freq=50)

    # Train
    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=EPOCHS, gamma=GAMMA,
        lambda_gan=LAMBDA_GAN, data_range=DATA_RANGE,
        verbose=True,
        log_to_wandb=True,              
    )

    # Save curves & CSV (still useful)
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

    # ---- NEW: log test metrics to wandb ----
    wandb.log({
        "test/SSIM": metrics["SSIM"],
        "test/PSNR": metrics["PSNR"],
        "test/MSE": metrics["MSE"],
        "test/MMD": metrics["MMD"],
    })

    metrics_txt = os.path.join(OUT_RUN, "test_metrics.txt")
    with open(metrics_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics to: {metrics_txt}")

    wandb.finish()
