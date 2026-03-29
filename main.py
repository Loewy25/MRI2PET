#!/usr/bin/env python3
import os
import torch
import wandb  # <-- add this

from mri2pet.config import (
    ROOT_DIR, OUT_DIR, RUN_NAME, OUT_RUN, VOL_DIR,
    EPOCHS, GAMMA, DATA_RANGE, BATCH_SIZE, RESIZE_TO,
    OVERSAMPLE_ENABLE, OVERSAMPLE_LABEL3_TARGET,
    AUG_ENABLE, AUG_PROB, AUG_FLIP_PROB,
    AUG_INTENSITY_PROB, AUG_NOISE_STD,
    AUG_SCALE_MIN, AUG_SCALE_MAX,
    AUG_SHIFT_MIN, AUG_SHIFT_MAX,
    ROI_HI_Q, ROI_HI_LAMBDA, ROI_HI_MIN_VOXELS,
    BRAAK_THRESHOLD, SPLITS_DIR, FOLD_INDEX, FOLD_CSV,
    MR_AMY_TAU_CDR_CSV, MR_COG_PET_CSV, DEMOGRAPHICS_CSV,
    LAMBDA_CON, LAMBDA_HIGH, LAMBDA_56, CONTRAST_TEMP,
)

from mri2pet.data import build_loaders_from_fold_csv
from mri2pet.models import Generator, CondPatchDiscriminator3D
from mri2pet.train_eval import train_paggan, evaluate_and_save
from mri2pet.plotting import save_loss_curves, save_history_csv


def init_wandb_run():
    service_wait = float(
        os.environ.get("WANDB_SERVICE_WAIT", os.environ.get("WANDB__SERVICE_WAIT", "300"))
    )
    settings = None
    try:
        settings = wandb.Settings(_service_wait=service_wait)
    except Exception:
        settings = None

    try:
        return wandb.init(
            project="mri2pet",
            name=RUN_NAME,
            dir=OUT_RUN,
            settings=settings,
            config={
                "root_dir": ROOT_DIR,
                "run_name": RUN_NAME,
                "epochs": EPOCHS,
                "gamma": GAMMA,
                "data_range": DATA_RANGE,
                "batch_size": BATCH_SIZE,
                "resize_to": RESIZE_TO,
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
                "roi_hi_q": ROI_HI_Q,
                "roi_hi_lambda": ROI_HI_LAMBDA,
                "roi_hi_min_voxels": ROI_HI_MIN_VOXELS,
                "splits_dir": SPLITS_DIR,
                "fold_index": FOLD_INDEX,
                "fold_csv": FOLD_CSV,
                "braak_threshold": BRAAK_THRESHOLD,
                "mr_amy_tau_cdr_csv": MR_AMY_TAU_CDR_CSV,
                "mr_cog_pet_csv": MR_COG_PET_CSV,
                "demographics_csv": DEMOGRAPHICS_CSV,
                "lambda_con": LAMBDA_CON,
                "lambda_high": LAMBDA_HIGH,
                "lambda_56": LAMBDA_56,
                "contrast_temp": CONTRAST_TEMP,
            },
        )
    except Exception as exc:
        print(f"[WARN] wandb init failed: {exc}")
        print("[WARN] Continuing with wandb disabled.")
        return None

if __name__ == "__main__":
    print(f"Data root: {ROOT_DIR}")
    print(f"Output root: {OUT_DIR}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run dir: {OUT_RUN}")
    print(f"Fold CSV: {FOLD_CSV}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb_run = init_wandb_run()

    # Build loaders
    if not os.path.isfile(FOLD_CSV):
        raise FileNotFoundError(f"Fold CSV not found: {FOLD_CSV}")
    train_loader, val_loader, test_loader, N, ntr, nva, nte = build_loaders_from_fold_csv(FOLD_CSV)
    print(f"Subjects: total={N}, train={ntr}, val={nva}, test={nte}")
    with torch.no_grad():
        sample = next(iter(train_loader))
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            mri0, pet0, meta0 = sample
            if isinstance(meta0, dict):
                sid0 = meta0.get("sid", "NA")
                flair_shape = tuple(meta0["flair"].shape)
                clinical_shape = tuple(meta0["clinical_vector"].shape)
            elif isinstance(meta0, list) and len(meta0) > 0:
                sid0 = meta0[0].get("sid", "NA")
                flair_shape = tuple(meta0[0]["flair"].shape)
                clinical_shape = tuple(meta0[0]["clinical_vector"].shape)
            else:
                sid0 = "NA"
                flair_shape = ()
                clinical_shape = ()
        else:
            mri0, pet0 = sample
            sid0 = "NA"
            flair_shape = ()
            clinical_shape = ()
        print(
            f"Sample tensor shapes: MRI {tuple(mri0.shape)}, PET {tuple(pet0.shape)}, "
            f"FLAIR {flair_shape}, clinical {clinical_shape}, SID {sid0}"
        )

    # Instantiate models
    G = Generator(in_ch=1, out_ch=1)
    D = CondPatchDiscriminator3D(in_ch=2)

    if wandb_run is not None:
        wandb.watch(G, log="gradients", log_freq=50)
        wandb.watch(D, log="gradients", log_freq=50)

    # Train
    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=EPOCHS, gamma=GAMMA,
        data_range=DATA_RANGE,
        verbose=True,
        log_to_wandb=(wandb_run is not None),
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

    if wandb_run is not None:
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

    if wandb_run is not None:
        wandb.finish()
