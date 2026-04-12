#!/usr/bin/env python3
import os
import torch
import wandb

from mri2pet.config import (
    ROOT_DIR, OUT_DIR, RUN_NAME, OUT_RUN, CKPT_DIR, VOL_DIR,
    EPOCHS, GAMMA, DATA_RANGE, BATCH_SIZE, EVAL_BATCH_SIZE, RESIZE_TO, LR_G, LR_D,
    OVERSAMPLE_ENABLE, OVERSAMPLE_LABEL3_TARGET,
    AUG_ENABLE, AUG_PROB, AUG_FLIP_PROB,
    AUG_INTENSITY_PROB, AUG_NOISE_STD,
    AUG_SCALE_MIN, AUG_SCALE_MAX,
    AUG_SHIFT_MIN, AUG_SHIFT_MAX,
    ROI_HI_Q, ROI_HI_LAMBDA, ROI_HI_MIN_VOXELS,
    MODEL_VARIANT, BASE_PRETRAIN_CKPT,
    FREEZE_BASE_EPOCHS, BASE_LR_MULT,
    LAMBDA_STAGE_ORD, LAMBDA_BRAAK, LAMBDA_DELTA_OUT,
    CLINICAL_DIM, PROMPT_HIDDEN_DIM,
    USE_CHECKPOINT, AMP_ENABLE,
    LR_PLATEAU_PATIENCE, EARLY_STOP_PATIENCE, VAL_ROI_WEIGHT,
    MASK_GLOBAL_RECON, USE_GT_STAGE_HINT_TRAIN,
    ABLATION_STEP, USE_FLAIR, USE_CLINICAL, USE_STAGE_PROMPT,
)

from mri2pet.data import build_loaders
from mri2pet.config import FOLD_CSV
from mri2pet.data import build_loaders_from_fold_csv
from mri2pet.models import Generator, CondPatchDiscriminator3D, PromptResidualBraakGenerator
from mri2pet.train_eval import train_paggan, train_prompt_residual_braak, evaluate_and_save
from mri2pet.plotting import save_loss_curves, save_history_csv
from mri2pet.data import CLINICAL_FEATURE_NAMES


def init_wandb_run():
    service_wait = float(
        os.environ.get("WANDB_SERVICE_WAIT", os.environ.get("WANDB__SERVICE_WAIT", "300"))
    )
    settings = None
    try:
        settings = wandb.Settings(_service_wait=service_wait)
    except Exception:
        settings = None

    wandb_config = {
        "root_dir": ROOT_DIR,
        "run_name": RUN_NAME,
        "model_variant": MODEL_VARIANT,
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
    }

    if MODEL_VARIANT == "prompt_residual_braak":
        wandb_config.update({
            "freeze_base_epochs": FREEZE_BASE_EPOCHS,
            "base_lr_mult": BASE_LR_MULT,
            "lambda_stage_ord": LAMBDA_STAGE_ORD,
            "lambda_braak": LAMBDA_BRAAK,
            "lambda_delta_out": LAMBDA_DELTA_OUT,
            "clinical_dim": CLINICAL_DIM,
            "prompt_hidden_dim": PROMPT_HIDDEN_DIM,
            "use_checkpoint": USE_CHECKPOINT,
            "amp_enable": AMP_ENABLE,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "val_roi_weight": VAL_ROI_WEIGHT,
            "ablation_step": ABLATION_STEP,
            "use_flair": USE_FLAIR,
            "use_clinical": USE_CLINICAL,
            "use_stage_prompt": USE_STAGE_PROMPT,
        })

    try:
        return wandb.init(
            project="mri2pet",
            name=RUN_NAME,
            dir=OUT_RUN,
            settings=settings,
            config=wandb_config,
        )
    except Exception as exc:
        print(f"[WARN] wandb init failed: {exc}")
        print("[WARN] Continuing with wandb disabled.")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("MRI2PET Training Run")
    print("=" * 70)
    print(f"Data root:      {ROOT_DIR}")
    print(f"Output root:    {OUT_DIR}")
    print(f"Run name:       {RUN_NAME}")
    print(f"Run dir:        {OUT_RUN}")
    print(f"Model variant:  {MODEL_VARIANT}")
    print(f"Fold CSV:       {FOLD_CSV}")
    print(f"Base ckpt:      {BASE_PRETRAIN_CKPT or '(none, training from scratch)'}")
    print(f"Resize to:      {RESIZE_TO}")
    print(f"Batch size:     {BATCH_SIZE}  (eval: {EVAL_BATCH_SIZE})")
    print(f"Epochs:         {EPOCHS}")
    print(f"LR_G:           {LR_G}  LR_D: {LR_D}")
    print(f"AMP:            {AMP_ENABLE}  Checkpoint: {USE_CHECKPOINT}")
    if MODEL_VARIANT == "prompt_residual_braak":
        step_desc = {1: "base+residual", 2: "+FLAIR", 3: "+Clinical", 4: "+Stage/CORAL", 5: "+Braak(full)"}
        print(f"Ablation step:  {ABLATION_STEP} ({step_desc.get(ABLATION_STEP, '?')})")
        print(f"  USE_FLAIR={USE_FLAIR}  USE_CLINICAL={USE_CLINICAL}  USE_STAGE_PROMPT={USE_STAGE_PROMPT}")
        print(f"Freeze base:    {FREEZE_BASE_EPOCHS} epochs, then lr_mult={BASE_LR_MULT}")
        print(f"Lambda stage:   {LAMBDA_STAGE_ORD}  braak: {LAMBDA_BRAAK}  delta_out: {LAMBDA_DELTA_OUT}")
        print(f"Mask global:    {MASK_GLOBAL_RECON}  GT stage hint: {USE_GT_STAGE_HINT_TRAIN}")
        print(f"LR patience:    {LR_PLATEAU_PATIENCE}  Early stop: {EARLY_STOP_PATIENCE}")
    print(f"Val score:      val_recon + {VAL_ROI_WEIGHT} * val_roi")
    print(f"Augmentation:   {AUG_ENABLE} (prob={AUG_PROB})")
    print(f"Oversample:     {OVERSAMPLE_ENABLE} (target_p3={OVERSAMPLE_LABEL3_TARGET})")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    wandb_run = init_wandb_run()

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

    # ---- Data sanity log ----
    import numpy as np
    def _data_sanity_log(loader, split_name):
        """Collect stage_ord class counts and clinical stats from a data loader."""
        stage_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        clin_vecs, braak_vecs = [], []
        n_samples, n_missing = 0, 0
        shapes = set()
        for batch in loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
                continue
            _, _, meta = batch
            metas = [meta] if isinstance(meta, dict) else meta
            for m in metas:
                n_samples += 1
                so = m.get("stage_ord", None)
                if so is not None:
                    stage_counts[int(so)] = stage_counts.get(int(so), 0) + 1
                else:
                    n_missing += 1
                cv = m.get("clinical_vector", None)
                if cv is not None:
                    clin_vecs.append(cv if isinstance(cv, np.ndarray) else cv.numpy())
                bv = m.get("braak_values", None)
                if bv is not None:
                    braak_vecs.append(bv if isinstance(bv, np.ndarray) else bv.numpy())
                t1_shape = m.get("cur_shape", None)
                if t1_shape is not None:
                    shapes.add(tuple(t1_shape) if not isinstance(t1_shape, tuple) else t1_shape)
        print(f"  [{split_name}] n={n_samples}, stage_ord counts={stage_counts}, missing_meta={n_missing}")
        if shapes:
            print(f"  [{split_name}] volume shapes: {shapes}")
        if clin_vecs:
            clin = np.stack(clin_vecs)
            summary = {CLINICAL_FEATURE_NAMES[i]: f"mean={clin[:, i].mean():.3f} std={clin[:, i].std():.3f}"
                       for i in range(min(clin.shape[1], len(CLINICAL_FEATURE_NAMES)))}
            print(f"  [{split_name}] clinical (normalized): {summary}")
        if braak_vecs:
            bk = np.stack(braak_vecs)
            print(f"  [{split_name}] braak (normalized): mean={bk.mean(0)} std={bk.std(0)}")
        return stage_counts, clin_vecs, braak_vecs

    print("-" * 70)
    print("DATA SANITY CHECK")
    print("  (NOTE: train counts reflect sampled distribution if oversampling is on)")
    train_stage, train_clin, train_braak = _data_sanity_log(train_loader, "train(sampled)")
    val_stage, _, _ = _data_sanity_log(val_loader, "val")
    test_stage, _, _ = _data_sanity_log(test_loader, "test")

    if wandb_run is not None:
        sanity = {"data/train_n": ntr, "data/val_n": nva, "data/test_n": nte}
        for k, v in train_stage.items():
            sanity[f"data/train_stage_{k}"] = v
        for k, v in val_stage.items():
            sanity[f"data/val_stage_{k}"] = v
        for k, v in test_stage.items():
            sanity[f"data/test_stage_{k}"] = v
        if train_braak:
            bk = np.stack(train_braak)
            sanity["data/braak_norm_mean_12"] = float(bk[:, 0].mean())
            sanity["data/braak_norm_mean_34"] = float(bk[:, 1].mean())
            sanity["data/braak_norm_mean_56"] = float(bk[:, 2].mean())
        wandb.log(sanity, step=0)
    print("-" * 70)

    # Instantiate models
    is_prompt_residual = (MODEL_VARIANT == "prompt_residual_braak")

    if is_prompt_residual:
        G = PromptResidualBraakGenerator(
            in_ch=1, out_ch=1,
            use_checkpoint=USE_CHECKPOINT,
            clinical_dim=CLINICAL_DIM,
            prompt_z_dim=PROMPT_HIDDEN_DIM,
        )
        # Load base pretrain checkpoint if specified
        if BASE_PRETRAIN_CKPT and os.path.isfile(BASE_PRETRAIN_CKPT):
            print(f"Loading base pretrain checkpoint: {BASE_PRETRAIN_CKPT}")
            ckpt = torch.load(BASE_PRETRAIN_CKPT, map_location="cpu")
            G.base.load_state_dict(ckpt, strict=True)
            print("Base weights loaded successfully.")
        elif BASE_PRETRAIN_CKPT:
            raise FileNotFoundError(
                f"BASE_PRETRAIN_CKPT not found: {BASE_PRETRAIN_CKPT}"
            )
    else:
        G = Generator(in_ch=1, out_ch=1)

    D = CondPatchDiscriminator3D(in_ch=2)

    def _count_params(m):
        return sum(p.numel() for p in m.parameters())
    def _count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Generator params:     {_count_params(G):,} ({_count_trainable(G):,} trainable)")
    print(f"Discriminator params: {_count_params(D):,}")
    if is_prompt_residual:
        print(f"  Base params:        {_count_params(G.base):,}")
        print(f"  New branch params:  {_count_params(G) - _count_params(G.base):,}")
    print("=" * 70)

    if wandb_run is not None:
        wandb.watch(G, log="gradients", log_freq=50)
        wandb.watch(D, log="gradients", log_freq=50)

    # Train
    if is_prompt_residual:
        out = train_prompt_residual_braak(
            G, D, train_loader, val_loader,
            device=device, epochs=EPOCHS, gamma=GAMMA,
            data_range=DATA_RANGE,
            verbose=True,
            log_to_wandb=(wandb_run is not None),
        )
    else:
        out = train_paggan(
            G, D, train_loader, val_loader,
            device=device, epochs=EPOCHS, gamma=GAMMA,
            data_range=DATA_RANGE,
            verbose=True,
            log_to_wandb=(wandb_run is not None),
        )

    # Save curves & CSV
    curves_path = os.path.join(OUT_RUN, "loss_curves.png")
    save_loss_curves(out["history"], curves_path)
    print(f"Saved loss curves to: {curves_path}")

    csv_path = os.path.join(OUT_RUN, "training_log.csv")
    save_history_csv(out["history"], csv_path)
    print(f"Saved training log CSV to: {csv_path}")

    # Evaluate + Save — clear VOL_DIR first to avoid stale leftovers from prior runs
    import shutil
    if os.path.isdir(VOL_DIR):
        shutil.rmtree(VOL_DIR)
        print(f"Cleared old VOL_DIR: {VOL_DIR}")
    os.makedirs(VOL_DIR, exist_ok=True)

    metrics = evaluate_and_save(
        G, test_loader, device=device,
        out_dir=VOL_DIR, data_range=DATA_RANGE,
        mmd_voxels=2048,
        is_prompt_residual=is_prompt_residual,
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
