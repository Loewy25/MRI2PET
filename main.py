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
    USE_BASELINE_CACHE, BASELINE_CACHE_DIR,
    DIFF_TIMESTEPS, DIFF_BETA_START, DIFF_BETA_END,
    DIFF_UNET_BASE_CH, DIFF_EMB_DIM, DIFF_LR, DIFF_WEIGHT_DECAY,
    DIFF_RESIDUAL_MEAN, DIFF_RESIDUAL_STD, DIFF_X0_CLIP,
    DIFF_LAMBDA_X0, DIFF_LAMBDA_ROI, DIFF_LAMBDA_BRAAK,
    DIFF_VAL_SAMPLE_STEPS, DIFF_TEST_SAMPLE_STEPS, DIFF_NUM_SAMPLES,
    CDRM_BASIS_DIR, CDRM_COEFF_CSV, CDRM_K_CAL, CDRM_K_DIS,
    CDRM_LR, CDRM_WEIGHT_DECAY, CDRM_LAMBDA_ROI,
    CDRM_LAMBDA_C_COEF, CDRM_LAMBDA_A_COEF,
    CDRM_A_STAGE_WEIGHT_2, CDRM_A_STAGE_WEIGHT_3,
    CDRM_T1_FREEZE, CDRM_STAT_DIM,
    CDRM_DISEASE_TARGET_MODE, CDRM_CONTRAST_LAMBDA, CDRM_CONTRAST_REF,
    FREEZE_BASE_EPOCHS, BASE_LR_MULT, DETACH_BASE_LATENT_FOR_PRIOR,
    LAMBDA_BRAAK, LAMBDA_DELTA_SUP,
    CLINICAL_DIM, PROMPT_HIDDEN_DIM,
    USE_FLAIR, USE_CLINICAL, USE_BRAAK_HEAD, USE_SPATIAL_PRIOR,
    SPATIAL_PRIOR_K, SPATIAL_PRIOR_LR_MULT,
    PRIOR_GAIN_INIT_B, PRIOR_GAIN_INIT_X4, PRIOR_GAIN_INIT_X3,
    USE_CHECKPOINT, AMP_ENABLE,
    LR_PLATEAU_PATIENCE, EARLY_STOP_PATIENCE, VAL_ROI_WEIGHT,
    MASK_GLOBAL_RECON,
    EVAL_ONLY, EVAL_CKPT,
)

from mri2pet.data import build_loaders
from mri2pet.config import FOLD_CSV
from mri2pet.data import build_loaders_from_fold_csv
from mri2pet.models import (
    Generator,
    CondPatchDiscriminator3D,
    ResidualSpatialPriorGenerator,
    ResidualDiffusionUNet3D,
    ResidualManifoldNet,
)
from mri2pet.train_eval import (
    train_paggan,
    train_residual_spatial_prior,
    train_residual_diffusion,
    train_residual_manifold,
    evaluate_and_save,
    evaluate_and_save_diffusion,
    evaluate_and_save_residual_manifold,
)
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
        "eval_only": EVAL_ONLY,
        "eval_ckpt": EVAL_CKPT,
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

    if MODEL_VARIANT in {"prompt_residual_braak", "residual_spatial_prior"}:
        wandb_config.update({
            "freeze_base_epochs": FREEZE_BASE_EPOCHS,
            "base_lr_mult": BASE_LR_MULT,
            "detach_base_latent_for_prior": DETACH_BASE_LATENT_FOR_PRIOR,
            "lambda_braak": LAMBDA_BRAAK,
            "lambda_delta_sup": LAMBDA_DELTA_SUP,
            "clinical_dim": CLINICAL_DIM,
            "prompt_hidden_dim": PROMPT_HIDDEN_DIM,
            "use_checkpoint": USE_CHECKPOINT,
            "amp_enable": AMP_ENABLE,
            "lr_plateau_patience": LR_PLATEAU_PATIENCE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "val_roi_weight": VAL_ROI_WEIGHT,
            "use_flair": USE_FLAIR,
            "use_clinical": USE_CLINICAL,
            "use_braak_head": USE_BRAAK_HEAD,
            "use_spatial_prior": USE_SPATIAL_PRIOR,
            "spatial_prior_k": SPATIAL_PRIOR_K,
            "spatial_prior_lr_mult": SPATIAL_PRIOR_LR_MULT,
            "prior_gain_init_b": PRIOR_GAIN_INIT_B,
            "prior_gain_init_x4": PRIOR_GAIN_INIT_X4,
            "prior_gain_init_x3": PRIOR_GAIN_INIT_X3,
        })
    if MODEL_VARIANT == "residual_diffusion":
        wandb_config.update({
            "use_baseline_cache": USE_BASELINE_CACHE,
            "baseline_cache_dir": BASELINE_CACHE_DIR,
            "diff_timesteps": DIFF_TIMESTEPS,
            "diff_beta_start": DIFF_BETA_START,
            "diff_beta_end": DIFF_BETA_END,
            "diff_unet_base_ch": DIFF_UNET_BASE_CH,
            "diff_emb_dim": DIFF_EMB_DIM,
            "diff_lr": DIFF_LR,
            "diff_weight_decay": DIFF_WEIGHT_DECAY,
            "diff_residual_mean": DIFF_RESIDUAL_MEAN,
            "diff_residual_std": DIFF_RESIDUAL_STD,
            "diff_x0_clip": DIFF_X0_CLIP,
            "diff_lambda_x0": DIFF_LAMBDA_X0,
            "diff_lambda_roi": DIFF_LAMBDA_ROI,
            "diff_lambda_braak": DIFF_LAMBDA_BRAAK,
            "diff_val_sample_steps": DIFF_VAL_SAMPLE_STEPS,
            "diff_test_sample_steps": DIFF_TEST_SAMPLE_STEPS,
            "diff_num_samples": DIFF_NUM_SAMPLES,
        })
    if MODEL_VARIANT == "residual_manifold":
        wandb_config.update({
            "use_baseline_cache": USE_BASELINE_CACHE,
            "baseline_cache_dir": BASELINE_CACHE_DIR,
            "base_pretrain_ckpt": BASE_PRETRAIN_CKPT,
            "cdrm_basis_dir": CDRM_BASIS_DIR,
            "cdrm_coeff_csv": CDRM_COEFF_CSV,
            "cdrm_k_cal": CDRM_K_CAL,
            "cdrm_k_dis": CDRM_K_DIS,
            "cdrm_lr": CDRM_LR,
            "cdrm_weight_decay": CDRM_WEIGHT_DECAY,
            "cdrm_lambda_roi": CDRM_LAMBDA_ROI,
            "cdrm_lambda_c_coef": CDRM_LAMBDA_C_COEF,
            "cdrm_lambda_a_coef": CDRM_LAMBDA_A_COEF,
            "cdrm_a_stage_weight_2": CDRM_A_STAGE_WEIGHT_2,
            "cdrm_a_stage_weight_3": CDRM_A_STAGE_WEIGHT_3,
            "cdrm_t1_freeze": CDRM_T1_FREEZE,
            "cdrm_stat_dim": CDRM_STAT_DIM,
            "cdrm_disease_target_mode": CDRM_DISEASE_TARGET_MODE,
            "cdrm_contrast_lambda": CDRM_CONTRAST_LAMBDA,
            "cdrm_contrast_ref": CDRM_CONTRAST_REF,
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


def log_wandb_output_files(wandb_run, run_name: str, paths):
    if wandb_run is None:
        return
    safe_name = run_name.replace("/", "_")
    artifact = wandb.Artifact(f"{safe_name}-outputs", type="run_outputs")
    added = []
    for path in paths:
        if path and os.path.isfile(path):
            artifact.add_file(path, name=os.path.basename(path))
            added.append(os.path.basename(path))
    if not added:
        return
    try:
        wandb_run.log_artifact(artifact)
        print(f"Logged W&B output artifact with files: {', '.join(added)}")
    except Exception as exc:
        print(f"[WARN] Failed to log W&B output artifact: {exc}")


def load_generator_weights_into_t1_backbone(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    for key in ("state_dict", "model_state_dict", "G", "generator"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    base_prefixed = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    if base_prefixed:
        state = base_prefixed
    model.t1_backbone.load_state_dict(state, strict=True)


def load_model_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt
    for key in ("state_dict", "model_state_dict", "G", "generator"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


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
    print(f"Eval only:      {EVAL_ONLY}  ckpt={EVAL_CKPT or '(auto from run dir)'}")
    print(f"LR_G:           {LR_G}  LR_D: {LR_D}")
    print(f"AMP:            {AMP_ENABLE}  Checkpoint: {USE_CHECKPOINT}")
    if MODEL_VARIANT == "residual_diffusion":
        print("Residual diffusion model:")
        print(f"  USE_BASELINE_CACHE={USE_BASELINE_CACHE}  BASELINE_CACHE_DIR={BASELINE_CACHE_DIR}")
        print(f"  timesteps={DIFF_TIMESTEPS} beta=[{DIFF_BETA_START}, {DIFF_BETA_END}]")
        print(f"  base_ch={DIFF_UNET_BASE_CH} emb_dim={DIFF_EMB_DIM} lr={DIFF_LR} wd={DIFF_WEIGHT_DECAY}")
        print(f"  residual mean/std={DIFF_RESIDUAL_MEAN}/{DIFF_RESIDUAL_STD}  x0_clip={DIFF_X0_CLIP}")
        print(
            f"  lambdas x0={DIFF_LAMBDA_X0} roi={DIFF_LAMBDA_ROI} braak={DIFF_LAMBDA_BRAAK} "
            f"val_steps={DIFF_VAL_SAMPLE_STEPS} test_steps={DIFF_TEST_SAMPLE_STEPS} K={DIFF_NUM_SAMPLES}"
        )
        if not (USE_BASELINE_CACHE and BASELINE_CACHE_DIR):
            raise RuntimeError("MODEL_VARIANT=residual_diffusion requires USE_BASELINE_CACHE=1 and BASELINE_CACHE_DIR")
    if MODEL_VARIANT == "residual_manifold":
        print("Residual manifold model:")
        print(f"  USE_BASELINE_CACHE={USE_BASELINE_CACHE}  BASELINE_CACHE_DIR={BASELINE_CACHE_DIR}")
        print(f"  basis_dir={CDRM_BASIS_DIR}  coeff_csv={CDRM_COEFF_CSV}")
        print(f"  K_cal={CDRM_K_CAL} K_dis={CDRM_K_DIS} lr={CDRM_LR} wd={CDRM_WEIGHT_DECAY}")
        print(
            f"  lambdas roi={CDRM_LAMBDA_ROI} c_coef={CDRM_LAMBDA_C_COEF} "
            f"a_coef={CDRM_LAMBDA_A_COEF} "
            f"t1_freeze={CDRM_T1_FREEZE} stat_dim={CDRM_STAT_DIM}"
        )
        print(
            f"  disease coefficient stage weights: "
            f"stage0/1=1.0 stage2={CDRM_A_STAGE_WEIGHT_2} stage3={CDRM_A_STAGE_WEIGHT_3}"
        )
        print(
            f"  basis mode={CDRM_DISEASE_TARGET_MODE} "
            f"contrast_lambda={CDRM_CONTRAST_LAMBDA} contrast_ref={CDRM_CONTRAST_REF}"
        )
        if not (USE_BASELINE_CACHE and BASELINE_CACHE_DIR):
            raise RuntimeError("MODEL_VARIANT=residual_manifold requires USE_BASELINE_CACHE=1 and BASELINE_CACHE_DIR")
        if not (CDRM_BASIS_DIR and os.path.isdir(CDRM_BASIS_DIR)):
            raise RuntimeError("MODEL_VARIANT=residual_manifold requires CDRM_BASIS_DIR")
        if not (CDRM_COEFF_CSV and os.path.isfile(CDRM_COEFF_CSV)):
            raise RuntimeError("MODEL_VARIANT=residual_manifold requires CDRM_COEFF_CSV")
        if not (BASE_PRETRAIN_CKPT and os.path.isfile(BASE_PRETRAIN_CKPT)):
            raise RuntimeError("MODEL_VARIANT=residual_manifold requires BASE_PRETRAIN_CKPT for T1 encoder")
    if MODEL_VARIANT in {"prompt_residual_braak", "residual_spatial_prior"}:
        print("Residual spatial prior model:")
        print(
            f"  USE_FLAIR={USE_FLAIR}  USE_CLINICAL={USE_CLINICAL}  "
            f"USE_BRAAK_HEAD={USE_BRAAK_HEAD}  USE_SPATIAL_PRIOR={USE_SPATIAL_PRIOR}"
        )
        print(f"Freeze base:    {FREEZE_BASE_EPOCHS} epochs, then lr_mult={BASE_LR_MULT}")
        print(
            f"Detach z_t1:    {DETACH_BASE_LATENT_FOR_PRIOR}  braak: {LAMBDA_BRAAK}  "
            f"delta_sup: {LAMBDA_DELTA_SUP}"
        )
        print(
            f"Spatial prior:  K={SPATIAL_PRIOR_K}  lr_mult={SPATIAL_PRIOR_LR_MULT}  "
            f"gain_b={PRIOR_GAIN_INIT_B}  gain_x4={PRIOR_GAIN_INIT_X4}  gain_x3={PRIOR_GAIN_INIT_X3}"
        )
        print(f"Mask global:    {MASK_GLOBAL_RECON}")
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
    is_prompt_residual = MODEL_VARIANT in {"prompt_residual_braak", "residual_spatial_prior"}
    is_residual_diffusion = (MODEL_VARIANT == "residual_diffusion")
    is_residual_manifold = (MODEL_VARIANT == "residual_manifold")

    if is_residual_diffusion:
        G = ResidualDiffusionUNet3D(
            in_ch=6,
            base_ch=DIFF_UNET_BASE_CH,
            emb_dim=DIFF_EMB_DIM,
            clinical_dim=CLINICAL_DIM,
            use_checkpoint=USE_CHECKPOINT,
        )
        D = None
        print(f"Diffusion branch will use cached PET_base from: {BASELINE_CACHE_DIR}")
    elif is_residual_manifold:
        G = ResidualManifoldNet(
            basis_dir=CDRM_BASIS_DIR,
            clinical_dim=CLINICAL_DIM,
            stat_dim=CDRM_STAT_DIM,
            t1_freeze=CDRM_T1_FREEZE,
            use_checkpoint=USE_CHECKPOINT,
        )
        if G.k_cal != CDRM_K_CAL or G.k_dis != CDRM_K_DIS:
            raise RuntimeError(
                f"Basis dimensions K_cal/K_dis={G.k_cal}/{G.k_dis} do not match "
                f"CDRM_K_CAL/CDRM_K_DIS={CDRM_K_CAL}/{CDRM_K_DIS}"
            )
        print(f"Loading T1 backbone checkpoint: {BASE_PRETRAIN_CKPT}")
        load_generator_weights_into_t1_backbone(G, BASE_PRETRAIN_CKPT)
        print("T1 backbone weights loaded successfully.")
        D = None
        print(f"Residual manifold branch will use cached PET_base from: {BASELINE_CACHE_DIR}")
    elif is_prompt_residual:
        G = ResidualSpatialPriorGenerator(
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

    if is_prompt_residual:
        D = CondPatchDiscriminator3D(in_ch=2)

    def _count_params(m):
        return sum(p.numel() for p in m.parameters())
    def _count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Generator params:     {_count_params(G):,} ({_count_trainable(G):,} trainable)")
    if D is not None:
        print(f"Discriminator params: {_count_params(D):,}")
    else:
        print("Discriminator params: (none for this branch)")
    if is_prompt_residual:
        print(f"  Base params:        {_count_params(G.base):,}")
        print(f"  New branch params:  {_count_params(G) - _count_params(G.base):,}")
    if is_residual_manifold:
        print(f"  T1 backbone params: {_count_params(G.t1_backbone):,}")
        print(f"  K_cal/K_dis:        {G.k_cal}/{G.k_dis}")
    print("=" * 70)

    if wandb_run is not None:
        wandb.watch(G, log="gradients", log_freq=50)
        if D is not None:
            wandb.watch(D, log="gradients", log_freq=50)

    # Train, unless this is a checkpoint-only evaluation rerun.
    if EVAL_ONLY:
        if EVAL_CKPT:
            eval_ckpt = EVAL_CKPT
        elif is_residual_diffusion:
            eval_ckpt = os.path.join(CKPT_DIR, "best_diffusion.pth")
        elif is_residual_manifold:
            eval_ckpt = os.path.join(CKPT_DIR, "best_residual_manifold.pth")
        else:
            eval_ckpt = os.path.join(CKPT_DIR, "best_G.pth")
        if not os.path.isfile(eval_ckpt):
            raise FileNotFoundError(f"EVAL_ONLY checkpoint not found: {eval_ckpt}")
        print(f"Evaluation-only mode: loading checkpoint {eval_ckpt}")
        load_model_weights(G, eval_ckpt)
        print("Evaluation-only checkpoint loaded successfully.")
    else:
        if is_residual_diffusion:
            out = train_residual_diffusion(
                G, train_loader, val_loader,
                device=device, epochs=EPOCHS,
                data_range=DATA_RANGE,
                verbose=True,
                log_to_wandb=(wandb_run is not None),
            )
        elif is_residual_manifold:
            out = train_residual_manifold(
                G, train_loader, val_loader,
                device=device, epochs=EPOCHS,
                data_range=DATA_RANGE,
                verbose=True,
                log_to_wandb=(wandb_run is not None),
            )
        elif is_prompt_residual:
            out = train_residual_spatial_prior(
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

    if is_residual_diffusion:
        metrics = evaluate_and_save_diffusion(
            G, test_loader, device=device,
            out_dir=VOL_DIR, data_range=DATA_RANGE,
            mmd_voxels=2048,
            sample_steps=DIFF_TEST_SAMPLE_STEPS,
            num_samples=DIFF_NUM_SAMPLES,
        )
    elif is_residual_manifold:
        metrics = evaluate_and_save_residual_manifold(
            G, test_loader, device=device,
            out_dir=VOL_DIR, data_range=DATA_RANGE,
            mmd_voxels=2048,
        )
    else:
        metrics = evaluate_and_save(
            G, test_loader, device=device,
            out_dir=VOL_DIR, data_range=DATA_RANGE,
            mmd_voxels=2048,
            is_prompt_residual=is_prompt_residual,
        )
    print("Test metrics:", metrics)

    if wandb_run is not None:
        test_log = {
            "test/SSIM": metrics["SSIM"],
            "test/PSNR": metrics["PSNR"],
            "test/MSE": metrics["MSE"],
            "test/MMD": metrics["MMD"],
        }
        for key in [
            "uncertainty_mean_brain",
            "uncertainty_mean_cortex",
            "mean_abs_delta",
            "base_vs_fake_recon_improvement",
            "base_vs_fake_roi_improvement",
            "mean_abs_res_cal",
            "mean_abs_res_dis",
        ]:
            if key in metrics:
                test_log[f"test/{key}"] = metrics[key]
        wandb.log(test_log)

    metrics_txt = os.path.join(OUT_RUN, "test_metrics.txt")
    with open(metrics_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics to: {metrics_txt}")

    if wandb_run is not None:
        output_files = [
            curves_path,
            csv_path,
            os.path.join(OUT_RUN, "per_subject_metrics.csv"),
            os.path.join(OUT_RUN, "test_metrics_summary.json"),
        ]
        if is_residual_diffusion:
            output_files.extend([
                os.path.join(OUT_RUN, "per_subject_diffusion.csv"),
                os.path.join(CKPT_DIR, "best_diffusion.pth"),
            ])
        elif is_residual_manifold:
            output_files.extend([
                os.path.join(OUT_RUN, "per_subject_manifold.csv"),
                os.path.join(OUT_RUN, "coefficients.csv"),
                os.path.join(CKPT_DIR, "best_residual_manifold.pth"),
                os.path.join(CDRM_BASIS_DIR, "basis_manifest.json"),
                os.path.join(CDRM_BASIS_DIR, "oracle_metrics.csv"),
                os.path.join(CDRM_BASIS_DIR, "coeff_targets.csv"),
            ])
        else:
            output_files.extend([
                os.path.join(OUT_RUN, "per_subject_aux.csv"),
                os.path.join(CKPT_DIR, "best_G.pth"),
                os.path.join(CKPT_DIR, "best_D.pth"),
            ])
        log_wandb_output_files(wandb_run, RUN_NAME, output_files)
        wandb.finish()
