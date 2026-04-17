#!/usr/bin/env python3
import csv
import json
import os
import shutil

import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader, Subset

from mri2pet.config import (
    AMP_ENABLE,
    CLINICAL_DIM,
    DATA_RANGE,
    FOLD_CSV,
    NUM_WORKERS,
    PIN_MEMORY,
    PROMPT_HIDDEN_DIM,
    RESAMPLE_BACK_TO_T1,
    ROOT_DIR,
    USE_CHECKPOINT,
)
from mri2pet.data import (
    KariAV1451Dataset,
    _collate_keep_meta,
    _compute_braak_stats,
    _compute_clinical_stats,
    _read_fold_csv_lists,
    _sid_for_item,
)
from mri2pet.losses import masked_mse, masked_psnr, mmd_gaussian, ssim3d_masked
from mri2pet.models import ResidualSpatialPriorGenerator
from mri2pet.train_eval import _extract_masks, _extract_new_variant_inputs, _meta_as_list
from mri2pet.utils import _meta_unbatch, _safe_name, _save_nifti


INFER_CKPT = os.environ.get("INFER_CKPT", "").strip()
if not INFER_CKPT:
    raise RuntimeError("INFER_CKPT must be set to the finished prompt-residual best_G.pth")

RUN_DIR = os.path.dirname(os.path.dirname(INFER_CKPT))
OUT_DIR = os.environ.get("TRAIN_INFER_OUT_DIR", os.path.join(RUN_DIR, "train_inference"))
VOL_DIR = os.path.join(OUT_DIR, "volumes")
CLEAR_CUDA_CACHE_EVERY = max(1, int(os.environ.get("CLEAR_CUDA_CACHE_EVERY", "8")))


def build_train_loader_from_fold_csv(fold_csv_path: str):
    train_sids, _val_sids, _test_sids, train_sid_to_label = _read_fold_csv_lists(fold_csv_path)

    ds = KariAV1451Dataset(root_dir=ROOT_DIR, sid_to_label=train_sid_to_label)
    sid_list = [_sid_for_item(item) for item in ds.items]
    sid_to_index = {sid: i for i, sid in enumerate(sid_list)}

    idx_train = []
    missing = []
    for sid in train_sids:
        if sid in sid_to_index:
            idx_train.append(sid_to_index[sid])
        else:
            missing.append(sid)
    if missing:
        raise RuntimeError(
            f"{len(missing)} training subjects from {fold_csv_path} not found on disk. "
            f"Examples: {missing[:8]}"
        )

    idx_train = sorted(idx_train)
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))
    braak_mean, braak_std = _compute_braak_stats(ds, idx_train)
    ds.set_braak_stats(braak_mean, braak_std)

    train_set = Subset(ds, idx_train)
    loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=_collate_keep_meta,
    )
    return loader, len(ds), len(idx_train)


def save_subject_volumes(subdir: str, meta: dict, mri_np: np.ndarray, pet_np: np.ndarray,
                         fake_np: np.ndarray, err_np: np.ndarray,
                         pet_base_np: np.ndarray, delta_np: np.ndarray) -> None:
    cur_shape = tuple(mri_np.shape)
    orig_shape = tuple(meta.get("orig_shape", cur_shape))

    if RESAMPLE_BACK_TO_T1 and tuple(orig_shape) != tuple(cur_shape):
        zf = (
            float(orig_shape[0]) / float(cur_shape[0]),
            float(orig_shape[1]) / float(cur_shape[1]),
            float(orig_shape[2]) / float(cur_shape[2]),
        )
        mri_np = nd_zoom(mri_np, zf, order=1)
        pet_np = nd_zoom(pet_np, zf, order=1)
        fake_np = nd_zoom(fake_np, zf, order=1)
        err_np = nd_zoom(err_np, zf, order=1)
        pet_base_np = nd_zoom(pet_base_np, zf, order=1)
        delta_np = nd_zoom(delta_np, zf, order=1)
        affine_to_use = meta.get("t1_affine", np.eye(4))
    else:
        resized_to = meta.get("resized_to", None)
        affine_to_use = meta.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

    _save_nifti(mri_np, affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
    _save_nifti(pet_np, affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
    _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
    _save_nifti(err_np, affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))
    _save_nifti(pet_base_np, affine_to_use, os.path.join(subdir, "PET_base.nii.gz"))
    _save_nifti(delta_np, affine_to_use, os.path.join(subdir, "PET_delta.nii.gz"))


if __name__ == "__main__":
    if not os.path.isfile(FOLD_CSV):
        raise FileNotFoundError(f"Fold CSV not found: {FOLD_CSV}")
    if not os.path.isfile(INFER_CKPT):
        raise FileNotFoundError(f"INFER_CKPT not found: {INFER_CKPT}")

    print("=" * 70)
    print("Prompt-Residual Train Inference")
    print("=" * 70)
    print(f"Fold CSV:       {FOLD_CSV}")
    print(f"Checkpoint:     {INFER_CKPT}")
    print(f"Output dir:     {OUT_DIR}")
    print(f"Data root:      {ROOT_DIR}")
    print(f"Resample back:  {RESAMPLE_BACK_TO_T1}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, n_total, n_train = build_train_loader_from_fold_csv(FOLD_CSV)
    print(f"Subjects: total={n_total}, train={n_train}")

    G = ResidualSpatialPriorGenerator(
        in_ch=1,
        out_ch=1,
        use_checkpoint=USE_CHECKPOINT,
        clinical_dim=CLINICAL_DIM,
        prompt_z_dim=PROMPT_HIDDEN_DIM,
    )
    print("Loading generator weights...")
    ckpt = torch.load(INFER_CKPT, map_location="cpu")
    G.load_state_dict(ckpt, strict=True)
    G.to(device)
    G.eval()

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        print(f"Cleared old output dir: {OUT_DIR}")
    os.makedirs(VOL_DIR, exist_ok=True)

    use_amp = AMP_ENABLE and device.type == "cuda"
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    rows = []
    aux_rows = []
    print("Starting training-set inference...")

    with torch.inference_mode():
        for i, batch in enumerate(loader, start=1):
            mri, pet, meta = batch
            meta = _meta_unbatch(meta)
            sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
            subdir = os.path.join(VOL_DIR, sid)
            os.makedirs(subdir, exist_ok=True)

            mri5 = mri.to(device, non_blocking=True)
            pet5 = pet.to(device, non_blocking=True)
            metas_list = _meta_as_list(meta, 1)
            flair5, clinical, _braak_gt = _extract_new_variant_inputs(metas_list, device)
            brain5, cortex5 = _extract_masks(metas_list, device)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                pet_hat, aux = G(mri5, flair5, clinical, brain5, cortex5, return_aux=True)

            fake_f = pet_hat.float()
            pet_f = pet5.float()
            brain_mask_np = meta.get("brain_mask", None)
            if brain_mask_np is not None:
                brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
            else:
                brain = (pet_f > 0).float()

            ssim_val = ssim3d_masked(fake_f, pet_f, brain, data_range=DATA_RANGE).item()
            psnr_val = masked_psnr(fake_f, pet_f, brain, data_range=DATA_RANGE)
            mse_val = masked_mse(fake_f, pet_f, brain).item()
            mmd_val = mmd_gaussian(pet_f, fake_f, num_voxels=2048, mask=brain)

            rows.append({
                "sid": sid,
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "MSE": mse_val,
                "MMD": mmd_val,
            })

            braak_pred_np = aux["braak_pred"].squeeze(0).float().cpu().numpy()
            braak_raw_gt = meta.get("braak_values_raw", None)
            prior_stats = aux["prior_stats"]
            aux_rows.append({
                "sid": sid,
                "braak_pred_12": float(braak_pred_np[0]),
                "braak_pred_34": float(braak_pred_np[1]),
                "braak_pred_56": float(braak_pred_np[2]),
                "braak_raw_gt_12": float(braak_raw_gt[0]) if braak_raw_gt is not None else "",
                "braak_raw_gt_34": float(braak_raw_gt[1]) if braak_raw_gt is not None else "",
                "braak_raw_gt_56": float(braak_raw_gt[2]) if braak_raw_gt is not None else "",
                "prior_in_cortex_mag": float(prior_stats["in_cortex_mag"].item()),
                "prior_out_cortex_mag": float(prior_stats["out_cortex_mag"].item()),
                "prior_in_out_ratio": float(
                    prior_stats["in_cortex_mag"].item() / (prior_stats["out_cortex_mag"].item() + 1e-12)
                ),
                "prior_router_entropy": float(prior_stats["router_entropy"].item()),
                "prior_router_top1_mean": float(prior_stats["router_top1_mean"].item()),
            })

            mri_np = mri5.squeeze(0).squeeze(0).cpu().numpy()
            pet_np = pet5.squeeze(0).squeeze(0).cpu().numpy()
            fake_np = pet_hat.squeeze(0).squeeze(0).float().cpu().numpy()
            err_np = np.abs(fake_np - pet_np)
            pet_base_np = aux["pet_base"].squeeze(0).squeeze(0).float().cpu().numpy()
            delta_np = aux["delta_pet"].squeeze(0).squeeze(0).float().cpu().numpy()

            save_subject_volumes(subdir, meta, mri_np, pet_np, fake_np, err_np, pet_base_np, delta_np)

            print(
                f"[{i}/{n_train}] {sid}  "
                f"SSIM={ssim_val:.4f}  PSNR={psnr_val:.3f}  MSE={mse_val:.6f}  MMD={mmd_val:.6f}"
            )

            del mri5, pet5, flair5, clinical, brain5, cortex5, pet_hat, aux, fake_f, pet_f, brain
            if device.type == "cuda" and (i % CLEAR_CUDA_CACHE_EVERY == 0 or i == n_train):
                torch.cuda.empty_cache()

    metrics_csv = os.path.join(OUT_DIR, "per_subject_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "SSIM", "PSNR", "MSE", "MMD"])
        for row in rows:
            w.writerow([row["sid"], row["SSIM"], row["PSNR"], row["MSE"], row["MMD"]])

    aux_csv = os.path.join(OUT_DIR, "per_subject_aux.csv")
    aux_cols = [
        "sid",
        "braak_pred_12", "braak_pred_34", "braak_pred_56",
        "braak_raw_gt_12", "braak_raw_gt_34", "braak_raw_gt_56",
        "prior_in_cortex_mag", "prior_out_cortex_mag", "prior_in_out_ratio",
        "prior_router_entropy", "prior_router_top1_mean",
    ]
    with open(aux_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(aux_cols)
        for row in aux_rows:
            w.writerow([row.get(c, "") for c in aux_cols])

    def _mean(vals, key):
        return float(np.mean([row[key] for row in vals])) if vals else float("nan")

    summary = {
        "N": len(rows),
        "SSIM": _mean(rows, "SSIM"),
        "PSNR": _mean(rows, "PSNR"),
        "MSE": _mean(rows, "MSE"),
        "MMD": _mean(rows, "MMD"),
        "per_subject_csv": metrics_csv,
        "per_subject_aux_csv": aux_csv,
        "volumes_dir": VOL_DIR,
    }
    summary_json = os.path.join(OUT_DIR, "metrics_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 70)
    print("Finished training-set inference")
    print(f"Per-subject metrics: {metrics_csv}")
    print(f"Per-subject aux:     {aux_csv}")
    print(f"Summary JSON:        {summary_json}")
    print(f"Volumes dir:         {VOL_DIR}")
    print(
        f"Mean metrics: SSIM={summary['SSIM']:.4f}  "
        f"PSNR={summary['PSNR']:.3f}  MSE={summary['MSE']:.6f}  MMD={summary['MMD']:.6f}"
    )
