#!/usr/bin/env python3
import os
import shutil

import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader, Subset

from mri2pet.config import (
    CLINICAL_DIM,
    FOLD_CSV,
    NUM_WORKERS,
    PIN_MEMORY,
    PROMPT_HIDDEN_DIM,
    RESAMPLE_BACK_TO_T1,
    ROOT_DIR,
)
from mri2pet.data import (
    KariAV1451Dataset,
    _collate_keep_meta,
    _compute_braak_stats,
    _compute_clinical_stats,
    _read_fold_csv_lists,
    _sid_for_item,
)
from mri2pet.models import ResidualSpatialPriorGenerator
from mri2pet.train_eval import _extract_masks, _extract_new_variant_inputs, _meta_as_list
from mri2pet.utils import _meta_unbatch, _safe_name, _save_nifti


INFER_CKPT = os.environ.get("INFER_CKPT", "").strip()
if not INFER_CKPT:
    raise RuntimeError("INFER_CKPT must be set")

RUN_DIR = os.path.dirname(os.path.dirname(INFER_CKPT))
OUT_DIR = os.environ.get("TRAIN_INFER_OUT_DIR", os.path.join(RUN_DIR, "train_inference"))
VOL_DIR = os.path.join(OUT_DIR, "volumes")
CLEAR_CUDA_CACHE_EVERY = max(1, int(os.environ.get("CLEAR_CUDA_CACHE_EVERY", "8")))


def build_train_loader(fold_csv_path: str):
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

    return DataLoader(
        Subset(ds, idx_train),
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        collate_fn=_collate_keep_meta,
    ), len(ds), len(idx_train)


def save_subject(subdir: str, meta: dict, mri_np: np.ndarray, pet_np: np.ndarray,
                 fake_np: np.ndarray, pet_base_np: np.ndarray, delta_np: np.ndarray) -> None:
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
        pet_base_np = nd_zoom(pet_base_np, zf, order=1)
        delta_np = nd_zoom(delta_np, zf, order=1)
        affine_to_use = meta.get("t1_affine", np.eye(4))
    else:
        resized_to = meta.get("resized_to", None)
        affine_to_use = meta.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

    _save_nifti(mri_np, affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
    _save_nifti(pet_np, affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
    _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
    _save_nifti(pet_base_np, affine_to_use, os.path.join(subdir, "PET_base.nii.gz"))
    _save_nifti(delta_np, affine_to_use, os.path.join(subdir, "PET_delta.nii.gz"))


if __name__ == "__main__":
    if not os.path.isfile(FOLD_CSV):
        raise FileNotFoundError(f"Fold CSV not found: {FOLD_CSV}")
    if not os.path.isfile(INFER_CKPT):
        raise FileNotFoundError(f"INFER_CKPT not found: {INFER_CKPT}")

    print("=" * 70)
    print("Prompt-Residual Training Inference")
    print("=" * 70)
    print(f"Fold CSV:    {FOLD_CSV}")
    print(f"Checkpoint:  {INFER_CKPT}")
    print(f"Output dir:  {OUT_DIR}")
    print(f"Data root:   {ROOT_DIR}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, n_total, n_train = build_train_loader(FOLD_CSV)
    print(f"Subjects: total={n_total}, train={n_train}")

    G = ResidualSpatialPriorGenerator(
        in_ch=1,
        out_ch=1,
        use_checkpoint=False,
        clinical_dim=CLINICAL_DIM,
        prompt_z_dim=PROMPT_HIDDEN_DIM,
    )
    print("Loading generator checkpoint...")
    G.load_state_dict(torch.load(INFER_CKPT, map_location="cpu"), strict=True)
    G.base.use_checkpoint = False
    G.to(device)
    G.eval()
    print(f"Base use_checkpoint: {G.base.use_checkpoint}")

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
        print(f"Cleared old output dir: {OUT_DIR}")
    os.makedirs(VOL_DIR, exist_ok=True)

    print("Starting inference...")
    with torch.inference_mode():
        for i, batch in enumerate(loader, start=1):
            mri, pet, meta = batch
            meta = _meta_unbatch(meta)
            sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
            subdir = os.path.join(VOL_DIR, sid)
            os.makedirs(subdir, exist_ok=True)

            mri_t = mri.to(device, non_blocking=True)
            pet_t = pet.to(device, non_blocking=True)
            mri5 = mri_t if mri_t.dim() == 5 else mri_t.unsqueeze(0)
            pet5 = pet_t if pet_t.dim() == 5 else pet_t.unsqueeze(0)
            metas_list = _meta_as_list(meta, 1)
            flair5, clinical, _ = _extract_new_variant_inputs(metas_list, device)
            brain5, cortex5 = _extract_masks(metas_list, device)

            pet_hat, aux = G(mri5, flair5, clinical, brain5, cortex5, return_aux=True)

            mri_np = mri5.squeeze(0).squeeze(0).cpu().numpy()
            pet_np = pet5.squeeze(0).squeeze(0).cpu().numpy()
            fake_np = pet_hat.squeeze(0).squeeze(0).float().cpu().numpy()
            pet_base_np = aux["pet_base"].squeeze(0).squeeze(0).float().cpu().numpy()
            delta_np = aux["delta_pet"].squeeze(0).squeeze(0).float().cpu().numpy()

            save_subject(subdir, meta, mri_np, pet_np, fake_np, pet_base_np, delta_np)
            print(f"[{i}/{n_train}] saved {sid}")

            del mri_t, pet_t, mri5, pet5, flair5, clinical, brain5, cortex5, pet_hat, aux
            if device.type == "cuda" and (i % CLEAR_CUDA_CACHE_EVERY == 0 or i == n_train):
                torch.cuda.empty_cache()

    print("-" * 70)
    print("Finished")
    print(f"Saved volumes to: {VOL_DIR}")
