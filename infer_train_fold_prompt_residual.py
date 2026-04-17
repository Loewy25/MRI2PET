#!/usr/bin/env python3
import os
import shutil

import torch
from torch.utils.data import DataLoader, Subset

from mri2pet.config import (
    ROOT_DIR,
    RUN_NAME,
    OUT_RUN,
    CKPT_DIR,
    FOLD_CSV,
    CLINICAL_DIM,
    PROMPT_HIDDEN_DIM,
    USE_CHECKPOINT,
    DATA_RANGE,
    NUM_WORKERS,
    PIN_MEMORY,
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
from mri2pet.train_eval import evaluate_and_save


INFER_CKPT = os.environ.get("INFER_CKPT", os.path.join(CKPT_DIR, "best_G.pth"))
TRAIN_INFER_ROOT = os.environ.get("TRAIN_INFER_ROOT", os.path.join(OUT_RUN, "train_inference"))
TRAIN_INFER_VOL_DIR = os.path.join(TRAIN_INFER_ROOT, "volumes")


def build_train_eval_loader_from_fold_csv(fold_csv_path: str):
    train_sids, _val_sids, _test_sids, train_sid_to_label = _read_fold_csv_lists(fold_csv_path)

    ds = KariAV1451Dataset(root_dir=ROOT_DIR, sid_to_label=train_sid_to_label)
    n_total = len(ds)

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
    return loader, n_total, len(idx_train)


if __name__ == "__main__":
    print("=" * 70)
    print("Prompt-Residual Training-Set Inference")
    print("=" * 70)
    print(f"Run name:       {RUN_NAME}")
    print(f"Data root:      {ROOT_DIR}")
    print(f"Fold CSV:       {FOLD_CSV}")
    print(f"Checkpoint:     {INFER_CKPT}")
    print(f"Output root:    {TRAIN_INFER_ROOT}")
    print("=" * 70)

    if not os.path.isfile(FOLD_CSV):
        raise FileNotFoundError(f"Fold CSV not found: {FOLD_CSV}")
    if not os.path.isfile(INFER_CKPT):
        raise FileNotFoundError(f"Inference checkpoint not found: {INFER_CKPT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, n_total, n_train = build_train_eval_loader_from_fold_csv(FOLD_CSV)
    print(f"Subjects: total={n_total}, train_infer={n_train}")

    G = ResidualSpatialPriorGenerator(
        in_ch=1,
        out_ch=1,
        use_checkpoint=USE_CHECKPOINT,
        clinical_dim=CLINICAL_DIM,
        prompt_z_dim=PROMPT_HIDDEN_DIM,
    )
    print(f"Loading generator checkpoint: {INFER_CKPT}")
    ckpt = torch.load(INFER_CKPT, map_location="cpu")
    G.load_state_dict(ckpt, strict=True)
    G.to(device)
    G.eval()

    if os.path.isdir(TRAIN_INFER_ROOT):
        shutil.rmtree(TRAIN_INFER_ROOT)
        print(f"Cleared old output root: {TRAIN_INFER_ROOT}")
    os.makedirs(TRAIN_INFER_VOL_DIR, exist_ok=True)

    metrics = evaluate_and_save(
        G,
        loader,
        device=device,
        out_dir=TRAIN_INFER_VOL_DIR,
        data_range=DATA_RANGE,
        mmd_voxels=2048,
        is_prompt_residual=True,
    )
    print("Training-set inference metrics:", metrics)
