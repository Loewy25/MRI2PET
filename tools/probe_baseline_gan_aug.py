#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OVERSAMPLE_ENABLE"] = "0"
os.environ["USE_BASELINE_CACHE"] = "0"

from mri2pet.config import BASE_PRETRAIN_CKPT, FOLD_CSV, RESIZE_TO, ROOT_DIR
from mri2pet.data import (
    KariAV1451Dataset,
    _compute_braak_stats,
    _compute_clinical_stats,
    _read_fold_csv_lists,
)
from mri2pet.models import Generator
from mri2pet.utils import _resized_affine_for_scipy_zoom, _safe_name, _save_nifti


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run frozen baseline GAN on transformed training samples for overfit/registration probes."
    )
    p.add_argument("--checkpoint", default=BASE_PRETRAIN_CKPT, help="Baseline Generator checkpoint.")
    p.add_argument("--root-dir", default=ROOT_DIR)
    p.add_argument("--fold-csv", default=FOLD_CSV)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--max-subjects", type=int, default=0, help="0 means all subjects in split.")
    p.add_argument("--lr-axis", type=int, default=0, help="Voxel axis for left/right flip. NIfTI x is usually axis 0.")
    p.add_argument("--rot-deg", type=float, default=5.0)
    p.add_argument("--shift-vox", type=float, default=4.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-volumes", type=int, default=1)
    p.add_argument("--save-volume-limit", type=int, default=30, help="Save NIfTI volumes for first N subjects only. 0 means no limit.")
    return p.parse_args()


def _load_generator(path: str, device: torch.device) -> Generator:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {path}")
    G = Generator(in_ch=1, out_ch=1, use_checkpoint=False).to(device)
    state = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model_state_dict", "G", "generator"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    if any(k.startswith("base.") for k in state.keys()):
        state = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    G.load_state_dict(state, strict=True)
    G.eval()
    return G


def _choose_sids(fold_csv: str, split: str):
    train_sids, val_sids, test_sids, train_sid_to_label = _read_fold_csv_lists(fold_csv)
    if split == "train":
        return train_sids, train_sid_to_label, train_sids
    if split == "val":
        return val_sids, train_sid_to_label, train_sids
    return test_sids, train_sid_to_label, train_sids


def _affine_for_meta(meta: Dict) -> np.ndarray:
    cur_shape = tuple(meta.get("cur_shape"))
    orig_shape = tuple(meta.get("orig_shape", cur_shape))
    affine = meta.get("model_affine", None)
    if affine is not None:
        return affine
    if tuple(orig_shape) == tuple(cur_shape):
        return meta.get("t1_affine", np.eye(4))
    return _resized_affine_for_scipy_zoom(meta.get("t1_affine", np.eye(4)), orig_shape, cur_shape)


def _masked_mae(fake: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(fake[mask] - true[mask])))


def _masked_mse(fake: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = fake[mask] - true[mask]
    return float(np.mean(diff * diff))


def _flip(vol: np.ndarray, axis: int, order: int) -> np.ndarray:
    return np.flip(vol, axis=axis).copy()


def _rot(vol: np.ndarray, angle: float, axes: Tuple[int, int], order: int) -> np.ndarray:
    return nd_rotate(vol, angle=float(angle), axes=axes, reshape=False, order=order, mode="constant", cval=0.0).astype(np.float32)


def _shift(vol: np.ndarray, shift_vec, order: int) -> np.ndarray:
    return nd_shift(vol, shift=shift_vec, order=order, mode="constant", cval=0.0).astype(np.float32)


def _apply_transform(vol: np.ndarray, name: str, args, order: int = 1) -> np.ndarray:
    lr_axis = int(args.lr_axis)
    shift = float(args.shift_vox)
    rot = float(args.rot_deg)
    if name == "identity":
        return vol.copy()
    if name == "flip_lr":
        return _flip(vol, lr_axis, order)
    if name == "rot_axial_pos":
        return _rot(vol, rot, axes=(0, 1), order=order)
    if name == "rot_axial_neg":
        return _rot(vol, -rot, axes=(0, 1), order=order)
    if name == "shift_lr_pos":
        vec = [0.0, 0.0, 0.0]
        vec[lr_axis] = shift
        return _shift(vol, vec, order=order)
    if name == "shift_lr_neg":
        vec = [0.0, 0.0, 0.0]
        vec[lr_axis] = -shift
        return _shift(vol, vec, order=order)
    raise ValueError(f"Unknown transform: {name}")


def _run_gan(G: Generator, mri_np: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(mri_np[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        y = G(x).squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    return y


def main():
    args = _parse_args()
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")
    if args.lr_axis not in (0, 1, 2):
        raise ValueError("--lr-axis must be 0, 1, or 2")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print("=" * 70, flush=True)
    print("[gan-probe] Baseline GAN transform probe", flush=True)
    print(f"[gan-probe] root={args.root_dir}", flush=True)
    print(f"[gan-probe] fold_csv={args.fold_csv}", flush=True)
    print(f"[gan-probe] checkpoint={args.checkpoint}", flush=True)
    print(f"[gan-probe] out_dir={args.out_dir}", flush=True)
    print(
        f"[gan-probe] split={args.split} resize_to={RESIZE_TO} "
        f"lr_axis={args.lr_axis} rot_deg={args.rot_deg} shift_vox={args.shift_vox} device={device}",
        flush=True,
    )
    print("=" * 70, flush=True)

    split_sids, train_sid_to_label, train_sids = _choose_sids(args.fold_csv, args.split)
    if args.max_subjects and args.max_subjects > 0:
        split_sids = split_sids[: int(args.max_subjects)]
        print(f"[gan-probe] limiting to first {len(split_sids)} subjects", flush=True)

    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO, sid_to_label=train_sid_to_label)
    sid_to_idx = {item["sid"]: i for i, item in enumerate(ds.items)}
    missing = [sid for sid in split_sids if sid not in sid_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} split subjects not found. Examples: {missing[:8]}")
    train_missing = [sid for sid in train_sids if sid not in sid_to_idx]
    if train_missing:
        raise RuntimeError(f"{len(train_missing)} train subjects not found. Examples: {train_missing[:8]}")
    idx_train = [sid_to_idx[sid] for sid in train_sids]
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))
    braak_mean, braak_std = _compute_braak_stats(ds, idx_train)
    ds.set_braak_stats(braak_mean, braak_std)

    G = _load_generator(args.checkpoint, device)
    transforms = ["identity", "flip_lr", "rot_axial_pos", "rot_axial_neg", "shift_lr_pos", "shift_lr_neg"]
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    rows = []
    t0 = time.time()

    for n, sid in enumerate(split_sids, start=1):
        subj_t0 = time.time()
        mri_t, pet_t, meta = ds[sid_to_idx[sid]]
        mri_np = mri_t.squeeze(0).numpy().astype(np.float32)
        pet_np = pet_t.squeeze(0).numpy().astype(np.float32)
        brain_np = meta["brain_mask"].astype(np.float32)
        cortex_np = meta["cortex_mask"].astype(np.float32)
        affine = _affine_for_meta(meta)

        identity_fake = None
        sid_safe = _safe_name(sid)
        for tname in transforms:
            item_t0 = time.time()
            mri_x = _apply_transform(mri_np, tname, args, order=1)
            pet_x = _apply_transform(pet_np, tname, args, order=1)
            brain_x = _apply_transform(brain_np, tname, args, order=0) > 0.5
            cortex_x = (_apply_transform(cortex_np, tname, args, order=0) > 0.5) & brain_x

            fake_x = _run_gan(G, mri_x, device)
            fake_x[~brain_x] = 0.0
            if tname == "identity":
                identity_fake = fake_x.copy()

            equiv_mae = float("nan")
            if identity_fake is not None and tname != "identity":
                fake_identity_x = _apply_transform(identity_fake, tname, args, order=1)
                fake_identity_x[~brain_x] = 0.0
                equiv_mae = _masked_mae(fake_x, fake_identity_x, brain_x)

            row = {
                "sid": sid,
                "stage_ord": int(meta.get("stage_ord", -1)),
                "transform": tname,
                "brain_mae": _masked_mae(fake_x, pet_x, brain_x),
                "brain_mse": _masked_mse(fake_x, pet_x, brain_x),
                "cortex_mae": _masked_mae(fake_x, pet_x, cortex_x),
                "equiv_mae_vs_transformed_identity_fake": equiv_mae,
                "fake_brain_mean": float(fake_x[brain_x].mean()) if np.any(brain_x) else float("nan"),
                "gt_brain_mean": float(pet_x[brain_x].mean()) if np.any(brain_x) else float("nan"),
                "fake_cortex_mean": float(fake_x[cortex_x].mean()) if np.any(cortex_x) else float("nan"),
                "gt_cortex_mean": float(pet_x[cortex_x].mean()) if np.any(cortex_x) else float("nan"),
                "sec": time.time() - item_t0,
            }
            rows.append(row)

            save_this_subject = int(args.save_volumes) and (
                int(args.save_volume_limit) <= 0 or n <= int(args.save_volume_limit)
            )
            if save_this_subject:
                out_sub = os.path.join(args.out_dir, sid_safe, tname)
                os.makedirs(out_sub, exist_ok=True)
                _save_nifti(mri_x, affine, os.path.join(out_sub, "MRI_input.nii.gz"))
                _save_nifti(pet_x, affine, os.path.join(out_sub, "PET_gt_transformed.nii.gz"))
                _save_nifti(fake_x, affine, os.path.join(out_sub, "PET_fake.nii.gz"))
                _save_nifti(np.abs(fake_x - pet_x).astype(np.float32), affine, os.path.join(out_sub, "PET_abs_error.nii.gz"))

            print(
                f"[gan-probe] {n}/{len(split_sids)} {sid} {tname} "
                f"brain_mae={row['brain_mae']:.5f} cortex_mae={row['cortex_mae']:.5f} "
                f"equiv_mae={equiv_mae:.5f} sec={row['sec']:.1f}",
                flush=True,
            )
        print(f"[gan-probe] subject {sid} done sec={time.time() - subj_t0:.1f}", flush=True)

    cols = [
        "sid", "stage_ord", "transform",
        "brain_mae", "brain_mse", "cortex_mae",
        "equiv_mae_vs_transformed_identity_fake",
        "fake_brain_mean", "gt_brain_mean",
        "fake_cortex_mean", "gt_cortex_mean",
        "sec",
    ]
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})

    print("[gan-probe] summaries", flush=True)
    for tname in transforms:
        vals = [r for r in rows if r["transform"] == tname]
        if not vals:
            continue
        brain_mae = np.asarray([r["brain_mae"] for r in vals], dtype=np.float64)
        cortex_mae = np.asarray([r["cortex_mae"] for r in vals], dtype=np.float64)
        equiv = np.asarray([r["equiv_mae_vs_transformed_identity_fake"] for r in vals], dtype=np.float64)
        equiv = equiv[np.isfinite(equiv)]
        msg = (
            f"[gan-probe] {tname}: brain_mae={brain_mae.mean():.6f} "
            f"cortex_mae={cortex_mae.mean():.6f}"
        )
        if equiv.size:
            msg += f" equiv_mae={equiv.mean():.6f}"
        print(msg, flush=True)

    print(f"[gan-probe] metrics_csv={metrics_path}", flush=True)
    print(f"[gan-probe] total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
