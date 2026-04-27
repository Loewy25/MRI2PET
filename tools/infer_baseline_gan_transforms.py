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
from mri2pet.data import KariAV1451Dataset, _compute_clinical_stats, _read_fold_csv_lists
from mri2pet.models import Generator
from mri2pet.utils import _resized_affine_for_scipy_zoom, _safe_name, _save_nifti


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a frozen baseline GAN on transformed split subjects and save PET_fake volumes."
    )
    p.add_argument("--checkpoint", default=BASE_PRETRAIN_CKPT, help="Baseline Generator checkpoint.")
    p.add_argument("--root-dir", default=ROOT_DIR)
    p.add_argument("--fold-csv", default=FOLD_CSV)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--max-subjects", type=int, default=0, help="0 means all subjects in split.")
    p.add_argument("--lr-axis", type=int, default=0, help="Voxel axis for left/right flip; NIfTI x is usually 0.")
    p.add_argument("--rot-deg", type=float, default=5.0)
    p.add_argument("--shift-vox", type=float, default=4.0)
    p.add_argument(
        "--transforms",
        default="identity,flip_lr,rot_pos,rot_neg,shift_lr_pos,shift_lr_neg",
        help="Comma-separated list: identity,flip_lr,rot_pos,rot_neg,shift_lr_pos,shift_lr_neg.",
    )
    p.add_argument("--save-input", type=int, default=1, help="Also save transformed MRI_input.nii.gz.")
    p.add_argument("--save-gt", type=int, default=1, help="Also save transformed PET_gt.nii.gz for visual reference.")
    p.add_argument("--save-diff", type=int, default=1, help="Also save abs-difference maps versus identity.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    if all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    if any(k.startswith("base.") for k in state.keys()):
        state = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    G.load_state_dict(state, strict=True)
    G.eval()
    return G


def _split_sids(fold_csv: str, split: str):
    train_sids, val_sids, test_sids, train_sid_to_label = _read_fold_csv_lists(fold_csv)
    if split == "train":
        return train_sids, train_sids, train_sid_to_label
    if split == "val":
        return val_sids, train_sids, train_sid_to_label
    return test_sids, train_sids, train_sid_to_label


def _affine_for_meta(meta: Dict) -> np.ndarray:
    cur_shape = tuple(meta.get("cur_shape"))
    orig_shape = tuple(meta.get("orig_shape", cur_shape))
    affine = meta.get("model_affine", None)
    if affine is not None:
        return affine
    if tuple(orig_shape) == tuple(cur_shape):
        return meta.get("t1_affine", np.eye(4))
    return _resized_affine_for_scipy_zoom(meta.get("t1_affine", np.eye(4)), orig_shape, cur_shape)


def _transform(vol: np.ndarray, name: str, lr_axis: int, rot_deg: float, shift_vox: float, order: int) -> np.ndarray:
    if name == "identity":
        return vol.copy()
    if name == "flip_lr":
        return np.flip(vol, axis=lr_axis).copy()
    if name == "rot_pos":
        return nd_rotate(vol, angle=rot_deg, axes=(0, 1), reshape=False, order=order, mode="constant", cval=0.0).astype(np.float32)
    if name == "rot_neg":
        return nd_rotate(vol, angle=-rot_deg, axes=(0, 1), reshape=False, order=order, mode="constant", cval=0.0).astype(np.float32)
    if name == "shift_lr_pos":
        shift_vec = [0.0, 0.0, 0.0]
        shift_vec[lr_axis] = shift_vox
        return nd_shift(vol, shift=shift_vec, order=order, mode="constant", cval=0.0).astype(np.float32)
    if name == "shift_lr_neg":
        shift_vec = [0.0, 0.0, 0.0]
        shift_vec[lr_axis] = -shift_vox
        return nd_shift(vol, shift=shift_vec, order=order, mode="constant", cval=0.0).astype(np.float32)
    raise ValueError(f"Unknown transform '{name}'")


def _run_gan(G: Generator, mri_np: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(mri_np[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        fake = G(x).squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    return fake


def main():
    args = _parse_args()
    if args.lr_axis not in (0, 1, 2):
        raise ValueError("--lr-axis must be 0, 1, or 2")
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")

    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    allowed = {"identity", "flip_lr", "rot_pos", "rot_neg", "shift_lr_pos", "shift_lr_neg"}
    bad = [t for t in transforms if t not in allowed]
    if bad:
        raise ValueError(f"Unknown transforms: {bad}. Allowed: {sorted(allowed)}")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    print("=" * 70, flush=True)
    print("[gan-transform] Frozen baseline GAN transform inference", flush=True)
    print(f"[gan-transform] root={args.root_dir}", flush=True)
    print(f"[gan-transform] fold_csv={args.fold_csv}", flush=True)
    print(f"[gan-transform] checkpoint={args.checkpoint}", flush=True)
    print(f"[gan-transform] out_dir={args.out_dir}", flush=True)
    print(f"[gan-transform] split={args.split} resize_to={RESIZE_TO}", flush=True)
    print(f"[gan-transform] transforms={transforms}", flush=True)
    print(f"[gan-transform] lr_axis={args.lr_axis} rot_deg={args.rot_deg} shift_vox={args.shift_vox}", flush=True)
    print("=" * 70, flush=True)

    split_sids, train_sids, train_sid_to_label = _split_sids(args.fold_csv, args.split)
    if args.max_subjects and args.max_subjects > 0:
        split_sids = split_sids[: int(args.max_subjects)]
        print(f"[gan-transform] limiting to first {len(split_sids)} subjects", flush=True)

    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO, sid_to_label=train_sid_to_label)
    sid_to_idx = {item["sid"]: i for i, item in enumerate(ds.items)}
    missing = [sid for sid in split_sids if sid not in sid_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} split subjects not found. Examples: {missing[:8]}")
    train_missing = [sid for sid in train_sids if sid not in sid_to_idx]
    if train_missing:
        raise RuntimeError(f"{len(train_missing)} train subjects not found. Examples: {train_missing[:8]}")

    ds.set_clinical_stats(_compute_clinical_stats(ds, [sid_to_idx[sid] for sid in train_sids]))
    G = _load_generator(args.checkpoint, device)

    manifest_path = os.path.join(args.out_dir, "outputs_manifest.csv")
    rows = []
    t0 = time.time()
    total = len(split_sids) * len(transforms)

    for s_idx, sid in enumerate(split_sids, start=1):
        subj_t0 = time.time()
        mri_t, pet_t, meta = ds[sid_to_idx[sid]]
        mri_np = mri_t.squeeze(0).numpy().astype(np.float32)
        pet_np = pet_t.squeeze(0).numpy().astype(np.float32)
        brain_np = meta["brain_mask"].astype(np.float32)
        brain_id = brain_np > 0.5
        fake_id = None
        if int(args.save_diff):
            fake_id = _run_gan(G, mri_np, device)
            fake_id[~brain_id] = 0.0
        affine = _affine_for_meta(meta)
        sid_safe = _safe_name(sid)

        for t_idx, transform_name in enumerate(transforms, start=1):
            item_t0 = time.time()
            out_sub = os.path.join(args.out_dir, sid_safe, transform_name)
            os.makedirs(out_sub, exist_ok=True)

            if transform_name == "identity" and fake_id is not None:
                mri_x = mri_np.copy()
                brain_x = brain_id.copy()
                fake_x = fake_id.copy()
            else:
                mri_x = _transform(mri_np, transform_name, args.lr_axis, args.rot_deg, args.shift_vox, order=1)
                brain_x = _transform(brain_np, transform_name, args.lr_axis, args.rot_deg, args.shift_vox, order=0) > 0.5
                fake_x = _run_gan(G, mri_x, device)
                fake_x[~brain_x] = 0.0

            fake_path = os.path.join(out_sub, "PET_fake.nii.gz")
            _save_nifti(fake_x, affine, fake_path)

            input_path = ""
            gt_path = ""
            mri_diff_path = ""
            fake_diff_path = ""
            gt_diff_path = ""
            if int(args.save_input):
                input_path = os.path.join(out_sub, "MRI_input.nii.gz")
                _save_nifti(mri_x, affine, input_path)
            if int(args.save_gt):
                if transform_name == "identity":
                    pet_x = pet_np.copy()
                else:
                    pet_x = _transform(pet_np, transform_name, args.lr_axis, args.rot_deg, args.shift_vox, order=1)
                pet_x[~brain_x] = 0.0
                gt_path = os.path.join(out_sub, "PET_gt.nii.gz")
                _save_nifti(pet_x, affine, gt_path)
            if int(args.save_diff):
                union_brain = np.logical_or(brain_id, brain_x)
                mri_diff = np.abs(mri_x - mri_np).astype(np.float32)
                mri_diff[~union_brain] = 0.0
                fake_diff = np.abs(fake_x - fake_id).astype(np.float32)
                fake_diff[~union_brain] = 0.0
                mri_diff_path = os.path.join(out_sub, "MRI_absdiff_from_identity.nii.gz")
                fake_diff_path = os.path.join(out_sub, "PET_fake_absdiff_from_identity.nii.gz")
                _save_nifti(mri_diff, affine, mri_diff_path)
                _save_nifti(fake_diff, affine, fake_diff_path)
                if int(args.save_gt):
                    pet_diff = np.abs(pet_x - pet_np).astype(np.float32)
                    pet_diff[~union_brain] = 0.0
                    gt_diff_path = os.path.join(out_sub, "PET_gt_absdiff_from_identity.nii.gz")
                    _save_nifti(pet_diff, affine, gt_diff_path)

            rows.append({
                "sid": sid,
                "stage_ord": int(meta.get("stage_ord", -1)),
                "transform": transform_name,
                "PET_fake": fake_path,
                "MRI_input": input_path,
                "PET_gt": gt_path,
                "MRI_absdiff_from_identity": mri_diff_path,
                "PET_fake_absdiff_from_identity": fake_diff_path,
                "PET_gt_absdiff_from_identity": gt_diff_path,
                "sec": time.time() - item_t0,
            })
            done = (s_idx - 1) * len(transforms) + t_idx
            print(
                f"[gan-transform] {done}/{total} sid={sid} transform={transform_name} "
                f"saved={fake_path} sec={time.time() - item_t0:.1f}",
                flush=True,
            )

        print(f"[gan-transform] subject {s_idx}/{len(split_sids)} {sid} done sec={time.time() - subj_t0:.1f}", flush=True)

    with open(manifest_path, "w", newline="") as f:
        cols = [
            "sid", "stage_ord", "transform", "PET_fake", "MRI_input", "PET_gt",
            "MRI_absdiff_from_identity", "PET_fake_absdiff_from_identity",
            "PET_gt_absdiff_from_identity", "sec",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[gan-transform] manifest={manifest_path}", flush=True)
    print(f"[gan-transform] total_outputs={len(rows)} total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
