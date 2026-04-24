#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OVERSAMPLE_ENABLE"] = "0"
os.environ["USE_BASELINE_CACHE"] = "0"

from mri2pet.config import BASELINE_CACHE_DIR, FOLD_CSV, RESIZE_TO, ROOT_DIR
from mri2pet.data import (
    KariAV1451Dataset,
    _compute_braak_stats,
    _compute_clinical_stats,
    _read_fold_csv_lists,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Compute train-fold residual stats from PET_true - PET_base.")
    p.add_argument("--cache-dir", default=BASELINE_CACHE_DIR, help="Directory containing *_pet_base.npy files.")
    p.add_argument("--root-dir", default=ROOT_DIR, help="Subject root directory.")
    p.add_argument("--fold-csv", default=FOLD_CSV, help="Fold CSV; only train subjects are used.")
    p.add_argument("--voxels-per-subject", type=int, default=20000, help="Sampled voxels per subject for p01/p99.")
    p.add_argument("--seed", type=int, default=1999)
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.cache_dir:
        raise RuntimeError("--cache-dir is required or BASELINE_CACHE_DIR must be set")
    if not os.path.isdir(args.cache_dir):
        raise FileNotFoundError(f"Cache dir not found: {args.cache_dir}")
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")

    print(f"[stats] root={args.root_dir}", flush=True)
    print(f"[stats] fold_csv={args.fold_csv}", flush=True)
    print(f"[stats] cache_dir={args.cache_dir}", flush=True)
    print(f"[stats] resize_to={RESIZE_TO}", flush=True)

    train_sids, _, _, train_sid_to_label = _read_fold_csv_lists(args.fold_csv)
    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO, sid_to_label=train_sid_to_label)
    sid_to_index = {item["sid"]: i for i, item in enumerate(ds.items)}
    missing = [sid for sid in train_sids if sid not in sid_to_index]
    if missing:
        raise RuntimeError(f"{len(missing)} train subjects not found on disk. Examples: {missing[:8]}")

    idx_train = [sid_to_index[sid] for sid in train_sids]
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))
    braak_mean, braak_std = _compute_braak_stats(ds, idx_train)
    ds.set_braak_stats(braak_mean, braak_std)

    rng = np.random.default_rng(args.seed)
    total_n = 0
    total_sum = 0.0
    total_sumsq = 0.0
    pct_samples = []
    t0 = time.time()

    for n, idx in enumerate(idx_train, start=1):
        item_t0 = time.time()
        _, pet, meta = ds[idx]
        sid = meta["sid"]
        base_path = os.path.join(args.cache_dir, f"{sid}_pet_base.npy")
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"{sid}: missing cached baseline PET: {base_path}")
        pet_np = pet.squeeze(0).numpy().astype(np.float32)
        base_np = np.load(base_path).astype(np.float32)
        brain = meta["brain_mask"].astype(bool)
        if base_np.shape != pet_np.shape:
            raise RuntimeError(f"{sid}: base shape {base_np.shape} != PET shape {pet_np.shape}")

        vals = (pet_np - base_np)[brain].astype(np.float64)
        total_n += int(vals.size)
        total_sum += float(vals.sum())
        total_sumsq += float(np.dot(vals, vals))

        if vals.size > 0 and args.voxels_per_subject > 0:
            k = min(int(args.voxels_per_subject), vals.size)
            sel = rng.choice(vals.size, size=k, replace=False)
            pct_samples.append(vals[sel].astype(np.float32))

        print(
            f"[stats] {n}/{len(idx_train)} {sid} voxels={vals.size} "
            f"subject_mean={vals.mean():.6f} subject_std={vals.std():.6f} "
            f"sec={time.time() - item_t0:.1f}",
            flush=True,
        )

    mean = total_sum / max(1, total_n)
    var = max(0.0, (total_sumsq / max(1, total_n)) - mean * mean)
    std = float(np.sqrt(var))
    if pct_samples:
        sample = np.concatenate(pct_samples, axis=0)
        p01, p99 = np.percentile(sample, [1, 99])
    else:
        p01 = p99 = float("nan")

    print("[stats] done", flush=True)
    print(f"[stats] train_subjects={len(idx_train)} brain_voxels={total_n}", flush=True)
    print(f"[stats] residual_mean={mean:.8f}", flush=True)
    print(f"[stats] residual_std={std:.8f}", flush=True)
    print(f"[stats] sampled_p01={p01:.8f} sampled_p99={p99:.8f}", flush=True)
    print("", flush=True)
    print(f"export DIFF_RESIDUAL_MEAN={mean:.8f}", flush=True)
    print(f"export DIFF_RESIDUAL_STD={std:.8f}", flush=True)
    print(f"[stats] total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
