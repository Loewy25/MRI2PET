#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OVERSAMPLE_ENABLE"] = "0"
os.environ["USE_BASELINE_CACHE"] = "0"

from mri2pet.config import BASELINE_CACHE_DIR, FOLD_CSV, RESIZE_TO, ROOT_DIR
from mri2pet.data import KariAV1451Dataset, _compute_clinical_stats, _read_fold_csv_lists


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Check whether high-cortex disease signal is positive raw residual "
            "or mainly PET contrast relative to the rest of brain."
        )
    )
    p.add_argument("--cache-dir", default=BASELINE_CACHE_DIR, help="Directory containing *_pet_base.npy files.")
    p.add_argument("--root-dir", default=ROOT_DIR, help="Subject root directory.")
    p.add_argument("--fold-csv", default=FOLD_CSV, help="Fold CSV defining train/validation/test subjects.")
    p.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test", "all"),
        help="Subjects to inspect. Use train for basis design; all is diagnostic only.",
    )
    p.add_argument("--hi-q", type=float, default=0.85, help="Quantile in true cortex used as high-cortex mask.")
    p.add_argument("--out-csv", default="", help="Optional per-subject CSV output path.")
    return p.parse_args()


def _mean(x: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask]))


def _fraction_positive(x: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(x[mask] > 0.0))


def _subject_sids(args, ds: KariAV1451Dataset) -> List[str]:
    train_sids, val_sids, test_sids, _ = _read_fold_csv_lists(args.fold_csv)
    if args.split == "train":
        sids = train_sids
    elif args.split == "val":
        sids = val_sids
    elif args.split == "test":
        sids = test_sids
    else:
        sids = train_sids + val_sids + test_sids

    sid_to_index = {item["sid"]: i for i, item in enumerate(ds.items)}
    missing = [sid for sid in sids if sid not in sid_to_index]
    if missing:
        raise RuntimeError(f"{len(missing)} subjects from fold CSV not found on disk. Examples: {missing[:8]}")
    return sids


def _summarize_group(rows: Iterable[Dict[str, float]], name: str):
    rows = list(rows)
    if not rows:
        return
    keys = [
        "true_brain_mean",
        "base_brain_mean",
        "brain_res_mean",
        "true_hi_ctx_mean",
        "base_hi_ctx_mean",
        "hi_ctx_res_mean",
        "hi_ctx_res_pos_frac",
        "true_hi_minus_nonctx",
        "base_hi_minus_nonctx",
        "contrast_gap_hi_vs_nonctx",
    ]
    print(f"\n[contrast] group={name} n={len(rows)}", flush=True)
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f"[contrast]   {key}: mean={vals.mean():.6f} std={vals.std():.6f}", flush=True)


def main():
    args = _parse_args()
    if not args.cache_dir:
        raise RuntimeError("--cache-dir is required or BASELINE_CACHE_DIR must be set")
    if not os.path.isdir(args.cache_dir):
        raise FileNotFoundError(f"Cache dir not found: {args.cache_dir}")
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")
    if not (0.0 < args.hi_q < 1.0):
        raise ValueError("--hi-q must be between 0 and 1")

    print(f"[contrast] root={args.root_dir}", flush=True)
    print(f"[contrast] fold_csv={args.fold_csv}", flush=True)
    print(f"[contrast] cache_dir={args.cache_dir}", flush=True)
    print(f"[contrast] split={args.split} resize_to={RESIZE_TO} hi_q={args.hi_q}", flush=True)

    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO)
    sid_to_index = {item["sid"]: i for i, item in enumerate(ds.items)}

    train_sids, _, _, _ = _read_fold_csv_lists(args.fold_csv)
    missing_train = [sid for sid in train_sids if sid not in sid_to_index]
    if missing_train:
        raise RuntimeError(
            f"{len(missing_train)} train subjects from fold CSV not found on disk. "
            f"Examples: {missing_train[:8]}"
        )
    idx_train = [sid_to_index[sid] for sid in train_sids]
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))

    sids = _subject_sids(args, ds)

    rows = []
    t0 = time.time()
    for n, sid in enumerate(sids, start=1):
        item_t0 = time.time()
        idx = sid_to_index[sid]
        _, pet, meta = ds[idx]
        pet_true = pet.squeeze(0).numpy().astype(np.float32)
        base_path = os.path.join(args.cache_dir, f"{sid}_pet_base.npy")
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"{sid}: missing cached PET_base: {base_path}")
        pet_base = np.load(base_path).astype(np.float32)
        if pet_base.shape != pet_true.shape:
            raise RuntimeError(f"{sid}: base shape {pet_base.shape} != PET shape {pet_true.shape}")

        brain = meta["brain_mask"].astype(bool)
        cortex = np.logical_and(meta["cortex_mask"].astype(bool), brain)
        nonctx = np.logical_and(brain, ~cortex)
        if not np.any(cortex):
            print(f"[contrast][WARN] {sid}: empty cortex mask, skipping", flush=True)
            continue

        cortex_true_vals = pet_true[cortex]
        hi_thr = float(np.quantile(cortex_true_vals, args.hi_q))
        hi_ctx = np.logical_and(cortex, pet_true >= hi_thr)
        residual = pet_true - pet_base

        true_hi = _mean(pet_true, hi_ctx)
        base_hi = _mean(pet_base, hi_ctx)
        true_nonctx = _mean(pet_true, nonctx)
        base_nonctx = _mean(pet_base, nonctx)
        true_hi_minus_nonctx = true_hi - true_nonctx
        base_hi_minus_nonctx = base_hi - base_nonctx

        row = {
            "sid": sid,
            "stage_ord": int(meta["stage_ord"]),
            "braak_max_raw": float(np.max(meta["braak_values_raw"])),
            "true_brain_mean": _mean(pet_true, brain),
            "base_brain_mean": _mean(pet_base, brain),
            "brain_res_mean": _mean(residual, brain),
            "true_cortex_mean": _mean(pet_true, cortex),
            "base_cortex_mean": _mean(pet_base, cortex),
            "cortex_res_mean": _mean(residual, cortex),
            "true_hi_ctx_mean": true_hi,
            "base_hi_ctx_mean": base_hi,
            "hi_ctx_res_mean": _mean(residual, hi_ctx),
            "hi_ctx_res_pos_frac": _fraction_positive(residual, hi_ctx),
            "true_nonctx_mean": true_nonctx,
            "base_nonctx_mean": base_nonctx,
            "true_hi_minus_nonctx": true_hi_minus_nonctx,
            "base_hi_minus_nonctx": base_hi_minus_nonctx,
            "contrast_gap_hi_vs_nonctx": true_hi_minus_nonctx - base_hi_minus_nonctx,
        }
        row["contrast_positive_raw_negative"] = int(
            row["contrast_gap_hi_vs_nonctx"] > 0.0 and row["hi_ctx_res_mean"] < 0.0
        )
        rows.append(row)

        print(
            f"[contrast] {n}/{len(sids)} {sid} stage={row['stage_ord']} "
            f"brain_res={row['brain_res_mean']:.5f} hi_ctx_res={row['hi_ctx_res_mean']:.5f} "
            f"hi_pos_frac={row['hi_ctx_res_pos_frac']:.3f} "
            f"contrast_gap={row['contrast_gap_hi_vs_nonctx']:.5f} "
            f"flag={row['contrast_positive_raw_negative']} sec={time.time() - item_t0:.1f}",
            flush=True,
        )

    by_stage = defaultdict(list)
    for row in rows:
        by_stage[int(row["stage_ord"])].append(row)

    print("\n[contrast] summaries", flush=True)
    _summarize_group(rows, "all")
    for stage in sorted(by_stage):
        _summarize_group(by_stage[stage], f"stage{stage}")

    high_rows = [r for r in rows if int(r["stage_ord"]) >= 3]
    flagged = [r for r in high_rows if r["contrast_positive_raw_negative"]]
    print(
        f"\n[contrast] high_stage_n={len(high_rows)} "
        f"contrast_positive_but_raw_hi_res_negative_n={len(flagged)}",
        flush=True,
    )
    if high_rows:
        print(
            "[contrast] high_stage_flag_fraction="
            f"{len(flagged) / max(1, len(high_rows)):.4f}",
            flush=True,
        )
    for row in flagged[:20]:
        print(
            f"[contrast][flag] {row['sid']} brain_res={row['brain_res_mean']:.5f} "
            f"hi_ctx_res={row['hi_ctx_res_mean']:.5f} "
            f"contrast_gap={row['contrast_gap_hi_vs_nonctx']:.5f}",
            flush=True,
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sid"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[contrast] wrote {args.out_csv}", flush=True)

    print(f"[contrast] total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
