#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OVERSAMPLE_ENABLE"] = "0"
os.environ["USE_BASELINE_CACHE"] = "0"

from mri2pet.config import (
    BASELINE_CACHE_DIR,
    CDRM_CAL_STAGE_MAX,
    CDRM_CONTRAST_LAMBDA,
    CDRM_CONTRAST_REF,
    CDRM_DIS_STAGE_MIN,
    CDRM_DISEASE_TARGET_MODE,
    CDRM_K_CAL,
    CDRM_K_DIS,
    FOLD_CSV,
    RESIZE_TO,
    ROOT_DIR,
)
from mri2pet.data import KariAV1451Dataset, _compute_clinical_stats, _read_fold_csv_lists
from mri2pet.residual_manifold import (
    fit_joint_coefficients,
    reconstruct_from_basis,
    solve_nonnegative_coefficients,
    solve_signed_coefficients,
)


def _parse_args():
    p = argparse.ArgumentParser(description="Build residual-manifold calibration/disease bases and oracle metrics.")
    p.add_argument("--cache-dir", default=BASELINE_CACHE_DIR, help="Directory with *_pet_base.npy files.")
    p.add_argument("--basis-dir", required=True, help="Output directory for basis files and coefficient targets.")
    p.add_argument("--root-dir", default=ROOT_DIR, help="Subject root directory.")
    p.add_argument("--fold-csv", default=FOLD_CSV, help="Fold CSV. Basis is built from train subjects only.")
    p.add_argument("--k-cal", type=int, default=CDRM_K_CAL)
    p.add_argument("--k-dis", type=int, default=CDRM_K_DIS)
    p.add_argument("--disease-target-mode", default=CDRM_DISEASE_TARGET_MODE,
                   choices=("abs_plus_contrast", "abs_only"))
    p.add_argument("--contrast-lambda", type=float, default=CDRM_CONTRAST_LAMBDA)
    p.add_argument("--contrast-ref", default=CDRM_CONTRAST_REF,
                   choices=("brain_minus_cortex", "brain", "wholebrain_nobg"))
    p.add_argument("--cal-stage-max", type=int, default=CDRM_CAL_STAGE_MAX,
                   help="Use only train subjects with stage_ord <= this value to build B_cal.")
    p.add_argument("--dis-stage-min", type=int, default=CDRM_DIS_STAGE_MIN,
                   help="Use train subjects with stage_ord >= this value to build B_dis.")
    p.add_argument("--hi-q", type=float, default=0.85)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--fit-iters", type=int, default=3)
    p.add_argument("--nmf-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=1999)
    p.add_argument("--max-subjects", type=int, default=0, help="Optional smoke-test limit on train subjects.")
    return p.parse_args()


def _sid_index(ds: KariAV1451Dataset) -> Dict[str, int]:
    return {item["sid"]: i for i, item in enumerate(ds.items)}


def _load_pet_base(cache_dir: str, sid: str, shape: Tuple[int, int, int]) -> np.ndarray:
    path = os.path.join(cache_dir, f"{sid}_pet_base.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{sid}: missing cached PET_base: {path}")
    arr = np.load(path).astype(np.float32)
    if arr.shape != tuple(shape):
        raise RuntimeError(f"{sid}: cached PET_base shape {arr.shape} != PET/model grid {shape}")
    return arr


def _load_subject(ds: KariAV1451Dataset, sid_to_idx: Dict[str, int], cache_dir: str, sid: str):
    _, pet_t, meta = ds[sid_to_idx[sid]]
    pet_true = pet_t.squeeze(0).numpy().astype(np.float32)
    pet_base = _load_pet_base(cache_dir, sid, pet_true.shape)
    brain = meta["brain_mask"].astype(bool)
    cortex = np.logical_and(meta["cortex_mask"].astype(bool), brain)
    stage = int(meta["stage_ord"])
    pet_base[~brain] = 0.0
    pet_true[~brain] = 0.0
    return pet_true, pet_base, brain, cortex, stage, meta


def _normalize_basis_map(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    out[~mask] = 0.0
    norm = float(np.sqrt(np.sum(out[mask].astype(np.float64) ** 2)))
    if norm <= 1e-8:
        return out
    return (out / norm).astype(np.float32)


def _build_calibration_basis(raw_residuals: np.ndarray, brain_any: np.ndarray, k_cal: int) -> np.ndarray:
    if k_cal < 1:
        raise ValueError("--k-cal must be >= 1")
    n, v = raw_residuals.shape
    shape = brain_any.shape
    print(f"[cdrm-basis] calibration PCA: n={n} voxels={v} k={k_cal}", flush=True)

    mean_flat = raw_residuals.mean(axis=0)
    raw_residuals -= mean_flat[None, :]
    C = (raw_residuals @ raw_residuals.T).astype(np.float64) / max(1, v)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    basis: List[np.ndarray] = []
    mean_map = mean_flat.reshape(shape)
    basis.append(_normalize_basis_map(mean_map, brain_any))

    n_pc = min(max(0, k_cal - 1), eigvals.shape[0])
    for j in range(n_pc):
        ev = float(max(eigvals[j], 1e-12))
        comp_flat = (eigvecs[:, j].astype(np.float32) @ raw_residuals) / np.sqrt(ev * max(1, v))
        comp = comp_flat.reshape(shape)
        basis.append(_normalize_basis_map(comp, brain_any))
        print(f"[cdrm-basis]   PC{j + 1}: eig={eigvals[j]:.6e}", flush=True)
    while len(basis) < k_cal:
        print(f"[cdrm-basis][WARN] padding B_cal component {len(basis)} with zeros", flush=True)
        basis.append(np.zeros(shape, dtype=np.float32))

    B_cal = np.stack(basis, axis=0).astype(np.float32)
    print(f"[cdrm-basis] B_cal shape={B_cal.shape}", flush=True)
    return B_cal


def _reference_mask(brain: np.ndarray, cortex: np.ndarray, ref_name: str) -> np.ndarray:
    if ref_name == "brain":
        return brain
    if ref_name == "wholebrain_nobg":
        print("[cdrm-basis][WARN] wholebrain_nobg mask is not loaded by the dataset; using brain_minus_cortex.", flush=True)
    ref = np.logical_and(brain, ~cortex)
    if not np.any(ref):
        return brain
    return ref


def _disease_target(
    pet_true: np.ndarray,
    pet_base_cal: np.ndarray,
    brain: np.ndarray,
    cortex: np.ndarray,
    mode: str,
    contrast_lambda: float,
    contrast_ref: str,
) -> np.ndarray:
    r_after = (pet_true - pet_base_cal).astype(np.float32)
    d_abs = np.maximum(r_after, 0.0) * cortex.astype(np.float32)
    if mode == "abs_only":
        return d_abs.astype(np.float32)

    ref = _reference_mask(brain, cortex, contrast_ref)
    true_ref_mean = float(pet_true[ref].mean()) if np.any(ref) else 0.0
    base_ref_mean = float(pet_base_cal[ref].mean()) if np.any(ref) else 0.0
    contrast_true = pet_true - true_ref_mean
    contrast_base = pet_base_cal - base_ref_mean
    d_contrast = np.maximum(contrast_true - contrast_base, 0.0) * cortex.astype(np.float32)
    return (d_abs + float(contrast_lambda) * d_contrast).astype(np.float32)


def _nmf_multiplicative(
    X: np.ndarray,
    k: int,
    iters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if k < 1:
        raise ValueError("--k-dis must be >= 1")
    X = np.maximum(np.asarray(X, dtype=np.float32), 0.0)
    n, v = X.shape
    if float(X.sum()) <= 0.0:
        raise RuntimeError("Disease target matrix is all zero; cannot build B_dis")

    rng = np.random.default_rng(seed)
    W = rng.random((n, k)).astype(np.float32) + 1e-3
    H = rng.random((k, v)).astype(np.float32) + 1e-3
    scale = float(max(X.mean(), 1e-4))
    H *= scale
    eps = np.float32(1e-6)

    print(f"[cdrm-basis] NMF disease basis: n={n} voxels={v} k={k} iters={iters}", flush=True)
    for it in range(1, int(iters) + 1):
        H *= (W.T @ X) / ((W.T @ W) @ H + eps)
        W *= (X @ H.T) / (W @ (H @ H.T) + eps)

        norms = np.sqrt(np.sum(H.astype(np.float64) ** 2, axis=1)).astype(np.float32)
        norms = np.maximum(norms, eps)
        H /= norms[:, None]
        W *= norms[None, :]

        if it == 1 or it % 10 == 0 or it == int(iters):
            recon = W @ H
            mse = float(np.mean((X - recon) ** 2))
            print(f"[cdrm-basis]   nmf_iter={it:03d}/{iters} mse={mse:.8f}", flush=True)
    return W.astype(np.float32), H.astype(np.float32)


def _masked_l1(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))


def _high_cortex_mask(pet_true: np.ndarray, cortex: np.ndarray, q: float) -> np.ndarray:
    if not np.any(cortex):
        return cortex
    thr = float(np.quantile(pet_true[cortex], float(q)))
    return np.logical_and(cortex, pet_true >= thr)


def _summarize_oracle(rows: List[Dict[str, float]], split_name: str = "train") -> None:
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[int(row["stage_ord"])].append(row)

    def _print_group(name: str, group_rows: List[Dict[str, float]]):
        if not group_rows:
            return
        print(f"[cdrm-oracle][{split_name}] summary group={name} n={len(group_rows)}", flush=True)
        for key in [
            "base_brain_l1", "cal_brain_l1", "oracle_brain_l1",
            "base_hi_l1", "cal_hi_l1", "oracle_hi_l1",
            "oracle_improve_brain", "oracle_improve_hi",
        ]:
            vals = np.asarray([r[key] for r in group_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                print(f"[cdrm-oracle][{split_name}]   {key}: mean={vals.mean():.6f} std={vals.std():.6f}", flush=True)

    _print_group("all", rows)
    for stage in sorted(by_stage):
        _print_group(f"stage{stage}", by_stage[stage])


def _write_oracle_metrics(
    *,
    csv_path: str,
    split_name: str,
    sids: List[str],
    ds: KariAV1451Dataset,
    sid_to_idx: Dict[str, int],
    cache_dir: str,
    B_cal: np.ndarray,
    B_dis: np.ndarray,
    ridge: float,
    fit_iters: int,
    hi_q: float,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    print(f"[cdrm-oracle][{split_name}] evaluating fitted oracle n={len(sids)}", flush=True)
    for i, sid in enumerate(sids):
        item_t0 = time.time()
        pet_true, pet_base, brain, cortex, stage, _ = _load_subject(ds, sid_to_idx, cache_dir, sid)
        residual = (pet_true - pet_base).astype(np.float32)
        c, a = fit_joint_coefficients(
            residual,
            B_cal,
            B_dis,
            brain,
            ridge=float(ridge),
            iters=int(fit_iters),
        )
        res_cal = np.einsum("k,kdhw->dhw", c, B_cal).astype(np.float32)
        res_total = reconstruct_from_basis(c, a, B_cal, B_dis)
        pet_cal = (pet_base + res_cal).astype(np.float32)
        pet_oracle = (pet_base + res_total).astype(np.float32)
        hi_ctx = _high_cortex_mask(pet_true, cortex, float(hi_q))
        row = {
            "sid": sid,
            "stage_ord": int(stage),
            "base_brain_l1": _masked_l1(pet_base, pet_true, brain),
            "cal_brain_l1": _masked_l1(pet_cal, pet_true, brain),
            "oracle_brain_l1": _masked_l1(pet_oracle, pet_true, brain),
            "base_hi_l1": _masked_l1(pet_base, pet_true, hi_ctx),
            "cal_hi_l1": _masked_l1(pet_cal, pet_true, hi_ctx),
            "oracle_hi_l1": _masked_l1(pet_oracle, pet_true, hi_ctx),
            "mean_abs_res_cal": float(np.mean(np.abs(res_cal[brain]))) if np.any(brain) else float("nan"),
            "mean_abs_res_dis": float(np.mean(np.abs((res_total - res_cal)[brain]))) if np.any(brain) else float("nan"),
            "mean_a": float(np.mean(a)) if a.size else float("nan"),
        }
        row["oracle_improve_brain"] = row["base_brain_l1"] - row["oracle_brain_l1"]
        row["oracle_improve_hi"] = row["base_hi_l1"] - row["oracle_hi_l1"]
        rows.append(row)
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == len(sids):
            print(
                f"[cdrm-oracle][{split_name}]   {i + 1}/{len(sids)} sid={sid} stage={stage} "
                f"base_hi={row['base_hi_l1']:.5f} oracle_hi={row['oracle_hi_l1']:.5f} "
                f"improve_hi={row['oracle_improve_hi']:.5f} "
                f"|cal|={row['mean_abs_res_cal']:.5f} |dis|={row['mean_abs_res_dis']:.5f} "
                f"sec={time.time() - item_t0:.1f}",
                flush=True,
            )

    cols = [
        "sid", "stage_ord",
        "base_brain_l1", "cal_brain_l1", "oracle_brain_l1",
        "base_hi_l1", "cal_hi_l1", "oracle_hi_l1",
        "oracle_improve_brain", "oracle_improve_hi",
        "mean_abs_res_cal", "mean_abs_res_dis", "mean_a",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    _summarize_oracle(rows, split_name=split_name)
    print(f"[cdrm-oracle][{split_name}] csv={csv_path}", flush=True)
    return rows


def main():
    args = _parse_args()
    if not args.cache_dir:
        raise RuntimeError("--cache-dir is required or BASELINE_CACHE_DIR must be set")
    if not os.path.isdir(args.cache_dir):
        raise FileNotFoundError(f"Cache dir not found: {args.cache_dir}")
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")
    if not (0.0 < float(args.hi_q) < 1.0):
        raise ValueError("--hi-q must be between 0 and 1")

    os.makedirs(args.basis_dir, exist_ok=True)
    t0 = time.time()
    print("=" * 70, flush=True)
    print("[cdrm-basis] Build residual manifold", flush=True)
    print(f"[cdrm-basis] root={args.root_dir}", flush=True)
    print(f"[cdrm-basis] fold_csv={args.fold_csv}", flush=True)
    print(f"[cdrm-basis] cache_dir={args.cache_dir}", flush=True)
    print(f"[cdrm-basis] basis_dir={args.basis_dir}", flush=True)
    print(
        f"[cdrm-basis] k_cal={args.k_cal} k_dis={args.k_dis} "
        f"mode={args.disease_target_mode} contrast_lambda={args.contrast_lambda} "
        f"contrast_ref={args.contrast_ref} cal_stage_max={args.cal_stage_max} "
        f"dis_stage_min={args.dis_stage_min} resize_to={RESIZE_TO}",
        flush=True,
    )
    print("=" * 70, flush=True)

    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO)
    sid_to_idx = _sid_index(ds)
    train_sids, val_sids, test_sids, _ = _read_fold_csv_lists(args.fold_csv)
    if args.max_subjects and args.max_subjects > 0:
        train_sids = train_sids[: int(args.max_subjects)]
        print(f"[cdrm-basis][SMOKE] limiting train subjects to {len(train_sids)}", flush=True)
    missing = [sid for sid in (train_sids + val_sids + test_sids) if sid not in sid_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} fold subjects not found on disk. Examples: {missing[:8]}")
    idx_train = [sid_to_idx[sid] for sid in train_sids]
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))

    first_pet, _, first_brain, _, _, _ = _load_subject(ds, sid_to_idx, args.cache_dir, train_sids[0])
    shape = tuple(first_pet.shape)
    n = len(train_sids)
    v = int(np.prod(shape))
    raw_residuals = np.empty((n, v), dtype=np.float32)
    brain_any = np.zeros(shape, dtype=bool)
    stages: List[int] = []

    print(f"[cdrm-basis] pass1 raw residual matrix n={n} shape={shape}", flush=True)
    for i, sid in enumerate(train_sids):
        item_t0 = time.time()
        pet_true, pet_base, brain, cortex, stage, _ = _load_subject(ds, sid_to_idx, args.cache_dir, sid)
        if pet_true.shape != shape:
            raise RuntimeError(f"{sid}: shape {pet_true.shape} != first shape {shape}")
        residual = (pet_true - pet_base).astype(np.float32)
        residual[~brain] = 0.0
        raw_residuals[i] = residual.reshape(-1)
        brain_any |= brain
        stages.append(stage)
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == n:
            print(
                f"[cdrm-basis]   residual {i + 1}/{n} sid={sid} stage={stage} "
                f"sec={time.time() - item_t0:.1f}",
                flush=True,
            )

    cal_indices = [i for i, st in enumerate(stages) if int(st) <= int(args.cal_stage_max)]
    if not cal_indices:
        raise RuntimeError(
            f"No train subjects with stage_ord <= cal_stage_max={args.cal_stage_max}; "
            "cannot build calibration basis."
        )
    cal_counts = defaultdict(int)
    for i in cal_indices:
        cal_counts[int(stages[i])] += 1
    print(
        f"[cdrm-basis] calibration pool: stage<= {args.cal_stage_max}, "
        f"n={len(cal_indices)}, counts={dict(sorted(cal_counts.items()))}",
        flush=True,
    )
    B_cal = _build_calibration_basis(raw_residuals[np.asarray(cal_indices)].copy(), brain_any, int(args.k_cal))
    del raw_residuals

    disease_rows = []
    disease_sids = []
    disease_stages = []
    print("[cdrm-basis] pass2 contrast-aware disease targets", flush=True)
    for i, sid in enumerate(train_sids):
        item_t0 = time.time()
        pet_true, pet_base, brain, cortex, stage, _ = _load_subject(ds, sid_to_idx, args.cache_dir, sid)
        residual = (pet_true - pet_base).astype(np.float32)
        c_cal = solve_signed_coefficients(residual, B_cal, brain, ridge=float(args.ridge))
        pet_base_cal = pet_base + np.einsum("k,kdhw->dhw", c_cal, B_cal)
        target = _disease_target(
            pet_true,
            pet_base_cal.astype(np.float32),
            brain,
            cortex,
            mode=args.disease_target_mode,
            contrast_lambda=float(args.contrast_lambda),
            contrast_ref=args.contrast_ref,
        )
        if stage >= int(args.dis_stage_min):
            disease_rows.append(target.reshape(-1))
            disease_sids.append(sid)
            disease_stages.append(stage)
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == n:
            nz = float(np.mean(target[cortex] > 0.0)) if np.any(cortex) else 0.0
            print(
                f"[cdrm-basis]   disease target {i + 1}/{n} sid={sid} stage={stage} "
                f"ctx_pos_frac={nz:.3f} sec={time.time() - item_t0:.1f}",
                flush=True,
            )

    if len(disease_rows) < int(args.k_dis):
        print(
            f"[cdrm-basis][WARN] only {len(disease_rows)} stage>={args.dis_stage_min} subjects "
            f"for k_dis={args.k_dis}; adding stage2+ subjects to disease-basis pool.",
            flush=True,
        )
        disease_rows = []
        disease_sids = []
        disease_stages = []
        for sid, stage in zip(train_sids, stages):
            if stage < 2:
                continue
            pet_true, pet_base, brain, cortex, _, _ = _load_subject(ds, sid_to_idx, args.cache_dir, sid)
            residual = (pet_true - pet_base).astype(np.float32)
            c_cal = solve_signed_coefficients(residual, B_cal, brain, ridge=float(args.ridge))
            pet_base_cal = pet_base + np.einsum("k,kdhw->dhw", c_cal, B_cal)
            target = _disease_target(
                pet_true, pet_base_cal.astype(np.float32), brain, cortex,
                mode=args.disease_target_mode,
                contrast_lambda=float(args.contrast_lambda),
                contrast_ref=args.contrast_ref,
            )
            disease_rows.append(target.reshape(-1))
            disease_sids.append(sid)
            disease_stages.append(stage)

    if not disease_rows:
        raise RuntimeError(
            "No stage>=2 disease targets were available for B_dis. "
            "Check fold labels or lower the disease-pool threshold in the script."
        )

    X_dis = np.stack(disease_rows, axis=0).astype(np.float32)
    _, H = _nmf_multiplicative(X_dis, int(args.k_dis), int(args.nmf_iters), int(args.seed))
    B_dis = H.reshape((int(args.k_dis),) + shape).astype(np.float32)
    for k in range(B_dis.shape[0]):
        B_dis[k] = _normalize_basis_map(np.maximum(B_dis[k], 0.0), brain_any)
    print(f"[cdrm-basis] B_dis shape={B_dis.shape} pool_n={len(disease_sids)}", flush=True)
    del X_dis

    np.save(os.path.join(args.basis_dir, "B_cal.npy"), B_cal.astype(np.float32))
    np.save(os.path.join(args.basis_dir, "B_dis.npy"), B_dis.astype(np.float32))
    print("[cdrm-basis] saved B_cal.npy and B_dis.npy", flush=True)

    coeff_csv = os.path.join(args.basis_dir, "coeff_targets.csv")
    oracle_csv = os.path.join(args.basis_dir, "oracle_metrics.csv")
    val_oracle_csv = os.path.join(args.basis_dir, "val_oracle_metrics.csv")
    test_oracle_csv = os.path.join(args.basis_dir, "test_oracle_metrics.csv")
    coeff_cols = (
        ["sid", "stage_ord"]
        + [f"c{k}" for k in range(B_cal.shape[0])]
        + [f"a{k}" for k in range(B_dis.shape[0])]
    )

    print("[cdrm-basis] pass3 train coefficient targets", flush=True)
    with open(coeff_csv, "w", newline="") as f_coeff:
        writer = csv.DictWriter(f_coeff, fieldnames=coeff_cols)
        writer.writeheader()
        for i, sid in enumerate(train_sids):
            item_t0 = time.time()
            pet_true, pet_base, brain, cortex, stage, _ = _load_subject(ds, sid_to_idx, args.cache_dir, sid)
            residual = (pet_true - pet_base).astype(np.float32)
            c, a = fit_joint_coefficients(
                residual,
                B_cal,
                B_dis,
                brain,
                ridge=float(args.ridge),
                iters=int(args.fit_iters),
            )
            coeff_row = {"sid": sid, "stage_ord": int(stage)}
            coeff_row.update({f"c{k}": float(c[k]) for k in range(B_cal.shape[0])})
            coeff_row.update({f"a{k}": float(a[k]) for k in range(B_dis.shape[0])})
            writer.writerow(coeff_row)

            if i == 0 or (i + 1) % 10 == 0 or (i + 1) == n:
                print(
                    f"[cdrm-basis]   coeff target {i + 1}/{n} sid={sid} stage={stage} "
                    f"sec={time.time() - item_t0:.1f}",
                    flush=True,
                )

    _write_oracle_metrics(
        csv_path=oracle_csv,
        split_name="train",
        sids=train_sids,
        ds=ds,
        sid_to_idx=sid_to_idx,
        cache_dir=args.cache_dir,
        B_cal=B_cal,
        B_dis=B_dis,
        ridge=float(args.ridge),
        fit_iters=int(args.fit_iters),
        hi_q=float(args.hi_q),
    )
    _write_oracle_metrics(
        csv_path=val_oracle_csv,
        split_name="val",
        sids=val_sids,
        ds=ds,
        sid_to_idx=sid_to_idx,
        cache_dir=args.cache_dir,
        B_cal=B_cal,
        B_dis=B_dis,
        ridge=float(args.ridge),
        fit_iters=int(args.fit_iters),
        hi_q=float(args.hi_q),
    )
    _write_oracle_metrics(
        csv_path=test_oracle_csv,
        split_name="test",
        sids=test_sids,
        ds=ds,
        sid_to_idx=sid_to_idx,
        cache_dir=args.cache_dir,
        B_cal=B_cal,
        B_dis=B_dis,
        ridge=float(args.ridge),
        fit_iters=int(args.fit_iters),
        hi_q=float(args.hi_q),
    )

    manifest = {
        "root_dir": args.root_dir,
        "fold_csv": args.fold_csv,
        "cache_dir": args.cache_dir,
        "basis_dir": args.basis_dir,
        "resize_to": RESIZE_TO,
        "shape": shape,
        "n_train": n,
        "n_validation": len(val_sids),
        "n_test": len(test_sids),
        "k_cal": int(args.k_cal),
        "k_dis": int(args.k_dis),
        "calibration_basis": "mean_residual_plus_signed_pca",
        "cal_stage_max": int(args.cal_stage_max),
        "calibration_pool_sids": [train_sids[i] for i in cal_indices],
        "calibration_pool_stages": [int(stages[i]) for i in cal_indices],
        "disease_basis": "nonnegative_nmf",
        "dis_stage_min": int(args.dis_stage_min),
        "disease_pool_sids": disease_sids,
        "disease_pool_stages": disease_stages,
        "disease_target_mode": args.disease_target_mode,
        "contrast_lambda": float(args.contrast_lambda),
        "contrast_ref": args.contrast_ref,
        "hi_q": float(args.hi_q),
        "ridge": float(args.ridge),
        "fit_iters": int(args.fit_iters),
        "nmf_iters": int(args.nmf_iters),
        "seed": int(args.seed),
        "total_sec": time.time() - t0,
        "outputs": {
            "B_cal": os.path.join(args.basis_dir, "B_cal.npy"),
            "B_dis": os.path.join(args.basis_dir, "B_dis.npy"),
            "coeff_targets": coeff_csv,
            "oracle_metrics": oracle_csv,
            "val_oracle_metrics": val_oracle_csv,
            "test_oracle_metrics": test_oracle_csv,
        },
    }
    with open(os.path.join(args.basis_dir, "basis_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[cdrm-basis] coeff_csv={coeff_csv}", flush=True)
    print(f"[cdrm-basis] oracle_csv={oracle_csv}", flush=True)
    print(f"[cdrm-basis] val_oracle_csv={val_oracle_csv}", flush=True)
    print(f"[cdrm-basis] test_oracle_csv={test_oracle_csv}", flush=True)
    print(f"[cdrm-basis] manifest={os.path.join(args.basis_dir, 'basis_manifest.json')}", flush=True)
    print(f"[cdrm-basis] total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
