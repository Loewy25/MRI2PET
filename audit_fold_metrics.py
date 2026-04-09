#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit script to compare, fold-by-fold, three things on the SAME experiment outputs:

1) INTERNAL_LOGGED      : per_subject_metrics.csv produced by evaluate_and_save()
2) SAVED_INTERNAL_STYLE : recompute metrics on saved PET_fake/PET_gt volumes using the SAME
                          style as train_eval.evaluate_and_save()  (mask by multiplication,
                          then compute SSIM/MSE/PSNR over the whole tensor)
3) SAVED_EXTERNAL_STYLE : recompute metrics on the SAME saved volumes using the external-style
                          masked metrics (WholeBrain only)

It also checks subject-set consistency:
- expected test subjects from fold CSV
- subjects in per_subject_metrics.csv
- subject folders found under run_dir/volumes

This is the most direct way to answer:
- are the saved volumes from the best checkpoint?
- do saved volumes match internal logged metrics?
- does the external evaluator disagree even on the same subjects/files?
- are there stale / extra / missing subject folders?
"""

import os
import csv
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

ATOL_AFFINE = 1e-4


def norm_key(x: str) -> str:
    return str(x).strip().lower()


def load_nii(path: str):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    return img, arr


def affines_close(a: nib.Nifti1Image, b: nib.Nifti1Image, atol=ATOL_AFFINE) -> bool:
    return np.allclose(a.affine, b.affine, atol=atol)


def to_tensor_5d(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))[None, None, ...].to(device)


# ---------------------- internal-style metrics ----------------------
def ssim3d_internal(x: torch.Tensor, y: torch.Tensor, ksize: int = 3,
                    k1: float = 0.01, k2: float = 0.03,
                    data_range: float = 3.5) -> torch.Tensor:
    pad = ksize // 2
    mu_x = F.avg_pool3d(x, ksize, stride=1, padding=pad)
    mu_y = F.avg_pool3d(y, ksize, stride=1, padding=pad)
    sigma_x = F.avg_pool3d(x * x, ksize, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, ksize, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, ksize, stride=1, padding=pad) - mu_x * mu_y
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return ssim_map.mean()


def psnr_internal(x: torch.Tensor, y: torch.Tensor, data_range: float = 3.5) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def mmd_gaussian(real: torch.Tensor,
                 fake: torch.Tensor,
                 num_voxels: int = 2048,
                 sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
                 mask: Optional[torch.Tensor] = None) -> float:
    B = real.size(0)
    dev = real.device
    total = 0.0
    n_valid = 0
    for i in range(B):
        r = real[i, 0].reshape(-1)
        f = fake[i, 0].reshape(-1)

        if mask is not None:
            m = (mask[i, 0].reshape(-1) > 0.5)
            idx_pool = torch.nonzero(m, as_tuple=False).reshape(-1)
            if idx_pool.numel() == 0:
                continue
            S = min(num_voxels, idx_pool.numel())
            sel = idx_pool[torch.randint(0, idx_pool.numel(), (S,), device=dev)]
            r_s = r[sel].view(S, 1)
            f_s = f[sel].view(S, 1)
        else:
            S = min(num_voxels, r.numel(), f.numel())
            ridx = torch.randint(0, r.numel(), (S,), device=dev)
            fidx = torch.randint(0, f.numel(), (S,), device=dev)
            r_s = r[ridx].view(S, 1)
            f_s = f[fidx].view(S, 1)

        d_rr = (r_s - r_s.t()).pow(2)
        d_ff = (f_s - f_s.t()).pow(2)
        d_rf = (r_s - f_s.t()).pow(2)

        mmd = 0.0
        for s in sigmas:
            Krr = torch.exp(-d_rr / (2 * s * s))
            Kff = torch.exp(-d_ff / (2 * s * s))
            Krf = torch.exp(-d_rf / (2 * s * s))
            mmd += Krr.mean() + Kff.mean() - 2 * Krf.mean()
        mmd /= len(sigmas)
        total += mmd.item()
        n_valid += 1
    return total / max(1, n_valid)


# ---------------------- external-style metrics ----------------------
def masked_mse_external(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (((x - y) ** 2) * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den


def masked_psnr_external(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                         data_range: float = 3.5, eps: float = 1e-6) -> torch.Tensor:
    mse = masked_mse_external(x, y, mask, eps=eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def ssim3d_masked_external(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                           ksize: int = 3, k1: float = 0.01, k2: float = 0.03,
                           data_range: float = 3.5, eps: float = 1e-6) -> torch.Tensor:
    pad = ksize // 2
    k = torch.ones((1, 1, ksize, ksize, ksize), device=x.device, dtype=x.dtype)
    def wavg(z):
        z_sum = F.conv3d(z * mask, k, padding=pad)
        m_sum = F.conv3d(mask,    k, padding=pad).clamp_min(eps)
        return z_sum / m_sum
    mu_x = wavg(x); mu_y = wavg(y)
    mu_x2 = mu_x * mu_x; mu_y2 = mu_y * mu_y; mu_xy = mu_x * mu_y
    sigma_x  = wavg(x * x) - mu_x2
    sigma_y  = wavg(y * y) - mu_y2
    sigma_xy = wavg(x * y) - mu_xy
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + eps)
    win_hits = (F.conv3d(mask, k, padding=pad) > 0).to(x.dtype)
    return (ssim_map * win_hits).sum() / win_hits.sum().clamp_min(eps)


# ---------------------- fold / subject helpers ----------------------
def read_fold_test_sids(fold_csv: str) -> List[str]:
    sids = []
    with open(fold_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        tcol = cols.get('test')
        if tcol is None:
            raise ValueError(f"No 'test' column in {fold_csv}")
        for row in reader:
            sid = (row.get(tcol) or '').strip()
            if sid:
                sids.append(sid)
    return sorted(sids)


def list_volume_subjects(vol_dir: str) -> List[str]:
    if not os.path.isdir(vol_dir):
        return []
    return sorted([d for d in os.listdir(vol_dir) if os.path.isdir(os.path.join(vol_dir, d))])


def read_internal_per_subject_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=['sid', 'SSIM', 'PSNR', 'MSE', 'MMD'])
    df = pd.read_csv(path)
    for c in ['SSIM', 'PSNR', 'MSE', 'MMD']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def build_roi_index_case_insensitive(roi_root: str) -> Dict[str, str]:
    idx = {}
    for d in os.listdir(roi_root):
        p = os.path.join(roi_root, d)
        if os.path.isdir(p):
            idx[norm_key(d)] = d
    return idx


def aggregate(vals: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1) if a.size > 1 else 0.0), 'n': int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dirs', nargs='+', required=True,
                    help='List of OUT_RUN directories, one per fold (contains volumes/ and per_subject_metrics.csv)')
    ap.add_argument('--fold-csvs', nargs='+', required=True,
                    help='List of fold CSV paths, same order as --run-dirs')
    ap.add_argument('--roi-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--data-range', type=float, default=3.5)
    ap.add_argument('--mmd-voxels', type=int, default=2048)
    ap.add_argument('--strict-affine', action='store_true')
    args = ap.parse_args()

    if len(args.run_dirs) != len(args.fold_csvs):
        raise ValueError('--run-dirs and --fold-csvs must have the same length')

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roi_idx = build_roi_index_case_insensitive(args.roi_root)

    fold_rows = []
    per_subject_rows = []

    for fold_i, (run_dir, fold_csv) in enumerate(zip(args.run_dirs, args.fold_csvs), start=1):
        run_dir = run_dir.rstrip('/')
        vol_dir = os.path.join(run_dir, 'volumes')
        internal_csv = os.path.join(run_dir, 'per_subject_metrics.csv')

        expected = sorted(read_fold_test_sids(fold_csv))
        expected_set = set(expected)
        vol_subjects = list_volume_subjects(vol_dir)
        vol_set = set(vol_subjects)
        internal_df = read_internal_per_subject_csv(internal_csv)
        internal_sids = sorted(internal_df['sid'].astype(str).tolist()) if len(internal_df) else []
        internal_set = set(internal_sids)

        missing_from_volumes = sorted(expected_set - vol_set)
        extra_in_volumes = sorted(vol_set - expected_set)
        missing_from_internal = sorted(expected_set - internal_set)
        extra_in_internal = sorted(internal_set - expected_set)

        # evaluate only the expected test set intersection with volumes
        eval_sids = sorted(expected_set & vol_set)

        internal_logged_ssim = []
        internal_logged_psnr = []
        internal_logged_mse = []
        internal_logged_mmd = []

        saved_internal_ssim = []
        saved_internal_psnr = []
        saved_internal_mse = []
        saved_internal_mmd = []

        saved_external_ssim = []
        saved_external_psnr = []
        saved_external_mse = []
        saved_external_mmd = []

        for sid in eval_sids:
            subj_dir = os.path.join(vol_dir, sid)
            fake_p = os.path.join(subj_dir, 'PET_fake.nii.gz')
            gt_p = os.path.join(subj_dir, 'PET_gt.nii.gz')
            roi_dir = roi_idx.get(norm_key(sid), None)
            if roi_dir is None:
                continue
            brain_p = os.path.join(args.roi_root, roi_dir, 'aseg_brainmask.nii.gz')
            if not (os.path.exists(fake_p) and os.path.exists(gt_p) and os.path.exists(brain_p)):
                continue

            fake_img, fake_np = load_nii(fake_p)
            gt_img, gt_np = load_nii(gt_p)
            brain_img, brain_np = load_nii(brain_p)

            if fake_np.shape != gt_np.shape or brain_np.shape != fake_np.shape:
                continue
            if args.strict_affine:
                if not affines_close(fake_img, gt_img):
                    continue
                if not affines_close(fake_img, brain_img):
                    continue

            x_fake = to_tensor_5d(fake_np, device)
            x_gt = to_tensor_5d(gt_np, device)
            brain = to_tensor_5d((brain_np > 0).astype(np.float32), device)

            # INTERNAL_LOGGED
            row = internal_df[internal_df['sid'].astype(str) == sid]
            if len(row) == 1:
                internal_logged_ssim.append(float(row['SSIM'].iloc[0]))
                internal_logged_psnr.append(float(row['PSNR'].iloc[0]))
                internal_logged_mse.append(float(row['MSE'].iloc[0]))
                internal_logged_mmd.append(float(row['MMD'].iloc[0]))
                int_ssim = float(row['SSIM'].iloc[0])
                int_psnr = float(row['PSNR'].iloc[0])
                int_mse  = float(row['MSE'].iloc[0])
                int_mmd  = float(row['MMD'].iloc[0])
            else:
                int_ssim = int_psnr = int_mse = int_mmd = float('nan')

            # SAVED_INTERNAL_STYLE (same style as train_eval on saved files)
            fake_m = x_fake * brain
            gt_m = x_gt * brain
            ssim_i = float(ssim3d_internal(fake_m, gt_m, data_range=args.data_range).item())
            psnr_i = float(psnr_internal(fake_m, gt_m, data_range=args.data_range))
            mse_i = float(F.mse_loss(fake_m, gt_m).item())
            mmd_i = float(mmd_gaussian(gt_m, fake_m, num_voxels=args.mmd_voxels, mask=brain))

            saved_internal_ssim.append(ssim_i)
            saved_internal_psnr.append(psnr_i)
            saved_internal_mse.append(mse_i)
            saved_internal_mmd.append(mmd_i)

            # SAVED_EXTERNAL_STYLE (WholeBrain only)
            ssim_e = float(ssim3d_masked_external(x_fake, x_gt, brain, data_range=args.data_range).item())
            psnr_e = float(masked_psnr_external(x_fake, x_gt, brain, data_range=args.data_range).item())
            mse_e = float(masked_mse_external(x_fake, x_gt, brain).item())
            mmd_e = float(mmd_gaussian(x_gt, x_fake, num_voxels=args.mmd_voxels, mask=brain))

            saved_external_ssim.append(ssim_e)
            saved_external_psnr.append(psnr_e)
            saved_external_mse.append(mse_e)
            saved_external_mmd.append(mmd_e)

            per_subject_rows.append({
                'fold': fold_i,
                'sid': sid,
                'internal_logged_SSIM': int_ssim,
                'saved_internal_SSIM': ssim_i,
                'saved_external_wholebrain_SSIM': ssim_e,
                'internal_logged_PSNR': int_psnr,
                'saved_internal_PSNR': psnr_i,
                'saved_external_wholebrain_PSNR': psnr_e,
                'internal_logged_MSE': int_mse,
                'saved_internal_MSE': mse_i,
                'saved_external_wholebrain_MSE': mse_e,
                'internal_logged_MMD': int_mmd,
                'saved_internal_MMD': mmd_i,
                'saved_external_wholebrain_MMD': mmd_e,
                'delta_logged_vs_saved_internal_SSIM': (ssim_i - int_ssim) if np.isfinite(int_ssim) else np.nan,
                'delta_saved_internal_vs_external_SSIM': ssim_e - ssim_i,
                'delta_logged_vs_saved_internal_PSNR': (psnr_i - int_psnr) if np.isfinite(int_psnr) else np.nan,
                'delta_saved_internal_vs_external_PSNR': psnr_e - psnr_i,
                'delta_logged_vs_saved_internal_MSE': (mse_i - int_mse) if np.isfinite(int_mse) else np.nan,
                'delta_saved_internal_vs_external_MSE': mse_e - mse_i,
            })

        row = {
            'fold': fold_i,
            'run_dir': run_dir,
            'fold_csv': fold_csv,
            'expected_test_n': len(expected_set),
            'volume_subject_n': len(vol_set),
            'internal_csv_subject_n': len(internal_set),
            'eval_intersection_n': len(eval_sids),
            'missing_from_volumes_n': len(missing_from_volumes),
            'extra_in_volumes_n': len(extra_in_volumes),
            'missing_from_internal_csv_n': len(missing_from_internal),
            'extra_in_internal_csv_n': len(extra_in_internal),
            'missing_from_volumes_examples': ';'.join(missing_from_volumes[:10]),
            'extra_in_volumes_examples': ';'.join(extra_in_volumes[:10]),
            'missing_from_internal_examples': ';'.join(missing_from_internal[:10]),
            'extra_in_internal_examples': ';'.join(extra_in_internal[:10]),
        }
        for prefix, vals_ssim, vals_psnr, vals_mse, vals_mmd in [
            ('internal_logged', internal_logged_ssim, internal_logged_psnr, internal_logged_mse, internal_logged_mmd),
            ('saved_internal', saved_internal_ssim, saved_internal_psnr, saved_internal_mse, saved_internal_mmd),
            ('saved_external_wholebrain', saved_external_ssim, saved_external_psnr, saved_external_mse, saved_external_mmd),
        ]:
            row[f'{prefix}_SSIM_mean'] = aggregate(vals_ssim)['mean']
            row[f'{prefix}_PSNR_mean'] = aggregate(vals_psnr)['mean']
            row[f'{prefix}_MSE_mean'] = aggregate(vals_mse)['mean']
            row[f'{prefix}_MMD_mean'] = aggregate(vals_mmd)['mean']
        fold_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_dir, 'fold_metric_audit.csv'), index=False)
    pd.DataFrame(per_subject_rows).to_csv(os.path.join(args.out_dir, 'per_subject_metric_audit.csv'), index=False)

    # pooled summary across folds (means of fold means)
    if fold_rows:
        df = pd.DataFrame(fold_rows)
        pooled = {}
        for col in [c for c in df.columns if c.endswith('_mean')]:
            pooled[col] = float(df[col].mean())
        with open(os.path.join(args.out_dir, 'pooled_metric_audit.json'), 'w') as f:
            json.dump(pooled, f, indent=2)

    print(f"[WRITE] {os.path.join(args.out_dir, 'fold_metric_audit.csv')}")
    print(f"[WRITE] {os.path.join(args.out_dir, 'per_subject_metric_audit.csv')}")
    print(f"[DONE] Audit complete.")


if __name__ == '__main__':
    main()
