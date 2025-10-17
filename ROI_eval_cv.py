#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, math, argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from math import log10
# NEW
import json

# ---------- Config / defaults ----------
ROI_FILES = {
    "Hippocampus":        "ROI_Hippocampus.nii.gz",
    "PosteriorCingulate": "ROI_PosteriorCingulate.nii.gz",
    "Precuneus":          "ROI_Precuneus.nii.gz",
    "TemporalLobe":       "ROI_TemporalLobe.nii.gz",
    "LimbicCortex":       "ROI_LimbicCortex.nii.gz",
}
ATOL_AFFINE = 1e-4
PRINT_EVERY = 1

# ---------- Metrics (exactly your old logic) ----------
def ssim3d(x: torch.Tensor, y: torch.Tensor, ksize: int = 3,
           k1: float = 0.01, k2: float = 0.03, data_range: float = 3.5) -> torch.Tensor:
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

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 3.5) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float('inf')
    return 10.0 * log10((data_range ** 2) / mse)

@torch.no_grad()
def mmd_gaussian(real: torch.Tensor,
                 fake: torch.Tensor,
                 num_voxels: int = 2048,
                 sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
                 mask: Optional[torch.Tensor] = None) -> float:
    B = real.size(0)
    dev = real.device
    total = 0.0
    for i in range(B):
        r = real[i, 0].reshape(-1)
        f = fake[i, 0].reshape(-1)

        if mask is not None:
            m = (mask[i, 0].reshape(-1) > 0.5)
            idx_pool = torch.nonzero(m, as_tuple=False).reshape(-1)
            if idx_pool.numel() == 0:
                return 0.0
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
    return total / B

# ---------- Masked metrics (bug fix) ----------
def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = (((x - y) ** 2) * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_psnr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, data_range: float = 3.5, eps: float = 1e-6) -> torch.Tensor:
    mse = masked_mse(x, y, mask, eps=eps)
    return 10.0 * torch.log10((data_range ** 2) / mse)

def ssim3d_masked(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                  ksize: int = 3, k1: float = 0.01, k2: float = 0.03,
                  data_range: float = 3.5, eps: float = 1e-6) -> torch.Tensor:
    """
    SSIM averaged only over windows that intersect the mask.
    Uses masked local statistics within each window.
    """
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

# ---------- IO helpers ----------
def load_nii(path: str):
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    return img, arr

def affines_close(a: nib.Nifti1Image, b: nib.Nifti1Image, atol=ATOL_AFFINE) -> bool:
    return np.allclose(a.affine, b.affine, atol=atol)

def to_tensor_5d(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))[None, None, ...].to(device)

def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---------- Core per-subject / per-ROI ----------
def compute_subject_roi_metrics(subj: str,
                                roi_root: str,
                                vols_root: str,
                                device: torch.device,
                                data_range: float,
                                mmd_voxels: int,
                                strict_affine: bool) -> List[Dict[str, str]]:
    subj_vol = os.path.join(vols_root, subj)
    subj_roi = os.path.join(roi_root,  subj)
    fake_p = os.path.join(subj_vol, "PET_fake.nii.gz")
    gt_p   = os.path.join(subj_vol, "PET_gt.nii.gz")

    if not (os.path.exists(fake_p) and os.path.exists(gt_p)):
        print(f"[WARN] Missing PET files for {subj}; skip.")
        return []

    fake_img, fake_np = load_nii(fake_p)
    gt_img,   gt_np   = load_nii(gt_p)

    if fake_np.shape != gt_np.shape:
        print(f"[ERROR] Shape mismatch PET fake{fake_np.shape} vs gt{gt_np.shape} for {subj}; skip.")
        return []
    if strict_affine and not affines_close(fake_img, gt_img):
        print(f"[WARN] Affine mismatch PET fake vs gt for {subj}; skipping (strict_affine).")
        return []

    x_fake = to_tensor_5d(fake_np, device)
    x_gt   = to_tensor_5d(gt_np,   device)

    rows: List[Dict[str, str]] = []

    # Track union of ROI masks (fallback if whole-brain mask is missing)
    roi_union_np: Optional[np.ndarray] = None

    for roi_name, roi_file in ROI_FILES.items():
        roi_path = os.path.join(subj_roi, roi_file)
        if not os.path.exists(roi_path):
            print(f"[WARN] ROI not found for {subj}: {roi_file}; skip.")
            continue

        roi_img, roi_np = load_nii(roi_path)
        if roi_np.shape != fake_np.shape:
            print(f"[WARN] ROI shape mismatch for {subj} {roi_name}: roi{roi_np.shape} vs pet{fake_np.shape}; skip.")
            continue
        if strict_affine and not affines_close(roi_img, fake_img):
            print(f"[WARN] ROI affine mismatch for {subj} {roi_name}; skip (strict_affine).")
            continue

        roi_bin = (roi_np > 0).astype(np.float32)
        voxels = int(np.count_nonzero(roi_bin))
        if voxels == 0:
            print(f"[WARN] ROI empty for {subj} {roi_name}; skip.")
            continue

        # accumulate union
        roi_union_np = roi_bin if roi_union_np is None else np.logical_or(roi_union_np > 0, roi_bin > 0).astype(np.float32)

        roi_t  = to_tensor_5d(roi_bin, device)

        # ---- Masked metrics (bug fix) ----
        ssim_v = float(ssim3d_masked(x_fake, x_gt, roi_t, data_range=data_range).item())
        mse_v  = float(masked_mse(x_fake, x_gt, roi_t).item())
        psnr_v = float(masked_psnr(x_fake, x_gt, roi_t, data_range=data_range).item())
        mmd_v  = float(mmd_gaussian(x_gt, x_fake, num_voxels=mmd_voxels, mask=roi_t))

        rows.append({
            "subject": subj,
            "roi": roi_name,
            "voxels": str(voxels),
            "SSIM": f"{ssim_v:.6f}",
            "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
            "MSE":  f"{mse_v:.8f}",
            "MMD":  f"{mmd_v:.8f}",
        })

    # ---- Whole-brain metrics (additional result) ----
    # Preferred: use aseg_brainmask if present; else fallback to union of ROIs (if any).
    wb_path = os.path.join(subj_roi, "aseg_brainmask.nii.gz")
    wb_mask_np: Optional[np.ndarray] = None
    if os.path.exists(wb_path):
        wb_img, wb_np = load_nii(wb_path)
        if wb_np.shape != fake_np.shape:
            raise TypeError("Mask Wrong Shape")
        elif strict_affine and not affines_close(wb_img, fake_img):
            raise TypeError("Mask Wrong Affine")
        else:
            wb_mask_np = (wb_np > 0).astype(np.float32)
            
    if wb_mask_np is None and roi_union_np is not None:
        raise TypeError("Noooo Mask")

    if wb_mask_np is not None:
        wb_vox = int(np.count_nonzero(wb_mask_np))
        if wb_vox > 0:
            wb_t = to_tensor_5d(wb_mask_np, device)
            ssim_wb = float(ssim3d_masked(x_fake, x_gt, wb_t, data_range=data_range).item())
            mse_wb  = float(masked_mse(x_fake, x_gt, wb_t).item())
            psnr_wb = float(masked_psnr(x_fake, x_gt, wb_t, data_range=data_range).item())
            mmd_wb  = float(mmd_gaussian(x_gt, x_fake, num_voxels=mmd_voxels, mask=wb_t))

            rows.append({
                "subject": subj,
                "roi": "WholeBrain",
                "voxels": str(wb_vox),
                "SSIM": f"{ssim_wb:.6f}",
                "PSNR": f"{psnr_wb:.6f}" if math.isfinite(psnr_wb) else "inf",
                "MSE":  f"{mse_wb:.8f}",
                "MMD":  f"{mmd_wb:.8f}",
            })
        else:
            print(f"[WARN] Whole-brain mask empty for {subj}; skipping WholeBrain row.")
    else:
        print(f"[WARN] No whole-brain mask available for {subj}; skipping WholeBrain row.")

    # ---- Whole-brain-noBG metrics (new) ----
    nbg_path = os.path.join(subj_roi, "mask_parenchyma_noBG.nii.gz")
    if os.path.exists(nbg_path):
        nbg_img, nbg_np = load_nii(nbg_path)
        if nbg_np.shape != fake_np.shape:
            raise TypeError("NoBG Mask Wrong Shape")
        elif strict_affine and not affines_close(nbg_img, fake_img):
            raise TypeError("NoBG Mask Wrong Affine")
        else:
            nbg_mask_np = (nbg_np > 0).astype(np.float32)
            nbg_vox = int(np.count_nonzero(nbg_mask_np))
            if nbg_vox > 0:
                nbg_t = to_tensor_5d(nbg_mask_np, device)
                ssim_nbg = float(ssim3d_masked(x_fake, x_gt, nbg_t, data_range=data_range).item())
                mse_nbg  = float(masked_mse(x_fake, x_gt, nbg_t).item())
                psnr_nbg = float(masked_psnr(x_fake, x_gt, nbg_t, data_range=data_range).item())
                mmd_nbg  = float(mmd_gaussian(x_gt, x_fake, num_voxels=mmd_voxels, mask=nbg_t))

                rows.append({
                    "subject": subj,
                    "roi": "WholeBrain_noBG",
                    "voxels": str(nbg_vox),
                    "SSIM": f"{ssim_nbg:.6f}",
                    "PSNR": f"{psnr_nbg:.6f}" if math.isfinite(psnr_nbg) else "inf",
                    "MSE":  f"{mse_nbg:.8f}",
                    "MMD":  f"{mmd_nbg:.8f}",
                })
            else:
                print(f"[WARN] WholeBrain_noBG mask empty for {subj}; skipping row.")
    else:
        print(f"[WARN] No WholeBrain_noBG mask for {subj}; skipping WholeBrain_noBG row.")

    return rows

# ---------- Aggregation (existing; unchanged) ----------
def summarize_long_rows(long_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_roi: Dict[str, Dict[str, List[float]]] = {}
    for r in long_rows:
        roi = r["roi"]
        by_roi.setdefault(roi, {"SSIM":[], "PSNR":[], "MSE":[], "MMD":[], "N_subj":set(), "Voxels":[]})
        for k in ["SSIM","PSNR","MSE","MMD"]:
            v = float(r[k]) if r[k] != "inf" else float("inf")
            by_roi[roi][k].append(v)
        by_roi[roi]["N_subj"].add(r["subject"])
        by_roi[roi]["Voxels"].append(int(r["voxels"]))

    summary = []
    for roi, d in by_roi.items():
        def mean_std(values: List[float]) -> Tuple[float, float]:
            vals = [v for v in values if math.isfinite(v)]
            if not vals:
                return (float("nan"), float("nan"))
            m = sum(vals)/len(vals)
            s = (sum((x-m)**2 for x in vals)/max(1, len(vals)-1))**0.5
            return (m, s)

        ssim_m, ssim_s = mean_std(d["SSIM"])
        psnr_m, psnr_s = mean_std(d["PSNR"])
        mse_m,  mse_s  = mean_std(d["MSE"])
        mmd_m,  mmd_s  = mean_std(d["MMD"])
        vox_med = int(np.median(d["Voxels"])) if d["Voxels"] else 0

        summary.append({
            "roi": roi,
            "N": str(len(d["N_subj"])),
            "SSIM_mean": f"{ssim_m:.6f}", "SSIM_std": f"{ssim_s:.6f}",
            "PSNR_mean": f"{psnr_m:.6f}", "PSNR_std": f"{psnr_s:.6f}",
            "MSE_mean":  f"{mse_m:.8f}",  "MSE_std":  f"{mse_s:.8f}",
            "MMD_mean":  f"{mmd_m:.8f}",  "MMD_std":  f"{mmd_s:.8f}",
            "Voxels_median": str(vox_med),
        })
    summary.sort(key=lambda x: x["roi"])
    return summary

# ---------- NEW: CI helpers ----------
def _tcrit(df: int) -> float:
    try:
        # try scipy if available
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df)) if df > 0 else float("nan")
    except Exception:
        return 1.96 if df > 0 else float("nan")  # normal approx fallback

def _mean_std_ci(values: List[float]) -> Tuple[float,float,int,float,float]:
    vals = [v for v in values if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    m = float(sum(vals)/n)
    sd = float((sum((x-m)**2 for x in vals)/max(1, n-1))**0.5) if n > 1 else float("nan")
    if n > 1:
        se = sd / (n**0.5)
        t = _tcrit(n-1)
        lo, hi = m - t*se, m + t*se
    else:
        lo = hi = float("nan")
    return m, sd, n, lo, hi

def summarize_with_ci_subject_level(long_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Per-ROI CI across all subjects pooled from all folds."""
    by_roi = {}
    for r in long_rows:
        roi = r["roi"]
        by_roi.setdefault(roi, {"SSIM":[], "PSNR":[], "MSE":[], "MMD":[], "SIDs":set()})
        for k in ["SSIM","PSNR","MSE","MMD"]:
            v = float(r[k]) if r[k] != "inf" else float("inf")
            by_roi[roi][k].append(v)
        by_roi[roi]["SIDs"].add(r["subject"])

    out = []
    for roi, d in sorted(by_roi.items()):
        ms = {}
        for k in ["SSIM","PSNR","MSE","MMD"]:
            mean, sd, n, lo, hi = _mean_std_ci(d[k])
            ms[k] = (mean, sd, n, lo, hi)
        out.append({
            "roi": roi,
            "N_subjects": str(len(d["SIDs"])),
            "SSIM_mean": f"{ms['SSIM'][0]:.6f}", "SSIM_std": f"{ms['SSIM'][1]:.6f}",
            "SSIM_lo95": f"{ms['SSIM'][3]:.6f}", "SSIM_hi95": f"{ms['SSIM'][4]:.6f}",
            "PSNR_mean": f"{ms['PSNR'][0]:.6f}", "PSNR_std": f"{ms['PSNR'][1]:.6f}",
            "PSNR_lo95": f"{ms['PSNR'][3]:.6f}", "PSNR_hi95": f"{ms['PSNR'][4]:.6f}",
            "MSE_mean":  f"{ms['MSE'][0]:.8f}",  "MSE_std":  f"{ms['MSE'][1]:.8f}",
            "MSE_lo95":  f"{ms['MSE'][3]:.8f}",  "MSE_hi95":  f"{ms['MSE'][4]:.8f}",
            "MMD_mean":  f"{ms['MMD'][0]:.8f}",  "MMD_std":  f"{ms['MMD'][1]:.8f}",
            "MMD_lo95":  f"{ms['MMD'][3]:.8f}",  "MMD_hi95":  f"{ms['MMD'][4]:.8f}",
        })
    return out

def summarize_with_ci_fold_level(long_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Per-ROI CI across folds (compute mean per fold, then CI over the 5 means)."""
    # collect per-fold, per-ROI lists
    by_fold_roi: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for r in long_rows:
        fold = r.get("fold", "fold?")
        roi  = r["roi"]
        by_fold_roi.setdefault(fold, {}).setdefault(roi, {"SSIM":[], "PSNR":[], "MSE":[], "MMD":[]})
        for k in ["SSIM","PSNR","MSE","MMD"]:
            v = float(r[k]) if r[k] != "inf" else float("inf")
            by_fold_roi[fold][roi][k].append(v)

    # compute mean per fold per ROI
    roi_to_fold_means: Dict[str, Dict[str, List[float]]] = {}
    for fold, roj in by_fold_roi.items():
        for roi, d in roj.items():
            for k in ["SSIM","PSNR","MSE","MMD"]:
                vals = [v for v in d[k] if math.isfinite(v)]
                if not vals:
                    continue
                m = float(sum(vals)/len(vals))
                roi_to_fold_means.setdefault(roi, {}).setdefault(k, []).append(m)

    # summarize across folds
    out = []
    for roi in sorted(roi_to_fold_means.keys()):
        row = {"roi": roi, "N_folds": "5"}  # nominally 5; if fewer, reflect actual below
        for k in ["SSIM","PSNR","MSE","MMD"]:
            fold_means = roi_to_fold_means[roi].get(k, [])
            mean, sd, n, lo, hi = _mean_std_ci(fold_means)
            row[f"{k}_mean_of_foldmeans"] = f"{mean:.6f}" if k in ("SSIM","PSNR") else f"{mean:.8f}"
            row[f"{k}_std_over_folds"]    = f"{sd:.6f}"   if k in ("SSIM","PSNR") else f"{sd:.8f}"
            row[f"{k}_lo95"]              = f"{lo:.6f}"   if k in ("SSIM","PSNR") else f"{lo:.8f}"
            row[f"{k}_hi95"]              = f"{hi:.6f}"   if k in ("SSIM","PSNR") else f"{hi:.8f}"
            row["N_folds"] = str(n)
        out.append(row)
    return out

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="ROI-specific evaluation (single fold or multi-fold).")
    ap.add_argument("--roi-root",  default="/scratch/l.peiwang/kari_brainv11",
                    help="Root with ROI masks per subject (ROI_*.nii.gz).")
    ap.add_argument("--vols-root", default="/home/l.peiwang/MRI2PET/MGDA_UB_dynamic_cos/volumes",
                    help="Root with saved PET volumes per subject (single fold).")
    # NEW: multi-fold runs; pass run dirs and we'll use <run>/volumes
    ap.add_argument("--runs", nargs="+", default=None,
                    help="List of run folders (each containing a 'volumes' subdir). If set, ignores --vols-root.")
    ap.add_argument("--out",       default="/home/l.peiwang/MRI2PET/MGDA_UB_dynamic_cos/roi_metrics",
                    help="Output dir for CSVs (combined summaries go here).")
    ap.add_argument("--data-range", type=float, default=3.5,
                    help="Data range for SSIM/PSNR (defaults to your project value 3.5).")
    ap.add_argument("--mmd-voxels", type=int,   default=2048,
                    help="Sample size for MMD (same as your project).")
    ap.add_argument("--strict-affine", action="store_true",
                    help="Require affine equality; otherwise only shape is enforced.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of subjects (debug).")
    return ap.parse_args()

def _eval_one_fold(roi_root: str, vols_root: str, out_dir_for_fold: str,
                   data_range: float, mmd_voxels: int, strict_affine: bool,
                   limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Run the original single-fold evaluation and return long rows (also write per-subject CSVs as before)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | data_range={data_range} | mmd_voxels={mmd_voxels}")
    print(f"[INFO] ROI root:  {roi_root}")
    print(f"[INFO] Vols root: {vols_root}")
    print(f"[INFO] Out dir:   {out_dir_for_fold}")

    vols_subs = set([d for d in os.listdir(vols_root)
                     if os.path.isdir(os.path.join(vols_root, d))])
    roi_subs  = set([d for d in os.listdir(roi_root)
                     if os.path.isdir(os.path.join(roi_root, d))])
    subjects  = sorted(list(vols_subs & roi_subs))
    if limit:
        subjects = subjects[:limit]
    print(f"[INFO] Found {len(subjects)} subjects with volumes + ROI.")

    all_rows: List[Dict[str, str]] = []
    for i, subj in enumerate(subjects, 1):
        if (i % PRINT_EVERY) == 0:
            print(f"[DEBUG] ({i}/{len(subjects)}) {subj}")
        rows = compute_subject_roi_metrics(
            subj=subj,
            roi_root=roi_root,
            vols_root=vols_root,
            device=device,
            data_range=data_range,
            mmd_voxels=mmd_voxels,
            strict_affine=strict_affine
        )
        if not rows:
            print(f"[WARN] No ROI rows for {subj}.")
            continue

        sub_csv = os.path.join(vols_root, subj, "roi_metrics.csv")
        write_csv(sub_csv, rows, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        all_rows.extend(rows)

    # Original single-fold summaries (kept identical, but we write under per-fold out)
    if all_rows:
        long_csv = os.path.join(out_dir_for_fold, "roi_metrics_long.csv")
        write_csv(long_csv, all_rows, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        print(f"[INFO] Wrote long CSV: {long_csv}")

        summary_rows = summarize_long_rows(all_rows)
        summ_csv = os.path.join(out_dir_for_fold, "roi_metrics_summary.csv")
        write_csv(summ_csv, summary_rows,
                  header=["roi","N","SSIM_mean","SSIM_std","PSNR_mean","PSNR_std",
                          "MSE_mean","MSE_std","MMD_mean","MMD_std","Voxels_median"])
        print(f"[INFO] Wrote summary CSV: {summ_csv}")
    else:
        print("[WARN] No rows generated for this fold.")
    return all_rows

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Single-fold mode (backward compatible)
    if not args.runs:
        fold_rows = _eval_one_fold(
            roi_root=args.roi_root,
            vols_root=args.vols_root,
            out_dir_for_fold=args.out,
            data_range=args.data_range,
            mmd_voxels=args.mmd_voxels,
            strict_affine=args.strict_affine,
            limit=args.limit
        )
        return

    # Multi-fold mode
    print("[INFO] Multi-fold mode. Runs:", args.runs)
    all_rows_all_folds: List[Dict[str, str]] = []
    per_fold_rows: Dict[str, List[Dict[str, str]]] = {}

    for run in args.runs:
        run = run.rstrip("/")
        fold_name = os.path.basename(run)
        vols_root = os.path.join(run, "volumes")
        if not os.path.isdir(vols_root):
            raise SystemExit(f"Missing volumes dir: {vols_root}")

        out_dir_for_fold = os.path.join(args.out, fold_name)
        os.makedirs(out_dir_for_fold, exist_ok=True)

        rows = _eval_one_fold(
            roi_root=args.roi_root,
            vols_root=vols_root,
            out_dir_for_fold=out_dir_for_fold,
            data_range=args.data_range,
            mmd_voxels=args.mmd_voxels,
            strict_affine=args.strict_affine,
            limit=args.limit
        )
        # tag with fold for later CI over folds
        for r in rows:
            r2 = dict(r)
            r2["fold"] = fold_name
            all_rows_all_folds.append(r2)
        per_fold_rows[fold_name] = rows

    # ---- Combined long CSV across all folds ----
    long_all_csv = os.path.join(args.out, "roi_metrics_long_all_folds.csv")
    write_csv(long_all_csv, all_rows_all_folds,
              header=["fold","subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
    print(f"[INFO] Wrote combined long CSV: {long_all_csv}")

    # ---- Subject-level CI over all subjects (pooled) ----
    subj_ci = summarize_with_ci_subject_level(all_rows_all_folds)
    subj_ci_csv = os.path.join(args.out, "roi_metrics_summary_subject_level_ci.csv")
    write_csv(subj_ci_csv, subj_ci,
              header=["roi","N_subjects",
                      "SSIM_mean","SSIM_std","SSIM_lo95","SSIM_hi95",
                      "PSNR_mean","PSNR_std","PSNR_lo95","PSNR_hi95",
                      "MSE_mean","MSE_std","MSE_lo95","MSE_hi95",
                      "MMD_mean","MMD_std","MMD_lo95","MMD_hi95"])
    print(f"[INFO] Wrote subject-level CI summary: {subj_ci_csv}")

    # ---- Fold-level CI (mean per fold → CI across folds) ----
    fold_ci = summarize_with_ci_fold_level(all_rows_all_folds)
    fold_ci_csv = os.path.join(args.out, "roi_metrics_summary_fold_level_ci.csv")
    write_csv(fold_ci_csv, fold_ci,
              header=["roi","N_folds",
                      "SSIM_mean_of_foldmeans","SSIM_std_over_folds","SSIM_lo95","SSIM_hi95",
                      "PSNR_mean_of_foldmeans","PSNR_std_over_folds","PSNR_lo95","PSNR_hi95",
                      "MSE_mean_of_foldmeans","MSE_std_over_folds","MSE_lo95","MSE_hi95",
                      "MMD_mean_of_foldmeans","MMD_std_over_folds","MMD_lo95","MMD_hi95"])
    print(f"[INFO] Wrote fold-level CI summary: {fold_ci_csv}")

if __name__ == "__main__":
    main()
