#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, math, argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from math import log10

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

        voxels = int(np.count_nonzero(roi_np))
        if voxels == 0:
            print(f"[WARN] ROI empty for {subj} {roi_name}; skip.")
            continue

        roi_t  = to_tensor_5d((roi_np > 0).astype(np.float32), device)
        fake_m = x_fake * roi_t
        gt_m   = x_gt   * roi_t

        # ---- EXACT old metrics (4): SSIM, PSNR, MSE, MMD ----
        ssim_v = float(ssim3d(fake_m, gt_m, data_range=data_range).item())
        psnr_v = float(psnr(fake_m,  gt_m, data_range=data_range))
        mse_v  = float(F.mse_loss(fake_m, gt_m).item())
        mmd_v  = float(mmd_gaussian(gt_m, fake_m, num_voxels=mmd_voxels, mask=roi_t))

        rows.append({
            "subject": subj,
            "roi": roi_name,
            "voxels": str(voxels),
            "SSIM": f"{ssim_v:.6f}",
            "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
            "MSE":  f"{mse_v:.8f}",
            "MMD":  f"{mmd_v:.8f}",
        })
    return rows

# ---------- Aggregation ----------
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

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="ROI-specific evaluation using the exact old logic (SSIM, PSNR, MSE, MMD).")
    ap.add_argument("--roi-root",  default="/scratch/l.peiwang/kari_brainv11",
                    help="Root with ROI masks per subject (ROI_*.nii.gz).")
    ap.add_argument("--vols-root", default="/home/l.peiwang/MRI2PET/MGDA_UB_dynamic_cos/volumes",
                    help="Root with saved PET volumes per subject (PET_fake.nii.gz, PET_gt.nii.gz).")
    ap.add_argument("--out",       default="/home/l.peiwang/MRI2PET/MGDA_UB_dynamic_cos/roi_metrics",
                    help="Output dir for CSVs.")
    ap.add_argument("--data-range", type=float, default=3.5,
                    help="Data range for SSIM/PSNR (defaults to your project value 3.5).")
    ap.add_argument("--mmd-voxels", type=int,   default=2048,
                    help="Sample size for MMD (same as your project).")
    ap.add_argument("--strict-affine", action="store_true",
                    help="Require affine equality; otherwise only shape is enforced.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of subjects (debug).")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device} | data_range={args.data_range} | mmd_voxels={args.mmd_voxels}")
    print(f"[INFO] ROI root:  {args.roi_root}")
    print(f"[INFO] Vols root: {args.vols_root}")
    print(f"[INFO] Out dir:   {args.out}")

    vols_subs = set([d for d in os.listdir(args.vols_root)
                     if os.path.isdir(os.path.join(args.vols_root, d))])
    roi_subs  = set([d for d in os.listdir(args.roi_root)
                     if os.path.isdir(os.path.join(args.roi_root, d))])
    subjects  = sorted(list(vols_subs & roi_subs))
    if args.limit:
        subjects = subjects[:args.limit]
    print(f"[INFO] Found {len(subjects)} subjects with volumes + ROI.")

    all_rows: List[Dict[str, str]] = []
    for i, subj in enumerate(subjects, 1):
        if (i % PRINT_EVERY) == 0:
            print(f"[DEBUG] ({i}/{len(subjects)}) {subj}")
        rows = compute_subject_roi_metrics(
            subj=subj,
            roi_root=args.roi_root,
            vols_root=args.vols_root,
            device=device,
            data_range=args.data_range,
            mmd_voxels=args.mmd_voxels,
            strict_affine=args.strict_affine
        )
        if not rows:
            print(f"[WARN] No ROI rows for {subj}.")
            continue

        sub_csv = os.path.join(args.vols_root, subj, "roi_metrics.csv")
        write_csv(sub_csv, rows, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        all_rows.extend(rows)

    if all_rows:
        long_csv = os.path.join(args.out, "roi_metrics_long.csv")
        write_csv(long_csv, all_rows, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        print(f"[INFO] Wrote long CSV: {long_csv}")

        summary_rows = summarize_long_rows(all_rows)
        summ_csv = os.path.join(args.out, "roi_metrics_summary.csv")
        write_csv(summ_csv, summary_rows,
                  header=["roi","N","SSIM_mean","SSIM_std","PSNR_mean","PSNR_std",
                          "MSE_mean","MSE_std","MMD_mean","MMD_std","Voxels_median"])
        print(f"[INFO] Wrote summary CSV: {summ_csv}")
    else:
        print("[WARN] No rows generated. Check paths and grid alignment.")

if __name__ == "__main__":
    main()
