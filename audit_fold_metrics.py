#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast fold audit for SSIM / PSNR only.

For each fold/run, compare three views of the SAME test subjects:

1) INTERNAL_LOGGED
   Mean SSIM / PSNR from run_dir/per_subject_metrics.csv
   (what evaluate_and_save() logged internally during the training script)

2) SAVED_INTERNAL_STYLE
   Recompute on saved PET_gt.nii.gz / PET_fake.nii.gz using the INTERNAL metric style:
      fake_m = fake * brain_mask
      gt_m   = gt   * brain_mask
      ssim3d(fake_m, gt_m)
      psnr(fake_m, gt_m)

3) SAVED_EXTERNAL_STYLE
   Recompute on the SAME saved files using the EXTERNAL masked metric style:
      ssim3d_masked(fake, gt, brain_mask)
      masked_psnr(fake, gt, brain_mask)

This keeps only WholeBrain and only SSIM / PSNR to make auditing fast.
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as nd_zoom
import pandas as pd
import torch
import torch.nn.functional as F

ATOL_AFFINE = 1e-4
WHOLEBRAIN_MASK = "aseg_brainmask.nii.gz"


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


# ---------------- Internal-style metrics ----------------
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


def psnr_internal(x: torch.Tensor, y: torch.Tensor, data_range: float = 3.5) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


# ---------------- External-style metrics ----------------
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
    pad = ksize // 2
    k = torch.ones((1, 1, ksize, ksize, ksize), device=x.device, dtype=x.dtype)

    def wavg(z):
        z_sum = F.conv3d(z * mask, k, padding=pad)
        m_sum = F.conv3d(mask, k, padding=pad).clamp_min(eps)
        return z_sum / m_sum

    mu_x = wavg(x)
    mu_y = wavg(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = wavg(x * x) - mu_x2
    sigma_y = wavg(y * y) - mu_y2
    sigma_xy = wavg(x * y) - mu_xy
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + eps)
    win_hits = (F.conv3d(mask, k, padding=pad) > 0).to(x.dtype)
    return (ssim_map * win_hits).sum() / win_hits.sum().clamp_min(eps)


# ---------------- Fold / subject helpers ----------------
def read_test_sids(fold_csv: str) -> List[str]:
    out: List[str] = []
    with open(fold_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        tcol = cols.get("test")
        if tcol is None:
            raise ValueError(f"No 'test' column found in {fold_csv}")
        for row in reader:
            sid = (row.get(tcol) or "").strip()
            if sid:
                out.append(sid)
    return out


def build_roi_index_case_insensitive(roi_root: str) -> Dict[str, str]:
    idx = {}
    for d in os.listdir(roi_root):
        p = os.path.join(roi_root, d)
        if os.path.isdir(p):
            idx[norm_key(d)] = d
    return idx


def resolve_run_and_vols(path: str) -> Tuple[str, str, str]:
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "volumes")):
        run_dir = path
        vols_dir = os.path.join(path, "volumes")
        metrics_csv = os.path.join(path, "per_subject_metrics.csv")
    else:
        run_dir = os.path.dirname(path)
        vols_dir = path
        metrics_csv = os.path.join(run_dir, "per_subject_metrics.csv")
    return run_dir, vols_dir, metrics_csv


def list_volume_subjects(vols_dir: str) -> List[str]:
    if not os.path.isdir(vols_dir):
        return []
    return sorted([d for d in os.listdir(vols_dir) if os.path.isdir(os.path.join(vols_dir, d))])


def safe_mean(vals: List[float]) -> float:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def compute_saved_metrics_for_subject(
    sid: str,
    vols_dir: str,
    roi_root: str,
    roi_idx: Dict[str, str],
    device: torch.device,
    data_range: float,
    strict_affine: bool,
) -> Optional[Dict[str, float]]:
    sid_dir = os.path.join(vols_dir, sid)
    fake_p = os.path.join(sid_dir, "PET_fake.nii.gz")
    gt_p = os.path.join(sid_dir, "PET_gt.nii.gz")
    if not (os.path.exists(fake_p) and os.path.exists(gt_p)):
        return None

    key = norm_key(sid)
    if key not in roi_idx:
        return None
    roi_subj = os.path.join(roi_root, roi_idx[key])
    mask_p = os.path.join(roi_subj, WHOLEBRAIN_MASK)
    if not os.path.exists(mask_p):
        return None

    fake_img, fake_np = load_nii(fake_p)
    gt_img, gt_np = load_nii(gt_p)
    mask_img, mask_np = load_nii(mask_p)

    if fake_np.shape != gt_np.shape:
        return None
    if fake_np.shape != mask_np.shape:
        zf = tuple(f / r for f, r in zip(fake_np.shape, mask_np.shape))
        mask_np = (nd_zoom(mask_np.astype(np.float32), zf, order=0) > 0.5).astype(np.uint8)
    if strict_affine and (not affines_close(fake_img, gt_img) or not affines_close(fake_img, mask_img)):
        return None

    x_fake = to_tensor_5d(fake_np, device)
    x_gt = to_tensor_5d(gt_np, device)
    mask = to_tensor_5d((mask_np > 0).astype(np.float32), device)

    # saved + internal style
    fake_m = x_fake * mask
    gt_m = x_gt * mask
    ssim_int = float(ssim3d(fake_m, gt_m, data_range=data_range).item())
    psnr_int = float(psnr_internal(fake_m, gt_m, data_range=data_range))

    # saved + external style
    ssim_ext = float(ssim3d_masked(x_fake, x_gt, mask, data_range=data_range).item())
    psnr_ext = float(masked_psnr(x_fake, x_gt, mask, data_range=data_range).item())

    return {
        "saved_internal_ssim": ssim_int,
        "saved_internal_psnr": psnr_int,
        "saved_external_ssim": ssim_ext,
        "saved_external_psnr": psnr_ext,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Fast fold audit: internal vs external SSIM/PSNR only.")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Fold run dirs (or volumes dirs).")
    ap.add_argument("--fold-csvs", nargs="+", required=True, help="Matching fold CSV files.")
    ap.add_argument("--roi-root", required=True, help="Root containing ROI/mask folders per subject.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--data-range", type=float, default=3.5)
    ap.add_argument("--strict-affine", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return ap.parse_args()


def main():
    args = parse_args()
    if len(args.run_dirs) != len(args.fold_csvs):
        raise ValueError("--run-dirs and --fold-csvs must have the same length")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    roi_idx = build_roi_index_case_insensitive(args.roi_root)

    fold_rows: List[Dict[str, object]] = []
    subj_rows: List[Dict[str, object]] = []

    all_logged_ssim, all_logged_psnr = [], []
    all_saved_int_ssim, all_saved_int_psnr = [], []
    all_saved_ext_ssim, all_saved_ext_psnr = [], []

    for i, (run_path, fold_csv) in enumerate(zip(args.run_dirs, args.fold_csvs), start=1):
        run_dir, vols_dir, metrics_csv = resolve_run_and_vols(run_path)
        expected = read_test_sids(fold_csv)
        expected_set = set(expected)
        vols_subjects = list_volume_subjects(vols_dir)
        vols_set = set(vols_subjects)

        # internal logged csv (if present)
        csv_subjects = set()
        logged_map: Dict[str, Dict[str, float]] = {}
        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            if "sid" in df.columns:
                for _, row in df.iterrows():
                    sid = str(row["sid"])
                    csv_subjects.add(sid)
                    logged_map[sid] = {
                        "internal_logged_ssim": float(row["SSIM"]) if "SSIM" in row else float("nan"),
                        "internal_logged_psnr": float(row["PSNR"]) if "PSNR" in row else float("nan"),
                    }

        common_subjects = sorted(expected_set & vols_set)

        logged_ssim_list, logged_psnr_list = [], []
        saved_int_ssim_list, saved_int_psnr_list = [], []
        saved_ext_ssim_list, saved_ext_psnr_list = [], []

        for sid in common_subjects:
            row = {
                "fold_index": i,
                "fold_csv": fold_csv,
                "run_dir": run_dir,
                "sid": sid,
            }

            if sid in logged_map:
                row.update(logged_map[sid])
                if math.isfinite(row["internal_logged_ssim"]):
                    logged_ssim_list.append(row["internal_logged_ssim"])
                if math.isfinite(row["internal_logged_psnr"]):
                    logged_psnr_list.append(row["internal_logged_psnr"])
            else:
                row["internal_logged_ssim"] = float("nan")
                row["internal_logged_psnr"] = float("nan")

            saved = compute_saved_metrics_for_subject(
                sid=sid,
                vols_dir=vols_dir,
                roi_root=args.roi_root,
                roi_idx=roi_idx,
                device=device,
                data_range=args.data_range,
                strict_affine=args.strict_affine,
            )
            if saved is not None:
                row.update(saved)
                saved_int_ssim_list.append(saved["saved_internal_ssim"])
                saved_int_psnr_list.append(saved["saved_internal_psnr"])
                saved_ext_ssim_list.append(saved["saved_external_ssim"])
                saved_ext_psnr_list.append(saved["saved_external_psnr"])
            else:
                row["saved_internal_ssim"] = float("nan")
                row["saved_internal_psnr"] = float("nan")
                row["saved_external_ssim"] = float("nan")
                row["saved_external_psnr"] = float("nan")

            subj_rows.append(row)

        fold_row = {
            "fold_index": i,
            "fold_csv": fold_csv,
            "run_dir": run_dir,
            "vols_dir": vols_dir,
            "metrics_csv": metrics_csv,
            "expected_test_n": len(expected_set),
            "volumes_subject_n": len(vols_set),
            "internal_csv_subject_n": len(csv_subjects),
            "matched_subject_n": len(common_subjects),
            "missing_from_volumes_n": len(expected_set - vols_set),
            "missing_from_internal_csv_n": len(expected_set - csv_subjects),
            "extra_in_volumes_n": len(vols_set - expected_set),
            "internal_logged_ssim": safe_mean(logged_ssim_list),
            "internal_logged_psnr": safe_mean(logged_psnr_list),
            "saved_internal_ssim": safe_mean(saved_int_ssim_list),
            "saved_internal_psnr": safe_mean(saved_int_psnr_list),
            "saved_external_ssim": safe_mean(saved_ext_ssim_list),
            "saved_external_psnr": safe_mean(saved_ext_psnr_list),
        }
        fold_rows.append(fold_row)

        all_logged_ssim.extend(logged_ssim_list)
        all_logged_psnr.extend(logged_psnr_list)
        all_saved_int_ssim.extend(saved_int_ssim_list)
        all_saved_int_psnr.extend(saved_int_psnr_list)
        all_saved_ext_ssim.extend(saved_ext_ssim_list)
        all_saved_ext_psnr.extend(saved_ext_psnr_list)

    pooled_row = {
        "fold_index": "POOLED",
        "fold_csv": "",
        "run_dir": "",
        "vols_dir": "",
        "metrics_csv": "",
        "expected_test_n": sum(int(r["expected_test_n"]) for r in fold_rows),
        "volumes_subject_n": sum(int(r["volumes_subject_n"]) for r in fold_rows),
        "internal_csv_subject_n": sum(int(r["internal_csv_subject_n"]) for r in fold_rows),
        "matched_subject_n": len(all_saved_ext_ssim),
        "missing_from_volumes_n": sum(int(r["missing_from_volumes_n"]) for r in fold_rows),
        "missing_from_internal_csv_n": sum(int(r["missing_from_internal_csv_n"]) for r in fold_rows),
        "extra_in_volumes_n": sum(int(r["extra_in_volumes_n"]) for r in fold_rows),
        "internal_logged_ssim": safe_mean(all_logged_ssim),
        "internal_logged_psnr": safe_mean(all_logged_psnr),
        "saved_internal_ssim": safe_mean(all_saved_int_ssim),
        "saved_internal_psnr": safe_mean(all_saved_int_psnr),
        "saved_external_ssim": safe_mean(all_saved_ext_ssim),
        "saved_external_psnr": safe_mean(all_saved_ext_psnr),
    }
    fold_rows.append(pooled_row)

    fold_csv_out = os.path.join(args.out_dir, "fold_metric_audit_fast.csv")
    subj_csv_out = os.path.join(args.out_dir, "per_subject_metric_audit_fast.csv")
    pooled_json_out = os.path.join(args.out_dir, "pooled_metric_audit_fast.json")

    pd.DataFrame(fold_rows).to_csv(fold_csv_out, index=False)
    pd.DataFrame(subj_rows).to_csv(subj_csv_out, index=False)
    with open(pooled_json_out, "w") as f:
        json.dump(pooled_row, f, indent=2)

    print(f"[WRITE] {fold_csv_out}")
    print(f"[WRITE] {subj_csv_out}")
    print(f"[WRITE] {pooled_json_out}")
    print("[DONE] Fast audit finished.")


if __name__ == "__main__":
    main()
