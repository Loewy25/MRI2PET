
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal "paper-style" comprehensive evaluation for GT vs Fake tau PET given:
- PET_gt.nii.gz and PET_fake.nii.gz per subject/session folder
- ROI masks per subject (same space)
- Meta CSV containing clinical/pathological columns (cdr, Centiloid, Braak*, etc.)

Outputs (ALL + optional grouped):
(A) Fidelity (voxelwise in masks): SSIM, PSNR, MSE, MMD -> long + summary (mean/std/95% CI)
(B) ROI-mean agreement (paper-style): Pearson r (+ Fisher CI), MAE/MAPE (mean/std/95% CI), CorrAVG
(C) Clinical utility (2B): CDR(>0) vs CDR(==0) AUROC (GT vs Fake) for 3 ROI summaries with bootstrap CI
(D) Braak monotonicity (block 3, lightweight): tau summary vs Braak stage group (means + CI, trend)
"""

import os, glob, argparse, math, re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F

# ----------------------- Config -----------------------
ATOL_AFFINE = 1e-4
PRINT_EVERY = 50

# 5 anatomical ROIs (same as your old logic).
ROI_FILES = {
    "Hippocampus":        "ROI_Hippocampus.nii.gz",
    "PosteriorCingulate": "ROI_PosteriorCingulate.nii.gz",
    "Precuneus":          "ROI_Precuneus.nii.gz",
    "TemporalLobe":       "ROI_TemporalLobe.nii.gz",
    "LimbicCortex":       "ROI_LimbicCortex.nii.gz",
}

# Extra masks (same as your old logic)
EXTRA_MASKS = {
    "WholeBrain":      "aseg_brainmask.nii.gz",
    "WholeBrain_noBG": "mask_parenchyma_noBG.nii.gz",
}

# For 2B: the three summaries you asked to run
CDR_SUMMARY_ROIS = ["Hippocampus", "TemporalLobe", "WholeBrain_noBG"]

LABEL_COLS = ["Centiloid","MTL","NEO","Braak1_2","Braak3_4","Braak5_6","cdr"]

# ----------------------- Metrics ----------------------
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

@torch.no_grad()
def mmd_gaussian(real: torch.Tensor, fake: torch.Tensor,
                 num_voxels: int = 2048, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
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

# ----------------------- IO helpers -------------------
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

def safe_fname(s: str) -> str:
    s = str(s)
    # replace characters that can break shells or paths
    s = s.replace(" ", "")
    s = s.replace(">=", "ge_").replace("<=", "le_").replace(">", "gt_").replace("<", "lt_")
    s = s.replace("=", "eq_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

# ----------------------- CI utils ---------------------
def _tcrit(df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df)) if df > 0 else float("nan")
    except Exception:
        return 1.96 if df > 0 else float("nan")

def _mean_std_ci(values: List[float]) -> Tuple[float,float,int,float,float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
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
        row = {"roi": roi, "N_subjects": str(len(d["SIDs"]))}
        for k in ["SSIM","PSNR","MSE","MMD"]:
            mean, sd, n, lo, hi = _mean_std_ci(d[k])
            fmt = (lambda z: f"{z:.6f}") if k in ("SSIM","PSNR") else (lambda z: f"{z:.8f}")
            row[f"{k}_mean"] = fmt(mean); row[f"{k}_std"] = fmt(sd)
            row[f"{k}_lo95"] = fmt(lo);   row[f"{k}_hi95"] = fmt(hi)
        out.append(row)
    return out

# ------------------ Subject discovery -----------------
def gather_subjects_from_vols(vols_roots: List[str]) -> Dict[str, str]:
    mapping = {}
    for vr in vols_roots:
        vr = vr.rstrip("/")
        if not os.path.isdir(vr):
            print(f"[WARN] missing vols_root: {vr}")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(vr, "*"))):
            if not os.path.isdir(subj_dir):
                continue
            sid = os.path.basename(subj_dir)
            if sid in mapping:
                continue
            fake_p = os.path.join(subj_dir, "PET_fake.nii.gz")
            gt_p   = os.path.join(subj_dir, "PET_gt.nii.gz")
            if os.path.exists(fake_p) and os.path.exists(gt_p):
                mapping[sid] = vr
    print(f"[DISCOVER] subjects with both PET_gt/fake: {len(mapping)}")
    return mapping

def build_roi_index_case_insensitive(roi_root: str) -> Dict[str, str]:
    idx = {}
    for d in os.listdir(roi_root):
        p = os.path.join(roi_root, d)
        if os.path.isdir(p):
            idx[norm_key(d)] = d
    return idx

def attach_roi_dirs(subject_to_volroot: Dict[str, str], roi_root: str) -> Dict[str, Tuple[str, str]]:
    roi_idx = build_roi_index_case_insensitive(roi_root)
    kept: Dict[str, Tuple[str,str]] = {}
    missed = []
    for sid, vr in subject_to_volroot.items():
        key = norm_key(sid)
        if key in roi_idx:
            kept[sid] = (vr, roi_idx[key])
        else:
            missed.append(sid)
    print(f"[FILTER] subjects having ROI masks (case-insensitive): {len(kept)} (dropped {len(missed)})")
    if missed:
        print(f"[HINT] Example missing (first 10): {missed[:10]}")
    return kept

# ------------------ Meta & splitting ------------------
def load_meta(meta_csv: str, session_col: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    cmap = {c.strip().lower(): c for c in df.columns}
    sess_key = session_col.strip().lower()
    if sess_key not in cmap:
        raise KeyError(f"session_col '{session_col}' not in meta CSV.")
    cols_present = [cmap[c.lower()] for c in LABEL_COLS if c.lower() in cmap]
    out = df[[cmap[sess_key]] + cols_present].copy()
    out["__key__"] = out[cmap[sess_key]].astype(str).map(lambda s: s.strip().lower())
    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    agg = {c: "max" for c in cols_present}  # keep your old behavior
    meta = out.groupby("__key__", as_index=False).agg(agg)
    return meta.set_index("__key__").to_dict(orient="index")

def build_groups(subjects: List[str],
                 meta_map: Dict[str, Dict[str,float]],
                 split_by: str,
                 thr: Optional[float],
                 thr_braak: float,
                 cdr_pos_thr: float) -> Dict[str, set]:
    groups: Dict[str,set] = {}
    if split_by == "none":
        return groups

    if split_by in ("centiloid","mtl","neo","cdr"):
        if thr is None:
            thr = 0.0 if split_by == "cdr" else (18.4 if split_by == "centiloid" else 1.2)
        col = {"centiloid":"Centiloid","mtl":"MTL","neo":"NEO","cdr":"cdr"}[split_by]
        pos, neg = set(), set()
        for s in subjects:
            lab = meta_map.get(norm_key(s), {})
            v = lab.get(col, np.nan)
            if np.isnan(v):
                continue
            if split_by == "cdr":
                (pos if v > thr else neg).add(s)
            else:
                (pos if v >= thr else neg).add(s)
        if split_by == "cdr":
            groups[f"cdr_pos> {thr}"] = pos
            groups[f"cdr_neg<= {thr}"] = neg
        else:
            groups[f"{split_by}_pos>= {thr}"] = pos
            groups[f"{split_by}_neg< {thr}"]  = neg

    elif split_by == "braak":
        s0, s1, s2, s3 = set(), set(), set(), set()
        for s in subjects:
            lab = meta_map.get(norm_key(s), {})
            b12 = lab.get("Braak1_2", np.nan)
            b34 = lab.get("Braak3_4", np.nan)
            b56 = lab.get("Braak5_6", np.nan)
            if any(np.isnan([b12, b34, b56])):
                continue
            stage = 0
            if b12 >= thr_braak: stage = max(stage, 1)
            if b34 >= thr_braak: stage = max(stage, 2)
            if b56 >= thr_braak: stage = max(stage, 3)
            (s0, s1, s2, s3)[stage].add(s)
        groups[f"braak_stage0<{thr_braak}"]  = s0
        groups[f"braak_stage1>={thr_braak}"] = s1
        groups[f"braak_stage2>={thr_braak}"] = s2
        groups[f"braak_stage3>={thr_braak}"] = s3
    else:
        raise ValueError(f"Unknown split_by: {split_by}")
    return groups

def compute_braak_stage_from_meta(meta_row: Dict[str, float], thr_braak: float) -> Optional[int]:
    b12 = meta_row.get("Braak1_2", np.nan)
    b34 = meta_row.get("Braak3_4", np.nan)
    b56 = meta_row.get("Braak5_6", np.nan)
    if any(np.isnan([b12, b34, b56])):
        return None
    stage = 0
    if b12 >= thr_braak: stage = max(stage, 1)
    if b34 >= thr_braak: stage = max(stage, 2)
    if b56 >= thr_braak: stage = max(stage, 3)
    return stage

# ---------------- AUC (no sklearn) --------------------
def roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if y.min() == y.max():
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n_pos = int(y.sum())
    n = len(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)

def bootstrap_auc_ci(y: np.ndarray,
                     scores: np.ndarray,
                     n_boot: int = 2000,
                     seed: int = 0) -> Tuple[float,float,float,int,int,int,str]:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores).astype(float)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0 or n < 3:
        return float("nan"), float("nan"), float("nan"), n, n_pos, n_neg, "only_one_class_or_too_few"
    auc_raw = roc_auc_from_scores(y, s)
    if not math.isfinite(auc_raw):
        return float("nan"), float("nan"), float("nan"), n, n_pos, n_neg, "auc_nan"
    # flip to make auc>=0.5
    if auc_raw < 0.5:
        s = -s
        auc = 1.0 - auc_raw
    else:
        auc = auc_raw
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        sb = s[idx]
        ab = roc_auc_from_scores(yb, sb)
        if math.isfinite(ab):
            aucs.append(ab)
    if len(aucs) < 50:
        return float(auc), float("nan"), float("nan"), n, n_pos, n_neg, "too_few_bootstrap_samples"
    lo, hi = np.quantile(np.asarray(aucs), [0.025, 0.975])
    return float(auc), float(lo), float(hi), n, n_pos, n_neg, ""

# ------------- Per-subject evaluation -----------------
def _compute_one_mask_metrics(x_fake, x_gt, mask_np, device, data_range, mmd_voxels, do_mmd: bool):
    roi_bin = (mask_np > 0).astype(np.float32)
    vox = int(roi_bin.sum())
    if vox == 0:
        return None
    roi_t = to_tensor_5d(roi_bin, device)
    ssim_v = float(ssim3d_masked(x_fake, x_gt, roi_t, data_range=data_range).item())
    mse_v  = float(masked_mse(x_fake, x_gt, roi_t).item())
    psnr_v = float(masked_psnr(x_fake, x_gt, roi_t, data_range=data_range).item())
    if do_mmd:
        mmd_v  = float(mmd_gaussian(x_gt, x_fake, num_voxels=mmd_voxels, mask=roi_t))
    else:
        mmd_v = float("nan")
    return ssim_v, psnr_v, mse_v, mmd_v, vox

def compute_subject_all(subj_for_paths: str,
                        emit_subject_id: str,
                        vols_root: str,
                        roi_root: str,
                        device: torch.device,
                        data_range: float,
                        mmd_voxels: int,
                        strict_affine: bool,
                        do_mmd: bool) -> Tuple[List[Dict[str,str]], Dict[str,float]]:
    """
    Returns:
      long_rows: list of per-ROI fidelity metrics rows (SSIM/PSNR/MSE/MMD)
      means: dict with ROI mean values (GT/Fake) for ROIs + extras
    """
    subj_vol = os.path.join(vols_root, emit_subject_id)
    subj_roi = os.path.join(roi_root,  subj_for_paths)

    fake_p = os.path.join(subj_vol, "PET_fake.nii.gz")
    gt_p   = os.path.join(subj_vol, "PET_gt.nii.gz")
    if not (os.path.exists(fake_p) and os.path.exists(gt_p)):
        return [], {}

    fake_img, fake_np = load_nii(fake_p)
    gt_img,   gt_np   = load_nii(gt_p)
    if fake_np.shape != gt_np.shape:
        return [], {}
    if strict_affine and not affines_close(fake_img, gt_img):
        return [], {}

    x_fake = to_tensor_5d(fake_np, device)
    x_gt   = to_tensor_5d(gt_np,   device)

    long_rows: List[Dict[str,str]] = []
    means: Dict[str,float] = {"subject": emit_subject_id}

    # evaluate a set of masks
    def handle_mask(roi_name: str, roi_path: str):
        nonlocal long_rows, means
        if not os.path.exists(roi_path):
            means[f"{roi_name}_GT"] = float("nan")
            means[f"{roi_name}_Fake"] = float("nan")
            return
        roi_img, roi_np = load_nii(roi_path)
        if roi_np.shape != fake_np.shape:
            means[f"{roi_name}_GT"] = float("nan")
            means[f"{roi_name}_Fake"] = float("nan")
            return
        if strict_affine and not affines_close(roi_img, fake_img):
            means[f"{roi_name}_GT"] = float("nan")
            means[f"{roi_name}_Fake"] = float("nan")
            return

        mask = (roi_np > 0)
        if mask.sum() == 0:
            means[f"{roi_name}_GT"] = float("nan")
            means[f"{roi_name}_Fake"] = float("nan")
            return

        # ROI means (paper-style)
        means[f"{roi_name}_GT"]   = float(gt_np[mask].mean())
        means[f"{roi_name}_Fake"] = float(fake_np[mask].mean())

        # Fidelity metrics (your old style)
        out = _compute_one_mask_metrics(x_fake, x_gt, roi_np, device, data_range, mmd_voxels, do_mmd=do_mmd)
        if out is None:
            return
        ssim_v, psnr_v, mse_v, mmd_v, vox = out
        long_rows.append({
            "subject": emit_subject_id,
            "roi": roi_name,
            "voxels": str(vox),
            "SSIM": f"{ssim_v:.6f}",
            "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
            "MSE":  f"{mse_v:.8f}",
            "MMD":  f"{mmd_v:.8f}" if math.isfinite(mmd_v) else "nan",
        })

    # 5 anatomical ROIs
    for roi_name, roi_file in ROI_FILES.items():
        handle_mask(roi_name, os.path.join(subj_roi, roi_file))

    # WholeBrain masks
    for roi_name, roi_file in EXTRA_MASKS.items():
        handle_mask(roi_name, os.path.join(subj_roi, roi_file))

    return long_rows, means

# ---------------- ROI mean agreement ------------------
def fisher_ci_for_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float,float]:
    if not math.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    zcrit = 1.96  # 95%
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)

def roi_mean_agreement_table(df_means: pd.DataFrame,
                             roi_list: List[str],
                             corravg_rois: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str,float]]:
    """
    Per ROI:
      Pearson r + Fisher CI
      MAE mean/std/CI
      MAPE mean/std/CI
      Bias mean/std/CI (Fake-GT)
    Also returns CorrAVG summary dict.
    """
    rows = []
    r_list = []
    r_rois = []

    for roi in roi_list:
        gt_col = f"{roi}_GT"
        fk_col = f"{roi}_Fake"
        if gt_col not in df_means.columns or fk_col not in df_means.columns:
            continue
        x = df_means[gt_col].to_numpy(dtype=float)
        y = df_means[fk_col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]
        n = int(len(x))
        if n < 3:
            r = float("nan"); rlo = float("nan"); rhi = float("nan")
        else:
            r = float(np.corrcoef(x, y)[0,1])
            rlo, rhi = fisher_ci_for_r(r, n)

        # per-subject errors
        mae_vals = np.abs(y - x)
        # avoid divide-by-zero; SUVR should not be 0 but be safe
        denom = np.clip(np.abs(x), 1e-6, None)
        mape_vals = (np.abs(y - x) / denom) * 100.0  # percent
        bias_vals = (y - x)

        mae_m, mae_sd, _, mae_lo, mae_hi = _mean_std_ci(list(mae_vals))
        mape_m, mape_sd, _, mape_lo, mape_hi = _mean_std_ci(list(mape_vals))
        bias_m, bias_sd, _, bias_lo, bias_hi = _mean_std_ci(list(bias_vals))

        rows.append({
            "roi": roi,
            "N": n,
            "Pearson_r": r,
            "r_lo95": rlo,
            "r_hi95": rhi,
            "MAE_mean": mae_m, "MAE_std": mae_sd, "MAE_lo95": mae_lo, "MAE_hi95": mae_hi,
            "MAPE_mean": mape_m, "MAPE_std": mape_sd, "MAPE_lo95": mape_lo, "MAPE_hi95": mape_hi,
            "Bias_mean": bias_m, "Bias_std": bias_sd, "Bias_lo95": bias_lo, "Bias_hi95": bias_hi,
        })

        if math.isfinite(r):
            r_list.append(r)
            r_rois.append(roi)

    out_df = pd.DataFrame(rows)

    # CorrAVG: mean correlation across chosen ROIs (default: 5 anatomical)
    corravg = {"CorrAVG": float("nan"), "CorrAVG_rois": ""}
    if corravg_rois is None:
        corravg_rois = list(ROI_FILES.keys())

    # pull r per ROI from out_df
    rois_in = []
    rs_in = []
    for roi in corravg_rois:
        sub = out_df[out_df["roi"] == roi]
        if len(sub) == 1 and math.isfinite(float(sub["Pearson_r"].iloc[0])):
            rs_in.append(float(sub["Pearson_r"].iloc[0]))
            rois_in.append(roi)
    if len(rs_in) > 0:
        corravg["CorrAVG"] = float(sum(rs_in) / len(rs_in))
        corravg["CorrAVG_rois"] = ",".join(rois_in)

    return out_df, corravg

# ---------------- Braak trend summary -----------------
def braak_trend_summary(df_means: pd.DataFrame,
                        tau_roi: str,
                        stage_col: str = "braak_stage") -> pd.DataFrame:
    """
    Lightweight Block 3:
    - group means (GT and Fake) of tau_roi by braak_stage (0..3)
    - trend: Pearson correlation between stage and tau
    """
    rows = []
    for modality in ("GT","Fake"):
        col = f"{tau_roi}_{modality}"
        if col not in df_means.columns:
            continue
        for stage in sorted(df_means[stage_col].dropna().unique()):
            d = df_means[df_means[stage_col] == stage][col].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            m, sd, n, lo, hi = _mean_std_ci(list(d))
            rows.append({
                "roi": tau_roi,
                "modality": modality,
                "stage": int(stage),
                "N": int(n),
                "mean": m,
                "std": sd,
                "lo95": lo,
                "hi95": hi
            })

        # trend across all subjects with stage
        sub = df_means[[stage_col, col]].copy()
        sub = sub[np.isfinite(sub[stage_col]) & np.isfinite(sub[col])]
        if len(sub) >= 3:
            stage_vals = sub[stage_col].to_numpy(dtype=float)
            tau_vals = sub[col].to_numpy(dtype=float)
            r = float(np.corrcoef(stage_vals, tau_vals)[0,1])
            rlo, rhi = fisher_ci_for_r(r, len(sub))
        else:
            r, rlo, rhi = float("nan"), float("nan"), float("nan")
        rows.append({
            "roi": tau_roi,
            "modality": modality,
            "stage": -1,  # -1 means "trend"
            "N": int(len(sub)),
            "mean": r,     # store r in mean field for this row
            "std": float("nan"),
            "lo95": rlo,
            "hi95": rhi
        })
    return pd.DataFrame(rows)

# ------------------------- Main -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Comprehensive tau PET evaluation (fidelity + CorrAVG/MAE/MAPE + CDR AUC + Braak trend).")
    ap.add_argument("--roi-root",  required=True, help="Root with ROI masks per subject.")
    ap.add_argument("--vols-roots", nargs="+", required=True,
                    help="One or more roots containing <SUBJECT> dirs with PET_gt.nii.gz & PET_fake.nii.gz.")
    ap.add_argument("--meta-csv",  required=True, help="Meta CSV with labels.")
    ap.add_argument("--session-col", default="TAU_PET_Session",
                    help="Column in meta CSV matching subject folder names.")
    ap.add_argument("--split-by", default="braak", choices=["none","centiloid","mtl","neo","cdr","braak"],
                    help="Which column drives cohort split (always includes ALL).")
    ap.add_argument("--thr", type=float, default=None,
                    help="Threshold for centiloid/mtl/neo/cdr splits. Defaults: centiloid=18.4, mtl/neo=1.2, cdr=0.")
    ap.add_argument("--thr-braak", type=float, default=1.2,
                    help="Threshold used to call PET-Braak sub-stages (default 1.2).")
    ap.add_argument("--cdr-pos-thr", type=float, default=0.0,
                    help="Positive class for AUC: cdr > this (default 0.0 => CDR>0).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--data-range", type=float, default=3.5)
    ap.add_argument("--mmd-voxels", type=int, default=2048)
    ap.add_argument("--skip-mmd", action="store_true", help="Skip MMD (fast). Default: compute MMD.")
    ap.add_argument("--strict-affine", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap resamples for AUC CI (default 2000).")
    ap.add_argument("--braak-trend-roi", default="Hippocampus",
                    choices=list(ROI_FILES.keys()) + list(EXTRA_MASKS.keys()),
                    help="ROI to use for the lightweight Braak trend summary (default Hippocampus).")
    return ap.parse_args()

def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # seeds (helps MMD reproducibility)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Discover subjects across volumes roots
    subj_to_volroot = gather_subjects_from_vols(args.vols_roots)
    subj_index = attach_roi_dirs(subj_to_volroot, args.roi_root)
    subjects = sorted(subj_index.keys())
    if args.limit:
        subjects = subjects[:args.limit]
    print(f"[INFO] Evaluating {len(subjects)} subjects (pooled).")

    # Meta
    meta = load_meta(args.meta_csv, args.session_col)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Main loops
    long_rows_all: List[Dict[str,str]] = []
    mean_rows: List[Dict[str,float]] = []

    for i, sid in enumerate(subjects, 1):
        vols_root, roi_dirname = subj_index[sid]
        if (i % PRINT_EVERY) == 0 or i == 1:
            print(f"[DBG] {i}/{len(subjects)} {sid}  roi_dir='{roi_dirname}'")
        lr, means = compute_subject_all(
            subj_for_paths=roi_dirname,
            emit_subject_id=sid,
            vols_root=vols_root,
            roi_root=args.roi_root,
            device=device,
            data_range=args.data_range,
            mmd_voxels=args.mmd_voxels,
            strict_affine=args.strict_affine,
            do_mmd=(not args.skip_mmd)
        )
        if lr:
            long_rows_all.extend(lr)
        if means:
            # attach key meta fields (including cdr and Braak stage)
            mrow = meta.get(norm_key(sid), {})
            for c in LABEL_COLS:
                means[c] = float(mrow.get(c, np.nan)) if mrow else float("nan")
            stage = compute_braak_stage_from_meta(mrow, args.thr_braak) if mrow else None
            means["braak_stage"] = float(stage) if stage is not None else float("nan")
            mean_rows.append(means)

    # ---- Outputs: per-subject long metrics
    long_all_csv = os.path.join(args.out, "roi_metrics_long_all.csv")
    write_csv(long_all_csv, long_rows_all, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
    print(f"[WRITE] {long_all_csv}")

    # ---- Block 1 (ALL): Fidelity summary (mean/std/95% CI)
    summ_all = summarize_with_ci_subject_level(long_rows_all)
    summ_all_csv = os.path.join(args.out, "roi_metrics_summary_subject_ci_all.csv")
    write_csv(summ_all_csv, summ_all,
              header=["roi","N_subjects",
                      "SSIM_mean","SSIM_std","SSIM_lo95","SSIM_hi95",
                      "PSNR_mean","PSNR_std","PSNR_lo95","PSNR_hi95",
                      "MSE_mean","MSE_std","MSE_lo95","MSE_hi95",
                      "MMD_mean","MMD_std","MMD_lo95","MMD_hi95"])
    print(f"[WRITE] {summ_all_csv}")

    # ---- Outputs: per-subject ROI means table
    df_means = pd.DataFrame(mean_rows)
    means_csv = os.path.join(args.out, "tau_roi_means_with_meta.csv")
    df_means.to_csv(means_csv, index=False)
    print(f"[WRITE] {means_csv}")

    if not long_rows_all:
        print("[WARN] No fidelity rows computed. Exiting after writing means.")
        return

    # ---- Build groups (always include ALL)
    groups = {"ALL": set(subjects)}
    groups.update(build_groups(subjects, meta, args.split_by, args.thr, args.thr_braak, args.cdr_pos_thr))

    # ---- Block 1: Fidelity summary by group
    for gname, gset in groups.items():
        if gname == "ALL":
            continue
        rows_g = [r for r in long_rows_all if r["subject"] in gset]
        if not rows_g:
            print(f"[INFO] Group '{gname}' has 0 fidelity rows; skipping fidelity summaries.")
            continue

        tag = "all" if gname == "ALL" else safe_fname(gname)
        long_g_csv = os.path.join(args.out, f"roi_metrics_long_{tag}.csv")
        write_csv(long_g_csv, rows_g, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        print(f"[WRITE] {long_g_csv}")

        summ = summarize_with_ci_subject_level(rows_g)
        summ_csv = os.path.join(args.out, f"roi_metrics_summary_subject_ci_{tag}.csv")
        write_csv(summ_csv, summ,
                  header=["roi","N_subjects",
                          "SSIM_mean","SSIM_std","SSIM_lo95","SSIM_hi95",
                          "PSNR_mean","PSNR_std","PSNR_lo95","PSNR_hi95",
                          "MSE_mean","MSE_std","MSE_lo95","MSE_hi95",
                          "MMD_mean","MMD_std","MMD_lo95","MMD_hi95"])
        print(f"[WRITE] {summ_csv}")

    # ---- Block 1 (paper-style addition): ROI mean agreement (r, MAE, MAPE, CorrAVG)
    roi_for_mean_agreement = list(ROI_FILES.keys()) + list(EXTRA_MASKS.keys())
    mean_agree_rows = []
    corravg_rows = []
    for gname, gset in groups.items():
        dfg = df_means[df_means["subject"].isin(gset)].copy()
        tag = "all" if gname == "ALL" else safe_fname(gname)
        agree_df, corravg = roi_mean_agreement_table(dfg, roi_for_mean_agreement, corravg_rois=list(ROI_FILES.keys()))
        agree_path = os.path.join(args.out, f"roi_mean_agreement_{tag}.csv")
        agree_df.to_csv(agree_path, index=False)
        print(f"[WRITE] {agree_path}")

        corravg_rows.append({"group": gname, "group_tag": tag, **corravg})

    corravg_df = pd.DataFrame(corravg_rows)
    corravg_path = os.path.join(args.out, f"corravg_summary_{safe_fname(args.split_by)}.csv")
    corravg_df.to_csv(corravg_path, index=False)
    print(f"[WRITE] {corravg_path}")

    # ---- Block 2B: CDR AUROC for 3 ROI summaries (GT vs Fake), by group
    auc_rows = []
    for gname, gset in groups.items():
        dfg = df_means[df_means["subject"].isin(gset)].copy()
        dfg = dfg[np.isfinite(dfg["cdr"].to_numpy(dtype=float))]
        if dfg.empty:
            continue
        y = (dfg["cdr"].to_numpy(dtype=float) > args.cdr_pos_thr).astype(int)

        for roi in CDR_SUMMARY_ROIS:
            for modality in ("GT","Fake"):
                col = f"{roi}_{modality}"
                if col not in dfg.columns:
                    continue
                x = dfg[col].to_numpy(dtype=float)
                ok = np.isfinite(x)
                yy = y[ok]
                xx = x[ok]
                auc, lo, hi, n, npos, nneg, note = bootstrap_auc_ci(
                    yy, xx, n_boot=args.n_bootstrap, seed=args.seed
                )
                auc_rows.append({
                    "group": gname,
                    "roi": roi,
                    "modality": modality,
                    "N": int(n),
                    "N_pos": int(npos),
                    "N_neg": int(nneg),
                    "AUC": auc,
                    "AUC_lo95": lo,
                    "AUC_hi95": hi,
                    "note": note
                })

    auc_df = pd.DataFrame(auc_rows)
    auc_path = os.path.join(args.out, f"cdr_auc_summary_{safe_fname(args.split_by)}.csv")
    auc_df.to_csv(auc_path, index=False)
    print(f"[WRITE] {auc_path}")

    # ---- Block 3 (lightweight): Braak monotonicity summary (uses meta-derived braak_stage)
    # Only meaningful if braak_stage exists
    if "braak_stage" in df_means.columns and df_means["braak_stage"].notna().any():
        bt = braak_trend_summary(df_means, tau_roi=args.braak_trend_roi, stage_col="braak_stage")
        bt_path = os.path.join(args.out, f"braak_trend_{safe_fname(args.braak_trend_roi)}.csv")
        bt.to_csv(bt_path, index=False)
        print(f"[WRITE] {bt_path}")
    else:
        print("[INFO] No braak_stage available; skipping Braak trend summary.")

if __name__ == "__main__":
    main()
