#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

# ----------------------- Config -----------------------
ATOL_AFFINE = 1e-4
PRINT_EVERY = 1

ROI_FILES = {
    "Hippocampus":        "ROI_Hippocampus.nii.gz",
    "PosteriorCingulate": "ROI_PosteriorCingulate.nii.gz",
    "Precuneus":          "ROI_Precuneus.nii.gz",
    "TemporalLobe":       "ROI_TemporalLobe.nii.gz",
    "LimbicCortex":       "ROI_LimbicCortex.nii.gz",
    "WholeBrain" :        "aseg_brainmask.nii.gz",
    "WholeBrain_noBG" :   "mask_parenchyma_noBG.nii.gz"
}
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

# ----------------------- CI utils ---------------------
def _tcrit(df: int) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df)) if df > 0 else float("nan")
    except Exception:
        return 1.96 if df > 0 else float("nan")

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
    """
    Return {subject -> vols_root_that_contains_it}. First hit wins; de-dups subjects across roots.
    Expects each vols_root to have: <vols_root>/<SUBJECT>/{PET_fake.nii.gz,PET_gt.nii.gz}
    """
    mapping = {}
    for vr in vols_roots:
        vr = vr.rstrip("/")
        if not os.path.isdir(vr):
            print(f"[WARN] missing vols_root: {vr}")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(vr, "*"))):
            if not os.path.isdir(subj_dir): continue
            sid = os.path.basename(subj_dir)
            if sid in mapping: continue
            fake_p = os.path.join(subj_dir, "PET_fake.nii.gz")
            gt_p   = os.path.join(subj_dir, "PET_gt.nii.gz")
            if os.path.exists(fake_p) and os.path.exists(gt_p):
                mapping[sid] = vr
    print(f"[DISCOVER] subjects with both PET_gt/fake: {len(mapping)}")
    return mapping

def intersect_with_roi(subject_map: Dict[str,str], roi_root: str) -> Dict[str,str]:
    roi_subjects = {d for d in os.listdir(roi_root) if os.path.isdir(os.path.join(roi_root, d))}
    kept = {s:root for s,root in subject_map.items() if s in roi_subjects}
    print(f"[FILTER] subjects having ROI masks too: {len(kept)} (dropped {len(subject_map) - len(kept)})")
    return kept

# ------------- Per-subject ROI evaluation -------------
def _compute_one_mask(x_fake, x_gt, mask_np, device, data_range, mmd_voxels):
    roi_bin = (mask_np > 0).astype(np.float32)
    if roi_bin.sum() == 0:
        return None
    roi_t = to_tensor_5d(roi_bin, device)
    ssim_v = float(ssim3d_masked(x_fake, x_gt, roi_t, data_range=data_range).item())
    mse_v  = float(masked_mse(x_fake, x_gt, roi_t).item())
    psnr_v = float(masked_psnr(x_fake, x_gt, roi_t, data_range=data_range).item())
    mmd_v  = float(mmd_gaussian(x_gt, x_fake, num_voxels=mmd_voxels, mask=roi_t))
    return ssim_v, psnr_v, mse_v, mmd_v, int(roi_bin.sum())

def compute_subject_roi_metrics(subj: str,
                                vols_root: str,
                                roi_root: str,
                                device: torch.device,
                                data_range: float,
                                mmd_voxels: int,
                                strict_affine: bool) -> List[Dict[str, str]]:
    subj_vol = os.path.join(vols_root, subj)
    subj_roi = os.path.join(roi_root,  subj)
    fake_p = os.path.join(subj_vol, "PET_fake.nii.gz")
    gt_p   = os.path.join(subj_vol, "PET_gt.nii.gz")
    if not (os.path.exists(fake_p) and os.path.exists(gt_p)):
        return []

    fake_img, fake_np = load_nii(fake_p)
    gt_img,   gt_np   = load_nii(gt_p)
    if fake_np.shape != gt_np.shape:
        return []
    if strict_affine and not affines_close(fake_img, gt_img):
        return []

    x_fake = to_tensor_5d(fake_np, device)
    x_gt   = to_tensor_5d(gt_np,   device)

    rows: List[Dict[str, str]] = []

    # 5 anatomical ROIs
    for roi_name, roi_file in ROI_FILES.items():
        roi_path = os.path.join(subj_roi, roi_file)
        if not os.path.exists(roi_path):  # skip missing ROI
            continue
        roi_img, roi_np = load_nii(roi_path)
        if roi_np.shape != fake_np.shape:
            continue
        if strict_affine and not affines_close(roi_img, fake_img):
            continue
        out = _compute_one_mask(x_fake, x_gt, roi_np, device, data_range, mmd_voxels)
        if out is None: 
            continue
        ssim_v, psnr_v, mse_v, mmd_v, vox = out
        rows.append({
            "subject": subj,
            "roi": roi_name,
            "voxels": str(vox),
            "SSIM": f"{ssim_v:.6f}",
            "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
            "MSE":  f"{mse_v:.8f}",
            "MMD":  f"{mmd_v:.8f}",
        })

    # WholeBrain (aseg_brainmask) if present
    wb_path = os.path.join(subj_roi, "aseg_brainmask.nii.gz")
    if os.path.exists(wb_path):
        wb_img, wb_np = load_nii(wb_path)
        if wb_np.shape == fake_np.shape and (not strict_affine or affines_close(wb_img, fake_img)):
            out = _compute_one_mask(x_fake, x_gt, wb_np, device, data_range, mmd_voxels)
            if out is not None:
                ssim_v, psnr_v, mse_v, mmd_v, vox = out
                rows.append({
                    "subject": subj, "roi": "WholeBrain", "voxels": str(vox),
                    "SSIM": f"{ssim_v:.6f}",
                    "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
                    "MSE":  f"{mse_v:.8f}",
                    "MMD":  f"{mmd_v:.8f}",
                })

    # WholeBrain_noBG if present
    nbg_path = os.path.join(subj_roi, "mask_parenchyma_noBG.nii.gz")
    if os.path.exists(nbg_path):
        nbg_img, nbg_np = load_nii(nbg_path)
        if nbg_np.shape == fake_np.shape and (not strict_affine or affines_close(nbg_img, fake_img)):
            out = _compute_one_mask(x_fake, x_gt, nbg_np, device, data_range, mmd_voxels)
            if out is not None:
                ssim_v, psnr_v, mse_v, mmd_v, vox = out
                rows.append({
                    "subject": subj, "roi": "WholeBrain_noBG", "voxels": str(vox),
                    "SSIM": f"{ssim_v:.6f}",
                    "PSNR": f"{psnr_v:.6f}" if math.isfinite(psnr_v) else "inf",
                    "MSE":  f"{mse_v:.8f}",
                    "MMD":  f"{mmd_v:.8f}",
                })

    return rows

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
    agg = {c: "max" for c in cols_present}
    meta = out.groupby("__key__", as_index=False).agg(agg)
    return meta.set_index("__key__").to_dict(orient="index")

def build_groups(subjects: List[str],
                 meta_map: Dict[str, Dict[str,float]],
                 split_by: str,
                 thr: Optional[float],
                 thr_braak: float) -> Dict[str, set]:
    groups: Dict[str,set] = {}
    if split_by in ("centiloid","mtl","neo","cdr"):
        if thr is None:
            thr = 0.0 if split_by == "cdr" else 18.4 if split_by == "centiloid" else 1.2
        col = {"centiloid":"Centiloid","mtl":"MTL","neo":"NEO","cdr":"cdr"}[split_by]
        pos, neg = set(), set()
        for s in subjects:
            lab = meta_map.get(s.strip().lower(), {})
            v = lab.get(col, np.nan)
            if np.isnan(v): continue
            (pos if v >= thr else neg).add(s)
        groups[f"{split_by}_pos>= {thr}"] = pos
        groups[f"{split_by}_neg< {thr}"]  = neg
    elif split_by == "braak":
        s0, s1, s2, s3 = set(), set(), set(), set()
        for s in subjects:
            lab = meta_map.get(s.strip().lower(), {})
            b12 = lab.get("Braak1_2", np.nan)
            b34 = lab.get("Braak3_4", np.nan)
            b56 = lab.get("Braak5_6", np.nan)
            if any(np.isnan([b12, b34, b56])): continue
            stage = 0
            if b12 >= thr_braak: stage = max(stage, 1)
            if b34 >= thr_braak: stage = max(stage, 2)
            if b56 >= thr_braak: stage = max(stage, 3)
            (s0, s1, s2, s3)[stage].add(s)
        groups[f"braak_stage0<{thr_braak}"] = s0
        groups[f"braak_stage1>={thr_braak}"] = s1
        groups[f"braak_stage2>={thr_braak}"] = s2
        groups[f"braak_stage3>={thr_braak}"] = s3
    else:
        raise ValueError(f"Unknown split_by: {split_by}")
    return groups

# ------------------------- Main -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Pooled subject-level ROI evaluation with cohort split (no fold math).")
    ap.add_argument("--roi-root",  required=True, help="Root with ROI masks per subject.")
    ap.add_argument("--vols-roots", nargs="+", required=True,
                    help="One or more roots containing <SUBJECT> dirs with PET_gt.nii.gz & PET_fake.nii.gz.")
    ap.add_argument("--meta-csv",  required=True, help="Meta CSV with labels.")
    ap.add_argument("--session-col", default="TAU_PET_Session",
                    help="Column in meta CSV matching subject folder names.")
    ap.add_argument("--split-by", required=True, choices=["centiloid","mtl","neo","cdr","braak"],
                    help="Which column drives the cohort split.")
    ap.add_argument("--thr", type=float, default=None,
                    help="Threshold for centiloid/mtl/neo/cdr splits (>= thr = positive). If omitted: centiloid=18.4, mtl=1.2, neo=1.2, cdr=0.")
    ap.add_argument("--thr-braak", type=float, default=1.2,
                    help="Threshold used to call Braak sub-stages (default 1.2).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--data-range", type=float, default=3.5)
    ap.add_argument("--mmd-voxels", type=int, default=2048)
    ap.add_argument("--strict-affine", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
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

    # Discover subjects across vols roots and intersect with ROI root
    subj_map = gather_subjects_from_vols(args.vols_roots)
    subj_map = intersect_with_roi(subj_map, args.roi_root)
    subjects = sorted(subj_map.keys())
    if args.limit: subjects = subjects[:args.limit]
    print(f"[INFO] Evaluating {len(subjects)} subjects (pooled across provided volumes roots).")

    # Compute per-subject ROI metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    long_rows: List[Dict[str,str]] = []
    for i, s in enumerate(subjects, 1):
        if (i % PRINT_EVERY) == 0:
            print(f"[DBG] {i}/{len(subjects)} {s}")
        rows = compute_subject_roi_metrics(
            subj=s,
            vols_root=subj_map[s],
            roi_root=args.roi_root,
            device=device,
            data_range=args.data_range,
            mmd_voxels=args.mmd_voxels,
            strict_affine=args.strict_affine
        )
        long_rows.extend(rows)

    # Write combined long CSV
    long_all_csv = os.path.join(args.out, "roi_metrics_long_all.csv")
    write_csv(long_all_csv, long_rows, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
    print(f"[WRITE] {long_all_csv}")

    if not long_rows:
        print("[WARN] No rows computed; exiting.")
        return

    # Load meta and build groups
    meta = load_meta(args.meta_csv, args.session_col)
    groups = build_groups(subjects, meta, args.split_by, args.thr, args.thr_braak)

    # For each group: filter, summarize, write CSVs
    for gname, gset in groups.items():
        rows_g = [r for r in long_rows if r["subject"] in gset]
        if not rows_g:
            print(f"[INFO] Group '{gname}' has 0 rows; skipping.")
            continue
        # long
        long_g_csv = os.path.join(args.out, f"roi_metrics_long_{gname}.csv")
        write_csv(long_g_csv, rows_g, header=["subject","roi","voxels","SSIM","PSNR","MSE","MMD"])
        print(f"[WRITE] {long_g_csv}")
        # subject-level CI summary
        summ = summarize_with_ci_subject_level(rows_g)
        summ_csv = os.path.join(args.out, f"roi_metrics_summary_subject_ci_{gname}.csv")
        write_csv(summ_csv, summ,
                  header=["roi","N_subjects",
                          "SSIM_mean","SSIM_std","SSIM_lo95","SSIM_hi95",
                          "PSNR_mean","PSNR_std","PSNR_lo95","PSNR_hi95",
                          "MSE_mean","MSE_std","MSE_lo95","MSE_hi95",
                          "MMD_mean","MMD_std","MMD_lo95","MMD_hi95"])
        print(f"[WRITE] {summ_csv}")

if __name__ == "__main__":
    main()
