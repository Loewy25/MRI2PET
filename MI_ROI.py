#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, argparse
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib

# ---- ROI filenames expected inside each subject folder ----
ROI_FILES: Dict[str, str] = {
    "Hippocampus":        "ROI_Hippocampus.nii.gz",
    "PosteriorCingulate": "ROI_PosteriorCingulate.nii.gz",
    "Precuneus":          "ROI_Precuneus.nii.gz",
    "TemporalLobe":       "ROI_TemporalLobe.nii.gz",
    "LimbicCortex":       "ROI_LimbicCortex.nii.gz",
}

# ---- IO helpers ----
def load_nii(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())

def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ---- Normalization (matches your dataset loader semantics) ----
def zscore_in_brain(mri: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    m = brain_mask > 0
    x = mri.astype(np.float32)
    vals = x[m]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mean = float(vals.mean())
    std  = float(vals.std() + 1e-6)
    z = (x - mean) / std
    z[~m] = 0.0
    return z

def pet_mask_outside(pet: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    x = pet.astype(np.float32).copy()
    x[~(brain_mask > 0)] = 0.0
    return x

# ---- Entropy / MI / NMI (histogram estimator, MI in bits; NMI = (Hx+Hy)/Hxy) ----
def _entropy_bits_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else float("nan")

def mi_nmi_bits(x: np.ndarray, y: np.ndarray, bins: int = 64) -> Tuple[float, float]:
    """
    Return (MI_bits, NMI_SU) where
      MI_bits = Hx + Hy - Hxy  (bits, base-2)
      NMI_SU  = 2*MI / (Hx + Hy)  in [0, 1]   # symmetric uncertainty
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan"), float("nan")

    xr = (float(x.min()), float(x.max()))
    yr = (float(y.min()), float(y.max()))
    if xr[1] <= xr[0]: xr = (xr[0], xr[0] + 1e-6)
    if yr[1] <= yr[0]: yr = (yr[0], yr[0] + 1e-6)

    H, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    if H.sum() == 0:
        return float("nan"), float("nan")

    Pxy = H / H.sum()
    Px  = Pxy.sum(axis=1)
    Py  = Pxy.sum(axis=0)

    def _H_bits(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p))) if p.size else float("nan")

    Hx  = _H_bits(Px)
    Hy  = _H_bits(Py)
    Hxy = _H_bits(Pxy.ravel())
    MI  = Hx + Hy - Hxy

    denom = (Hx + Hy)
    NMI_SU = (2.0 * MI) / denom if np.isfinite(denom) and denom > 0 else float("nan")
    return float(MI), float(NMI_SU)


def mi_nmi_in_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray, bins: int = 64) -> Tuple[float, float, int]:
    m = (mask > 0)
    n = int(m.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    MI, NMI = mi_nmi_bits(x[m], y[m], bins=bins)
    return MI, NMI, n

# ---- Per-subject computation ----
def compute_subject(subj_dir: str, bins: int, verbose: bool) -> List[Dict[str, str]]:
    sid  = os.path.basename(subj_dir.rstrip("/"))
    t1_p = os.path.join(subj_dir, "T1_masked.nii.gz")
    pt_p = os.path.join(subj_dir, "PET_in_T1_masked.nii.gz")
    bm_p = os.path.join(subj_dir, "aseg_brainmask.nii.gz")

    required = [t1_p, pt_p, bm_p]
    if not all(os.path.exists(p) for p in required):
        print(f"[MISS] {sid}: missing one of {', '.join(os.path.basename(p) for p in required)}")
        return []

    t1  = load_nii(t1_p)
    pet = load_nii(pt_p)
    bm  = load_nii(bm_p)

    if t1.shape != pet.shape or bm.shape != t1.shape:
        print(f"[WARN] {sid}: shape mismatch (T1 {t1.shape}, PET {pet.shape}, mask {bm.shape}); skipping.")
        return []

    if verbose:
        print(f"[DBG] {sid}: shape={t1.shape} | T1[min,max]=({t1.min():.4g},{t1.max():.4g}) | PET[min,max]=({pet.min():.4g},{pet.max():.4g})")

    # Normalized counterparts (MRI z-score within brain; PET masked outside brain)
    t1n  = zscore_in_brain(t1, bm)
    petn = pet_mask_outside(pet, bm)

    rows: List[Dict[str, str]] = []

    # Whole brain (brain mask)
    mi_raw,  nmi_raw,  vox_b = mi_nmi_in_mask(t1,  pet,  bm, bins=bins)
    mi_norm, nmi_norm, _     = mi_nmi_in_mask(t1n, petn, bm, bins=bins)
    rows.append({
        "subject": sid, "region": "WholeBrain", "voxels": str(vox_b),
        "MI_raw_bits":  f"{mi_raw:.6f}"  if np.isfinite(mi_raw)  else "nan",
        "NMI_raw":      f"{nmi_raw:.6f}" if np.isfinite(nmi_raw) else "nan",
        "MI_norm_bits": f"{mi_norm:.6f}" if np.isfinite(mi_norm) else "nan",
        "NMI_norm":     f"{nmi_norm:.6f}"if np.isfinite(nmi_norm)else "nan",
    })

    # ROI-specific (from the SAME folder)
    for roi_name, roi_file in ROI_FILES.items():
        rp = os.path.join(subj_dir, roi_file)
        if not os.path.exists(rp):
            if verbose: print(f"[WARN] {sid}: missing {roi_file}")
            continue
        r = load_nii(rp)
        if r.shape != t1.shape:
            print(f"[WARN] {sid}: {roi_file} shape {r.shape} != MRI {t1.shape}; skip.")
            continue

        mi_raw,  nmi_raw,  vox_r = mi_nmi_in_mask(t1,  pet,  r, bins=bins)
        mi_norm, nmi_norm, _     = mi_nmi_in_mask(t1n, petn, r, bins=bins)
        rows.append({
            "subject": sid, "region": roi_name, "voxels": str(vox_r),
            "MI_raw_bits":  f"{mi_raw:.6f}"  if np.isfinite(mi_raw)  else "nan",
            "NMI_raw":      f"{nmi_raw:.6f}" if np.isfinite(nmi_raw) else "nan",
            "MI_norm_bits": f"{mi_norm:.6f}" if np.isfinite(mi_norm) else "nan",
            "NMI_norm":     f"{nmi_norm:.6f}"if np.isfinite(nmi_norm)else "nan",
        })

    return rows

# ---- Aggregation (mean & std per region, raw and normalized) ----
def summarize(long_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for r in long_rows:
        reg = r["region"]
        bucket.setdefault(reg, {"MI_raw": [], "NMI_raw": [], "MI_norm": [], "NMI_norm": []})
        def f(key): 
            v = r[key]
            return float(v) if v != "nan" else np.nan
        bucket[reg]["MI_raw"].append(f("MI_raw_bits"))
        bucket[reg]["NMI_raw"].append(f("NMI_raw"))
        bucket[reg]["MI_norm"].append(f("MI_norm_bits"))
        bucket[reg]["NMI_norm"].append(f("NMI_norm"))

    def mean_std(arr: List[float]) -> Tuple[float, float, int]:
        a = np.array([x for x in arr if np.isfinite(x)], dtype=float)
        n = a.size
        if n == 0: return float("nan"), float("nan"), 0
        m = float(a.mean())
        s = float(a.std(ddof=1)) if n > 1 else 0.0
        return m, s, n

    out = []
    for reg in sorted(bucket.keys()):
        mi_r_m, mi_r_s, n_mi_r   = mean_std(bucket[reg]["MI_raw"])
        nmi_r_m, nmi_r_s, n_nmi_r= mean_std(bucket[reg]["NMI_raw"])
        mi_n_m, mi_n_s, n_mi_n   = mean_std(bucket[reg]["MI_norm"])
        nmi_n_m, nmi_n_s, n_nmi_n= mean_std(bucket[reg]["NMI_norm"])
        out.append({
            "region": reg,
            "N_MI_raw":   str(n_mi_r),
            "MI_raw_mean_bits":  f"{mi_r_m:.6f}"  if np.isfinite(mi_r_m)  else "nan",
            "MI_raw_std_bits":   f"{mi_r_s:.6f}"  if np.isfinite(mi_r_s)  else "nan",
            "N_NMI_raw":  str(n_nmi_r),
            "NMI_raw_mean":      f"{nmi_r_m:.6f}" if np.isfinite(nmi_r_m) else "nan",
            "NMI_raw_std":       f"{nmi_r_s:.6f}" if np.isfinite(nmi_r_s) else "nan",
            "N_MI_norm":  str(n_mi_n),
            "MI_norm_mean_bits": f"{mi_n_m:.6f}"  if np.isfinite(mi_n_m)  else "nan",
            "MI_norm_std_bits":  f"{mi_n_s:.6f}"  if np.isfinite(mi_n_s)  else "nan",
            "N_NMI_norm": str(n_nmi_n),
            "NMI_norm_mean":     f"{nmi_n_m:.6f}" if np.isfinite(nmi_n_m) else "nan",
            "NMI_norm_std":      f"{nmi_n_s:.6f}" if np.isfinite(nmi_n_s) else "nan",
        })
    return out

# ---- CLI ----
def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute MRI–PET MI (bits) and NMI per subject (WholeBrain + ROIs), both RAW and NORMALIZED, from a single root."
    )
    ap.add_argument("--root-dir", required=True, help="Root containing subject folders.")
    ap.add_argument("--out",      default="mi_results", help="Output dir for global CSVs.")
    ap.add_argument("--bins",     type=int, default=64, help="Histogram bins for MI/NMI.")
    ap.add_argument("--limit",    type=int, default=None, help="Limit number of subjects (debug).")
    ap.add_argument("--verbose",  action="store_true", help="Print debug info.")
    return ap.parse_args()

def main():
    args = parse_args()

    req = {"T1_masked.nii.gz", "PET_in_T1_masked.nii.gz", "aseg_brainmask.nii.gz"}
    cand = [os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
            if os.path.isdir(os.path.join(args.root_dir, d))]
    subjects = [d for d in sorted(cand) if req.issubset(set(os.listdir(d)))]
    if args.limit: subjects = subjects[:args.limit]

    print(f"[INFO] Found {len(subjects)} subjects under {args.root_dir}")

    all_rows: List[Dict[str, str]] = []
    for i, sd in enumerate(subjects, 1):
        sid = os.path.basename(sd.rstrip("/"))
        print(f"[{i}/{len(subjects)}] {sid}")
        rows = compute_subject(sd, bins=args.bins, verbose=args.verbose)
        if rows:
            sub_csv = os.path.join(sd, "mi_metrics.csv")
            write_csv(sub_csv, rows,
                      header=["subject","region","voxels","MI_raw_bits","NMI_raw","MI_norm_bits","NMI_norm"])
            all_rows.extend(rows)

    if all_rows:
        os.makedirs(args.out, exist_ok=True)
        long_csv = os.path.join(args.out, "mi_long.csv")
        write_csv(long_csv, all_rows,
                  header=["subject","region","voxels","MI_raw_bits","NMI_raw","MI_norm_bits","NMI_norm"])
        summ_csv = os.path.join(args.out, "mi_summary.csv")
        write_csv(summ_csv, summarize(all_rows),
                  header=["region",
                          "N_MI_raw","MI_raw_mean_bits","MI_raw_std_bits",
                          "N_NMI_raw","NMI_raw_mean","NMI_raw_std",
                          "N_MI_norm","MI_norm_mean_bits","MI_norm_std_bits",
                          "N_NMI_norm","NMI_norm_mean","NMI_norm_std"])
        print(f"[INFO] Wrote {long_csv}")
        print(f"[INFO] Wrote {summ_csv}")
    else:
        print("[WARN] No MI/NMI rows produced.")

if __name__ == "__main__":
    main()
