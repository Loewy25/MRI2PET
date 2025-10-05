#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, csv, argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib

# ----- Your ROIs (exact filenames expected in each subject's ROI folder) -----
ROI_FILES: Dict[str, str] = {
    "Hippocampus":        "ROI_Hippocampus.nii.gz",
    "PosteriorCingulate": "ROI_PosteriorCingulate.nii.gz",
    "Precuneus":          "ROI_Precuneus.nii.gz",
    "TemporalLobe":       "ROI_TemporalLobe.nii.gz",
    "LimbicCortex":       "ROI_LimbicCortex.nii.gz",
}

# ----- Simple IO -----
def load_nii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    return np.asarray(img.get_fdata()), img.affine

def write_csv(path: str, rows: List[Dict[str, str]], header: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ----- Classic histogram MI (bits) -----
def entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else float("nan")

def mi_bits(x: np.ndarray, y: np.ndarray, bins: int = 64) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return float("nan")
    xr = (float(x.min()), float(x.max()))
    yr = (float(y.min()), float(y.max()))
    # minimal guard for degenerate ranges
    if xr[1] <= xr[0]: xr = (xr[0], xr[0] + 1e-6)
    if yr[1] <= yr[0]: yr = (yr[0], yr[0] + 1e-6)

    H, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    Hx  = entropy_bits(Px)
    Hy  = entropy_bits(Py)
    Hxy = entropy_bits(Pxy.ravel())
    return Hx + Hy - Hxy

def mi_in_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray, bins: int = 64) -> Tuple[float, int]:
    m = (mask > 0)
    if m.sum() == 0:
        return float("nan"), 0
    return mi_bits(x[m], y[m], bins=bins), int(m.sum())

# ----- Per-subject computation -----
def compute_subject(
    subj_dir: str,
    roi_root: Optional[str],
    bins: int,
    verbose: bool
) -> List[Dict[str, str]]:
    sid = os.path.basename(subj_dir.rstrip("/"))
    t1_p  = os.path.join(subj_dir, "T1_masked.nii.gz")
    pet_p = os.path.join(subj_dir, "PET_in_T1_masked.nii.gz")
    msk_p = os.path.join(subj_dir, "aseg_brainmask.nii.gz")

    if not (os.path.exists(t1_p) and os.path.exists(pet_p) and os.path.exists(msk_p)):
        print(f"[MISS] {sid}: required files not found; skip.")
        return []

    t1,  _ = load_nii(t1_p)
    pet, _ = load_nii(pet_p)
    msk, _ = load_nii(msk_p)

    if verbose:
        print(f"[DBG] {sid} shape={t1.shape}  T1[min,max]=({t1.min():.4g},{t1.max():.4g})  PET[min,max]=({pet.min():.4g},{pet.max():.4g})")

    rows: List[Dict[str, str]] = []

    # Whole-brain MI (brain mask voxels)
    mi_b, v_b = mi_in_mask(t1, pet, msk, bins=bins)
    rows.append({"subject": sid, "region": "WholeBrain", "voxels": str(v_b), "MI_bits": f"{mi_b:.6f}" if np.isfinite(mi_b) else "nan"})

    # ROI MI (exact subject folder match under roi_root)
    if roi_root:
        roi_dir = os.path.join(roi_root, sid)
        if os.path.isdir(roi_dir):
            for roi_name, roi_file in ROI_FILES.items():
                rp = os.path.join(roi_dir, roi_file)
                if not os.path.exists(rp):
                    if verbose: print(f"[WARN] {sid}: missing {roi_file}")
                    continue
                r, _ = load_nii(rp)
                if r.shape != t1.shape:
                    print(f"[WARN] {sid}: {roi_file} shape {r.shape} != MRI {t1.shape}; skip ROI.")
                    continue
                mi_r, v_r = mi_in_mask(t1, pet, r, bins=bins)
                rows.append({"subject": sid, "region": roi_name, "voxels": str(v_r), "MI_bits": f"{mi_r:.6f}" if np.isfinite(mi_r) else "nan"})
        else:
            print(f"[NOTE] {sid}: no ROI folder at {roi_dir}; skipping ROIs.")

    return rows

# ----- Aggregation (mean & std per region) -----
def summarize(long_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    acc: Dict[str, List[float]] = {}
    for r in long_rows:
        reg = r["region"]
        val = float(r["MI_bits"]) if r["MI_bits"] != "nan" else np.nan
        acc.setdefault(reg, []).append(val)

    out = []
    for reg in sorted(acc.keys()):
        vals = np.array([v for v in acc[reg] if np.isfinite(v)], dtype=float)
        n = vals.size
        mean = float(vals.mean()) if n else float("nan")
        std  = float(vals.std(ddof=1)) if n > 1 else (0.0 if n == 1 else float("nan"))
        out.append({
            "region": reg,
            "N": str(n),
            "MI_mean_bits": f"{mean:.6f}" if np.isfinite(mean) else "nan",
            "MI_std_bits":  f"{std:.6f}"  if np.isfinite(std)  else "nan",
        })
    return out

# ----- CLI -----
def parse_args():
    ap = argparse.ArgumentParser(description="Compute MRI–PET mutual information (bits): WholeBrain + ROI per subject.")
    ap.add_argument("--root-dir", required=True, help="Training root with subject folders (*_av1451_*).")
    ap.add_argument("--roi-root",  default=None, help="ROI root with per-subject ROI folders (same shape/grid).")
    ap.add_argument("--out",       default="mi_results", help="Output dir for global CSVs.")
    ap.add_argument("--bins",      type=int, default=64, help="Histogram bins for MI (default 64).")
    ap.add_argument("--limit",     type=int, default=None, help="Limit number of subjects (debug).")
    ap.add_argument("--verbose",   action="store_true", help="Print debug info (shapes, missing files).")
    return ap.parse_args()

def main():
    args = parse_args()
    patterns = [os.path.join(args.root_dir, "*_av1451_*"),
                os.path.join(args.root_dir, "*_AV1451_*")]
    subjects = sorted([d for p in patterns for d in glob.glob(p) if os.path.isdir(d)])
    if args.limit: subjects = subjects[:args.limit]
    print(f"[INFO] Found {len(subjects)} subject folders under {args.root_dir}")

    all_rows: List[Dict[str, str]] = []
    for i, sd in enumerate(subjects, 1):
        sid = os.path.basename(sd.rstrip("/"))
        print(f"[{i}/{len(subjects)}] {sid}")
        rows = compute_subject(sd, args.roi_root, bins=args.bins, verbose=args.verbose)
        if rows:
            sub_csv = os.path.join(sd, "mi_metrics.csv")
            write_csv(sub_csv, rows, header=["subject","region","voxels","MI_bits"])
            all_rows.extend(rows)

    if all_rows:
        os.makedirs(args.out, exist_ok=True)
        long_csv = os.path.join(args.out, "mi_long.csv")
        write_csv(long_csv, all_rows, header=["subject","region","voxels","MI_bits"])
        summ_csv = os.path.join(args.out, "mi_summary.csv")
        write_csv(summ_csv, summarize(all_rows), header=["region","N","MI_mean_bits","MI_std_bits"])
        print(f"[INFO] Wrote {long_csv}")
        print(f"[INFO] Wrote {summ_csv}")
    else:
        print("[WARN] No MI rows produced.")

if __name__ == "__main__":
    main()
