#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create one exclusion mask per subject:
  mask_parenchyma_noBG.nii.gz  = brain parenchyma minus basal ganglia

Mirrors your ROI script's logic:
- Subject discovery under OUT_ROOT
- T1 resolution/affine as target
- PUP CNDA timestamp -> closest FreeSurfer run -> aseg.mgz
- Resample aseg to T1 if grid/affine differ (nearest)
- Build final mask and store alongside your ROIs

Requires: nibabel, numpy, FreeSurfer (mri_vol2vol on PATH)
"""

import os, re, glob, shutil, subprocess, sys
from datetime import datetime
import numpy as np
import nibabel as nib

# =========================
# CONFIG
# =========================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_brainv11"  # same parent used by your ROI script
RUN_LIMIT   = None
ATOL_AFFINE = 1e-4
DEBUG       = True

# If your OUT_ROOT subjects correspond to a different tracer token, change this:
TRACER_TOKEN = "AV1451"   # e.g., "T807" or keep "AV1451" to mirror your ROI script

# --- Label sets ---
# Strict basal ganglia: Caudate, Putamen, Pallidum, Nucleus Accumbens (L/R)
BG_IDS_STRICT = {11, 50, 12, 51, 13, 52, 26, 58}
INCLUDE_VENTRAL_DC = False  # set True if your PI wants to exclude VentralDC too
BG_IDS = (BG_IDS_STRICT | {28, 60}) if INCLUDE_VENTRAL_DC else BG_IDS_STRICT

# Parenchyma = tissue mask (GM+WM + cerebellum + subcortical + brainstem + VentralDC); excludes CSF
# Matches your earlier KEEP_LABELS
PARENCHYMA_IDS = {
    2, 41, 3, 42, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51,
    13, 52, 17, 53, 18, 54, 26, 58, 28, 60, 16
}

# =========================
# LOGGING
# =========================
def log(msg, level="INFO"):
    if DEBUG or level in ("WARN", "ERROR"):
        print(f"[{level}] {msg}", flush=True)

# =========================
# HELPERS (mirror your ROI script)
# =========================
def extract_subject_code(name: str):
    """Get '1092_385' from '1092_385_AV1451_v1'."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", name)
    return m.group(0) if m else None

def find_pup_subject_dir(pup_root, subj_code):
    """Locate PUP subject dir that contains subj_code and the tracer token."""
    cands = []
    for d in glob.glob(os.path.join(pup_root, "*")):
        b = os.path.basename(d)
        if os.path.isdir(d) and subj_code in b and (TRACER_TOKEN.lower() in b.lower()):
            cands.append(d)
    cands.sort()
    return cands[-1] if cands else None

def find_pup_nifti_dir(pup_dir):
    """Pick .../CNDA*/NIFTI_GZ (latest if multiple)."""
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ")) if pup_dir else []
    if not hits: return None
    hits.sort()
    return hits[-1]

def parse_cnda_timestamp_from_name(name: str):
    """Parse 'YYYYmmddHHMMSS' from a CNDA folder name string."""
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m: return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None

def find_fs_subject_dir(fs_root, subj_code):
    """Locate the freesurfer subject folder whose name contains subj_code."""
    cands = [d for d in glob.glob(os.path.join(fs_root, "*"))
             if os.path.isdir(d) and subj_code in os.path.basename(d)]
    if not cands: return None
    def rank(name):
        n = name.lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2
    cands.sort(key=lambda p: rank(os.path.basename(p)))
    return cands[0]

def _run_dirs(fs_subject_dir):
    return glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))

def _find_label_in_run(run_dir, fname):
    hits = glob.glob(os.path.join(run_dir, "DATA", "*", "mri", fname))
    hits.sort()
    return hits[-1] if hits else None

def find_fs_aseg_closest(fs_subject_dir, target_dt):
    """
    Return (aseg_path, used_run_dir) from the FS run closest to target_dt.
    Fallback: first aseg.mgz found anywhere under the subject tree.
    """
    runs = _run_dirs(fs_subject_dir)
    best, best_diff = None, float("inf")
    for rd in runs:
        dt = parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if diff < best_diff:
            best, best_diff = rd, diff
    if best:
        aseg = _find_label_in_run(best, "aseg.mgz")
        if aseg: return aseg, best
    # Fallback search
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files:
            return os.path.join(root, "aseg.mgz"), None
    return None, None

def shapes_affines_match(a_path, b_path, atol=ATOL_AFFINE):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok   = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and aff_ok, shape_ok, aff_ok, ia.shape, ib.shape, ia.affine, ib.affine

def resample_label_to_target(label_path, target_path, out_path):
    """Use FreeSurfer mri_vol2vol (nearest) to resample label → target grid."""
    if not shutil.which("mri_vol2vol"):
        raise RuntimeError("mri_vol2vol not found. Source your FreeSurfer env.")
    cmd = [
        "mri_vol2vol", "--mov", label_path, "--targ", target_path,
        "--regheader", "--interp", "nearest", "--o", out_path
    ]
    log("RUN: " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-20:])
        log("mri_vol2vol stderr (tail):\n" + tail, level="ERROR")
        raise RuntimeError(f"mri_vol2vol failed: {r.returncode}")
    return out_path

# =========================
# CORE
# =========================
def main():
    # Sanity
    for p, isdir in [(OUT_ROOT, True), (PUP_ROOT, True), (FS_ROOT, True)]:
        if isdir and not os.path.isdir(p):
            log(f"Missing required dir: {p}", level="ERROR"); sys.exit(1)

    # Subject list (same pattern as your ROI script)
    subjects = [d for d in sorted(os.listdir(OUT_ROOT))
                if os.path.isdir(os.path.join(OUT_ROOT, d))
                and re.search(r"^\d{4}_\d{3,4}_", d)]
    if RUN_LIMIT:
        subjects = subjects[:RUN_LIMIT]
    log(f"Found {len(subjects)} subjects under {OUT_ROOT}")

    ok = skipped = 0

    for i, subj in enumerate(subjects, 1):
        subj_dir  = os.path.join(OUT_ROOT, subj)
        subj_code = extract_subject_code(subj)
        log(f"\n[{i}/{len(subjects)}] Subject: {subj} (code={subj_code})")

        # --- T1: prefer local; fallback to PUP ---
        t1_path = os.path.join(subj_dir, "T1.nii.gz")
        if not os.path.exists(t1_path):
            pup_dir   = find_pup_subject_dir(PUP_ROOT, subj_code) if subj_code else None
            nifti_dir = find_pup_nifti_dir(pup_dir) if pup_dir else None
            t1_alt    = os.path.join(nifti_dir, "T1.nii.gz") if nifti_dir else None
            if t1_alt and os.path.exists(t1_alt):
                t1_path = t1_alt
                log(f"  WARN: T1.nii.gz not in subject folder; using PUP T1: {t1_path}")
            else:
                log("  ERROR: Missing T1.nii.gz; SKIP.", level="ERROR")
                skipped += 1;  continue

        # --- Closest FS aseg.mgz by CNDA timestamp ---
        pup_dir   = find_pup_subject_dir(PUP_ROOT, subj_code) if subj_code else None
        nifti_dir = find_pup_nifti_dir(pup_dir) if pup_dir else None
        pup_dt    = parse_cnda_timestamp_from_name(os.path.basename(os.path.dirname(nifti_dir))) if nifti_dir else None

        fs_dir    = find_fs_subject_dir(FS_ROOT, subj_code) if subj_code else None
        if not fs_dir:
            log("  ERROR: No matching FreeSurfer subject dir; SKIP.", level="ERROR")
            skipped += 1;  continue

        aseg_path, used_run = find_fs_aseg_closest(fs_dir, pup_dt)
        if not aseg_path:
            log("  ERROR: aseg.mgz not found; SKIP.", level="ERROR")
            skipped += 1;  continue

        log(f"  FS subject dir: {fs_dir}")
        log(f"  FS run used: {used_run if used_run else 'fallback-search'}")
        log(f"  aseg: {aseg_path}")

        # Optional transparency: report Δt to chosen run (if both times known)
        if used_run and pup_dt:
            run_dt = parse_cnda_timestamp_from_name(os.path.basename(used_run))
            if run_dt:
                dt_hours = abs((run_dt - pup_dt).total_seconds()) / 3600.0
                log(f"  Δt(run vs PUP) ≈ {dt_hours:.2f} hours")

        # --- Ensure aseg aligns with T1 (shape + affine); resample if needed ---
        try:
            same, shape_ok, aff_ok, a_shape, t_shape, a_aff, t_aff = shapes_affines_match(aseg_path, t1_path, ATOL_AFFINE)
        except Exception as e:
            log(f"  ERROR: Could not load volumes to compare shapes/affines: {e}", level="ERROR")
            skipped += 1;  continue

        if not same:
            log(f"  RESAMPLE: aseg → T1 (shape_ok={shape_ok}, affine_ok={aff_ok})")
            log(f"    aseg.shape={a_shape}  T1.shape={t_shape}")
            log("    aseg.affine:\n" + np.array2string(a_aff, precision=5))
            log("    T1.affine:\n"   + np.array2string(t_aff, precision=5))
            aseg_inT1 = os.path.join(subj_dir, "aseg_inT1.nii.gz")
            try:
                resample_label_to_target(aseg_path, t1_path, aseg_inT1)
            except Exception as e:
                log(f"  ERROR: mri_vol2vol failed: {e}", level="ERROR")
                skipped += 1;  continue
        else:
            aseg_inT1 = aseg_path
            log("  aseg aligns with T1 → no resample")

        # --- Build masks in memory ---
        aseg_img  = nib.load(aseg_inT1)
        aseg_data = np.asanyarray(aseg_img.dataobj)

        # BG bool mask
        bg_mask = np.isin(aseg_data, list(BG_IDS))

        # Parenchyma: prefer existing aseg_brainmask.nii.gz; if mismatch, rebuild from aseg_inT1
        paren_path = os.path.join(subj_dir, "aseg_brainmask.nii.gz")
        if os.path.exists(paren_path):
            paren_img  = nib.load(paren_path)
            paren_mask = np.asanyarray(paren_img.dataobj).astype(bool)
            same_grid  = (paren_mask.shape == bg_mask.shape) and np.allclose(paren_img.affine, aseg_img.affine, atol=ATOL_AFFINE)
            if not same_grid:
                log("  WARN: Existing parenchyma mask grid/affine mismatch; rebuilding from aseg_inT1")
                paren_mask = np.isin(aseg_data, list(PARENCHYMA_IDS))
        else:
            paren_mask = np.isin(aseg_data, list(PARENCHYMA_IDS))

        # Safety check on shapes
        if paren_mask.shape != bg_mask.shape:
            log(f"  ERROR: Mask grid mismatch (parenchyma {paren_mask.shape} vs BG {bg_mask.shape}); SKIP.", level="ERROR")
            skipped += 1;  continue

        # Final composite: parenchyma minus BG
        final_mask = (paren_mask & ~bg_mask).astype(np.uint8)
        out_path   = os.path.join(subj_dir, "mask_parenchyma_noBG.nii.gz")
        nib.save(nib.Nifti1Image(final_mask, aseg_img.affine), out_path)
        log(f"  Wrote: {out_path} (voxels={int(final_mask.sum())})")

        ok += 1

    # Summary
    log(f"\nDONE. Wrote no-BG masks for {ok} subjects; skipped {skipped}.")
    if skipped:
        log("Check errors above for shape/affine issues or missing inputs.", level="WARN")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled error: {e}", level="ERROR")
        sys.exit(2)
