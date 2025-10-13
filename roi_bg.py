#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Make mask_parenchyma_noBG.nii.gz per subject, but **force visit=v1** matching:
# - Pick PUP subject dir that matches subj_code + TRACER_TOKEN + v1
# - Pick FreeSurfer subject dir that matches subj_code + v1
# - Then proceed as before (resample aseg→T1 if needed; parenchyma \ BG)

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

OUT_ROOT  = "/scratch/l.peiwang/kari_brainv11"
RUN_LIMIT   = None
ATOL_AFFINE = 1e-4
DEBUG       = True

TRACER_TOKEN = "AV1451"   # change to "T807" etc if needed
VISIT_TOKEN  = "v1"       # << enforce v1 only

# Strict BG set (Caudate, Putamen, Pallidum, Accumbens; L/R)
BG_IDS_STRICT = {11, 50, 12, 51, 13, 52, 26, 58}
INCLUDE_VENTRAL_DC = False
BG_IDS = (BG_IDS_STRICT | {28, 60}) if INCLUDE_VENTRAL_DC else BG_IDS_STRICT

# Parenchyma fallback set (tissue only; matches your KEEP_LABELS)
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
# HELPERS
# =========================
def extract_subject_code(name: str):
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", name)
    return m.group(0) if m else None

def has_visit_token(s: str, visit: str) -> bool:
    s = s.lower(); visit = visit.lower()
    # match visit token as a standalone token (avoid v10 matching v1)
    return re.search(rf'(^|[^a-z0-9]){re.escape(visit)}($|[^a-z0-9])', s) is not None

def find_pup_subject_dir_v1(pup_root, subj_code, tracer_token, visit_token):
    """
    Find PUP subject dir whose basename contains subj_code AND tracer_token AND visit_token (v1).
    If multiple, return the last in lexicographic order.
    """
    cands = []
    for d in glob.glob(os.path.join(pup_root, "*")):
        if not os.path.isdir(d): 
            continue
        b = os.path.basename(d)
        if (subj_code in b) and (tracer_token.lower() in b.lower()) and has_visit_token(b, visit_token):
            cands.append(d)
    cands.sort()
    return cands[-1] if cands else None

def find_pup_nifti_dir(pup_dir):
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ")) if pup_dir else []
    hits.sort()
    return hits[-1] if hits else None

def parse_cnda_timestamp_from_name(name: str):
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m: return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None

def _rank_fs_name(name: str):
    n = name.lower()
    if "mri" in n: return 0
    if "mmr" in n: return 1
    return 2

def find_fs_subject_dir_v1(fs_root, subj_code, visit_token):
    """
    Find FS subject dir whose basename contains subj_code AND visit_token (v1).
    Search one and two levels deep; prefer names with 'mri' over 'mmr'.
    """
    matches = []

    # pass 1: immediate children
    for d in glob.glob(os.path.join(fs_root, "*")):
        if os.path.isdir(d) and (subj_code in os.path.basename(d)) and has_visit_token(os.path.basename(d), visit_token):
            matches.append(d)

    # pass 2: two levels deep if needed
    if not matches:
        for d in glob.glob(os.path.join(fs_root, "*", "*")):
            if os.path.isdir(d) and (subj_code in os.path.basename(d)) and has_visit_token(os.path.basename(d), visit_token):
                matches.append(d)

    if not matches:
        return None

    matches.sort(key=lambda p: _rank_fs_name(os.path.basename(p)))
    return matches[0]

def _run_dirs(fs_subject_dir):
    return glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))

def _find_label_in_run(run_dir, fname):
    hits = glob.glob(os.path.join(run_dir, "DATA", "*", "mri", fname))
    hits.sort()
    return hits[-1] if hits else None

def find_fs_aseg_within_dir(fs_subject_dir, target_dt=None):
    """
    Within a *visit-locked* FS subject dir, pick aseg from the run whose CNDA timestamp
    is closest to target_dt; if target_dt is None, pick the lexicographically last run with aseg.
    """
    runs = _run_dirs(fs_subject_dir)
    if not runs:
        return None, None

    if target_dt is None:
        runs.sort()
        for rd in reversed(runs):
            aseg = _find_label_in_run(rd, "aseg.mgz")
            if aseg: return aseg, rd
        return None, None

    best, best_diff, best_aseg = None, float("inf"), None
    for rd in runs:
        dt = parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if diff < best_diff:
            aseg = _find_label_in_run(rd, "aseg.mgz")
            if aseg:
                best, best_diff, best_aseg = rd, diff, aseg
    return (best_aseg, best) if best_aseg else (None, None)

def shapes_affines_match(a_path, b_path, atol=ATOL_AFFINE):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok   = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and aff_ok, shape_ok, aff_ok, ia.shape, ib.shape, ia.affine, ib.affine

def resample_label_to_target(label_path, target_path, out_path):
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

        # --- T1: prefer local; fallback to PUP v1 ---
        pup_dir   = find_pup_subject_dir_v1(PUP_ROOT, subj_code, TRACER_TOKEN, VISIT_TOKEN) if subj_code else None
        if not pup_dir:
            log(f"  ERROR: No PUP dir matching subj_code+{TRACER_TOKEN}+{VISIT_TOKEN}; SKIP.", level="ERROR")
            skipped += 1;  continue
        nifti_dir = find_pup_nifti_dir(pup_dir)
        if not nifti_dir:
            log("  ERROR: No NIFTI_GZ under PUP v1 dir; SKIP.", level="ERROR")
            skipped += 1;  continue
        pup_cnda_name = os.path.basename(os.path.dirname(nifti_dir))
        pup_dt = parse_cnda_timestamp_from_name(pup_cnda_name)

        t1_path = os.path.join(subj_dir, "T1.nii.gz")
        if not os.path.exists(t1_path):
            t1_alt = os.path.join(nifti_dir, "T1.nii.gz")
            if t1_alt and os.path.exists(t1_alt):
                t1_path = t1_alt
                log(f"  WARN: T1.nii.gz not in subject folder; using PUP v1 T1: {t1_path}")
            else:
                log("  ERROR: Missing T1.nii.gz (subject and PUP v1); SKIP.", level="ERROR")
                skipped += 1;  continue

        # --- FS subject dir: must be v1 ---
        fs_dir = find_fs_subject_dir_v1(FS_ROOT, subj_code, VISIT_TOKEN) if subj_code else None
        if not fs_dir:
            log(f"  ERROR: No FreeSurfer subject dir matching subj_code+{VISIT_TOKEN}; SKIP.", level="ERROR")
            skipped += 1;  continue
        log(f"  PUP v1 dir: {pup_dir}")
        log(f"  FS v1 subject dir: {fs_dir}")

        # --- aseg.mgz picked ONLY within that FS v1 subject dir (closest to PUP v1 time) ---
        aseg_path, used_run = find_fs_aseg_within_dir(fs_dir, target_dt=pup_dt)
        if not aseg_path:
            log("  ERROR: aseg.mgz not found under FS v1 subject dir; SKIP.", level="ERROR")
            skipped += 1;  continue
        log(f"  FS run used: {used_run if used_run else 'N/A'}")
        if used_run and pup_dt:
            run_dt = parse_cnda_timestamp_from_name(os.path.basename(used_run))
            if run_dt:
                dt_hours = abs((run_dt - pup_dt).total_seconds()) / 3600.0
                log(f"  Δt(run vs PUP v1) ≈ {dt_hours:.2f} hours")

        # --- Ensure aseg aligns with T1 (shape+affine) ---
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

        # --- Build final mask: (parenchyma \ BG) ---
        aseg_img  = nib.load(aseg_inT1)
        aseg_data = np.asanyarray(aseg_img.dataobj)

        # BG
        bg_mask = np.isin(aseg_data, list(BG_IDS))

        # Parenchyma: prefer existing subject-level aseg_brainmask.nii.gz; else derive from aseg_inT1
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

        if paren_mask.shape != bg_mask.shape:
            log(f"  ERROR: Mask grid mismatch (parenchyma {paren_mask.shape} vs BG {bg_mask.shape}); SKIP.", level="ERROR")
            skipped += 1;  continue

        final_mask = (paren_mask & ~bg_mask).astype(np.uint8)
        out_path   = os.path.join(subj_dir, "mask_parenchyma_noBG.nii.gz")
        nib.save(nib.Nifti1Image(final_mask, aseg_img.affine), out_path)
        log(f"  Wrote: {out_path} (voxels={int(final_mask.sum())})")

        ok += 1

    log(f"\nDONE. Wrote no-BG masks for {ok} subjects; skipped {skipped}.")
    if skipped:
        log("Check errors above (missing v1 folders, missing aseg, or grid mismatches).", level="WARN")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled error: {e}", level="ERROR")
        sys.exit(2)
