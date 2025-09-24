#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, json, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib

# =========================
# CONFIG (edit if needed)
# =========================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_brainv11"   # subject folders live here (target for ROI masks)
LUT_PATH  = "/export/freesurfer/freesurfer-7.4.1/FreeSurferColorLUT.txt"

RUN_LIMIT   = None     # e.g., 5 for quick test; None = all
ATOL_AFFINE = 1e-4
DEBUG       = True

# ROI definitions (Desikan–Killiany names)
TEMPORAL_BASE = [
    "superiortemporal","middletemporal","inferiortemporal",
    "fusiform","transversetemporal","temporalpole","bankssts"
]
LIMBIC_BASE = [
    "posteriorcingulate","isthmuscingulate",
    "caudalanteriorcingulate","rostralanteriorcingulate",
    "parahippocampal","entorhinal"
]

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
    """Get '1092_385' from '1092_385_AV1451_v1'."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", name)
    return m.group(0) if m else None

def find_pup_subject_dir(pup_root, subj_code):
    """Locate PUP subject dir that contains subj_code and looks like AV1451."""
    cands = []
    for d in glob.glob(os.path.join(pup_root, "*")):
        b = os.path.basename(d)
        if os.path.isdir(d) and subj_code in b and ("av1451" in b.lower()):
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
    # Slight preference ordering
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

def find_fs_labels_closest(fs_subject_dir, target_dt):
    """
    Return (aseg_path, aparc_path, used_run_dir) from the FS run closest to target_dt.
    Only DK 'aparc+aseg.mgz' is supported for cortical ROIs here.
    Fallback: search entire subject tree.
    """
    runs = _run_dirs(fs_subject_dir)
    best, best_diff = None, float("inf")
    for rd in runs:
        dt = parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if diff < best_diff:
            best, best_diff = rd, diff

    if best:
        aseg  = _find_label_in_run(best, "aseg.mgz")
        aparc = _find_label_in_run(best, "aparc+aseg.mgz")  # prefer DK
        if aseg and aparc:
            return aseg, aparc, best

    # Fallback search across subject tree
    found_aseg, found_aparc = None, None
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files and not found_aseg:
            found_aseg = os.path.join(root, "aseg.mgz")
        if "aparc+aseg.mgz" in files and not found_aparc:
            found_aparc = os.path.join(root, "aparc+aseg.mgz")
    return found_aseg, found_aparc, None

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
        "mri_vol2vol",
        "--mov", label_path,
        "--targ", target_path,
        "--regheader",
        "--interp", "nearest",
        "--o", out_path
    ]
    log("RUN: " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-20:])
        log("mri_vol2vol stderr (tail):\n" + tail, level="ERROR")
        raise RuntimeError(f"mri_vol2vol failed: {r.returncode}")
    return out_path

def read_fs_lut(lut_path=LUT_PATH):
    """Return dict with both directions: name->id and id->name."""
    lut = {}
    with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): 
                continue
            parts = ln.split()
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1]
                lut[idx] = name      # id -> name
                lut[name] = idx      # name -> id
    return lut

def ids_for_hemi_names(base_names, lut):
    """From ['precuneus'] get IDs for ctx-lh-precuneus & ctx-rh-precuneus."""
    ids = []
    for bn in base_names:
        for hemi in ("lh", "rh"):
            key = f"ctx-{hemi}-{bn}"
            if key in lut:
                ids.append(lut[key])
            else:
                log(f"LUT missing: {key}", level="WARN")
    return ids

def write_mask_from_labels(label_path, id_list, out_path, dtype=np.uint8):
    img  = nib.load(label_path)
    data = np.asanyarray(img.dataobj)
    mask = np.isin(data, list(id_list)).astype(dtype)
    nib.save(nib.Nifti1Image(mask, img.affine), out_path)
    return int(mask.sum())

# =========================
# MAIN
# =========================
def main():
    # Sanity checks
    assert os.path.isdir(OUT_ROOT), f"OUT_ROOT not found: {OUT_ROOT}"
    assert os.path.isdir(PUP_ROOT), f"PUP_ROOT not found: {PUP_ROOT}"
    assert os.path.isdir(FS_ROOT),  f"FS_ROOT not found: {FS_ROOT}"
    assert os.path.isfile(LUT_PATH), f"LUT not found: {LUT_PATH}"

    lut = read_fs_lut(LUT_PATH)
    log(f"LUT loaded from {LUT_PATH}")

    # Find subject folders under OUT_ROOT
    all_subject_dirs = [d for d in sorted(os.listdir(OUT_ROOT))
                        if os.path.isdir(os.path.join(OUT_ROOT, d))
                        and re.search(r"^\d{4}_\d{3,4}_", d)]
    if RUN_LIMIT:
        all_subject_dirs = all_subject_dirs[:RUN_LIMIT]

    log(f"Found {len(all_subject_dirs)} subjects under {OUT_ROOT}")

    n_done = 0
    summary = []

    for i, subj in enumerate(all_subject_dirs, 1):
        subj_dir  = os.path.join(OUT_ROOT, subj)
        subj_code = extract_subject_code(subj)

        log(f"\n[{i}/{len(all_subject_dirs)}] Subject: {subj}  (code={subj_code})")

        # Working T1 (prefer T1.nii.gz in subject folder; else pull from PUP)
        t1_path = os.path.join(subj_dir, "T1.nii.gz")
        if not os.path.exists(t1_path):
            pup_dir   = find_pup_subject_dir(PUP_ROOT, subj_code) if subj_code else None
            nifti_dir = find_pup_nifti_dir(pup_dir) if pup_dir else None
            alt_t1 = os.path.join(nifti_dir, "T1.nii.gz") if nifti_dir else None
            if alt_t1 and os.path.exists(alt_t1):
                t1_path = alt_t1
                log(f"  WARN: T1.nii.gz not in OUT_ROOT; using PUP T1: {t1_path}")
            else:
                log("  ERROR: Missing T1.nii.gz; SKIP.", level="ERROR")
                continue

        # PUP CNDA time (choose closest FS run)
        pup_dir   = find_pup_subject_dir(PUP_ROOT, subj_code) if subj_code else None
        nifti_dir = find_pup_nifti_dir(pup_dir) if pup_dir else None
        pup_dt    = parse_cnda_timestamp_from_name(os.path.basename(os.path.dirname(nifti_dir))) if nifti_dir else None
        log(f"  PUP NIFTI: {nifti_dir if nifti_dir else 'N/A'}; CNDA time: {pup_dt if pup_dt else 'N/A'}")

        # FS subject and labels
        fs_dir = find_fs_subject_dir(FS_ROOT, subj_code) if subj_code else None
        if not fs_dir:
            log("  ERROR: No matching FreeSurfer subject dir; SKIP.", level="ERROR")
            continue
        log(f"  FS subject dir: {fs_dir}")

        aseg_path, aparc_path, used_run = find_fs_labels_closest(fs_dir, pup_dt)
        log(f"  FS run used: {used_run if used_run else 'fallback-search'}")
        log(f"  aseg: {aseg_path}")
        log(f"  aparc+aseg: {aparc_path}")

        if not aseg_path or not aparc_path:
            log("  ERROR: Missing aseg or aparc+aseg; SKIP.", level="ERROR")
            continue

        # Ensure label grids match T1 (resample if needed)
        need_aseg_out  = os.path.join(subj_dir, "aseg_inT1.nii.gz")
        need_aparc_out = os.path.join(subj_dir, "aparc_inT1.nii.gz")

        sameA, shapeA, affA, *_ = shapes_affines_match(aseg_path, t1_path, atol=ATOL_AFFINE)
        sameP, shapeP, affP, *_ = shapes_affines_match(aparc_path, t1_path, atol=ATOL_AFFINE)

        if sameA:
            aseg_inT1 = aseg_path
            log("  aseg aligns with T1 (shape+affine) → no resample")
        else:
            log(f"  WARN: aseg needs resample (shape_ok={shapeA}, affine_ok={affA})")
            aseg_inT1 = resample_label_to_target(aseg_path, t1_path, need_aseg_out)

        if sameP:
            aparc_inT1 = aparc_path
            log("  aparc+aseg aligns with T1 (shape+affine) → no resample")
        else:
            log(f"  WARN: aparc+aseg needs resample (shape_ok={shapeP}, affine_ok={affP})")
            aparc_inT1 = resample_label_to_target(aparc_path, t1_path, need_aparc_out)

        # ---- Build ROI masks ----
        created = {}

        # 1) Hippocampus (aseg)
        hip_ids = []
        for nm in ("Left-Hippocampus","Right-Hippocampus"):
            if nm in lut:
                hip_ids.append(lut[nm])
        if not hip_ids:  # fallback if LUT missing names
            hip_ids = [17, 53]
            log("  WARN: using fallback hippocampus IDs [17, 53]", level="WARN")
        hip_path = os.path.join(subj_dir, "ROI_Hippocampus.nii.gz")
        vox_hip  = write_mask_from_labels(aseg_inT1, hip_ids, hip_path)
        created[Path(hip_path).name] = vox_hip

        # 2) PCC (aparc)
        pcc_ids = ids_for_hemi_names(["posteriorcingulate"], lut)
        if pcc_ids:
            pcc_path = os.path.join(subj_dir, "ROI_PosteriorCingulate.nii.gz")
            vox_pcc  = write_mask_from_labels(aparc_inT1, pcc_ids, pcc_path)
            created[Path(pcc_path).name] = vox_pcc
        else:
            log("  ERROR: PCC labels not found in LUT; skipping PCC", level="ERROR")

        # 3) Precuneus (aparc)
        pcun_ids = ids_for_hemi_names(["precuneus"], lut)
        if pcun_ids:
            pcun_path = os.path.join(subj_dir, "ROI_Precuneus.nii.gz")
            vox_pcun  = write_mask_from_labels(aparc_inT1, pcun_ids, pcun_path)
            created[Path(pcun_path).name] = vox_pcun
        else:
            log("  ERROR: Precuneus labels not found in LUT; skipping Precuneus", level="ERROR")

        # 4) Temporal Lobe (aparc)
        temp_ids = ids_for_hemi_names(TEMPORAL_BASE, lut)
        if temp_ids:
            temp_path = os.path.join(subj_dir, "ROI_TemporalLobe.nii.gz")
            vox_temp  = write_mask_from_labels(aparc_inT1, temp_ids, temp_path)
            created[Path(temp_path).name] = vox_temp
        else:
            log("  ERROR: Temporal lobe labels not found in LUT; skipping Temporal", level="ERROR")

        # 5) Limbic Cortex (aparc)
        limb_ids = ids_for_hemi_names(LIMBIC_BASE, lut)
        if limb_ids:
            limb_path = os.path.join(subj_dir, "ROI_LimbicCortex.nii.gz")
            vox_limb  = write_mask_from_labels(aparc_inT1, limb_ids, limb_path)
            created[Path(limb_path).name] = vox_limb
        else:
            log("  ERROR: Limbic labels not found in LUT; skipping Limbic", level="ERROR")

        # Log voxel counts
        log("  ROI voxel counts:\n" + json.dumps(created, indent=2))
        summary.append({"subject": subj, "created": created})
        n_done += 1

    # =========================
    # SUMMARY
    # =========================
    log(f"\nDONE. Created ROI masks for {n_done} subjects out of {len(all_subject_dirs)}.")
    # Optional: write a summary JSON next to OUT_ROOT
    try:
        summ_path = os.path.join(OUT_ROOT, "roi_creation_summary.json")
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary written: {summ_path}")
    except Exception as e:
        log(f"Could not write summary JSON: {e}", level="WARN")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        log(str(e), level="ERROR")
        sys.exit(1)
    except Exception as e:
        log(f"Unhandled error: {e}", level="ERROR")
        sys.exit(2)
