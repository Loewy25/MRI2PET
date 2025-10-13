#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, shutil, subprocess, sys, json
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib

# =========================
# CONFIG
# =========================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_brainv11"   # subject folders already exist here
LUT_PATH  = "/scratch/l.peiwang/FreeSurferColorLUT.txt"

TRACER_TOKEN = "AV1451"   # e.g., "AV1451" or "T807"
VISIT_TOKEN  = "v1"       # enforce visit v1

RUN_LIMIT   = None        # None = all subjects; or small int to test
ATOL_AFFINE = 1e-4
DEBUG       = True

# --- Label sets ---
# Parenchyma (tissue only: GM+WM + cerebellum + subcortical + brainstem + VentralDC)
KEEP_LABELS = {
    2, 41, 3, 42, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51,
    13, 52, 17, 53, 18, 54, 26, 58, 28, 60, 16
}

# Basal ganglia exclusion (strict): Caudate, Putamen, Pallidum, Accumbens (L/R)
BG_IDS_STRICT = {11, 50, 12, 51, 13, 52, 26, 58}
INCLUDE_VENTRAL_DC = False
BG_IDS = (BG_IDS_STRICT | {28, 60}) if INCLUDE_VENTRAL_DC else BG_IDS_STRICT

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
def has_visit_token(s: str, visit: str) -> bool:
    s = s.lower(); visit = visit.lower()
    # match 'v1' as a token (avoid v10->v1 false positive)
    return re.search(rf'(^|[^a-z0-9]){re.escape(visit)}($|[^a-z0-9])', s) is not None

def extract_subject_code(name: str):
    """Get '1092_385' from '1092_385_...'; tolerant to tails."""
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", name)
    return m.group(0) if m else None

def find_pup_subject_dir_v1(pup_root, subj_code, tracer_token, visit_token):
    """PUP dir must contain subj_code + tracer_token + visit_token (v1)."""
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
    """Pick .../CNDA*/NIFTI_GZ (latest if multiple)."""
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

def find_fs_subject_dir_v1(fs_root, subj_code, visit_token):
    """FreeSurfer subject dir must contain subj_code + visit_token (v1)."""
    matches = []
    # level 1
    for d in glob.glob(os.path.join(fs_root, "*")):
        if os.path.isdir(d):
            b = os.path.basename(d)
            if (subj_code in b) and has_visit_token(b, visit_token):
                matches.append(d)
    # level 2 fallback
    if not matches:
        for d in glob.glob(os.path.join(fs_root, "*", "*")):
            if os.path.isdir(d):
                b = os.path.basename(d)
                if (subj_code in b) and has_visit_token(b, visit_token):
                    matches.append(d)
    if not matches: return None
    # prefer names with 'mri' then 'mmr'
    def rank(name):
        n = name.lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2
    matches.sort(key=lambda p: rank(os.path.basename(p)))
    return matches[0]

def _run_dirs(fs_subject_dir):
    return glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))

def _find_label_in_run(run_dir, fname):
    hits = glob.glob(os.path.join(run_dir, "DATA", "*", "mri", fname))
    hits.sort()
    return hits[-1] if hits else None

def find_fs_labels_closest_within_dir(fs_subject_dir, target_dt):
    """
    Inside a visit-locked FS subject dir, pick the run with CNDA time closest to target_dt,
    returning aseg.mgz and aparc+aseg.mgz. If target_dt is None, pick lexicographically last.
    """
    runs = _run_dirs(fs_subject_dir)
    if not runs: return None, None, None

    if target_dt is None:
        runs.sort()
        for rd in reversed(runs):
            aseg  = _find_label_in_run(rd, "aseg.mgz")
            aparc = _find_label_in_run(rd, "aparc+aseg.mgz")
            if aseg and aparc:
                return aseg, aparc, rd
        return None, None, None

    best, best_diff, best_aseg, best_aparc = None, float("inf"), None, None
    for rd in runs:
        dt = parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        aseg  = _find_label_in_run(rd, "aseg.mgz")
        aparc = _find_label_in_run(rd, "aparc+aseg.mgz")
        if aseg and aparc and (diff < best_diff):
            best, best_diff, best_aseg, best_aparc = rd, diff, aseg, aparc
    return best_aseg, best_aparc, best

def shapes_affines_match(a_path, b_path, atol=ATOL_AFFINE):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok   = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and aff_ok, shape_ok, aff_ok, ia.shape, ib.shape, ia.affine, ib.affine

def resample_label_to_target(label_path, target_path, out_path):
    """mri_vol2vol nearest-neighbor label resample label → target grid."""
    if not shutil.which("mri_vol2vol"):
        raise RuntimeError("mri_vol2vol not found on PATH. Source FreeSurfer.")
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

def read_fs_lut(lut_path=LUT_PATH):
    lut = {}
    with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0]); name = parts[1]
                lut[idx] = name; lut[name] = idx
    return lut

def ids_for_hemi_names(base_names, lut):
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

def apply_mask_to_image(image_path, mask_bool, mask_affine, out_path):
    img = nib.load(image_path)
    data = np.asanyarray(img.dataobj)
    # grid/affine check
    if data.shape != mask_bool.shape or not np.allclose(img.affine, mask_affine, atol=ATOL_AFFINE):
        log(f"  ERROR: Masking grid mismatch for {Path(image_path).name}: "
            f"img.shape={data.shape} vs mask.shape={mask_bool.shape}", level="ERROR")
        return False, 0
    masked = (data * mask_bool.astype(data.dtype))
    nib.save(nib.Nifti1Image(masked, img.affine, img.header), out_path)
    return True, int(mask_bool.sum())

# =========================
# MAIN
# =========================
def main():
    # Sanity
    for p, isdir in [(OUT_ROOT, True), (PUP_ROOT, True), (FS_ROOT, True)]:
        if isdir and not os.path.isdir(p):
            log(f"Missing required dir: {p}", level="ERROR"); sys.exit(1)
    if not os.path.isfile(LUT_PATH):
        log(f"LUT not found: {LUT_PATH}", level="ERROR"); sys.exit(1)

    lut = read_fs_lut(LUT_PATH)

    # Subject list under OUT_ROOT
    subjects = [d for d in sorted(os.listdir(OUT_ROOT))
                if os.path.isdir(os.path.join(OUT_ROOT, d))
                and re.search(r"^\d{4}_\d{3,4}_", d)]
    if RUN_LIMIT:
        subjects = subjects[:RUN_LIMIT]
    log(f"Found {len(subjects)} subject folders under {OUT_ROOT}")

    # Counters
    ok = 0
    skip_no_t1 = skip_no_pet = 0
    skip_no_pup_v1 = skip_no_nifti = 0
    skip_no_fs_v1 = skip_no_labels = 0
    vol2vol_fail = 0
    resample_aseg = resample_aparc = 0

    for i, subj in enumerate(subjects, 1):
        subj_dir  = os.path.join(OUT_ROOT, subj)
        subj_code = extract_subject_code(subj)
        log(f"\n[{i}/{len(subjects)}] {subj} (code={subj_code})")

        # T1/PET (we do NOT re-register; just read them and later write masked versions)
        t1_path  = os.path.join(subj_dir, "T1.nii.gz")
        pet_path = os.path.join(subj_dir, "PET_in_T1.nii.gz")

        if not os.path.exists(t1_path):
            log("  ERROR: Missing T1.nii.gz in subject folder; SKIP.", level="ERROR")
            skip_no_t1 += 1;  continue
        if not os.path.exists(pet_path):
            log("  WARN: PET_in_T1.nii.gz not found; PET masking will be skipped.", level="WARN")
            # don't early-continue; we still build masks/ROIs

        # PUP v1 → NIFTI_GZ and timestamp
        if not subj_code:
            log("  ERROR: Cannot extract subject code; SKIP.", level="ERROR")
            skip_no_pup_v1 += 1;  continue
        pup_dir = find_pup_subject_dir_v1(PUP_ROOT, subj_code, TRACER_TOKEN, VISIT_TOKEN)
        if not pup_dir:
            log(f"  ERROR: No PUP dir matching {subj_code}+{TRACER_TOKEN}+{VISIT_TOKEN}; SKIP.", level="ERROR")
            skip_no_pup_v1 += 1;  continue
        nifti_dir = find_pup_nifti_dir(pup_dir)
        if not nifti_dir:
            log("  ERROR: No NIFTI_GZ under PUP v1 dir; SKIP.", level="ERROR")
            skip_no_nifti += 1;  continue
        pup_cnda = os.path.basename(os.path.dirname(nifti_dir))
        pup_dt   = parse_cnda_timestamp_from_name(pup_cnda)
        log(f"  PUP v1 NIFTI: {nifti_dir}; CNDA={pup_cnda}; time={pup_dt if pup_dt else 'N/A'}")

        # FS v1 dir
        fs_dir = find_fs_subject_dir_v1(FS_ROOT, subj_code, VISIT_TOKEN)
        if not fs_dir:
            log(f"  ERROR: No FS v1 subject dir for {subj_code}; SKIP.", level="ERROR")
            skip_no_fs_v1 += 1;  continue
        log(f"  FS v1 subject dir: {fs_dir}")

        # Labels from closest run within FS v1
        aseg_path, aparc_path, used_run = find_fs_labels_closest_within_dir(fs_dir, pup_dt)
        if not (aseg_path and aparc_path):
            log("  ERROR: Missing aseg.mgz or aparc+aseg.mgz in FS v1; SKIP.", level="ERROR")
            skip_no_labels += 1;  continue
        log(f"  FS run used: {used_run if used_run else 'N/A'}")
        if used_run and pup_dt:
            run_dt = parse_cnda_timestamp_from_name(os.path.basename(used_run))
            if run_dt:
                dt_hours = abs((run_dt - pup_dt).total_seconds())/3600.0
                log(f"  Δt(run vs PUP v1) ≈ {dt_hours:.2f} h")

        # Ensure labels are on T1 grid (resample if shape or affine differ)
        try:
            sameA, shapeA, affA, a_shape, t_shapeA, a_aff, t_affA = shapes_affines_match(aseg_path, t1_path, ATOL_AFFINE)
            sameP, shapeP, affP, p_shape, t_shapeP, p_aff, t_affP = shapes_affines_match(aparc_path, t1_path, ATOL_AFFINE)
        except Exception as e:
            log(f"  ERROR: Failed to load volumes for grid/affine check: {e}", level="ERROR")
            skip_no_labels += 1;  continue

        if not sameA:
            resample_aseg += 1
            log(f"  RESAMPLE aseg → T1 (shape_ok={shapeA}, affine_ok={affA})")
            log(f"    aseg.shape={a_shape}  T1.shape={t_shapeA}")
            log("    aseg.affine:\n" + np.array2string(a_aff, precision=5))
            log("    T1.affine:\n"   + np.array2string(t_affA, precision=5))
            aseg_inT1 = os.path.join(subj_dir, "aseg_inT1.nii.gz")
            try:
                resample_label_to_target(aseg_path, t1_path, aseg_inT1)
            except Exception as e:
                log(f"  ERROR: mri_vol2vol failed for aseg: {e}", level="ERROR")
                vol2vol_fail += 1;  continue
        else:
            aseg_inT1 = aseg_path
            log("  aseg aligns with T1 → no resample")

        if not sameP:
            resample_aparc += 1
            log(f"  RESAMPLE aparc+aseg → T1 (shape_ok={shapeP}, affine_ok={affP})")
            log(f"    aparc.shape={p_shape}  T1.shape={t_shapeP}")
            log("    aparc.affine:\n" + np.array2string(p_aff, precision=5))
            log("    T1.affine:\n"    + np.array2string(t_affP, precision=5))
            aparc_inT1 = os.path.join(subj_dir, "aparc_inT1.nii.gz")
            try:
                resample_label_to_target(aparc_path, t1_path, aparc_inT1)
            except Exception as e:
                log(f"  ERROR: mri_vol2vol failed for aparc+aseg: {e}", level="ERROR")
                vol2vol_fail += 1;  continue
        else:
            aparc_inT1 = aparc_path
            log("  aparc+aseg aligns with T1 → no resample")

        # --- Parenchyma mask (overwrite) ---
        paren_path = os.path.join(subj_dir, "aseg_brainmask.nii.gz")
        vox_paren  = write_mask_from_labels(aseg_inT1, KEEP_LABELS, paren_path)
        log(f"  Wrote parenchyma mask: {paren_path} (voxels={vox_paren})")

        # --- Basal ganglia exclusion + final composite (overwrite) ---
        bg_path    = os.path.join(subj_dir, "mask_basalganglia.nii.gz")  # optional QC file
        vox_bg     = write_mask_from_labels(aseg_inT1, BG_IDS, bg_path)
        log(f"  Wrote BG mask (QC): {bg_path} (voxels={vox_bg})")

        paren_img  = nib.load(paren_path)
        paren_bool = np.asanyarray(paren_img.dataobj).astype(bool)
        bg_bool    = np.asanyarray(nib.load(bg_path).dataobj).astype(bool)
        if paren_bool.shape != bg_bool.shape or not np.allclose(paren_img.affine, nib.load(bg_path).affine, atol=ATOL_AFFINE):
            log("  ERROR: Parenchyma vs BG mask grid mismatch; SKIP masking step.", level="ERROR")
            continue
        final_bool = (paren_bool & ~bg_bool).astype(np.uint8)
        final_path = os.path.join(subj_dir, "mask_parenchyma_noBG.nii.gz")
        nib.save(nib.Nifti1Image(final_bool, paren_img.affine), final_path)
        log(f"  Wrote final metric mask: {final_path} (voxels={int(final_bool.sum())})")

        # --- ROI masks (overwrite) ---
        created = {}

        # Hippocampus from aseg
        hip_ids = []
        for nm in ("Left-Hippocampus","Right-Hippocampus"):
            if nm in lut: hip_ids.append(lut[nm])
        if not hip_ids: hip_ids = [17, 53]; log("  WARN: using fallback hippocampus IDs [17, 53]", level="WARN")
        hip_path = os.path.join(subj_dir, "ROI_Hippocampus.nii.gz")
        created[Path(hip_path).name] = write_mask_from_labels(aseg_inT1, hip_ids, hip_path)

        # PCC / Precuneus / Temporal / Limbic from aparc
        def _mk_aparc_roi(base_names, out_name):
            ids = ids_for_hemi_names(base_names, lut)
            if not ids:
                log(f"  ERROR: LUT missing IDs for {base_names}; skipping {out_name}", level="ERROR")
                return 0
            out = os.path.join(subj_dir, out_name)
            return write_mask_from_labels(aparc_inT1, ids, out)

        created["ROI_PosteriorCingulate.nii.gz"] = _mk_aparc_roi(["posteriorcingulate"], "ROI_PosteriorCingulate.nii.gz")
        created["ROI_Precuneus.nii.gz"]          = _mk_aparc_roi(["precuneus"], "ROI_Precuneus.nii.gz")
        created["ROI_TemporalLobe.nii.gz"]       = _mk_aparc_roi(TEMPORAL_BASE, "ROI_TemporalLobe.nii.gz")
        created["ROI_LimbicCortex.nii.gz"]       = _mk_aparc_roi(LIMBIC_BASE, "ROI_LimbicCortex.nii.gz")

        log("  ROI voxel counts:\n" + json.dumps(created, indent=2))

        # --- Masked images (overwrite; no re-registration) ---
        # T1_masked
        t1_masked_path = os.path.join(subj_dir, "T1_masked.nii.gz")
        ok_t1, vox = apply_mask_to_image(t1_path, final_bool.astype(bool), paren_img.affine, t1_masked_path)
        if ok_t1:
            log(f"  Wrote: {t1_masked_path} (mask voxels={vox})")
        else:
            log("  WARN: T1 masking skipped due to grid mismatch.", level="WARN")

        # PET_in_T1_masked (only if PET exists and grids match)
        if os.path.exists(pet_path):
            pet_masked_path = os.path.join(subj_dir, "PET_in_T1_masked.nii.gz")
            ok_pet, vox = apply_mask_to_image(pet_path, final_bool.astype(bool), paren_img.affine, pet_masked_path)
            if ok_pet:
                log(f"  Wrote: {pet_masked_path} (mask voxels={vox})")
            else:
                log("  WARN: PET masking skipped due to grid mismatch.", level="WARN")
        else:
            skip_no_pet += 1

        ok += 1

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Processed OK                     : {ok}")
    print(f"Skipped (no T1)                  : {skip_no_t1}")
    print(f"Skipped (no PET_in_T1 present)   : {skip_no_pet}")
    print(f"Skipped (no PUP v1)              : {skip_no_pup_v1}")
    print(f"Skipped (no PUP v1 NIFTI_GZ)     : {skip_no_nifti}")
    print(f"Skipped (no FS v1)               : {skip_no_fs_v1}")
    print(f"Skipped (missing labels)         : {skip_no_labels}")
    print(f"VOL2VOL failures                 : {vol2vol_fail}")
    print(f"DEBUG: resampled aseg count      : {resample_aseg}")
    print(f"DEBUG: resampled aparc count     : {resample_aparc}")
    print(f"Output root                      : {OUT_ROOT}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled error: {e}", level="ERROR")
        sys.exit(2)
