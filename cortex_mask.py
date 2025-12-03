#!/usr/bin/env python3
import os, re, glob
import numpy as np
import nibabel as nib
from datetime import datetime

# ====== EDIT THESE ======
BASE_ROOT    = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT     = os.path.join(BASE_ROOT, "pup")
FS_ROOT      = os.path.join(BASE_ROOT, "freesurfers")
DATASET_ROOT = "/scratch/l.peiwang/kari_brainv33_top300"   # your *new* dataset root
OUT_NAME     = "aseg_cortexmask.nii.gz"
# FreeSurfer aseg labels: 3=Left-Cerebral-Cortex, 42=Right-Cerebral-Cortex
CORTEX_LABELS = {3, 42}
# ========================

def die(msg):
    print(f"[FATAL] {msg}")
    raise SystemExit(1)

def extract_subject_code(folder_name):
    parts = folder_name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", folder_name)
    return m.group(0) if m else None

def find_pup_nifti_dir(pup_dir):
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ"))
    if not hits:
        return None
    hits.sort()
    return hits[-1]

def parse_cnda_timestamp(name):
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None

def find_fs_subject_dir(fs_root, subj_code):
    candidates = [d for d in glob.glob(os.path.join(fs_root, "*"))
                  if os.path.isdir(d) and subj_code in os.path.basename(d)]
    if not candidates:
        return None
    def rank(p):
        n = os.path.basename(p).lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2
    candidates.sort(key=rank)
    return candidates[0]

def find_aseg_mgz_closest(fs_subject_dir, target_dt):
    run_dirs = glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))
    best, best_diff = None, None

    for run in run_dirs:
        dt = parse_cnda_timestamp(os.path.basename(run))
        hits = glob.glob(os.path.join(run, "DATA", "*", "mri", "aseg.mgz"))
        if not hits:
            continue
        hits.sort()
        aseg = hits[-1]
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if best is None or diff < best_diff:
            best, best_diff = aseg, diff

    if best:
        return best

    # fallback: any aseg.mgz under subject dir
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files:
            return os.path.join(root, "aseg.mgz")
    return None

def main():
    if not os.path.isdir(DATASET_ROOT):
        die(f"DATASET_ROOT not found: {DATASET_ROOT}")

    subjects = sorted([d for d in os.listdir(DATASET_ROOT)
                       if os.path.isdir(os.path.join(DATASET_ROOT, d))])

    if not subjects:
        die(f"No subject folders found under {DATASET_ROOT}")

    print(f"[info] Found {len(subjects)} subject folders in dataset root.")

    for subj_folder in subjects:
        out_dir = os.path.join(DATASET_ROOT, subj_folder)
        t1_path = os.path.join(out_dir, "T1.nii.gz")
        if not os.path.exists(t1_path):
            die(f"{subj_folder}: missing {t1_path}")

        pup_dir = os.path.join(PUP_ROOT, subj_folder)
        if not os.path.isdir(pup_dir):
            die(f"{subj_folder}: PUP folder not found: {pup_dir}")

        nifti_dir = find_pup_nifti_dir(pup_dir)
        if not nifti_dir:
            die(f"{subj_folder}: no NIFTI_GZ under {pup_dir}")

        pup_cnda_name = os.path.basename(os.path.dirname(nifti_dir))
        pup_dt = parse_cnda_timestamp(pup_cnda_name)

        subj_code = extract_subject_code(subj_folder)
        if not subj_code:
            die(f"{subj_folder}: cannot extract subject code")

        fs_subject_dir = find_fs_subject_dir(FS_ROOT, subj_code)
        if not fs_subject_dir:
            die(f"{subj_folder}: no freesurfer subject matching '{subj_code}' under {FS_ROOT}")

        aseg_path = find_aseg_mgz_closest(fs_subject_dir, pup_dt)
        if not aseg_path or not os.path.exists(aseg_path):
            die(f"{subj_folder}: aseg.mgz not found under {fs_subject_dir}")

        # Build cortex-only mask from aseg labels {3, 42}
        aseg_img = nib.load(aseg_path)
        lab = np.asanyarray(aseg_img.dataobj)
        mask = np.isin(lab, list(CORTEX_LABELS)).astype(np.uint8)

        if mask.sum() == 0:
            die(f"{subj_folder}: cortex mask is EMPTY (labels {CORTEX_LABELS}) from {aseg_path}")

        # Hard checks: mask must match dataset T1 grid
        t1_img = nib.load(t1_path)
        if t1_img.shape != mask.shape:
            die(f"{subj_folder}: shape mismatch: T1 {t1_img.shape} vs aseg/mask {mask.shape}")
        if not np.allclose(t1_img.affine, aseg_img.affine, atol=1e-3):
            die(f"{subj_folder}: affine mismatch between T1 and aseg (bug to fix)")

        out_mask = os.path.join(out_dir, OUT_NAME)
        out_img = nib.Nifti1Image(mask, t1_img.affine)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, out_mask)

        print(f"[OK] {subj_folder} -> {out_mask}")

    print("[done] all cortex masks generated.")

if __name__ == "__main__":
    main()
