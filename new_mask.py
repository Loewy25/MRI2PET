import os, re, glob, shutil
import numpy as np
import nibabel as nib
import subprocess
from datetime import datetime

# =================== CONFIG ===================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_brain3"   # <--- EDIT this if needed
os.makedirs(OUT_ROOT, exist_ok=True)

# Labels to keep for a clean brain parenchyma mask (GM+WM, cerebellum, subcortical, brainstem, VentralDC)
KEEP_LABELS = {
    2, 41,        # Cerebral WM (L/R)
    3, 42,        # Cerebral Cortex (L/R)
    7, 46,        # Cerebellum WM (L/R)
    8, 47,        # Cerebellum Cortex (L/R)
    10, 49,       # Thalamus-Proper (L/R)
    11, 50,       # Caudate (L/R)
    12, 51,       # Putamen (L/R)
    13, 52,       # Pallidum (L/R)
    17, 53,       # Hippocampus (L/R)
    18, 54,       # Amygdala (L/R)
    26, 58,       # Accumbens (L/R)
    28, 60,       # VentralDC (L/R)
    16            # Brainstem
}

# =================== HELPERS ===================

def extract_subject_code(pup_folder_name):
    """Get '1092_385' from '1092_385_av1451_v1'."""
    parts = pup_folder_name.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    m = re.search(r"\d{4}_\d{3,4}", pup_folder_name)  # allow 3-4 digit tail
    return m.group(0) if m else None

def find_pup_nifti_dir(pup_dir):
    """Find .../CNDA*/NIFTI_GZ inside a PUP subject folder (assumes >=1)."""
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ"))
    if not hits:
        return None
    hits.sort()
    return hits[-1]

def _parse_cnda_timestamp_from_name(name):
    """Parse 'YYYYmmddHHMMSS' timestamp from a CNDA folder name."""
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None

def choose_native_t1(nifti_dir):
    """Use native T1.nii.gz (final reference space)."""
    path = os.path.join(nifti_dir, "T1.nii.gz")
    return path if os.path.exists(path) else None

def choose_t1001_strict(nifti_dir):
    """T1001.nii.gz (used only to estimate T1001->T1 transform)."""
    path = os.path.join(nifti_dir, "T1001.nii.gz")
    return path if os.path.exists(path) else None

def choose_pet_strict(nifti_dir):
    """Strict: *_msum_SUVR.nii.gz (no g8 or other suffix)."""
    matches = glob.glob(os.path.join(nifti_dir, "*_msum_SUVR.nii.gz"))
    if not matches:
        return None
    matches.sort()
    return matches[0]

def find_fs_subject_dir(fs_root, subj_code):
    """Locate the freesurfer subject folder whose name contains subj_code (e.g., '1092_385')."""
    candidates = [d for d in glob.glob(os.path.join(fs_root, "*"))
                  if os.path.isdir(d) and subj_code in os.path.basename(d)]
    if not candidates:
        return None
    def rank(name):
        n = name.lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2
    candidates.sort(key=lambda p: rank(os.path.basename(p)))
    return candidates[0]

def find_aseg_mgz_closest(fs_subject_dir, target_dt):
    """
    If multiple CNDA freesurfer runs exist, pick the one with timestamp closest to PUP CNDA.
    Return its DATA/*/mri/aseg.mgz path.
    """
    run_dirs = glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))
    best_aseg, best_diff = None, None
    for run in run_dirs:
        dt = _parse_cnda_timestamp_from_name(os.path.basename(run))
        aseg_hits = glob.glob(os.path.join(run, "DATA", "*", "mri", "aseg.mgz"))
        if not aseg_hits:
            continue
        aseg_hits.sort()
        aseg = aseg_hits[-1]
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if best_aseg is None or diff < best_diff:
            best_aseg, best_diff = aseg, diff
    if best_aseg:
        return best_aseg
    # Fallback: any aseg under the subject tree
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files:
            return os.path.join(root, "aseg.mgz")
    return None

def _ensure_3d_nifti(in_path, out_dir, tag):
    """If a NIfTI is 4D with one frame, write a 3D temp for CLI use."""
    try:
        img = nib.load(in_path)
        if img.ndim == 4 and img.shape[-1] == 1:
            out = os.path.join(out_dir, f"_{tag}_3d_tmp.nii.gz")
            nib.save(nib.Nifti1Image(img.get_fdata()[..., 0], img.affine, img.header), out)
            return out, True
    except Exception:
        pass
    return in_path, False

def make_aseg_mask_nifti(aseg_path, out_path):
    """
    Create NIfTI mask from aseg.mgz using KEEP_LABELS.
    No reslicing here (assumes aseg/T1 share the same grid, which you've verified).
    """
    aseg_img = nib.load(aseg_path)
    lab      = aseg_img.get_fdata()
    mask     = np.isin(lab, list(KEEP_LABELS)).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, aseg_img.affine), out_path)
    return out_path

# =================== MAIN ===================

pup_subjects = [d for d in os.listdir(PUP_ROOT)
                if os.path.isdir(os.path.join(PUP_ROOT, d)) and ("av1451" in d.lower())]
pup_subjects.sort()

total = len(pup_subjects)
ok = skip_no_nifti = skip_no_t1native = skip_no_t1001 = skip_no_pet = skip_no_code = skip_no_fs = skip_no_aseg = 0
flirt_fail = 0

print(f"Found {total} PUP AV1451 subjects.\n")

for subj_folder in pup_subjects[:2]:
    pup_dir   = os.path.join(PUP_ROOT, subj_folder)
    nifti_dir = find_pup_nifti_dir(pup_dir)
    if not nifti_dir:
        print(f"[SKIP:NIFTI] {subj_folder}  (no NIFTI_GZ)")
        skip_no_nifti += 1
        continue

    # PUP CNDA timestamp for closest FS-run selection
    pup_cnda_name = os.path.basename(os.path.dirname(nifti_dir))
    pup_dt = _parse_cnda_timestamp_from_name(pup_cnda_name)

    # Pick files
    t1_native = choose_native_t1(nifti_dir)     # final reference space
    t1_1001   = choose_t1001_strict(nifti_dir)  # only to estimate T1001->T1
    pet_path  = choose_pet_strict(nifti_dir)
    if not t1_native:
        print(f"[SKIP:T1]    {subj_folder}  (missing T1.nii.gz)")
        skip_no_t1native += 1
        continue
    if not t1_1001:
        print(f"[SKIP:T1001] {subj_folder}  (missing T1001.nii.gz)")
        skip_no_t1001 += 1
        continue
    if not pet_path:
        print(f"[SKIP:PET]   {subj_folder}  (missing *_msum_SUVR.nii.gz)")
        skip_no_pet += 1
        continue

    subj_code = extract_subject_code(subj_folder)
    if not subj_code:
        print(f"[SKIP:CODE]  {subj_folder}  (cannot extract subject code)")
        skip_no_code += 1
        continue

    fs_dir = find_fs_subject_dir(FS_ROOT, subj_code)
    if not fs_dir:
        print(f"[SKIP:FS]    {subj_folder}  (no freesurfer subject matching '{subj_code}')")
        skip_no_fs += 1
        continue

    aseg_path = find_aseg_mgz_closest(fs_dir, pup_dt)
    if not aseg_path or not os.path.exists(aseg_path):
        print(f"[SKIP:ASEG]  {subj_folder}  (aseg.mgz not found under {fs_dir})")
        skip_no_aseg += 1
        continue

    out_dir = os.path.join(OUT_ROOT, subj_folder)
    os.makedirs(out_dir, exist_ok=True)

    # --- Save reference T1 (native) ---
    dst_t1 = os.path.join(out_dir, os.path.basename(t1_native))  # T1.nii.gz
    shutil.copy2(t1_native, dst_t1)

    # --- Register T1001 -> T1 (rigid) and apply to PET -> PET_in_T1 ---
    flirt = shutil.which("flirt")
    if not flirt:
        print(f"[FAIL:FLIRT] {subj_folder}  (flirt not found on PATH)")
        flirt_fail += 1
        continue

    t1_for_cli, squeezed = _ensure_3d_nifti(dst_t1, out_dir, "t1")
    t1001_for_cli, _     = _ensure_3d_nifti(t1_1001, out_dir, "t1001")

    mat_path = os.path.join(out_dir, "T1001_to_T1.mat")
    dst_pet  = os.path.join(out_dir, "PET_in_T1.nii.gz")

    # 1) estimate transform (6 DOF, normalized MI)
    try:
        subprocess.run([flirt, "-in", t1001_for_cli, "-ref", t1_for_cli, "-omat", mat_path,
                        "-dof", "6", "-cost", "normmi"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL:FLIRT] {subj_folder}  (flirt estimate exit {e.returncode})")
        flirt_fail += 1
        if squeezed and os.path.exists(t1_for_cli):
            try: os.remove(t1_for_cli)
            except Exception: pass
        continue

    # 2) apply to PET (one resample; trilinear)
    try:
        subprocess.run([flirt, "-in", pet_path, "-ref", t1_for_cli, "-applyxfm", "-init", mat_path,
                        "-interp", "trilinear", "-out", dst_pet],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL:FLIRT] {subj_folder}  (flirt apply exit {e.returncode})")
        flirt_fail += 1
        if squeezed and os.path.exists(t1_for_cli):
            try: os.remove(t1_for_cli)
            except Exception: pass
        continue

    if squeezed and os.path.exists(t1_for_cli):
        try: os.remove(t1_for_cli)
        except Exception: pass

    # --- Build aseg-based mask (labels only; no reslice) ---
    out_mask = os.path.join(out_dir, "aseg_brainmask.nii.gz")
    make_aseg_mask_nifti(aseg_path, out_mask)

    print(f"[OK] {subj_folder}")
    print(f"     T1        -> {dst_t1}")
    print(f"     PET(new)  -> {dst_pet}")
    print(f"     MASK(new) -> {out_mask}")
    ok += 1

print("\n=== SUMMARY ===")
print(f"Total av1451 subjects : {total}")
print(f"Processed OK          : {ok}")
print(f"Skipped (no NIFTI_GZ) : {skip_no_nifti}")
print(f"Skipped (no T1)       : {skip_no_t1native}")
print(f"Skipped (no T1001)    : {skip_no_t1001}")
print(f"Skipped (no PET SUVR) : {skip_no_pet}")
print(f"Skipped (no subj code): {skip_no_code}")
print(f"Skipped (no FS match) : {skip_no_fs}")
print(f"Skipped (no ASEG)     : {skip_no_aseg}")
print(f"FLIRT failures        : {flirt_fail}")
print(f"Output root           : {OUT_ROOT}")

