import os, re, glob, shutil
import numpy as np
import nibabel as nib

# =================== CONFIG ===================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_brain/aseg_mask_bundle_v1"   # <--- EDIT this
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
    m = re.search(r"\d{4}_\d{3}", pup_folder_name)
    return m.group(0) if m else None

def find_pup_nifti_dir(pup_dir):
    """Find .../CNDA*/NIFTI_GZ inside a PUP subject folder (assumes exactly one)."""
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ"))
    if not hits:
        return None
    hits.sort()
    return hits[-1]

def choose_t1_strict(nifti_dir):
    """Strict: must be exactly T1001.nii.gz."""
    path = os.path.join(nifti_dir, "T1001.nii.gz")
    return path if os.path.exists(path) else None

def choose_pet_strict(nifti_dir):
    """Strict: must end with _msum_SUVR.nii.gz (no g8 or other suffix)."""
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
    # Prefer '*MRI*' over '*mMR*' over others
    def rank(name):
        n = name.lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2
    candidates.sort(key=lambda p: rank(os.path.basename(p)))
    return candidates[0]

def find_aseg_mgz(fs_subject_dir):
    """Find aseg.mgz under freesurfer subject dir."""
    pat1 = os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*", "DATA", "*", "mri", "aseg.mgz")
    hits = glob.glob(pat1)
    if hits:
        hits.sort()
        return hits[-1]
    # Fallback search within that subject tree
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files:
            return os.path.join(root, "aseg.mgz")
    return None

def make_aseg_mask_nifti(aseg_path, out_path):
    """Create NIfTI mask from aseg.mgz using KEEP_LABELS."""
    aseg_img  = nib.load(aseg_path)
    lab       = aseg_img.get_fdata()
    mask      = np.isin(lab, list(KEEP_LABELS)).astype(np.uint8)
    nii_img   = nib.Nifti1Image(mask, aseg_img.affine)
    nib.save(nii_img, out_path)
    return out_path

# =================== MAIN ===================

pup_subjects = [d for d in os.listdir(PUP_ROOT)
                if os.path.isdir(os.path.join(PUP_ROOT, d)) and ("av1451" in d.lower())]
pup_subjects.sort()

total = len(pup_subjects)
ok = skip_no_nifti = skip_no_t1 = skip_no_pet = skip_no_code = skip_no_fs = skip_no_aseg = 0

print(f"Found {total} PUP AV1451 subjects.\n")

for subj_folder in pup_subjects:
    pup_dir   = os.path.join(PUP_ROOT, subj_folder)
    nifti_dir = find_pup_nifti_dir(pup_dir)
    if not nifti_dir:
        print(f"[SKIP:NIFTI] {subj_folder}  (no NIFTI_GZ)")
        skip_no_nifti += 1
        continue

    t1_path  = choose_t1_strict(nifti_dir)
    pet_path = choose_pet_strict(nifti_dir)
    if not t1_path:
        print(f"[SKIP:T1]    {subj_folder}  (missing T1001.nii.gz)")
        skip_no_t1 += 1
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

    aseg_path = find_aseg_mgz(fs_dir)
    if not aseg_path or not os.path.exists(aseg_path):
        print(f"[SKIP:ASEG]  {subj_folder}  (aseg.mgz not found under {fs_dir})")
        skip_no_aseg += 1
        continue

    out_dir = os.path.join(OUT_ROOT, subj_folder)
    os.makedirs(out_dir, exist_ok=True)

    # Copy MRI & PET as-is
    dst_t1  = os.path.join(out_dir, os.path.basename(t1_path))
    dst_pet = os.path.join(out_dir, os.path.basename(pet_path))
    shutil.copy2(t1_path, dst_t1)
    shutil.copy2(pet_path, dst_pet)

    # Build aseg-based mask
    out_mask = os.path.join(out_dir, "aseg_brainmask.nii.gz")
    make_aseg_mask_nifti(aseg_path, out_mask)

    print(f"[OK] {subj_folder}")
    print(f"     T1   -> {dst_t1}")
    print(f"     PET  -> {dst_pet}")
    print(f"     MASK -> {out_mask}")
    ok += 1

print("\n=== SUMMARY ===")
print(f"Total av1451 subjects : {total}")
print(f"Processed OK          : {ok}")
print(f"Skipped (no NIFTI_GZ) : {skip_no_nifti}")
print(f"Skipped (no T1001)    : {skip_no_t1}")
print(f"Skipped (no PET SUVR) : {skip_no_pet}")
print(f"Skipped (no subj code): {skip_no_code}")
print(f"Skipped (no FS match) : {skip_no_fs}")
print(f"Skipped (no ASEG)     : {skip_no_aseg}")
print(f"Output root           : {OUT_ROOT}")
