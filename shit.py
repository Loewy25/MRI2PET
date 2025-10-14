import os
import nibabel as nib
import numpy as np

BASE_DIR = "/scratch/l.peiwang/kari_brainv11"

target_files = [
    "T1.nii.gz",
    "PET_in_T1.nii.gz",
    "aseg_brainmask.nii.gz",
    "mask_parenchyma_noBG.nii.gz",
    "ROI_Hippocampus.nii.gz",
    "ROI_LimbicCortex.nii.gz",
    "ROI_PosteriorCingulate.nii.gz",
    "ROI_Precuneus.nii.gz",
    "ROI_TemporalLobe.nii.gz",
]

def affine_diff(a1, a2, tol=1e-5):
    return not np.allclose(a1, a2, atol=tol)

for sub in sorted(os.listdir(BASE_DIR)):
    if not ("av1451" in sub.lower() and os.path.isdir(os.path.join(BASE_DIR, sub))):
        continue

    path = os.path.join(BASE_DIR, sub)
    ref_path = os.path.join(path, "T1.nii.gz")
    if not os.path.exists(ref_path):
        print(f"[WARN] {sub}: Missing reference T1_masked.nii.gz ❌")
        continue

    ref_img = nib.load(ref_path)
    ref_shape, ref_affine = ref_img.shape, ref_img.affine

    all_good = True
    for f in target_files:
        fpath = os.path.join(path, f)
        if not os.path.exists(fpath):
            print(f"  [MISS] {f} — file not found ❌")
            all_good = False
            continue
        img = nib.load(fpath)
        if img.shape != ref_shape or affine_diff(img.affine, ref_affine):
            print(f"  [MISMATCH] {f}")
            if img.shape != ref_shape:
                print(f"     ↳ shape {img.shape} ≠ ref {ref_shape}")
            if affine_diff(img.affine, ref_affine):
                print(f"     ↳ affine mismatch")
            all_good = False

    if all_good:
        print(f"[YAAAAA ✅] {sub} — all shapes & affines match perfectly!")
    else:
        print(f"[CHECK ❗] {sub} — some mismatches above.\n")
