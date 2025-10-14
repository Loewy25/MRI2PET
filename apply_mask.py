#!/usr/bin/env python3
import os, glob, nibabel as nib
import numpy as np

ROOT = "/scratch/l.peiwang/kari_brainv11"  # adjust as needed
PATTERN = "1092_*_AV1451_*"

def apply_mask(img_path, mask, out_path):
    img = nib.load(img_path)
    data = img.get_fdata()
    masked = data * mask
    nib.save(nib.Nifti1Image(masked, img.affine, img.header), out_path)

for subj in sorted(glob.glob(os.path.join(ROOT, PATTERN))):
    t1  = os.path.join(subj, "T1.nii.gz")
    pet = os.path.join(subj, "PET_in_T1.nii.gz")
    msk = os.path.join(subj, "aseg_brainmask.nii.gz")
    if not (os.path.exists(t1) and os.path.exists(pet) and os.path.exists(msk)):
        continue

    mask = (nib.load(msk).get_fdata() > 0).astype(np.float32)
    apply_mask(t1, mask, os.path.join(subj, "T1_masked.nii.gz"))
    apply_mask(pet, mask, os.path.join(subj, "PET_in_T1_masked.nii.gz"))
    print(f"✓ regenerated masks for {os.path.basename(subj)}")

print("All done.")
