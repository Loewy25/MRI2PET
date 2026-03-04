#!/usr/bin/env python3
import os
import nibabel as nib

BASE_DIR = os.environ.get("BASE_DIR", "/scratch/l.peiwang/kari_all")

ok = 0
skipped = 0
shape_mismatch = 0

for dirpath, _, filenames in os.walk(BASE_DIR):
    files = set(filenames)

    if not {"T1.nii.gz", "PET_in_T1.nii.gz", "aseg_brainmask.nii.gz"}.issubset(files):
        continue

    mri_path = os.path.join(dirpath, "T1.nii.gz")
    pet_path = os.path.join(dirpath, "PET_in_T1.nii.gz")
    mask_path = os.path.join(dirpath, "aseg_brainmask.nii.gz")

    try:
        mri_img = nib.load(mri_path)
        pet_img = nib.load(pet_path)
        mask_img = nib.load(mask_path)
    except Exception as e:
        print(f"[SKIP] {dirpath} (load error: {e})")
        skipped += 1
        continue

    mask_data = mask_img.get_fdata() > 0
    mri_data = mri_img.get_fdata()
    pet_data = pet_img.get_fdata()

    if mri_data.shape != mask_data.shape or pet_data.shape != mask_data.shape:
        print(
            f"[SKIP] {dirpath} (shape mismatch: "
            f"T1={mri_data.shape}, PET={pet_data.shape}, MASK={mask_data.shape})"
        )
        shape_mismatch += 1
        continue

    mri_masked_data = mri_data * mask_data
    pet_masked_data = pet_data * mask_data

    mri_out = os.path.join(dirpath, "T1_masked.nii.gz")
    pet_out = os.path.join(dirpath, "PET_in_T1_masked.nii.gz")

    nib.save(nib.Nifti1Image(mri_masked_data, mri_img.affine, mri_img.header), mri_out)
    nib.save(nib.Nifti1Image(pet_masked_data, pet_img.affine, pet_img.header), pet_out)

    print(
        f"[OK] {dirpath} | Mask coverage: {mask_data.mean() * 100:.2f}% | "
        "Saved: T1_masked.nii.gz, PET_in_T1_masked.nii.gz"
    )
    ok += 1

print("\n=== SUMMARY ===")
print(f"Root            : {BASE_DIR}")
print(f"Masked subjects : {ok}")
print(f"Skipped (load)  : {skipped}")
print(f"Skipped (shape) : {shape_mismatch}")
