import os
import nibabel as nib
import numpy as np

BASE_DIR = "/scratch/l.peiwang/kari_brainv11"

for subj in sorted(os.listdir(BASE_DIR)):
    subj_dir = os.path.join(BASE_DIR, subj)
    if not os.path.isdir(subj_dir):
        continue
    
    mri_path  = os.path.join(subj_dir, "T1.nii.gz")
    pet_path  = os.path.join(subj_dir, "PET_in_T1.nii.gz")
    mask_path = os.path.join(subj_dir, "aseg_brainmask.nii.gz")
    
    if not (os.path.exists(mri_path) and os.path.exists(pet_path) and os.path.exists(mask_path)):
        print(f"[SKIP] {subj} (missing one of MRI/PET/Mask)")
        continue
    
    # Load
    mri_img  = nib.load(mri_path)
    pet_img  = nib.load(pet_path)
    mask_img = nib.load(mask_path)
    
    # Mask data
    mask_data = mask_img.get_fdata() > 0
    mri_masked_data = mri_img.get_fdata() * mask_data
    pet_masked_data = pet_img.get_fdata() * mask_data
    
    # Save masked
    mri_out = os.path.join(subj_dir, "T1_masked.nii.gz")
    pet_out = os.path.join(subj_dir, "PET_in_T1_masked.nii.gz")
    
    nib.save(nib.Nifti1Image(mri_masked_data, mri_img.affine, mri_img.header), mri_out)
    nib.save(nib.Nifti1Image(pet_masked_data, pet_img.affine, pet_img.header), pet_out)
    
    # Print summary
    print(f"[OK] {subj} | Mask coverage: {mask_data.mean()*100:.2f}% | Saved: T1_masked.nii.gz, PET_in_T1_masked.nii.gz")


