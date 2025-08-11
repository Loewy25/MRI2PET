import os, glob
import nibabel as nib
import numpy as np
from nilearn.image import index_img, resample_to_img, new_img_like

# ==== CONFIG ====
BASE_DIR = "/ceph/chpc/mapped/benz04_kari/pup"     # where the av1451 folders are
OUT_DIR  = "/scratch/l.peiwang/kari_brain" # output folder
os.makedirs(OUT_DIR, exist_ok=True)

# ==== HELPERS ====
def squeeze(img):
    return index_img(img, 0) if len(img.shape) == 4 and img.shape[3] == 1 else img

# ==== MAIN LOOP ====
folders = [f for f in os.listdir(BASE_DIR) if "av1451" in f.lower()]
print(f"Found {len(folders)} AV1451 folders.")

count_ok = 0
for folder in folders:
    try:
        folder_path = os.path.join(BASE_DIR, folder)
        nifti_dir = glob.glob(os.path.join(folder_path, "CNDA*", "NIFTI_GZ"))
        if not nifti_dir:
            print(f"[SKIP] No NIFTI_GZ in {folder}")
            continue
        nifti_dir = nifti_dir[0]

        # find files
        t1_file  = glob.glob(os.path.join(nifti_dir, "T1.nii.gz"))
        pet_file = glob.glob(os.path.join(nifti_dir, "*msum_SUVR.nii.gz"))
        mask_file= glob.glob(os.path.join(nifti_dir, "BrainMask.nii.gz"))

        if not (t1_file and pet_file and mask_file):
            print(f"[SKIP] Missing file(s) in {folder}")
            continue

        t1_img   = squeeze(nib.load(t1_file[0]))
        pet_img  = squeeze(nib.load(pet_file[0]))
        mask_img = squeeze(nib.load(mask_file[0]))

        # reorient to RAS+
        t1_img   = nib.as_closest_canonical(t1_img)
        pet_img  = nib.as_closest_canonical(pet_img)
        mask_img = nib.as_closest_canonical(mask_img)

        # resample PET & mask to T1 grid
        pet_on_t1  = resample_to_img(pet_img, t1_img, interpolation="continuous")
        mask_on_t1 = resample_to_img(mask_img, t1_img, interpolation="nearest")

        mask = mask_on_t1.get_fdata() > 0.5
        t1_brain  = new_img_like(t1_img,  t1_img.get_fdata()  * mask)
        pet_brain = new_img_like(pet_on_t1, pet_on_t1.get_fdata() * mask)

        # save
        base = os.path.basename(folder)
        out_t1  = os.path.join(OUT_DIR, f"{base}_T1_brain.nii.gz")
        out_pet = os.path.join(OUT_DIR, f"{base}_PET_brain.nii.gz")
        nib.save(t1_brain, out_t1)
        nib.save(pet_brain, out_pet)

        print(f"[OK] {folder} -> saved T1 & PET brain")
        count_ok += 1

    except Exception as e:
        print(f"[ERROR] {folder}: {e}")

print(f"Processing complete: {count_ok}/{len(folders)} folders processed successfully.")
