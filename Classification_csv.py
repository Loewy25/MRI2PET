#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset_manifest.csv for MRI→PET synthesis experiments.

- Find PET_fake in:   <fake_dir>/volumes/<SUBJECT>/PET_fake.nii.gz   (dirs passed in --fake_dirs)
- For each SUBJECT, set real MRI/PET from original dataset:
      <orig_root>/<SUBJECT>/T1_masked.nii.gz            -> mri
      <orig_root>/<SUBJECT>/PET_in_T1_masked.nii.gz     -> pet_gt
- Join meta CSV on --session_col (e.g., TAU_PET_Session) to keep raw numeric labels:
      Centiloid, MTL, NEO, Braak1_2, Braak3_4, Braak5_6, cdr
- Write a single CSV with columns:
  subject, mri, pet_gt, pet_fake, fake_fold, Centiloid, MTL, NEO, Braak1_2, Braak3_4, Braak5_6, cdr
"""

import os, glob, argparse, pandas as pd, numpy as np

LABEL_COLS = ["Centiloid","MTL","NEO","Braak1_2","Braak3_4","Braak5_6","cdr"]

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def find_pet_fake(fake_dirs):
    """Return dict: subject -> {'pet_fake': path, 'fake_fold': fold} (first hit wins)."""
    found = {}
    dups = []
    # fake_dirs is a list of full paths
    for i, fd in enumerate(fake_dirs):
        # fold index 1-based (i+1)
        current_fold = i + 1
        vol_dir = os.path.join(fd, "volumes")
        if not os.path.isdir(vol_dir):
            print(f"[WARN] missing volumes dir: {vol_dir}")
            continue
        for subj_dir in sorted(glob.glob(os.path.join(vol_dir, "*"))):
            if not os.path.isdir(subj_dir):
                continue
            subj = os.path.basename(subj_dir)
            pet_fake = os.path.join(subj_dir, "PET_fake.nii.gz")
            if os.path.exists(pet_fake):
                if subj in found:
                    dups.append(subj)
                    continue
                found[subj] = {"pet_fake": pet_fake, "fake_fold": current_fold}
    if dups:
        print(f"[INFO] duplicate subjects across folds (kept first): {len(set(dups))}")
    print(f"[DISCOVER] subjects with PET_fake: {len(found)}")
    return found

def attach_real_modalities(rows: dict, orig_root: str):
    """Add real MRI/PET paths if they exist under orig_root/<SUBJECT>/..."""
    keep = {}
    miss_mri, miss_pet = 0, 0
    for subj, info in rows.items():
        sd = os.path.join(orig_root, subj)
        mri = os.path.join(sd, "T1_masked.nii.gz")
        pet = os.path.join(sd, "PET_in_T1_masked.nii.gz")
        ok_mri = os.path.exists(mri)
        ok_pet = os.path.exists(pet)
        if ok_mri and ok_pet:
            keep[subj] = {**info, "mri": mri, "pet_gt": pet}
        else:
            miss_mri += (not ok_mri)
            miss_pet += (not ok_pet)
    print(f"[FILTER] kept {len(keep)} subjects with all three files; "
          f"missing MRI={miss_mri}, missing PET_gt={miss_pet}")
    return keep

def load_meta(meta_csv: str, session_col: str):
    """Return dict key-> {label cols}, taking MAX per key for duplicates."""
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    cmap = {c.strip().lower(): c for c in df.columns}
    sess_key = session_col.strip().lower()
    if sess_key not in cmap:
        raise KeyError(f"session_col '{session_col}' not in meta CSV. Columns: {list(df.columns)}")
    # Select available label columns
    cols_present = [cmap[c.lower()] for c in LABEL_COLS if c.lower() in cmap]
    out = df[[cmap[sess_key]] + cols_present].copy()
    out["__key__"] = out[cmap[sess_key]].astype(str).map(norm_key)
    # numeric coercion
    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # aggregate by max
    agg = {c: "max" for c in cols_present}
    meta = out.groupby("__key__", as_index=False).agg(agg)
    return meta.set_index("__key__").to_dict(orient="index")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dirs", nargs="+", required=True,
                    help="List of fake PET fold directories, e.g. path/to/fold1 path/to/fold2 ...")
    ap.add_argument("--orig_root", required=True,
                    help="Original dataset root with <SUBJECT>/T1_masked.nii.gz and PET_in_T1_masked.nii.gz")
    ap.add_argument("--meta_csv", required=True, help="Meta CSV with labels")
    ap.add_argument("--session_col", default="TAU_PET_Session",
                    help="Column in meta CSV that matches SUBJECT names")
    ap.add_argument("--out_csv", required=True, help="Output manifest CSV path")
    args = ap.parse_args()

    # 1) gather fake PETs
    rows = find_pet_fake(args.fake_dirs)

    # 2) attach real MRI/PET
    rows = attach_real_modalities(rows, args.orig_root)

    if not rows:
        raise SystemExit("No subjects with all required files were found.")

    # 3) attach meta labels
    meta = load_meta(args.meta_csv, args.session_col)
    records = []
    miss_meta = 0
    for subj, info in rows.items():
        key = norm_key(subj)
        labels = meta.get(key, None)
        if labels is None:
            miss_meta += 1
            labels = {}
        rec = {
            "subject": subj,
            "mri": info["mri"],
            "pet_gt": info["pet_gt"],
            "pet_fake": info["pet_fake"],
            "fake_fold": info["fake_fold"],
        }
        # add only the label columns that exist in meta
        for name in LABEL_COLS:
            rec[name] = labels.get(name, np.nan)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df = df[["subject","mri","pet_gt","pet_fake","fake_fold"] + LABEL_COLS]  # column order

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[WRITE] {args.out_csv}  rows={len(df)}  meta_missing={miss_meta}")

if __name__ == "__main__":
    main()


