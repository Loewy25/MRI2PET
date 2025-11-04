#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build dataset_manifest.csv with CDR labels (0, 0.5, 1), no classification.

- Expects five fold dirs under --root:
    ab_MGDA_UB_v33_500_1 .. ab_MGDA_UB_v33_500_5
- Each subject folder: volumes/<SUBJECT>/{MRI.nii.gz, PET_gt.nii.gz, PET_fake.nii.gz}
- Labels are pulled from --meta_csv using --session_col to match SUBJECT and --cdr_col for CDR.
  If multiple rows per subject, we take the MAX CDR (0 < 0.5 < 1).
- Writes:  <outdir>/dataset_manifest.csv  with columns:
    subject,label,mri,pet_gt,pet_fake
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- helpers ----------

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def strict_fold_dirs(root: str, base_name: str, fold_ids):
    dirs = []
    for i in fold_ids:
        d = os.path.join(root, f"{base_name}{i}")
        if not (os.path.isdir(d) and os.path.isdir(os.path.join(d, "volumes"))):
            raise AssertionError(f"Missing expected fold or its 'volumes': {d}")
        dirs.append(d)
    return dirs

def collect_subjects_from_folds(fold_dirs):
    """Return {subject_id: {'mri':..., 'pet_gt':..., 'pet_fake':...}}"""
    subjects = {}
    dups = []
    for fd in fold_dirs:
        vol_dir = os.path.join(fd, "volumes")
        for subj_dir in sorted(glob.glob(os.path.join(vol_dir, "*"))):
            if not os.path.isdir(subj_dir):
                continue
            sid = os.path.basename(subj_dir)
            mri = os.path.join(subj_dir, "MRI.nii.gz")
            pet_gt = os.path.join(subj_dir, "PET_gt.nii.gz")
            pet_fake = os.path.join(subj_dir, "PET_fake.nii.gz")
            if not (os.path.exists(mri) and os.path.exists(pet_gt) and os.path.exists(pet_fake)):
                continue
            if sid in subjects:
                dups.append(sid); continue
            subjects[sid] = {"mri": mri, "pet_gt": pet_gt, "pet_fake": pet_fake}
    if not subjects:
        raise RuntimeError("No complete subjects found.")
    if dups:
        print(f"[WARN] duplicates skipped: {len(set(dups))} (e.g., {dups[:6]})")
    print(f"[DISCOVER] total unique subjects: {len(subjects)}")
    return subjects

def load_cdr_labels(meta_csv, subject_ids, session_col, cdr_col):
    """
    Build {subject_key -> CDR} using MAX CDR per subject key.
    - session_col must match the folder SUBJECT names (case/space-insensitive).
    - cdr_col expected values in {0, 0.5, 1} (coerced to numeric).
    - Drop NaN CDR rows. Subjects without valid CDR are excluded by caller.
    """
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    cols = {c.strip().lower(): c for c in df.columns}
    if session_col.strip().lower() not in cols or cdr_col.strip().lower() not in cols:
        raise KeyError(f"Missing columns in meta CSV: {session_col}, {cdr_col}")
    sess, cdr = cols[session_col.strip().lower()], cols[cdr_col.strip().lower()]

    df[cdr] = pd.to_numeric(df[cdr], errors="coerce")
    df["__key__"] = df[sess].astype(str).map(norm_key)

    keys = {norm_key(s) for s in subject_ids}
    matched = df[df["__key__"].isin(keys)].copy()
    if matched.empty:
        raise ValueError("No subjects matched folder names vs SESSION_COL.")

    n_nan = matched[cdr].isna().sum()
    if n_nan > 0:
        print(f"[WARN] dropping {n_nan}/{len(matched)} matched rows with NaN {cdr_col}")
        matched = matched.dropna(subset=[cdr])

    if matched.empty:
        raise ValueError("No matched rows with valid CDR after dropping NaNs.")

    # Take MAX CDR per subject key (0 < 0.5 < 1)
    per_key_max = matched.groupby("__key__")[cdr].max()

    # Sanity: restrict to {0, 0.5, 1} if values are slightly off due to float
    def snap(v):
        if np.isclose(v, 0.0): return 0.0
        if np.isclose(v, 0.5): return 0.5
        if np.isclose(v, 1.0): return 1.0
        return float(v)  # keep as-is if other valid values exist
    labels = per_key_max.map(snap)
    return dict(labels)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root containing ab_MGDA_UB_v33_500_1..5")
    ap.add_argument("--meta_csv", required=True, help="CSV with session + CDR columns")
    ap.add_argument("--session_col", default="TAU_PET_Session", help="Column matching SUBJECT folder names")
    ap.add_argument("--cdr_col", default="cdr", help="CDR column (values 0/0.5/1)")
    ap.add_argument("--outdir", default="./svm_results", help="Where to write dataset_manifest.csv")
    args = ap.parse_args()

    print("[ARGS]", vars(args), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) discover subjects
    folds = strict_fold_dirs(args.root, "ab_MGDA_UB_v33_500_", [1, 2, 3, 4, 5])
    for f in folds:
        print("FOLD:", f, flush=True)
    subj = collect_subjects_from_folds(folds)

    # 2) build labels from CDR
    sids = sorted(subj.keys())
    labels = load_cdr_labels(args.meta_csv, sids, args.session_col, args.cdr_col)

    keep = [s for s in sids if norm_key(s) in labels]
    drop = [s for s in sids if norm_key(s) not in labels]
    if drop:
        print(f"[WARN] subjects excluded (missing CDR): {len(drop)} e.g. {drop[:6]}", flush=True)
    if not keep:
        raise RuntimeError("No labeled subjects after join.")

    y_vals = [labels[norm_key(s)] for s in keep]
    counts = Counter(y_vals)
    print("[LABELS] CDR counts:", dict(counts), flush=True)

    # 3) write manifest (NO classification)
    manifest = pd.DataFrame(
        [{"subject": s, "label": labels[norm_key(s)], **subj[s]} for s in keep]
    ).sort_values("subject")
    mpath = os.path.join(args.outdir, "dataset_manifest.csv")
    manifest.to_csv(mpath, index=False)
    print("[WRITE]", mpath, flush=True)
    print(manifest.head(5).to_string(index=False))

if __name__ == "__main__":
    main()




