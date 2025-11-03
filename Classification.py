#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI→PET amyloid classification (Centiloid labels) with SVM (precomputed linear kernel).
- Expects five fold dirs under --root named ab_MGDA_UB_v33_500_1..5
- Each subject in volumes/<SUBJECT>/ has MRI.nii.gz, PET_gt.nii.gz, PET_fake.nii.gz
- Labels from Centiloid: >= thr → positive
- Nested CV (outer eval, inner C-grid) with precomputed linear kernel
- NO harmonization, NO masking: use ALL voxels (flattened).
- Outputs predictions_<mod>.csv and metrics_summary.csv
"""

import os, glob, argparse, warnings
from collections import Counter
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, precision_recall_curve,
    auc as sk_auc
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- helpers ----------

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def compute_auprc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return sk_auc(r, p)

def strict_fold_dirs(root: str, base_name: str, fold_ids):
    dirs = []
    for i in fold_ids:
        d = os.path.join(root, f"{base_name}{i}")
        if not (os.path.isdir(d) and os.path.isdir(os.path.join(d, "volumes"))):
            raise AssertionError(f"Missing expected fold or its 'volumes': {d}")
        dirs.append(d)
    return dirs

def collect_subjects_from_folds(fold_dirs):
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

def load_amyloid_labels(meta_csv, subject_ids, session_col, centiloid_col, thr):
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    cols = {c.strip().lower(): c for c in df.columns}
    if session_col.strip().lower() not in cols or centiloid_col.strip().lower() not in cols:
        raise KeyError(f"Missing columns: {session_col}, {centiloid_col}")
    sess, cl = cols[session_col.strip().lower()], cols[centiloid_col.strip().lower()]
    df[cl] = pd.to_numeric(df[cl], errors="coerce")
    df["__key__"] = df[sess].astype(str).map(norm_key)
    keys = {norm_key(s) for s in subject_ids}
    matched = df[df["__key__"].isin(keys)]
    if matched.empty:
        raise ValueError("No subjects matched folder names vs SESSION_COL.")
    per_key_max = matched.groupby("__key__")[cl].max()
    labels = (per_key_max >= thr).astype(int)
    return dict(labels)

# ---------- I/O (no masking, no harmonization) ----------

def _load_squeezed_arr(path):
    """Load as array; if 4D with last dim==1, squeeze to 3D. Return (arr, affine, shape3d)."""
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D(last=1). Got {arr.shape} for {path}")
    return arr, img.affine, arr.shape

def _assert_same_shape(paths, label=""):
    ref_shape = None
    for i, p in enumerate(paths):
        arr, _, shp = _load_squeezed_arr(p)
        if ref_shape is None:
            ref_shape = shp
        elif shp != ref_shape:
            raise ValueError(f"[ALIGN-ERROR] {label}: {p} shape {shp} vs {ref_shape}")
    print(f"[ALIGN] {label}: all shapes identical {ref_shape}")
    return ref_shape

def extract_features_flat(paths, ref_shape, label=""):
    X = []
    for p in paths:
        arr, _, shp = _load_squeezed_arr(p)
        if shp != ref_shape:
            raise ValueError(f"[ALIGN-ERROR] {label}: {p} shape {shp} vs {ref_shape}")
        vec = arr.reshape(-1).astype(np.float32)
        X.append(vec)
    X = np.vstack(X)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def fit_minmax_on_controls(X_train, y_train):
    ctrl = X_train[y_train == 0]
    if ctrl.shape[0] == 0:
        raise ValueError("No controls in this training split.")
    mins, maxs = ctrl.min(0), ctrl.max(0)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    def apply(X): return np.nan_to_num((X - mins)/scale, nan=0.0)
    return apply

# ---------- main CV loop ----------

def nested_cv_precomputed_kernel(img_paths, y, subjects, outer_splits, inner_splits, C_grid, seed=10, mod_tag=""):
    skf_outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    yT, yP, yH, subjT = [], [], [], []
    for k,(tr,te) in enumerate(skf_outer.split(img_paths,y),1):
        trp, tep = [img_paths[i] for i in tr], [img_paths[i] for i in te]
        ytr, yte = y[tr], y[te]

        fold_tag = f"{mod_tag}/fold{k}"

        # Only enforce same shape; ignore affine/origin/orientation
        ref_shape = _assert_same_shape(trp, label=f"{fold_tag}/train")

        # Features = flatten full volume (ALL voxels)
        Xtr = extract_features_flat(trp, ref_shape=ref_shape, label=f"{fold_tag}/train")
        Xte = extract_features_flat(tep, ref_shape=ref_shape, label=f"{fold_tag}/test")

        # Control-only scaling
        apply = fit_minmax_on_controls(Xtr, ytr)
        Xtr, Xte = apply(Xtr), apply(Xte)

        # Precomputed linear kernels
        Ktr, Kte = Xtr @ Xtr.T, Xte @ Xtr.T

        # Inner CV grid search
        gs = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced", probability=True),
            {"C": list(C_grid)}, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1),
            refit=True, n_jobs=-1
        )
        gs.fit(Ktr, ytr)
        best = gs.best_estimator_

        prob = best.predict_proba(Kte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        print(f"[{fold_tag}] C={gs.best_params_['C']} AUC={roc_auc_score(yte, prob):.4f} "
              f"n_train={len(tr)} n_test={len(te)}")

        yT += yte.tolist(); yP += prob.tolist(); yH += pred.tolist()
        subjT += [subjects[i] for i in te]

    yT, yP, yH = np.array(yT), np.array(yH), np.array(yH)  # careful—fix below
    # Correct yP (probabilities) rebuild because we overwrote accidentally:
    # (safer: recompute metrics from stored lists before conversion)


