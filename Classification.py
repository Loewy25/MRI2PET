#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI→PET amyloid classification (Centiloid labels) with SVM (precomputed linear kernel).
- Reads exactly five fold dirs under --root named ab_MGDA_UB_v33_500_1..5
- Aggregates all subjects in their 'volumes/<SUBJECT>/' folders
- Loads Centiloid labels from CSV (>= thr → positive)
- Nested CV (outer=eval, inner=hyperparameter tuning) with precomputed linear kernel
- NO resampling/harmonization; instead we assert identical shape & affine and print them.
- Outputs predictions_<mod>.csv and metrics_summary.csv
"""

import os, glob, argparse, warnings
from collections import Counter
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import image as nimg
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_brain_mask
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, precision_recall_curve,
    auc as sk_auc
)

warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")

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
                dups.append(sid)
                continue
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

# ---------- alignment check (no resampling) ----------

def _load_squeezed(img_path):
    img = nimg.load_img(img_path)
    if len(img.shape) == 4 and img.shape[-1] == 1:
        img = nimg.index_img(img, 0)
    return img

def _affine_equal(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)

def _affine_str(a):
    return np.array2string(a, formatter={'float_kind': lambda x: f"{x: .5f}"})

def _assert_all_aligned(imgs, tol=1e-5, label=""):
    """Assert every image has the same shape and affine as imgs[0]."""
    ref = imgs[0]
    mismatches = []
    for i, im in enumerate(imgs[1:], start=1):
        shape_ok = (im.shape == ref.shape)
        affine_ok = _affine_equal(im.affine, ref.affine, atol=tol)
        if not (shape_ok and affine_ok):
            mismatches.append((i, im.shape, shape_ok, affine_ok))
    if mismatches:
        msg = [f"[ALIGN-ERROR] {label}: {len(mismatches)} mismatch(es) vs reference:"]
        for (i, shp, shape_ok, affine_ok) in mismatches[:10]:
            msg.append(f"  - img#{i}: shape={shp} (ok={shape_ok}), affine_ok={affine_ok}")
        raise ValueError("\n".join(msg))
    return ref

def build_masker_from_training(train_img_paths, verbose_tag=""):
    # load training images (no resample), assert alignment, print reference
    train_imgs = [_load_squeezed(p) for p in train_img_paths]
    ref_img = _assert_all_aligned(train_imgs, label=f"{verbose_tag}/train")
    print(f"[ALIGN] {verbose_tag} TRAIN reference shape: {ref_img.shape}")
    print(f"[AFFINE] {verbose_tag} TRAIN reference affine:\n{_affine_str(ref_img.affine)}")
    mask_img = compute_brain_mask(ref_img)
    masker = NiftiMasker(mask_img=mask_img, standardize=False, smoothing_fwhm=None)
    masker.fit(train_imgs)  # learn mask voxels from TRAIN only
    return masker, ref_img

def extract_features(masker, img_paths, ref_img, verbose_tag=""):
    # load images (no resample), assert alignment to ref, then transform
    imgs = [_load_squeezed(p) for p in img_paths]
    _assert_all_aligned([ref_img] + imgs, label=f"{verbose_tag}/all")
    X = masker.transform(imgs)
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
        trp,tep = [img_paths[i] for i in tr],[img_paths[i] for i in te]
        ytr,yte = y[tr],y[te]

        # Build masker on TRAIN (no resampling), check alignment and print shape/affine once
        fold_tag = f"{mod_tag}/fold{k}"
        masker, ref = build_masker_from_training(trp, verbose_tag=fold_tag)

        # Transform features (with strict alignment check for both train and test)
        Xtr = extract_features(masker, trp, ref_img=ref, verbose_tag=f"{fold_tag}/train")
        Xte = extract_features(masker, tep, ref_img=ref, verbose_tag=f"{fold_tag}/test")

        # Control-only min-max scaling
        apply = fit_minmax_on_controls(Xtr,ytr)
        Xtr,Xte = apply(Xtr),apply(Xte)

        # Precomputed linear kernels
        Ktr,Kte = Xtr@Xtr.T, Xte@Xtr.T

        # Inner CV grid search
        gs = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced", probability=True),
            {"C": list(C_grid)}, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1),
            refit=True, n_jobs=-1
        )
        gs.fit(Ktr,ytr)
        best = gs.best_estimator_

        prob = best.predict_proba(Kte)[:,1]
        pred = (prob>=0.5).astype(int)
        print(f"[{fold_tag}] C={gs.best_params_['C']} AUC={roc_auc_score(yte,prob):.4f} "
              f"n_train={len(tr)} n_test={len(te)}")

        yT += yte.tolist()
        yP += prob.tolist()
        yH += pred.tolist()
        subjT += [subjects[i] for i in te]

    yT,yP,yH=np.array(yT),np.array(yP),np.array(yH)
    tn,fp,fn,tp = confusion_matrix(yT,yH).ravel()
    metrics=dict(
        AUC=roc_auc_score(yT,yP),AUPRC=compute_auprc(yT,yP),
        Accuracy=accuracy_score(yT,yH),
        BalancedAccuracy=balanced_accuracy_score(yT,yH),
        F1=f1_score(yT,yH),Sensitivity=recall_score(yT,yH),
        Specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan,
        PPV=precision_score(yT,yH,zero_division=1),
        NPV=tn/(tn+fn) if (tn+fn)>0 else np.nan
    )
    preds=pd.DataFrame({"subject":subjT,"y_true":yT,"y_prob":yP,"y_pred":yH})
    return metrics,preds

# ---------- main ----------

def main():
    import sys
    # make prints show up even under Slurm
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--session_col", default="TAU_PET_Session")
    ap.add_argument("--centiloid_col", default="Centiloid")
    ap.add_argument("--centiloid_thr", type=float, default=18.4)
    ap.add_argument("--modalities", nargs="+", default=["pet_gt", "pet_fake", "mri"])
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--c_grid", nargs="+", type=float, default=[1, 10, 100, 0.1, 0.01, 0.001])
    ap.add_argument("--outdir", default="./svm_results")
    args = ap.parse_args()

    print("[ARGS]", vars(args), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    # Folds & subjects
    folds = strict_fold_dirs(args.root, "ab_MGDA_UB_v33_500_", [1, 2, 3, 4, 5])
    for f in folds:
        print("FOLD:", f, flush=True)
    subj = collect_subjects_from_folds(folds)

    # Labels
    sids = sorted(subj.keys())
    labels = load_amyloid_labels(
        args.meta_csv, sids, args.session_col, args.centiloid_col, args.centiloid_thr
    )
    keep = [s for s in sids if norm_key(s) in labels]
    drop = [s for s in sids if norm_key(s) not in labels]
    if drop:
        print(f"[WARN] unlabeled excluded: {len(drop)} e.g. {drop[:6]}", flush=True)
    if not keep:
        raise RuntimeError("No labeled subjects after join.")
    y = np.array([labels[norm_key(s)] for s in keep], dtype=int)
    print("[LABELS] 0/1:", dict(Counter(y)), flush=True)

    # Manifest
    manifest = pd.DataFrame(
        [{"subject": s, "label": int(labels[norm_key(s)]), **subj[s]} for s in keep]
    ).sort_values("subject")
    mpath = os.path.join(args.outdir, "dataset_manifest.csv")
    manifest.to_csv(mpath, index=False)
    print("[WRITE]", mpath, flush=True)

    # Per-modality CV
    rows = []
    for mod in args.modalities:
        print("\n=== Running SVM for", mod, "===", flush=True)
        paths = [subj[s][mod] for s in keep]

        metrics, preds = nested_cv_precomputed_kernel(
            img_paths=paths,
            y=y,
            subjects=keep,
            outer_splits=args.outer,
            inner_splits=args.inner,
            C_grid=tuple(args.c_grid),
            seed=10,
            mod_tag=mod,
        )

        ppath = os.path.join(args.outdir, f"predictions_{mod}.csv")
        preds.to_csv(ppath, index=False)
        print("[WRITE]", ppath, flush=True)

        print(
            f"[{mod}] AUC={metrics['AUC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
            f"Acc={metrics['Accuracy']:.4f}  BalAcc={metrics['BalancedAccuracy']:.4f}",
            flush=True,
        )
        rows.append({"modality": mod, **{k: float(v) for k, v in metrics.items()}})

    summary = pd.DataFrame(rows).set_index("modality")
    spath = os.path.join(args.outdir, "metrics_summary.csv")
    summary.to_csv(spath)
    print("\n[WRITE]", spath, flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()


