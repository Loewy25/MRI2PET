#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, warnings
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, precision_recall_curve,
    auc as sk_auc
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- utils ----------

def compute_auprc(y_true, y_prob):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return sk_auc(r, p)

def _load_squeezed_arr(path):
    img = nib.load(path)
    arr = np.asarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 4D(last=1). Got {arr.shape} for {path}")
    return arr

def _assert_same_shape(paths, tag=""):
    ref_shape = None
    for p in paths:
        arr = _load_squeezed_arr(p)
        shp = arr.shape
        if ref_shape is None:
            ref_shape = shp
        elif shp != ref_shape:
            raise ValueError(f"[ALIGN-ERROR] {tag}: {p} shape {shp} vs {ref_shape}")
    print(f"[ALIGN] {tag}: all shapes identical {ref_shape}")
    return ref_shape

def _extract_features_flat(paths, ref_shape, tag=""):
    X = []
    for p in paths:
        arr = _load_squeezed_arr(p)
        if arr.shape != ref_shape:
            raise ValueError(f"[ALIGN-ERROR] {tag}: {p} shape {arr.shape} vs {ref_shape}")
        vec = arr.reshape(-1).astype(np.float32)
        X.append(vec)
    X = np.vstack(X)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def _fit_minmax_controls(X_train, y_train):
    ctrl = X_train[y_train == 0]
    if ctrl.shape[0] == 0:
        raise ValueError("No controls in this training split.")
    mins = ctrl.min(0)
    maxs = ctrl.max(0)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    def apply(X):
        X = (X - mins) / scale
        return np.nan_to_num(X, nan=0.0).astype(np.float32, copy=False)
    return apply

# ---------- core ----------

def run_from_manifest(manifest_csv, modality, outdir, outer, inner, c_grid, seed):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(manifest_csv)
    needed = {"subject", "label", modality}
    if not needed.issubset(df.columns):
        raise KeyError(f"CSV must contain columns {needed}. Got: {list(df.columns)}")

    # Drop rows with missing paths or labels
    df = df.dropna(subset=["label", modality]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Ensure files exist; filter if not
    bad = [p for p in df[modality].tolist() if not os.path.exists(p)]
    if bad:
        print(f"[WARN] {len(bad)} missing files (e.g., {bad[:3]}). Excluding those rows.")
        df = df[df[modality].apply(os.path.exists)].copy()
    if df.empty:
        raise RuntimeError("No valid rows after filtering paths/labels.")

    subjects = df["subject"].astype(str).tolist()
    y = df["label"].to_numpy(dtype=int)
    paths = df[modality].astype(str).tolist()

    print(f"[DATA] N={len(df)}  pos/neg={dict(Counter(y))}  modality={modality}")
    ref_shape = _assert_same_shape(paths, tag=f"{modality}/all")

    # Nested CV (precomputed linear kernel)
    skf_outer = StratifiedKFold(n_splits=outer, shuffle=True, random_state=seed)
    yT, yP, yH, subjT = [], [], [], []

    for k, (tr, te) in enumerate(skf_outer.split(paths, y), 1):
        trp = [paths[i] for i in tr]; tep = [paths[i] for i in te]
        ytr = y[tr]; yte = y[te]

        print(f"[{modality}/fold{k}] y_train={dict(Counter(ytr))} y_test={dict(Counter(yte))}")

        Xtr = _extract_features_flat(trp, ref_shape, tag=f"{modality}/train")
        Xte = _extract_features_flat(tep, ref_shape, tag=f"{modality}/test")

        apply = _fit_minmax_controls(Xtr, ytr)
        Xtr = apply(Xtr); Xte = apply(Xte)

        Ktr = (Xtr @ Xtr.T).astype(np.float32, copy=False)
        Kte = (Xte @ Xtr.T).astype(np.float32, copy=False)

        gs = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced", probability=True),
            {"C": list(c_grid)}, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=inner, shuffle=True, random_state=1),
            refit=True, n_jobs=1  # keep memory sane
        )
        gs.fit(Ktr, ytr)
        best = gs.best_estimator_

        prob = best.predict_proba(Kte)[:, 1]
        auc = roc_auc_score(yte, prob)
        auc_inv = roc_auc_score(yte, 1.0 - prob)
        print(f"[{modality}/fold{k}] C={gs.best_params_['C']}  AUC={auc:.4f}  (1-AUC={auc_inv:.4f}) "
              f"n_train={len(tr)} n_test={len(te)}")

        pred = (prob >= 0.5).astype(int)
        yT.extend(yte.tolist()); yP.extend(prob.tolist()); yH.extend(pred.tolist())
        subjT.extend([subjects[i] for i in te])

    yT = np.array(yT); yP = np.array(yP); yH = np.array(yH)
    tn, fp, fn, tp = confusion_matrix(yT, yH).ravel()
    metrics = dict(
        AUC=roc_auc_score(yT, yP),
        AUPRC=compute_auprc(yT, yP),
        Accuracy=accuracy_score(yT, yH),
        BalancedAccuracy=balanced_accuracy_score(yT, yH),
        F1=f1_score(yT, yH),
        Sensitivity=recall_score(yT, yH),
        Specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan,
        PPV=precision_score(yT, yH, zero_division=1),
        NPV=tn/(tn+fn) if (tn+fn)>0 else np.nan
    )

    preds = pd.DataFrame({"subject": subjT, "y_true": yT, "y_prob": yP, "y_pred": yH})
    ppath = os.path.join(outdir, f"predictions_{modality}.csv")
    preds.to_csv(ppath, index=False); print("[WRITE]", ppath)

    summary = pd.DataFrame([{"modality": modality, **{k: float(v) for k,v in metrics.items()}}]).set_index("modality")
    spath = os.path.join(outdir, "metrics_summary.csv")
    summary.to_csv(spath); print("[WRITE]", spath)
    print(summary.to_string())
    return metrics

# ---------- cli ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to dataset_manifest.csv")
    ap.add_argument("--modality", required=True, choices=["mri","pet_gt","pet_fake"])
    ap.add_argument("--outdir", default="./svm_results_manifest")
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--c_grid", nargs="+", type=float, default=[1, 10, 100, 0.1, 0.01, 0.001])
    ap.add_argument("--seed", type=int, default=10)
    args = ap.parse_args()

    print("[ARGS]", vars(args))
    run_from_manifest(args.manifest, args.modality, args.outdir, args.outer, args.inner, args.c_grid, args.seed)

if __name__ == "__main__":
    main()
