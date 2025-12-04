#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, glob
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ---------------- Binary label rule: {0,1} -> 0, {2,3} -> 1 ----------------
def braak_label(row, thr=1.2):
    """
    Binary Braak label:
      0 -> original stage 0 or 1
      1 -> original stage 2 or 3
    """
    b56 = float(row["Braak5_6"])
    b34 = float(row["Braak3_4"])
    b12 = float(row["Braak1_2"])

    # original 0–3 staging
    if b56 > thr:
        stage = 3
    elif b34 > thr:
        stage = 2
    elif b12 > thr:
        stage = 1
    else:
        stage = 0

    # collapse to binary
    return 1 if stage >= 2 else 0

def load_nii(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return img, data

def roi_stats(pet, mask):
    vox = pet[mask > 0]
    vox = vox[np.isfinite(vox)]
    if vox.size == 0:
        return None
    return [
        float(np.mean(vox)),
        float(np.median(vox)),
        float(np.percentile(vox, 90)),
        float(np.percentile(vox, 95)),
        int(vox.size)
    ]

# ---------------- XGBoost model: binary ----------------
def build_xgb(seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",   # <--- binary
        eval_metric="logloss",
        random_state=seed,
        n_jobs=8
    )

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pet-col", default="pet_gt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=1.2)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[FATAL] CSV not found: {args.csv}", flush=True); sys.exit(1)

    df = pd.read_csv(args.csv)
    needed = ["subject", args.pet_col, "Braak1_2", "Braak3_4", "Braak5_6"]
    for c in needed:
        if c not in df.columns:
            print(f"[FATAL] missing column {c}", flush=True); sys.exit(1)

    # masks expected in the SAME folder as the PET file
    MASK_LIST = [
        "mask_basalganglia.nii.gz",
        "mask_parenchyma_noBG.nii.gz",
        "ROI_Hippocampus.nii.gz",
        "ROI_PosteriorCingulate.nii.gz",
        "ROI_Precuneus.nii.gz",
        "ROI_TemporalLobe.nii.gz",
        "ROI_LimbicCortex.nii.gz",
        "aseg_brainmask.nii.gz",
    ]

    X_list, y_list = [], []
    skip = 0

    for idx, row in df.iterrows():
        pet_path = str(row[args.pet_col]).strip()
        if not os.path.exists(pet_path):
            skip += 1
            continue

        pet_dir = os.path.dirname(pet_path)

        # load PET
        try:
            _, pet = load_nii(pet_path)
        except Exception as e:
            print(f"[skip] PET load failed for {pet_path}: {e}")
            skip += 1
            continue

        if pet.ndim != 3:
            print(f"[skip] PET not 3D after squeeze: {pet_path}, shape={pet.shape}")
            skip += 1
            continue

        feats = []
        ok = True

        for mname in MASK_LIST:
            mpath = os.path.join(pet_dir, mname)
            if not os.path.exists(mpath):
                print(f"[skip] missing mask {mname} in {pet_dir}")
                ok = False
                break

            try:
                _, mask = load_nii(mpath)
            except Exception as e:
                print(f"[skip] mask load failed {mpath}: {e}")
                ok = False
                break

            if mask.ndim != 3:
                print(f"[skip] mask {mpath} not 3D: shape={mask.shape}")
                ok = False
                break

            if mask.shape != pet.shape:
                print(f"[skip] shape mismatch: PET={pet.shape}, {mname}={mask.shape}")
                ok = False
                break

            st = roi_stats(pet, mask)
            if st is None:
                print(f"[skip] empty ROI {mname} in {pet_dir}")
                ok = False
                break

            feats += st

        if not ok:
            skip += 1
            continue

        y = braak_label(row, thr=args.thr)
        X_list.append(feats)
        y_list.append(int(y))

    if len(X_list) < 20:
        print(f"[FATAL] too few usable subjects: {len(X_list)} (skipped={skip})", flush=True)
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"[info] final usable subjects: {len(X)} (skipped={skip})")
    print(f"[info] X shape = {X.shape}")
    print(f"[info] class counts: {np.bincount(y)}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    accs, aucs, cms = [], [], []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = build_xgb(seed=fold)
        model.fit(X[tr], y[tr])

        proba = model.predict_proba(X[te])[:, 1]  # P(class=1)
        pred  = (proba >= 0.5).astype(int)

        acc = accuracy_score(y[te], pred)
        auc = roc_auc_score(y[te], proba)
        cm  = confusion_matrix(y[te], pred, labels=[0, 1])

        accs.append(acc)
        aucs.append(auc)
        cms.append(cm)

        print(f"Fold{fold}: ACC={acc:.3f}, AUC={auc:.3f}")
        print(cm)

    print("=== FINAL (binary: {0,1}->0, {2,3}->1) ===")
    print("ACC mean:", float(np.mean(accs)), "std:", float(np.std(accs)))
    print("AUC mean:", float(np.mean(aucs)), "std:", float(np.std(aucs)))
    print("Confusion sum:\n", sum(cms))

if __name__ == "__main__":
    main()

