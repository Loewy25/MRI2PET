#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ---------------- Labels (same as your DL code) ----------------
def braak_label(row, thr=1.2):
    b56 = float(row["Braak5_6"])
    b34 = float(row["Braak3_4"])
    b12 = float(row["Braak1_2"])
    if b56 > thr: return 3
    elif b34 > thr: return 2
    elif b12 > thr: return 1
    else: return 0

def macro_auc_safe(y_true, proba, n_classes):
    aucs = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        if yb.min() == 0 and yb.max() == 1:
            aucs.append(roc_auc_score(yb, proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")

def info(msg): print(f"[info] {msg}", flush=True)
def warn(msg): print(f"[warn] {msg}", flush=True)

# ---------------- Feature extraction ----------------
def load_nii(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    return img, data

def affine_close(A, B, atol=1e-4):
    return np.allclose(A, B, atol=atol)

def roi_stats(pet_data, mask_data):
    vox = pet_data[mask_data > 0]
    vox = vox[np.isfinite(vox)]
    if vox.size == 0:
        return None
    return {
        "mean": float(np.mean(vox)),
        "median": float(np.median(vox)),
        "p90": float(np.percentile(vox, 90)),
        "p95": float(np.percentile(vox, 95)),
        "nvox": int(vox.size),
    }

def find_masks(subject_dir, patterns):
    masks = []
    for pat in patterns:
        masks.extend(sorted([p for p in glob_glob(os.path.join(subject_dir, pat)) if os.path.isfile(p)]))
    # unique keep order
    seen = set()
    out = []
    for m in masks:
        if m not in seen:
            seen.add(m); out.append(m)
    return out

def glob_glob(pat):
    import glob
    return glob.glob(pat)

# ---------------- Model ----------------
def build_model(seed=0, n_classes=4):
    # best choice: XGBoost if installed
    try:
        from xgboost import XGBClassifier
        return ("xgboost",
                XGBClassifier(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    random_state=seed,
                    n_jobs=8,
                ))
    except Exception as e:
        # fallback (still works, but usually a bit worse)
        from sklearn.ensemble import HistGradientBoostingClassifier
        warn(f"XGBoost not available ({e}). Falling back to HistGradientBoostingClassifier.")
        return ("hgb",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=400,
                    random_state=seed
                ))

# ---------------- Main pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: subject, pet_gt, Braak1_2, Braak3_4, Braak5_6")
    ap.add_argument("--dataset-root", required=True, help="Root containing subject folders with masks")
    ap.add_argument("--pet-col", default="pet_gt", help="Which CSV column is the PET path (default: pet_gt)")
    ap.add_argument("--thr", type=float, default=1.2, help="Threshold in your braak_label rule")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--atol-affine", type=float, default=1e-4)
    ap.add_argument("--mask", action="append", default=[
        "ROI_*.nii.gz",
        "mask_*.nii.gz",
        "aseg_brainmask.nii.gz",
    ], help="Mask glob(s) relative to each subject folder. Can pass multiple times.")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[FATAL] CSV not found: {args.csv}", flush=True); sys.exit(1)
    if not os.path.isdir(args.dataset_root):
        print(f"[FATAL] dataset root not found: {args.dataset_root}", flush=True); sys.exit(1)

    df = pd.read_csv(args.csv)
    needed = ["subject", args.pet_col, "Braak1_2", "Braak3_4", "Braak5_6"]
    for c in needed:
        if c not in df.columns:
            print(f"[FATAL] missing column in CSV: {c}", flush=True); sys.exit(1)

    # keep rows where PET exists + subject folder exists + braak fields are numeric
    rows = []
    for idx, r in df.iterrows():
        subj = str(r["subject"]).strip()
        petp = r[args.pet_col]
        if not isinstance(petp, str) or not os.path.exists(petp):
            continue
        sdir = os.path.join(args.dataset_root, subj)
        if not os.path.isdir(sdir):
            continue
        try:
            _ = braak_label(r, thr=args.thr)
        except Exception:
            continue
        rows.append(idx)

    df = df.loc[rows].reset_index(drop=True)
    if len(df) == 0:
        print("[FATAL] No usable rows after checking PET paths + subject folders + Braak columns.", flush=True)
        sys.exit(1)

    info(f"Using {len(df)} rows after basic filtering.")
    info(f"PET column: {args.pet_col} | dataset root: {args.dataset_root}")
    info(f"Mask patterns: {args.mask}")

    # discover mask list using first subject that has masks
    import glob
    mask_names = None
    for i in range(len(df)):
        subj = str(df.loc[i, "subject"]).strip()
        sdir = os.path.join(args.dataset_root, subj)
        found = []
        for pat in args.mask:
            found += glob.glob(os.path.join(sdir, pat))
        found = [p for p in found if os.path.isfile(p)]
        found = sorted(set(found))
        if found:
            mask_names = [os.path.basename(p) for p in found]
            info(f"Feature mask-set discovered from subject '{subj}': {len(mask_names)} masks")
            break
    if not mask_names:
        print("[FATAL] Could not find any masks in any subject folder using your --mask patterns.", flush=True)
        sys.exit(1)

    # build feature matrix
    X_list, y_list, keep_subjects = [], [], []
    skipped = 0

    for i in range(len(df)):
        subj = str(df.loc[i, "subject"]).strip()
        pet_path = str(df.loc[i, args.pet_col]).strip()
        sdir = os.path.join(args.dataset_root, subj)

        y = braak_label(df.loc[i], thr=args.thr)

        # load PET
        try:
            pet_img, pet = load_nii(pet_path)
        except Exception as e:
            warn(f"{subj}: failed to load PET: {pet_path} ({e})"); skipped += 1; continue
        if pet.ndim == 4 and pet.shape[-1] == 1:
            pet = pet[..., 0]
        if pet.ndim != 3:
            warn(f"{subj}: PET is not 3D after squeeze: shape={pet.shape}"); skipped += 1; continue

        feats = []
        ok = True

        for mname in mask_names:
            mpath = os.path.join(sdir, mname)
            if not os.path.exists(mpath):
                warn(f"{subj}: missing mask {mname} in {sdir} -> SKIP subject")
                ok = False; break

            try:
                m_img, m = load_nii(mpath)
            except Exception as e:
                warn(f"{subj}: failed to load mask {mname}: {e}")
                ok = False; break

            if m.ndim == 4 and m.shape[-1] == 1:
                m = m[..., 0]
            if m.ndim != 3:
                warn(f"{subj}: mask {mname} not 3D: shape={m.shape}")
                ok = False; break

            # geometry check: must match PET
            if m.shape != pet.shape or (not affine_close(m_img.affine, pet_img.affine, atol=args.atol_affine)):
                warn(f"{subj}: PET/mask geometry mismatch for {mname} -> SKIP subject")
                warn(f"      PET shape={pet.shape} | mask shape={m.shape}")
                ok = False; break

            st = roi_stats(pet, m)
            if st is None:
                warn(f"{subj}: mask {mname} gives EMPTY ROI -> SKIP subject")
                ok = False; break

            # add features in fixed order
            feats += [st["mean"], st["median"], st["p90"], st["p95"], float(st["nvox"])]

        if not ok:
            skipped += 1
            continue

        X_list.append(feats)
        y_list.append(int(y))
        keep_subjects.append(subj)

    if len(X_list) < 40:
        print(f"[FATAL] Too few usable subjects after mask/PET checks: {len(X_list)} (skipped={skipped})", flush=True)
        sys.exit(1)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    info(f"Built features: X={X.shape} | y classes={np.bincount(y, minlength=4)} | skipped={skipped}")

    # CV train/eval
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)
    accs, aucs, cms = [], [], []

    model_name, model = build_model(seed=0, n_classes=4)
    info(f"Model: {model_name}")

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        model_fold_name, model_fold = build_model(seed=fold, n_classes=4)
        model_fold.fit(X[tr], y[tr])

        # probability
        if hasattr(model_fold, "predict_proba"):
            proba = model_fold.predict_proba(X[te])
        else:
            # HistGradientBoosting has predict_proba
            proba = model_fold.predict_proba(X[te])

        pred = np.argmax(proba, axis=1)

        acc = accuracy_score(y[te], pred)
        auc = macro_auc_safe(y[te], proba, n_classes=4)
        cm  = confusion_matrix(y[te], pred, labels=[0,1,2,3])

        accs.append(acc); aucs.append(auc); cms.append(cm)

        info(f"Fold{fold}: ACC={acc:.3f} AUC(macro)={auc:.3f}")
        info(f"Fold{fold} confusion:\n{cm}")

    info(f"== FINAL ({args.folds}-fold CV) ==")
    info(f"ACC mean={np.mean(accs):.3f} std={np.std(accs):.3f}")
    info(f"AUC mean={np.nanmean(aucs):.3f} std={np.nanstd(aucs):.3f}")
    info(f"Confusion sum:\n{np.sum(cms, axis=0)}")

    # optional: dump features for inspection
    feat_cols = []
    for mname in mask_names:
        base = os.path.splitext(os.path.splitext(mname)[0])[0]  # handles .nii.gz
        feat_cols += [f"{base}__mean", f"{base}__median", f"{base}__p90", f"{base}__p95", f"{base}__nvox"]
    out_feat = pd.DataFrame(X, columns=feat_cols)
    out_feat.insert(0, "subject", keep_subjects)
    out_feat.insert(1, "y_braak_class", y)
    feat_path = os.path.join(os.getcwd(), "pet_mask_features.csv")
    out_feat.to_csv(feat_path, index=False)
    info(f"Wrote features table: {feat_path}")

if __name__ == "__main__":
    main()
