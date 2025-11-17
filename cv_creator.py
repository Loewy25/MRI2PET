#!/usr/bin/env python3
"""
Create five CSVs (fold1..fold5) with columns: train,validation,test.
Discovery matches your KariAV1451Dataset (T807/t807 dirs, requires T1/PET and mask unless overridden).
Splits are *stratified by Braak stage* computed from your metadata CSV:

- Stage logic (unchanged): highest positive stage at threshold BRAAK_THR
    V/VI >= thr → "V/VI"
    else III/IV >= thr → "III/IV"
    else I/II >= thr → "I/II"
    else "0"

- For fold f (1..5):
    test = subjects assigned to fold f
    val  = subjects assigned to (f+1) mod 5
    train = remaining subjects

Result files:
  <root>/cv_folds/fold1.csv ... fold5.csv
Each CSV: columns [train, validation, test], one subject per cell, rows padded with blanks.
"""

import os, glob, csv, argparse, random
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# ========= DEFAULTS (edit on CLI if needed) =========
DEFAULT_META_CSV = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
# Prefer top300 if present; fall back gracefully.
DEFAULT_ROOTS = [
    "/scratch/l.peiwang/kari_brainv33_top300",
    "/scratch/l.peiwang/kari_brain_top300",
    "/scratch/l.peiwang/kari_brainv33",
]

# Column names / thresholds exactly per your logic
TAU_COL     = "TAU_PET_Session"
BRAAK_COLS  = [("Braak1_2","I/II"), ("Braak3_4","III/IV"), ("Braak5_6","V/VI")]
BRAAK_THR   = 1.2
# ====================================================

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def get_col(df: pd.DataFrame, name: str) -> str:
    mapping = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key not in mapping:
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")
    return mapping[key]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def braak_stage_from_row(row: pd.Series,
                         braak_cols: List[Tuple[str, str]],
                         thr: float) -> str:
    # Unchanged: pick highest stage meeting threshold
    vals = {label: to_float(row[col]) for col, label in braak_cols}
    if vals["V/VI"]   >= thr: return "V/VI"
    if vals["III/IV"] >= thr: return "III/IV"
    if vals["I/II"]   >= thr: return "I/II"
    return "0"

def find_dataset_root() -> str:
    for p in DEFAULT_ROOTS:
        if os.path.isdir(p):
            return p
    # fallback to first in list
    return DEFAULT_ROOTS[0]

def list_subject_dirs(root: str,
                      require_mask: bool = True) -> List[str]:
    patterns = [os.path.join(root, "*T807*"), os.path.join(root, "**")]
    cand_dirs = []
    for p in patterns:
        cand_dirs.extend(glob.glob(p))
    cand_dirs = sorted([d for d in cand_dirs if os.path.isdir(d)])

    sids = []
    for d in cand_dirs:
        t1  = os.path.join(d, "T1_masked.nii.gz")
        pet = os.path.join(d, "PET_in_T1_masked.nii.gz")
        msk = os.path.join(d, "aseg_brainmask.nii.gz")
        if os.path.exists(t1) and os.path.exists(pet):
            if (not require_mask) or os.path.exists(msk):
                sids.append(os.path.basename(d))
    return sids

def load_stage_labels(meta_csv: str,
                      subjects: List[str],
                      tau_col: str,
                      braak_cols: List[Tuple[str, str]],
                      braak_thr: float) -> Dict[str, str]:
    """
    Return mapping: norm_key(sid) -> stage in {"V/VI","III/IV","I/II","0"}.
    If multiple CSV rows map to the same session key, choose the HIGHEST stage
    (consistent with your precedence rule).
    """
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    tau_c = get_col(df, tau_col)

    # Ensure Braak columns exist; if they appear with different capitalization/spacing, normalize
    df = df.copy()
    for col, _ in braak_cols:
        real = get_col(df, col)
        if real != col:
            df.rename(columns={real: col}, inplace=True)

    # Coerce to numeric for braak columns
    for col, _ in braak_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["__key__"] = df[tau_c].map(norm_key)
    subject_keys = {norm_key(s) for s in subjects}

    matched = df[df["__key__"].isin(subject_keys)].copy()
    if matched.empty:
        return {}

    # Compute stage per row with your exact logic
    matched["__stage__"] = matched.apply(
        lambda r: braak_stage_from_row(r, braak_cols, braak_thr), axis=1
    )

    # Resolve duplicates: keep the highest stage per key
    order = {"0": 0, "I/II": 1, "III/IV": 2, "V/VI": 3}
    by_key = matched.groupby("__key__")["__stage__"].apply(
        lambda s: max(s, key=lambda x: order.get(x, -1))
    )

    return dict(by_key)

def stratified_assign(sids_by_stage: Dict[str, List[str]],
                      k: int,
                      seed: int) -> Dict[int, List[str]]:
    """
    Per-stage round-robin into k folds after deterministic shuffle.
    Returns: fold_idx -> list of sids (test set for that fold).
    """
    rng = random.Random(seed)
    folds = {i: [] for i in range(k)}
    # deterministic but balanced per stage
    for stage in ["V/VI", "III/IV", "I/II", "0"]:
        group = list(sids_by_stage.get(stage, []))
        rng.shuffle(group)
        for i, sid in enumerate(group):
            folds[i % k].append(sid)
    # sort each fold for readability
    for i in range(k):
        folds[i] = sorted(folds[i])
    return folds

def write_fold_csv(out_csv: str, train: List[str], val: List[str], test: List[str]):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    L = max(len(train), len(val), len(test))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train", "validation", "test"])
        for i in range(L):
            w.writerow([
                train[i] if i < len(train) else "",
                val[i]   if i < len(val)   else "",
                test[i]  if i < len(test)  else "",
            ])

def main():
    parser = argparse.ArgumentParser("Make 5 CSVs for 5-fold CV with Braak-stratified splits.")
    parser.add_argument("--root", default=find_dataset_root(), help="Dataset root with subject folders.")
    parser.add_argument("--meta_csv", default=DEFAULT_META_CSV, help="Metadata CSV path.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=1999, help="Deterministic shuffle seed.")
    parser.add_argument("--val_offset", type=int, default=1, help="Validation = (test_fold + val_offset) % k.")
    parser.add_argument("--allow-missing-mask", action="store_true",
                        help="Include subjects without aseg_brainmask.nii.gz.")
    parser.add_argument("--outdir", default=None, help="Output dir for CSVs (default: <root>/cv_folds).")
    parser.add_argument("--tau_col", default=TAU_COL)
    parser.add_argument("--braak_thr", type=float, default=BRAAK_THR)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    outdir = args.outdir or os.path.join(root, "cv_folds")

    # 1) discover subjects as before
    subjects = list_subject_dirs(root, require_mask=(not args.allow_missing_mask))
    if len(subjects) == 0:
        raise SystemExit(f"No valid subjects found under {root}")

    # 2) compute stage labels from metadata CSV using your exact logic
    label_map = load_stage_labels(
        args.meta_csv, subjects, tau_col=args.tau_col,
        braak_cols=BRAAK_COLS, braak_thr=args.braak_thr
    )

    # Keep only labeled subjects (we can't stratify unlabeled); warn if any dropped
    labeled = [sid for sid in subjects if norm_key(sid) in label_map]
    dropped = [sid for sid in subjects if norm_key(sid) not in label_map]
    if len(labeled) == 0:
        raise SystemExit("No subjects matched between folder names and TAU_PET_Session in the CSV.")
    if dropped:
        print(f"[WARN] {len(dropped)} subjects have no Braak label in CSV and will be excluded from CV.")
        # print a few for QC
        print("       Examples:", dropped[:10])

    # 3) bucket by stage
    sids_by_stage: Dict[str, List[str]] = defaultdict(list)
    for sid in labeled:
        stage = label_map[norm_key(sid)]  # one of {"V/VI","III/IV","I/II","0"}
        sids_by_stage[stage].append(sid)

    # quick summary of stage counts
    total_counts = {stage: len(sids_by_stage.get(stage, [])) for stage in ["V/VI","III/IV","I/II","0"]}
    print("Stage counts among included subjects:", total_counts)

    # 4) stratified per-stage round-robin → test folds
    k = args.k
    test_folds = stratified_assign(sids_by_stage, k=k, seed=args.seed)

    # 5) write per-fold CSVs; val = next fold; train = everything else
    for f in range(k):
        test_set = set(test_folds[f])
        val_set  = set(test_folds[(f + args.val_offset) % k])
        train_set = sorted([sid for sid in labeled if sid not in test_set | val_set])
        val_list  = sorted(list(val_set))
        test_list = sorted(list(test_set))

        out_csv = os.path.join(outdir, f"fold{f+1}.csv")  # 1-based file names
        write_fold_csv(out_csv, train_set, val_list, test_list)

        # sanity print: distribution per split
        def dist(lst):
            c = Counter(label_map[norm_key(s)] for s in lst)
            return {k: c.get(k, 0) for k in ["V/VI","III/IV","I/II","0"]}
        print(f"fold{f+1}: Ntrain={len(train_set)} Nval={len(val_list)} Ntest={len(test_list)}")
        print(f"         stage dist train={dist(train_set)} val={dist(val_list)} test={dist(test_list)}")

    print(f"\nWrote stratified fold CSVs to: {outdir}")

if __name__ == "__main__":
    main()
