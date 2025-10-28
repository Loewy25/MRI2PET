#!/usr/bin/env python3
"""
Create five CSVs (fold1..fold5) with columns: train,validation,test.
Discovery matches your KariAV1451Dataset (T807/t807 dirs, requires T1/PET and mask unless overridden).

This version stratifies by **amyloid status** using Centiloid:
  - Aβ–  if Centiloid < 18.4
  - Aβ+  if Centiloid >= 18.4
(Bucket names are 'negative' and 'positive' for readability.)

For fold f (1..5):
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

# Column names / thresholds kept compatible with your original CLI.
# TAU_COL still provides the key used to match subject folders <-> CSV rows.
TAU_COL     = "TAU_PET_Session"
BRAAK_COLS  = [("Braak1_2","I/II"), ("Braak3_4","III/IV"), ("Braak5_6","V/VI")]  # unused in amyloid split, kept for compatibility
BRAAK_THR   = 18.4  # Centiloid threshold for Aβ+: <18.4 = negative, >=18.4 = positive
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

# Kept from original for compatibility (not used in amyloid split)
def braak_stage_from_row(row: pd.Series,
                         braak_cols: List[Tuple[str, str]],
                         thr: float) -> str:
    # Unchanged legacy helper; unused here.
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
    patterns = [os.path.join(root, "*T807*"), os.path.join(root, "*t807*")]
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
    Return mapping: norm_key(sid) -> one of {"positive","negative"} based on Centiloid.
      - Aβ+ (Centiloid >= braak_thr)  -> "positive"
      - Aβ– (Centiloid <  braak_thr)  -> "negative"

    We keep the signature and downstream usage unchanged; 'braak_cols' is ignored here.
    The subject-row join key remains tau_col (TAU_PET_Session), consistent with your folders.
    """
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Keying column unchanged (matches your T807 folder names)
    tau_c = get_col(df, tau_col)

    # Centiloid column (MR-free CLs recommended)
    cl_c = get_col(df, "Centiloid")
    df[cl_c] = pd.to_numeric(df[cl_c], errors="coerce")

    df["__key__"] = df[tau_c].map(norm_key)
    subject_keys = {norm_key(s) for s in subjects}

    matched = df[df["__key__"].isin(subject_keys)].copy()
    if matched.empty:
        return {}

    # Amyloid labels; NaN CLs fall into "negative" (same behavior as old code treated NaN → "0")
    matched["__stage__"] = np.where(matched[cl_c] >= braak_thr, "positive", "negative")

    # Resolve duplicates per key by taking the "higher" class (positive > negative)
    order = {"negative": 0, "positive": 1}
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

    Uses stage order ["positive","negative"] if present; otherwise falls back to sorted keys.
    """
    rng = random.Random(seed)
    folds = {i: [] for i in range(k)}

    pref_order = ["positive", "negative"]
    extras = [s for s in sorted(sids_by_stage.keys()) if s not in pref_order]
    stage_order = [s for s in pref_order if s in sids_by_stage] + extras

    for stage in stage_order:
        group = list(sids_by_stage.get(stage, []))
        rng.shuffle(group)
        for i, sid in enumerate(group):
            folds[i % k].append(sid)

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
    parser = argparse.ArgumentParser("Make 5 CSVs for 5-fold CV with amyloid-stratified splits (Centiloid).")
    parser.add_argument("--root", default=find_dataset_root(), help="Dataset root with subject folders.")
    parser.add_argument("--meta_csv", default=DEFAULT_META_CSV, help="Metadata CSV path.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=1999, help="Deterministic shuffle seed.")
    parser.add_argument("--val_offset", type=int, default=1, help="Validation = (test_fold + val_offset) % k.")
    parser.add_argument("--allow-missing-mask", action="store_true",
                        help="Include subjects without aseg_brainmask.nii.gz.")
    parser.add_argument("--outdir", default=None, help="Output dir for CSVs (default: <root>/cv_folds).")
    parser.add_argument("--tau_col", default=TAU_COL)
    parser.add_argument("--braak_thr", type=float, default=BRAAK_THR,
                        help="Centiloid threshold for Aβ positivity (default: 18.4).")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    outdir = args.outdir or os.path.join(root, "cv_folds")

    # 1) discover subjects as before
    subjects = list_subject_dirs(root, require_mask=(not args.allow_missing_mask))
    if len(subjects) == 0:
        raise SystemExit(f"No valid subjects found under {root}")

    # 2) compute amyloid labels from metadata CSV using Centiloid only
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
        print(f"[WARN] {len(dropped)} subjects have no amyloid label in CSV and will be excluded from CV.")
        print("       Examples:", dropped[:10])

    # 3) bucket by amyloid status
    sids_by_stage: Dict[str, List[str]] = defaultdict(list)
    for sid in labeled:
        stage = label_map[norm_key(sid)]  # "positive" or "negative"
        sids_by_stage[stage].append(sid)

    # quick summary of stage counts
    stage_names = ["positive", "negative"]
    # ensure both appear for reporting; zeros if absent
    total_counts = {stage: len(sids_by_stage.get(stage, [])) for stage in stage_names}
    print("Amyloid counts among included subjects:", total_counts)

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
            return {k: c.get(k, 0) for k in stage_names}
        print(f"fold{f+1}: Ntrain={len(train_set)} Nval={len(val_list)} Ntest={len(test_list)}")
        print(f"         amyloid dist train={dist(train_set)} val={dist(val_list)} test={dist(test_list)}")

    print(f"\nWrote stratified fold CSVs to: {outdir}")

if __name__ == "__main__":
    main()
