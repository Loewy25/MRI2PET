#!/usr/bin/env python3
"""
Create five CSVs (fold1..fold5) with columns: train,validation,test,label.

This combines the existing logic from:
- cv_creator.py (subject discovery + Braak-stratified 5-fold split creation)
- add_label.py  (append train-only numeric label column to each fold CSV)

Split logic and label logic are intentionally unchanged.
"""

import os
import csv
import argparse
import random
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

# ========= DEFAULTS (edit on CLI if needed) =========
DEFAULT_META_CSV = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
DEFAULT_ROOTS = [
    "/scratch/l.peiwang/kari_brainv33_top300",
    "/scratch/l.peiwang/kari_brain_top300",
    "/scratch/l.peiwang/kari_brainv33",
]

TAU_COL = "TAU_PET_Session"
BRAAK_COLS = [("Braak1_2", "I/II"), ("Braak3_4", "III/IV"), ("Braak5_6", "V/VI")]
BRAAK_THR = 1.2
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


def braak_stage_from_row(
    row: pd.Series, braak_cols: List[Tuple[str, str]], thr: float
) -> str:
    # Unchanged: pick highest stage meeting threshold
    vals = {label: to_float(row[col]) for col, label in braak_cols}
    if vals["V/VI"] >= thr:
        return "V/VI"
    if vals["III/IV"] >= thr:
        return "III/IV"
    if vals["I/II"] >= thr:
        return "I/II"
    return "0"


def find_dataset_root() -> str:
    for p in DEFAULT_ROOTS:
        if os.path.isdir(p):
            return p
    return DEFAULT_ROOTS[0]


def list_subject_dirs(root: str, require_mask: bool = True) -> List[str]:
    """
    Scan all folders under root (no naming constraints).
    Keep folders that contain:
      - T1_masked.nii.gz
      - PET_in_T1_masked.nii.gz
      - aseg_brainmask.nii.gz (if require_mask=True)
    """
    sids = []
    seen = set()

    for dirpath, _, filenames in os.walk(root):
        files = set(filenames)
        has_t1 = "T1_masked.nii.gz" in files
        has_pet = "PET_in_T1_masked.nii.gz" in files
        has_mask = "aseg_brainmask.nii.gz" in files

        if has_t1 and has_pet and ((not require_mask) or has_mask):
            sid = os.path.basename(dirpath.rstrip(os.sep))
            if sid and sid not in seen:
                sids.append(sid)
                seen.add(sid)

    return sorted(sids)


def load_stage_labels(
    meta_csv: str,
    subjects: List[str],
    tau_col: str,
    braak_cols: List[Tuple[str, str]],
    braak_thr: float,
) -> Dict[str, str]:
    """
    Return mapping: norm_key(sid) -> stage in {"V/VI","III/IV","I/II","0"}.
    If multiple CSV rows map to the same session key, choose the highest stage.
    """
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    tau_c = get_col(df, tau_col)

    df = df.copy()
    for col, _ in braak_cols:
        real = get_col(df, col)
        if real != col:
            df.rename(columns={real: col}, inplace=True)

    for col, _ in braak_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["__key__"] = df[tau_c].map(norm_key)
    subject_keys = {norm_key(s) for s in subjects}

    matched = df[df["__key__"].isin(subject_keys)].copy()
    if matched.empty:
        return {}

    matched["__stage__"] = matched.apply(
        lambda r: braak_stage_from_row(r, braak_cols, braak_thr), axis=1
    )

    order = {"0": 0, "I/II": 1, "III/IV": 2, "V/VI": 3}
    by_key = matched.groupby("__key__")["__stage__"].apply(
        lambda s: max(s, key=lambda x: order.get(x, -1))
    )

    return dict(by_key)


def stratified_assign(
    sids_by_stage: Dict[str, List[str]], k: int, seed: int
) -> Dict[int, List[str]]:
    """
    Per-stage round-robin into k folds after deterministic shuffle.
    Returns: fold_idx -> list of sids (test set for that fold).
    """
    rng = random.Random(seed)
    folds = {i: [] for i in range(k)}
    for stage in ["V/VI", "III/IV", "I/II", "0"]:
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
            w.writerow(
                [
                    train[i] if i < len(train) else "",
                    val[i] if i < len(val) else "",
                    test[i] if i < len(test) else "",
                ]
            )


def build_sid_to_label(meta_csv: str, sids: List[str]) -> Dict[str, int]:
    """
    Build mapping from subject folder name -> integer label 0/1/2/3.
    """
    df = pd.read_csv(meta_csv)
    df.columns = [c.strip() for c in df.columns]

    for col in [TAU_COL, "Braak1_2", "Braak3_4", "Braak5_6"]:
        if col not in df.columns:
            raise RuntimeError(f"Metadata CSV missing required column: {col}")

    df["_key"] = df[TAU_COL].apply(norm_key)

    for col in ["Braak1_2", "Braak3_4", "Braak5_6"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sid_to_label: Dict[str, int] = {}
    for sid in sorted(set(sids)):
        if not sid:
            continue
        key = norm_key(sid)
        rows = df[df["_key"] == key]
        if rows.empty:
            raise RuntimeError(f"No metadata row found for subject '{sid}' (key {key})")

        labels = []
        for _, row in rows.iterrows():
            b12 = row["Braak1_2"]
            b34 = row["Braak3_4"]
            b56 = row["Braak5_6"]

            if np.isnan(b12) and np.isnan(b34) and np.isnan(b56):
                labels.append(0)
                continue

            if b56 >= BRAAK_THR:
                labels.append(3)
            elif b34 >= BRAAK_THR:
                labels.append(2)
            elif b12 >= BRAAK_THR:
                labels.append(1)
            else:
                labels.append(0)

        sid_to_label[sid] = max(labels)

    return sid_to_label


def process_one_fold(path: str, sid_to_label: Dict[str, int]):
    df = pd.read_csv(path)

    if "train" not in df.columns:
        raise RuntimeError(f"'train' column not found in {path}")

    labels = []
    for sid in df["train"]:
        if isinstance(sid, float) and np.isnan(sid):
            labels.append("")
        elif sid == "" or sid is None:
            labels.append("")
        else:
            sid_str = str(sid).strip()
            if sid_str not in sid_to_label:
                raise RuntimeError(f"No label found for train subject '{sid_str}' in {path}")
            labels.append(sid_to_label[sid_str])

    df["label"] = labels
    df.to_csv(path, index=False)


def add_labels_to_folds(folds_dir: str, meta_csv: str):
    fold_files = [
        f
        for f in os.listdir(folds_dir)
        if f.lower().startswith("fold") and f.lower().endswith(".csv")
    ]
    if not fold_files:
        raise RuntimeError(f"No fold*.csv files found in {folds_dir}")

    all_train_sids: List[str] = []
    for fname in fold_files:
        df_fold = pd.read_csv(os.path.join(folds_dir, fname))
        if "train" not in df_fold.columns:
            raise RuntimeError(f"'train' column not found in {fname}")
        all_train_sids.extend([str(x).strip() for x in df_fold["train"] if str(x).strip()])

    sid_to_label = build_sid_to_label(meta_csv, all_train_sids)

    for fname in sorted(fold_files):
        process_one_fold(os.path.join(folds_dir, fname), sid_to_label)


def main():
    parser = argparse.ArgumentParser(
        "Make 5 CSVs for 5-fold CV with Braak-stratified splits and train label column."
    )
    parser.add_argument(
        "--root", default=find_dataset_root(), help="Dataset root with subject folders."
    )
    parser.add_argument(
        "--meta_csv", default=DEFAULT_META_CSV, help="Metadata CSV path."
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=1999, help="Deterministic shuffle seed.")
    parser.add_argument(
        "--val_offset", type=int, default=1, help="Validation = (test_fold + val_offset) % k."
    )
    parser.add_argument(
        "--allow-missing-mask",
        action="store_true",
        help="Include subjects without aseg_brainmask.nii.gz.",
    )
    parser.add_argument(
        "--outdir", default=None, help="Output dir for CSVs (default: <root>/cv_folds)."
    )
    parser.add_argument("--tau_col", default=TAU_COL)
    parser.add_argument("--braak_thr", type=float, default=BRAAK_THR)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    outdir = args.outdir or os.path.join(root, "cv_folds")

    subjects = list_subject_dirs(root, require_mask=(not args.allow_missing_mask))
    if len(subjects) == 0:
        raise SystemExit(f"No valid subjects found under {root}")

    label_map = load_stage_labels(
        args.meta_csv,
        subjects,
        tau_col=args.tau_col,
        braak_cols=BRAAK_COLS,
        braak_thr=args.braak_thr,
    )

    labeled = [sid for sid in subjects if norm_key(sid) in label_map]
    dropped = [sid for sid in subjects if norm_key(sid) not in label_map]
    if len(labeled) == 0:
        raise SystemExit(
            "No subjects matched between folder names and TAU_PET_Session in the CSV."
        )
    if dropped:
        print(
            f"[WARN] {len(dropped)} subjects have no Braak label in CSV and will be excluded from CV."
        )
        print("       Examples:", dropped[:10])

    sids_by_stage: Dict[str, List[str]] = defaultdict(list)
    for sid in labeled:
        stage = label_map[norm_key(sid)]
        sids_by_stage[stage].append(sid)

    total_counts = {
        stage: len(sids_by_stage.get(stage, [])) for stage in ["V/VI", "III/IV", "I/II", "0"]
    }
    print("Stage counts among included subjects:", total_counts)

    k = args.k
    test_folds = stratified_assign(sids_by_stage, k=k, seed=args.seed)

    for f in range(k):
        test_set = set(test_folds[f])
        val_set = set(test_folds[(f + args.val_offset) % k])
        train_set = sorted([sid for sid in labeled if sid not in test_set | val_set])
        val_list = sorted(list(val_set))
        test_list = sorted(list(test_set))

        out_csv = os.path.join(outdir, f"fold{f + 1}.csv")
        write_fold_csv(out_csv, train_set, val_list, test_list)

        def dist(lst):
            c = Counter(label_map[norm_key(s)] for s in lst)
            return {kk: c.get(kk, 0) for kk in ["V/VI", "III/IV", "I/II", "0"]}

        print(f"fold{f + 1}: Ntrain={len(train_set)} Nval={len(val_list)} Ntest={len(test_list)}")
        print(f"         stage dist train={dist(train_set)} val={dist(val_list)} test={dist(test_list)}")

    print(f"\nWrote stratified fold CSVs to: {outdir}")

    add_labels_to_folds(outdir, args.meta_csv)
    print("Done: each foldX.csv now has a 'label' column for the train subject.")


if __name__ == "__main__":
    main()
