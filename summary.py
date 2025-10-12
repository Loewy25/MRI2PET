#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from collections import Counter

# ========= EDIT THESE PATHS =========
CSV_PATH   = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
TRAIN_DIR  = "/scratch/l.peiwang/kari_brainv11"        # your dataset folder
TEST_DIR   = "/home/l.peiwang/MRI2PET/MGDA_UB_c_stable_contra_2432_batch1_hierachy_ROI_NOMEMORY_nomultiview/volumes"   # your testing dataset folder (or None)
# ========= HYPERPARAMETERS =========
TAU_COL       = "TAU_PET_Session"  # column used to match folder names
AMYLOID_COL   = "Centiloid"        # could also be "TSS" etc.
AMYLOID_THR   = 20.0               # amyloid+ if value >= this (change to your rule)
BRAAK_COLS    = [("Braak1_2","I/II"), ("Braak3_4","III/IV"), ("Braak5_6","V/VI")]
BRAAK_THR     = 1.2                # stage is positive if >= this (adjust as needed)
CDR_COL       = "cdr"
# ===================================

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def list_subjects(root):
    if not root or not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))])

def get_col(df, name):
    # case-insensitive lookup with whitespace tolerance
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

def braak_stage_from_row(row, braak_cols, thr):
    # assign the highest positive stage among V/VI -> III/IV -> I/II; else '0'
    vals = {label: to_float(row[col]) for col, label in braak_cols}
    if vals["V/VI"] >= thr:   return "V/VI"
    if vals["III/IV"] >= thr: return "III/IV"
    if vals["I/II"] >= thr:   return "I/II"
    return "0"

def summarize_folder(df, folder_path):
    subjects = list_subjects(folder_path)
    if not subjects:
        return {
            "folder": folder_path, "n_folders": 0, "n_matched": 0,
            "unmatched": [], "amyloid": {}, "braak": {}, "cdr": {}
        }

    # resolve column names in a tolerant way
    tau_c  = get_col(df, TAU_COL)
    amy_c  = get_col(df, AMYLOID_COL)
    cdr_c  = get_col(df, CDR_COL)
    # ensure Braak cols exist and are numeric-friendly
    for col, _ in BRAAK_COLS:
        bc = get_col(df, col)
        if bc != col:
            df.rename(columns={bc: col}, inplace=True)

    # build a fast lookup for TAU_PET_Session (case-insensitive)
    df = df.copy()
    df["__key__"] = df[tau_c].map(norm_key)
    subject_keys = {norm_key(s) for s in subjects}
    matched = df[df["__key__"].isin(subject_keys)]

    # unmatched folder names for QC
    matched_keys = set(matched["__key__"])
    unmatched = sorted([s for s in subjects if norm_key(s) not in matched_keys])

    # amyloid counts
    amy_vals = pd.to_numeric(matched[amy_c], errors="coerce")
    amy_pos = (amy_vals >= AMYLOID_THR)
    amy_counts = {"positive": int(amy_pos.sum()),
                  "negative": int((~amy_pos).sum())}

    # Braak staging
    # coerce to numeric for each needed Braak column
    for col, _ in BRAAK_COLS:
        matched[col] = pd.to_numeric(matched[col], errors="coerce")
    stages = matched.apply(lambda r: braak_stage_from_row(r, BRAAK_COLS, BRAAK_THR), axis=1)
    braak_counts = dict(Counter(stages))

    # CDR distribution
    cdr_counts = dict(matched[cdr_c].value_counts(dropna=False).sort_index())

    return {
        "folder": folder_path,
        "n_folders": len(subjects),
        "n_matched": int(len(matched)),
        "unmatched": unmatched,
        "amyloid": amy_counts,
        "braak": braak_counts,
        "cdr": cdr_counts
    }

def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]  # trim spaces in headers

    results = []
    for path in [TRAIN_DIR, TEST_DIR]:
        if path:
            results.append(summarize_folder(df, path))

    # pretty print
    for r in results:
        print("\n=== SUMMARY:", r["folder"], "===")
        print(f"Subject folders        : {r['n_folders']}")
        print(f"Rows matched in CSV    : {r['n_matched']}")
        print(f"Amyloid ({AMYLOID_COL} >= {AMYLOID_THR}): {r['amyloid']}")
        print(f"Braak (thr={BRAAK_THR})               : {r['braak']}")
        print(f"CDR counts                             : {r['cdr']}")
        if r["unmatched"]:
            print(f"Unmatched folder names ({len(r['unmatched'])}): {r['unmatched']}")

if __name__ == "__main__":
    main()
