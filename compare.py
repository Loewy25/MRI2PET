#!/usr/bin/env python3
import os
import sys
import pandas as pd
from typing import Optional, Set, Tuple

# ==== EDIT THESE ====
CSV_PATH    = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
FOLDER_PATH = "/ceph/chpc/mapped/benz04_kari/pup"
OUT_DIR     = "/scratch/l.peiwang"
# ====================

def norm(s: str) -> str:
    return str(s).strip().lower()

def find_col(df: pd.DataFrame, name: str) -> str:
    m = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key not in m:
        raise KeyError(f"Column '{name}' not found. Columns: {list(df.columns)}")
    return m[key]

def csv_sessions(df: pd.DataFrame, tau_col: str, keyword: Optional[str]) -> Set[str]:
    """
    Return a set of normalized session names from the CSV TAU_PET_Session column.
    If keyword is provided (e.g., 'av1451' or 't807'), filter rows to those containing it (case-insensitive).
    If keyword is None, return ALL non-empty sessions.
    """
    col = df[tau_col]
    # Drop NaNs before casting to str to avoid 'nan' string leaking into results
    col = col.dropna().astype(str).str.strip()
    if keyword:
        mask = col.str.contains(keyword, case=False, na=False)
        col = col[mask]
    # Remove empty strings after filtering (just in case)
    col = col[col.str.len() > 0]
    return set(col.str.lower())

def folder_sessions(folder_path: str, keyword: Optional[str]) -> Set[str]:
    """
    Return a set of normalized subdirectory names under folder_path.
    If keyword is provided, keep only directories whose names contain it (case-insensitive).
    If keyword is None, return ALL subdirectories.
    """
    try:
        entries = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"ERROR: Folder path not found: {folder_path}")
        sys.exit(1)

    result = set()
    for d in entries:
        full = os.path.join(folder_path, d)
        if os.path.isdir(full):
            d_norm = d.lower()
            if (keyword is None) or (keyword.lower() in d_norm):
                result.add(d_norm.strip())
    return result

def compare_sets(csv_set: Set[str], dir_set: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    common = csv_set & dir_set
    only_csv = csv_set - dir_set
    only_dir = dir_set - csv_set
    return common, only_csv, only_dir

def write_report(out_path: str, label: str,
                 only_csv: Set[str], only_dir: Set[str],
                 csv_count: int, dir_count: int, common_count: int) -> None:
    """
    Write a TXT file with just the two mismatch lists.
    Include counts at the top for quick reference.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Comparison: {label}\n")
        f.write(f"CSV count: {csv_count}\n")
        f.write(f"Folder count: {dir_count}\n")
        f.write(f"Common: {common_count}\n")
        f.write("\n")

        f.write(f"-- Only in CSV (n={len(only_csv)}) --\n")
        for x in sorted(only_csv):
            f.write(f"{x}\n")

        f.write("\n")
        f.write(f"-- Only in folder path (n={len(only_dir)}) --\n")
        for x in sorted(only_dir):
            f.write(f"{x}\n")

def run_one(df: pd.DataFrame, tau_col: str, label: str, keyword: Optional[str]) -> None:
    csv_set = csv_sessions(df, tau_col, keyword)
    dir_set = folder_sessions(FOLDER_PATH, keyword)
    common, only_csv, only_dir = compare_sets(csv_set, dir_set)

    out_file = os.path.join(OUT_DIR, f"tau_pet_compare_{label}.txt")
    write_report(out_file, label, only_csv, only_dir,
                 csv_count=len(csv_set),
                 dir_count=len(dir_set),
                 common_count=len(common))

    # Also print a brief console summary + where to find the file
    print(f"[{label}] CSV count={len(csv_set)} | Folders count={len(dir_set)} | Common={len(common)}")
    print(f"[{label}] Only in CSV: {len(only_csv)} | Only in folder: {len(only_dir)}")
    print(f"[{label}] Report written: {out_file}\n")

def main():
    # Basic path checks
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not os.path.isdir(FOLDER_PATH):
        print(f"ERROR: Folder path not found: {FOLDER_PATH}")
        sys.exit(1)

    # Load CSV and normalize headers
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    tau_col = find_col(df, "TAU_PET_Session")

    # Run the three comparisons you requested
    run_one(df, tau_col, label="av1451", keyword="av1451")
    run_one(df, tau_col, label="t807",   keyword="t807")
    run_one(df, tau_col, label="all",    keyword=None)

if __name__ == "__main__":
    main()

