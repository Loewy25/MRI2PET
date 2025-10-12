#!/usr/bin/env python3
import os
import sys
import pandas as pd

# --------- EDIT THESE ---------
CSV_PATH    = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
FOLDER_PATH = "/scratch/l.peiwang/kari_brainv11"
# ------------------------------

def normalize(s: str) -> str:
    return str(s).strip().lower()

def find_col_case_insensitive(df: pd.DataFrame, target: str) -> str:
    target_l = target.strip().lower()
    mapping = {c.strip().lower(): c for c in df.columns}
    if target_l not in mapping:
        raise KeyError(f"Column '{target}' not found in CSV. Columns present: {list(df.columns)}")
    return mapping[target_l]

def main():
    # basic checks
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not os.path.isdir(FOLDER_PATH):
        print(f"ERROR: Folder path not found: {FOLDER_PATH}")
        sys.exit(1)

    # read CSV (trim header spaces)
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # locate TAU_PET_Session column
    tau_col = find_col_case_insensitive(df, "TAU_PET_Session")

    # CSV: sessions containing "av1451" (case-insensitive)
    series = df[tau_col].astype(str)
    mask = series.str.contains("av1451", case=False, na=False)
    csv_sessions = set(series[mask].map(normalize))

    # Folders: names containing "av1451" (case-insensitive)
    folders = set(
        normalize(d)
        for d in os.listdir(FOLDER_PATH)
        if os.path.isdir(os.path.join(FOLDER_PATH, d)) and ("av1451" in d.lower())
    )

    # Compare
    common   = csv_sessions & folders
    only_csv = csv_sessions - folders
    only_dir = folders - csv_sessions

    # Report
    print(f"CSV (av1451) count     : {len(csv_sessions)}")
    print(f"Folders (av1451) count : {len(folders)}")
    print(f"Common                 : {len(common)}")
    print(f"Only in CSV            : {len(only_csv)}")
    print(f"Only in folder path    : {len(only_dir)}")

    # Show samples to debug quickly
    def preview(name_set, title):
        if not name_set:
            print(f"\n-- {title}: (none)")
            return
        items = sorted(list(name_set))[:20]
        print(f"\n-- {title} (showing up to 20 of {len(name_set)}):")
        for s in items:
            print("  ", s)

    preview(only_csv, "Only in CSV")
    preview(only_dir, "Only in Folder Path")

if __name__ == "__main__":
    main()



