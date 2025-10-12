#!/usr/bin/env python3
import os
import sys
import pandas as pd

# ==== EDIT THESE ====
CSV_PATH    = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
FOLDER_PATH = "/scratch/l.peiwang/kari_brainv11"
# ====================

def norm(s: str) -> str:
    return str(s).strip().lower()

def find_col(df: pd.DataFrame, name: str) -> str:
    m = {c.strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key not in m:
        raise KeyError(f"Column '{name}' not found. Columns: {list(df.columns)}")
    return m[key]

def main():
    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}"); sys.exit(1)
    if not os.path.isdir(FOLDER_PATH):
        print(f"ERROR: Folder path not found: {FOLDER_PATH}"); sys.exit(1)

    # Read CSV and normalize headers
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    tau_col = find_col(df, "TAU_PET_Session")

    # CSV side: values containing 'av1451' (case-insensitive)
    s = df[tau_col].astype(str).str.strip().str.lower()
    csv_sessions = set(s[s.str.contains("av1451", na=False)])

    # Folder side: names containing 'av1451' (case-insensitive)
    folders = set(
        norm(d) for d in os.listdir(FOLDER_PATH)
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

    # Show a few to debug quickly
    def show(title, items):
        items = sorted(items)
        print(f"\n-- {title} (showing up to 20 of {len(items)}) --")
        for x in items[:20]:
            print("  ", x)

    show("Only in CSV", only_csv)
    show("Only in Folder", only_dir)

if __name__ == "__main__":
    main()




