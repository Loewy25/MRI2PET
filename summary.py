#!/usr/bin/env python3
import os
import pandas as pd

# ====== edit these ======
CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
FOLDER_PATH = "/scratch/l.peiwang/kari_brainv11"
# ========================

# load csv and find TAU_PET_Session column (case-insensitive)
df = pd.read_csv(CSV_PATH)
col = [c for c in df.columns if c.strip().lower() == "tau_pet_session"][0]

# filter rows whose TAU_PET_Session contains "av1451" (case-insensitive)
csv_sessions = set(
    s.strip().lower()
    for s in df[col].dropna().astype(str)
    if "av1451" in s.lower()
)

# list folders that contain "av1451" (case-insensitive)
folders = set(
    d.lower()
    for d in os.listdir(FOLDER_PATH)
    if os.path.isdir(os.path.join(FOLDER_PATH, d)) and "av1451" in d.lower()
)

# compare
common   = csv_sessions & folders
only_csv = csv_sessions - folders
only_dir = folders - csv_sessions

# summary
print(f"Total CSV sessions (av1451): {len(csv_sessions)}")
print(f"Total folders (av1451):     {len(folders)}")
print(f"Common:                     {len(common)}")
print(f"Only in CSV:                {len(only_csv)}")
print(f"Only in folder path:        {len(only_dir)}")


