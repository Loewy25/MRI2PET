import os
import pandas as pd

# ======= set your paths here =======
CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
FOLDER_PATH = "/scratch/l.peiwang/kari_brainv11"
# ===================================

df = pd.read_csv(CSV_PATH)

# find the TAU_PET_Session column (case-insensitive)
col = next((c for c in df.columns if c.strip().lower() == "tau_pet_session"), None)
if not col:
    raise SystemExit("❌ No 'TAU_PET_Session' column found.")

csv_sessions = {
    str(v).strip().lower()
    for v in df[col].astype(str)
    if "av1451" in str(v).lower()
}

dir_sessions = {
    d.name.strip().lower()
    for d in os.scandir(FOLDER_PATH)
    if d.is_dir() and "av1451" in d.name.lower()
}

common = csv_sessions & dir_sessions
only_csv = csv_sessions - dir_sessions
only_dir = dir_sessions - csv_sessions

print(f"CSV (av1451) count : {len(csv_sessions)}")
print(f"Folders (av1451) count : {len(dir_sessions)}")
print(f"Common : {len(common)}")
print(f"Only in CSV : {len(only_csv)}")
print(f"Only in folder path : {len(only_dir)}")

if only_csv:
    print("\n-- Only in CSV (up to 20) --")
    print("\n".join(list(only_csv)[:20]))
if only_dir:
    print("\n-- Only in Folder (up to 20) --")
    print("\n".join(list(only_dir)[:20]))



