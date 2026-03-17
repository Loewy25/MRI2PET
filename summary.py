#!/usr/bin/env python3
import os
import pandas as pd

KARI_ROOT = "/scratch/l.peiwang/kari_all"
CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"

TAU_COL = "TAU_PET_Session"
BRAAK_COL = "Braak5_6"
CDR_COL = "cdr"   # change if your column name is different
FLAIR_FILE = "FLAIR_in_T1.nii.gz"

def norm(s):
    return str(s).strip().lower().replace("-", "_")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

# build lookup
braak_map = {}
cdr_map = {}

for _, r in df.iterrows():
    key = norm(r[TAU_COL])
    braak_map[key] = r.get(BRAAK_COL, None)
    cdr_map[key] = r.get(CDR_COL, None)

# =========================
# SCAN DATASET
# =========================
folders = [f for f in os.listdir(KARI_ROOT)
           if os.path.isdir(os.path.join(KARI_ROOT, f))]

total = len(folders)
flair_cnt = 0

braak_counts = {"1_2":0, "3_4":0, "5_6":0, "missing":0}
cdr_low = 0
cdr_high = 0
cdr_missing = 0

for f in folders:
    path = os.path.join(KARI_ROOT, f)

    # FLAIR
    if os.path.exists(os.path.join(path, FLAIR_FILE)):
        flair_cnt += 1

    key = norm(f)

    # Braak
    b = braak_map.get(key, None)
    try:
        b = float(b)
    except:
        b = None

    if b is None:
        braak_counts["missing"] += 1
    elif b <= 2:
        braak_counts["1_2"] += 1
    elif b <= 4:
        braak_counts["3_4"] += 1
    else:
        braak_counts["5_6"] += 1

    # CDR
    c = cdr_map.get(key, None)
    try:
        c = float(c)
    except:
        c = None

    if c is None:
        cdr_missing += 1
    elif c <= 1.2:
        cdr_low += 1
    else:
        cdr_high += 1

# =========================
# PRINT SUMMARY
# =========================
print("\n=== BASIC ===")
print(f"Total subjects: {total}")
print(f"With FLAIR: {flair_cnt} ({flair_cnt/total:.2%})")

print("\n=== BRAAK ===")
for k,v in braak_counts.items():
    print(f"{k}: {v} ({v/total:.2%})")

print("\n=== CDR (threshold=1.2) ===")
print(f"<=1.2: {cdr_low} ({cdr_low/total:.2%})")
print(f">1.2 : {cdr_high} ({cdr_high/total:.2%})")
print(f"missing: {cdr_missing}")

