#!/usr/bin/env python3
import os
import pandas as pd

KARI_ROOT = "/scratch/l.peiwang/kari_all"
CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"

TAU_COL = "TAU_PET_Session"
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
lookup = {}
for _, r in df.iterrows():
    key = norm(r[TAU_COL])
    lookup[key] = r

# =========================
# SCAN DATASET
# =========================
folders = [f for f in os.listdir(KARI_ROOT)
           if os.path.isdir(os.path.join(KARI_ROOT, f))]

total = len(folders)
flair_cnt = 0

braak = {"1_2":0, "3_4":0, "5_6":0, "missing":0}
cdr_low = 0
cdr_high = 0
cdr_missing = 0

for f in folders:
    path = os.path.join(KARI_ROOT, f)

    # FLAIR
    if os.path.exists(os.path.join(path, FLAIR_FILE)):
        flair_cnt += 1

    key = norm(f)
    row = lookup.get(key, None)

    if row is None:
        braak["missing"] += 1
        cdr_missing += 1
        continue

    # =========================
    # BRAAK (hierarchical)
    # =========================
    b56 = row.get("Braak5_6", 0)
    b34 = row.get("Braak3_4", 0)
    b12 = row.get("Braak1_2", 0)

    try: b56 = float(b56)
    except: b56 = 0
    try: b34 = float(b34)
    except: b34 = 0
    try: b12 = float(b12)
    except: b12 = 0

    if b56 > 0:
        braak["5_6"] += 1
    elif b34 > 0:
        braak["3_4"] += 1
    elif b12 > 0:
        braak["1_2"] += 1
    else:
        braak["missing"] += 1

    # =========================
    # CDR (threshold 1.2)
    # =========================
    c = row.get("cdr", None)
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
# PRINT
# =========================
print("\n=== BASIC ===")
print(f"Total subjects: {total}")
print(f"With FLAIR: {flair_cnt} ({flair_cnt/total:.2%})")

print("\n=== BRAAK (hierarchical) ===")
for k,v in braak.items():
    print(f"{k}: {v} ({v/total:.2%})")

print("\n=== CDR (threshold=1.2) ===")
print(f"<=1.2: {cdr_low} ({cdr_low/total:.2%})")
print(f">1.2 : {cdr_high} ({cdr_high/total:.2%})")
print(f"missing: {cdr_missing}")
