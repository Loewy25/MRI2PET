#!/usr/bin/env python3
import os
import pandas as pd

KARI_ROOT = "/scratch/l.peiwang/kari_all"
CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"

TAU_COL = "TAU_PET_Session"
FLAIR_FILE = "FLAIR_in_T1.nii.gz"
BRAAK_THRESHOLD = 1.2

def norm(s):
    return str(s).strip().lower().replace("-", "_")

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())

lookup = {}
for _, r in df.iterrows():
    lookup[norm(r[TAU_COL])] = r

# =========================
# SCAN DATASET
# =========================
folders = [f for f in os.listdir(KARI_ROOT)
           if os.path.isdir(os.path.join(KARI_ROOT, f))]

total = len(folders)
flair_cnt = 0

braak = {"1_2": 0, "3_4": 0, "5_6": 0, "none": 0, "missing": 0}
cdr = {"0": 0, "0.5": 0, "1": 0, "other": 0, "missing": 0}

for f in folders:
    path = os.path.join(KARI_ROOT, f)

    if os.path.exists(os.path.join(path, FLAIR_FILE)):
        flair_cnt += 1

    row = lookup.get(norm(f), None)
    if row is None:
        braak["missing"] += 1
        cdr["missing"] += 1
        continue

    # -------------------------
    # BRAAK: hierarchical, threshold = 1.2
    # -------------------------
    try:
        b12 = float(row["Braak1_2"])
    except:
        b12 = None
    try:
        b34 = float(row["Braak3_4"])
    except:
        b34 = None
    try:
        b56 = float(row["Braak5_6"])
    except:
        b56 = None

    if b12 is None and b34 is None and b56 is None:
        braak["missing"] += 1
    elif b56 is not None and b56 > BRAAK_THRESHOLD:
        braak["5_6"] += 1
    elif b34 is not None and b34 > BRAAK_THRESHOLD:
        braak["3_4"] += 1
    elif b12 is not None and b12 > BRAAK_THRESHOLD:
        braak["1_2"] += 1
    else:
        braak["none"] += 1

    # -------------------------
    # CDR: exact groups 0 / 0.5 / 1
    # -------------------------
    try:
        c = float(row["cdr"])
    except:
        c = None

    if c is None:
        cdr["missing"] += 1
    elif c == 0:
        cdr["0"] += 1
    elif c == 0.5:
        cdr["0.5"] += 1
    elif c == 1:
        cdr["1"] += 1
    else:
        cdr["other"] += 1

# =========================
# PRINT
# =========================
print("\n=== BASIC ===")
print(f"Total subjects: {total}")
print(f"With FLAIR: {flair_cnt} ({flair_cnt/total:.2%})")

print(f"\n=== BRAAK (hierarchical, threshold={BRAAK_THRESHOLD}) ===")
for k, v in braak.items():
    print(f"{k}: {v} ({v/total:.2%})")

print("\n=== CDR ===")
for k, v in cdr.items():
    print(f"{k}: {v} ({v/total:.2%})")
