#!/usr/bin/env python3
import os
import re
import shutil
from collections import defaultdict

import pandas as pd

CSV_PATH = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"

OLD_ROOT = "/scratch/l.peiwang/kari_all"
ARCHIVE_ROOT = "/scratch/l.peiwang/kari_all_non_selected"
NEW_ROOT = "/scratch/l.peiwang/kari_all"

TAU_COL = "TAU_PET_Session"
BRAAK_COL = "Braak5_6"
FLAIR_FILE = "FLAIR_in_T1.nii.gz"

SELECTED_MANIFEST = "/scratch/l.peiwang/kari_all_selected_manifest.csv"
ALL_MANIFEST = "/scratch/l.peiwang/kari_all_all_visits_manifest.csv"


def norm_name(s):
    s = str(s).strip().lower()
    s = s.replace(".zip", "")
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "", s)
    return s


def parse_braak(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    return float(m.group(0))


def get_subject_id(folder):
    parts = folder.split("_")
    # Important fix for names like 1092_001_AV1451_v1
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[0] + "_" + parts[1]
    return parts[0]


print("\n=== Load CSV ===")
df = pd.read_csv(CSV_PATH)
print("Columns in CSV:")
print(list(df.columns))

if TAU_COL not in df.columns:
    raise ValueError(f"Missing column: {TAU_COL}")
if BRAAK_COL not in df.columns:
    raise ValueError(f"Missing column: {BRAAK_COL}")

# session -> max Braak5_6
session_to_braak = {}
for _, row in df[[TAU_COL, BRAAK_COL]].dropna(subset=[TAU_COL]).iterrows():
    sess = norm_name(row[TAU_COL])
    braak = parse_braak(row[BRAAK_COL])

    if sess not in session_to_braak:
        session_to_braak[sess] = braak
    else:
        old = session_to_braak[sess]
        vals = [v for v in [old, braak] if v is not None]
        session_to_braak[sess] = max(vals) if vals else None

print(f"\nMapped {len(session_to_braak)} TAU_PET_Session entries to Braak values")

print("\n=== Scan current kari_all ===")
if not os.path.exists(OLD_ROOT):
    raise RuntimeError(f"OLD_ROOT not found: {OLD_ROOT}")

rows = []
groups = defaultdict(list)

for folder in sorted(os.listdir(OLD_ROOT)):
    path = os.path.join(OLD_ROOT, folder)
    if not os.path.isdir(path):
        continue

    sess_key = norm_name(folder)
    has_flair = os.path.exists(os.path.join(path, FLAIR_FILE))
    braak = session_to_braak.get(sess_key, None)
    sid = get_subject_id(folder)

    row = {
        "folder": folder,
        "subject_id": sid,
        "has_flair": has_flair,
        "braak": braak,
        "selected": 0,
        "reason": "",
    }
    rows.append(row)
    groups[sid].append(row)

print(f"Total visit folders: {len(rows)}")
print(f"Total subjects: {len(groups)}")

print("\n=== Select one visit per subject ===")
selected = []

for sid, visits in sorted(groups.items()):
    flair_visits = [v for v in visits if v["has_flair"]]

    if flair_visits:
        pool = flair_visits
        pool_name = "FLAIR_ONLY"
    else:
        pool = visits
        pool_name = "ALL_VISITS_NO_FLAIR"

    valid_braak = [v for v in pool if v["braak"] is not None]
    if valid_braak:
        max_braak = max(v["braak"] for v in valid_braak)
        finalists = [v for v in pool if v["braak"] == max_braak]
        braak_reason = f"max_braak={max_braak}"
    else:
        finalists = pool
        braak_reason = "all_braak_missing"

    # fallback: lexicographically largest folder name
    chosen = sorted(finalists, key=lambda x: x["folder"])[-1]
    chosen["selected"] = 1
    chosen["reason"] = f"{pool_name};{braak_reason};folder_tiebreak"
    selected.append(chosen)

    visit_text = " | ".join(
        f"{v['folder']} [flair={v['has_flair']}, braak={v['braak']}]"
        for v in sorted(visits, key=lambda x: x["folder"])
    )
    print(f"{sid} -> {chosen['folder']}   ({chosen['reason']})")
    print(f"    {visit_text}")

# save manifests before rename
all_df = pd.DataFrame(rows).sort_values(["subject_id", "folder"])
sel_df = pd.DataFrame(selected).sort_values(["subject_id", "folder"])

all_df.to_csv(ALL_MANIFEST, index=False)
sel_df.to_csv(SELECTED_MANIFEST, index=False)

print("\n=== Manifest summary ===")
print(f"All-visits manifest: {ALL_MANIFEST}")
print(f"Selected manifest  : {SELECTED_MANIFEST}")
print(f"Selected folders   : {len(sel_df)}")
print(f"Selected with flair: {int(sel_df['has_flair'].sum())}")
print(f"Selected no flair  : {len(sel_df) - int(sel_df['has_flair'].sum())}")

print("\n=== Rename and copy ===")
if os.path.exists(ARCHIVE_ROOT):
    raise RuntimeError(f"Archive root already exists, stop: {ARCHIVE_ROOT}")

print(f"Renaming:\n  {OLD_ROOT}\n-> {ARCHIVE_ROOT}")
os.rename(OLD_ROOT, ARCHIVE_ROOT)

print(f"Creating new empty folder:\n  {NEW_ROOT}")
os.makedirs(NEW_ROOT, exist_ok=False)

for folder in sel_df["folder"]:
    src = os.path.join(ARCHIVE_ROOT, folder)
    dst = os.path.join(NEW_ROOT, folder)
    print(f"Copy: {src} -> {dst}")
    shutil.copytree(src, dst)

print("\nDone.")
print(f"Old full dataset archived at: {ARCHIVE_ROOT}")
print(f"New selected dataset at    : {NEW_ROOT}")
