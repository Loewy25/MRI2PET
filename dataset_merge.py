#!/usr/bin/env python3
import os, sys, shutil, pandas as pd

CSV   = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
SRC_ROOTS = ["/scratch/l.peiwang/kari_brainv33", "/scratch/l.peiwang/kari_brainv11"]  # try v33 first
DEST  = "/scratch/l.peiwang/braak_merged_300"    # <--- change if you want
LIMIT = 300
THRESH = 1.2
SESSION_COL = "TAU_PET_Session"

# ---- load ----
df = pd.read_csv(CSV)
need_cols = ["ID", SESSION_COL, "TAU_PET_Date", "Braak1_2", "Braak3_4", "Braak5_6"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in CSV: {missing}")

df["TAU_PET_Date"] = pd.to_datetime(df["TAU_PET_Date"], errors="coerce")
for c in ["Braak1_2","Braak3_4","Braak5_6"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- stage assignment (highest > THRESH wins) ----
def stage_and_priority(r):
    if r["Braak5_6"] > THRESH: return ("5/6", 3)
    if r["Braak3_4"] > THRESH: return ("3/4", 2)
    if r["Braak1_2"] > THRESH: return ("1/2", 1)
    return ("0", 0)

df[["BraakStage","priority"]] = df.apply(stage_and_priority, axis=1, result_type="expand")

# one row per subject: highest stage, then latest TAU_PET_Date
df = (df.dropna(subset=[SESSION_COL])
        .sort_values(["priority","TAU_PET_Date"], ascending=[False, False])
        .drop_duplicates(subset=["ID"], keep="first"))

print("Subjects by stage (after de-dupe):")
print(df["BraakStage"].value_counts().to_string(), flush=True)

# ---- prioritized selection up to LIMIT ----
ordered = pd.concat([df[df["priority"]==p] for p in (3,2,1,0)], ignore_index=True)
sel = ordered.head(LIMIT)
print(f"\nSelected {len(sel)} subjects (target {LIMIT}). Stage breakdown:")
print(sel["BraakStage"].value_counts().to_string(), flush=True)

# ---- copy ----
os.makedirs(DEST, exist_ok=True)

def find_src(session):
    for root in SRC_ROOTS:
        p = os.path.join(root, session)
        if os.path.isdir(p): return p
    raise FileNotFoundError(f"Session folder '{session}' not found in {SRC_ROOTS}")

print(f"\nCopying to: {DEST}\n")
for i, r in enumerate(sel.itertuples(index=False), 1):
    session = str(getattr(r, SESSION_COL))
    src = find_src(session)
    dst = os.path.join(DEST, session)
    print(f"[{i:03}/{len(sel)}] {session}  <--  {src}")
    shutil.copytree(src, dst)   # will error if dst exists (good)

print("\nDone.")
