#!/usr/bin/env python3
import os, shutil, pandas as pd

CSV   = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
SRC_ROOTS = ["/scratch/l.peiwang/kari_brainv33", "/scratch/l.peiwang/kari_brainv11"]
DEST  = "/scratch/l.peiwang/braak_merged_300"
LIMIT = 300
THRESH = 1.2
SESSION_COL = "TAU_PET_Session"

df = pd.read_csv(CSV)
df["TAU_PET_Date"] = pd.to_datetime(df["TAU_PET_Date"], errors="coerce")
for c in ["Braak1_2","Braak3_4","Braak5_6"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def stage_and_priority(r):
    if r["Braak5_6"] > THRESH: return ("5/6", 3)
    if r["Braak3_4"] > THRESH: return ("3/4", 2)
    if r["Braak1_2"] > THRESH: return ("1/2", 1)
    return ("0", 0)

df[["BraakStage","priority"]] = df.apply(stage_and_priority, axis=1, result_type="expand")
df = (df.dropna(subset=[SESSION_COL])
        .sort_values(["priority","TAU_PET_Date"], ascending=[False, False])
        .drop_duplicates(subset=["ID"], keep="first"))

ordered = pd.concat([df[df["priority"]==p] for p in (3,2,1,0)], ignore_index=True)
sel = ordered.head(LIMIT)

os.makedirs(DEST, exist_ok=True)

def find_src(session):
    for root in SRC_ROOTS:
        p = os.path.join(root, session)
        if os.path.isdir(p): return p
    return None  # instead of raise

print(f"\nCopying to {DEST} ...\n")
copied = 0
for i, r in enumerate(sel.itertuples(index=False), 1):
    session = str(getattr(r, SESSION_COL))
    src = find_src(session)
    if src is None:
        print(f"[SKIP] {session} not found in roots.")
        continue
    dst = os.path.join(DEST, session)
    if os.path.exists(dst):
        print(f"[EXIST] {dst}, skipping.")
        continue
    print(f"[{copied+1:03}/{LIMIT}] Copying {session}")
    shutil.copytree(src, dst)
    copied += 1

print(f"\nDone. Successfully copied {copied} folders (out of {LIMIT} planned).")
