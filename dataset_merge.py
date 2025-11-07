#!/usr/bin/env python3
import os, shutil, pandas as pd
from collections import Counter

CSV        = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
SRC_ROOTS  = ["/scratch/l.peiwang/kari_brainv33", "/scratch/l.peiwang/kari_brainv11"]
DEST       = "/scratch/l.peiwang/braak_merged_300"
LIMIT      = 300
THRESH     = 1.2
SESSION_COL= "TAU_PET_Session"

# --- load ---
need = ["ID", SESSION_COL, "TAU_PET_Date", "Braak1_2", "Braak3_4", "Braak5_6"]
df = pd.read_csv(CSV)
missing = [c for c in need if c not in df.columns]
if missing: raise KeyError(f"Missing columns: {missing}")

df["TAU_PET_Date"] = pd.to_datetime(df["TAU_PET_Date"], errors="coerce")
for c in ["Braak1_2","Braak3_4","Braak5_6"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def stage_priority(r):
    if r["Braak5_6"] > THRESH: return ("5/6",3)
    if r["Braak3_4"] > THRESH: return ("3/4",2)
    if r["Braak1_2"] > THRESH: return ("1/2",1)
    return ("0",0)

df[["BraakStage","priority"]] = df.apply(stage_priority, axis=1, result_type="expand")
df = (df.dropna(subset=[SESSION_COL])
        .sort_values(["priority","TAU_PET_Date"], ascending=[False, False])
        .drop_duplicates(subset=["ID"], keep="first"))

ordered = pd.concat([df[df["priority"]==p] for p in (3,2,1,0)], ignore_index=True)
print("Candidates by stage:")
print(ordered["BraakStage"].value_counts().to_string())

os.makedirs(DEST, exist_ok=True)

def find_src(session):
    for root in SRC_ROOTS:
        p = os.path.join(root, session)
        if os.path.isdir(p): return p
    return None

copied = 0
seen_sessions = set()
by_stage = Counter()
skips_missing = 0
skips_dupe = 0
print(f"\nCopying into {DEST} (target {LIMIT})...\n")

for r in ordered.itertuples(index=False):
    if copied >= LIMIT: break
    session = str(getattr(r, SESSION_COL))
    stage   = getattr(r, "BraakStage")
    if session in seen_sessions:
        skips_dupe += 1
        continue
    seen_sessions.add(session)

    src = find_src(session)
    if src is None:
        print(f"[SKIP] missing: {session}")
        skips_missing += 1
        continue

    dst = os.path.join(DEST, session)
    if os.path.exists(dst):
        print(f"[EXIST] {dst} — skipping copy, counting as included.")
        # count it as included if already there
        copied += 1
        by_stage[stage] += 1
        continue

    print(f"[{copied+1:03}/{LIMIT}] {session}  <-  {src}")
    shutil.copytree(src, dst)
    copied += 1
    by_stage[stage] += 1

print("\n=== Summary ===")
print(f"Copied: {copied}/{LIMIT}")
if copied < LIMIT:
    print("WARNING: Ran out of candidates before reaching limit.")
print("Included per stage:", dict(by_stage))
print(f"Skipped (missing folders): {skips_missing}")
print(f"Skipped (duplicate sessions in CSV): {skips_dupe}")

