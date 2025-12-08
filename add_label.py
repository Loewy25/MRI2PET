#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ---- paths you care about ----
ROOT_DIR   = "/scratch/l.peiwang/kari_brainv33_top300"
FOLDS_DIR  = os.path.join(ROOT_DIR, "CV5_braak_strat")   # where fold1..fold5.csv live
META_CSV   = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"

# column names in metadata
TAU_COL      = "TAU_PET_Session"
BRAAK1_2_COL = "Braak1_2"
BRAAK3_4_COL = "Braak3_4"
BRAAK5_6_COL = "Braak5_6"
BRAAK_THR    = 1.2  # threshold


def norm_key(s: str) -> str:
    return str(s).strip().lower()


def build_sid_to_label(meta_csv: str, sids):
    """
    Build mapping from subject folder name -> integer label 0/1/2/3.
    Raises if anything is missing or inconsistent.
    """
    df = pd.read_csv(meta_csv)
    # strip column names
    df.columns = [c.strip() for c in df.columns]

    # make sure needed columns exist
    for col in [TAU_COL, BRAAK1_2_COL, BRAAK3_4_COL, BRAAK5_6_COL]:
        if col not in df.columns:
            raise RuntimeError(f"Metadata CSV missing required column: {col}")

    # normalize TAU_PET_Session for matching
    df["_key"] = df[TAU_COL].apply(norm_key)

    # ensure braak columns are numeric
    for col in [BRAAK1_2_COL, BRAAK3_4_COL, BRAAK5_6_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sid_to_label = {}
    for sid in sorted(set(sids)):
        if not sid:
            continue
        key = norm_key(sid)
        rows = df[df["_key"] == key]
        if rows.empty:
            raise RuntimeError(f"No metadata row found for subject '{sid}' (key {key})")

        # there *might* be multiple rows; pick the highest stage across them
        labels = []
        for _, row in rows.iterrows():
            b12 = row[BRAAK1_2_COL]
            b34 = row[BRAAK3_4_COL]
            b56 = row[BRAAK5_6_COL]

            # basic sanity
            if np.isnan(b12) and np.isnan(b34) and np.isnan(b56):
                # treat as 0
                labels.append(0)
                continue

            if b56 >= BRAAK_THR:
                labels.append(3)  # V/VI
            elif b34 >= BRAAK_THR:
                labels.append(2)  # III/IV
            elif b12 >= BRAAK_THR:
                labels.append(1)  # I/II
            else:
                labels.append(0)  # stage 0

        sid_to_label[sid] = max(labels)

    return sid_to_label


def process_one_fold(path: str, sid_to_label):
    print(f"Processing {path}")
    df = pd.read_csv(path)

    if "train" not in df.columns:
        raise RuntimeError(f"'train' column not found in {path}")

    labels = []
    for sid in df["train"]:
        if isinstance(sid, float) and np.isnan(sid):
            # empty cell at bottom; no label
            labels.append("")
        elif sid == "" or sid is None:
            labels.append("")
        else:
            sid_str = str(sid).strip()
            if sid_str not in sid_to_label:
                raise RuntimeError(f"No label found for train subject '{sid_str}' in {path}")
            labels.append(sid_to_label[sid_str])

    df["label"] = labels
    df.to_csv(path, index=False)


def main():
    # find fold csvs
    fold_files = [f for f in os.listdir(FOLDS_DIR)
                  if f.lower().startswith("fold") and f.lower().endswith(".csv")]
    if not fold_files:
        raise RuntimeError(f"No fold*.csv files found in {FOLDS_DIR}")

    # collect all train SIDs once
    all_train_sids = []
    for fname in fold_files:
        df_fold = pd.read_csv(os.path.join(FOLDS_DIR, fname))
        if "train" not in df_fold.columns:
            raise RuntimeError(f"'train' column not found in {fname}")
        all_train_sids.extend([str(x).strip() for x in df_fold["train"] if str(x).strip()])

    print(f"Found {len(set(all_train_sids))} unique train subjects across folds.")

    # build mapping from SID -> label (0/1/2/3) using metadata
    sid_to_label = build_sid_to_label(META_CSV, all_train_sids)

    # now modify each fold in place
    for fname in sorted(fold_files):
        process_one_fold(os.path.join(FOLDS_DIR, fname), sid_to_label)

    print("Done: each foldX.csv now has a 'label' column for the train subject.")


if __name__ == "__main__":
    main()
