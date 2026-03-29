#!/usr/bin/env python3
from pathlib import Path
import shutil

from summary_flair_clinical import (
    CSV1,
    CSV2,
    CSV3,
    DATASET,
    FIELDS2,
    FIELDS3,
    ROOT,
    collapse,
    has_value,
    norm,
    read_csv_any,
)

TARGET = ROOT / "kari_flair_all"


def main():
    for path in [DATASET, CSV1, CSV2, CSV3]:
        if not path.exists():
            raise FileNotFoundError(path)

    need1 = ["TAU_PET_Session", "MR_Session"]
    need2 = ["MR_Session", "ID"] + FIELDS2
    need3 = ["ID"] + FIELDS3

    df1 = read_csv_any(CSV1, need1)
    df2 = read_csv_any(CSV2, need2)
    df3 = read_csv_any(CSV3, need3, allow_empty=True)

    idx1 = collapse(df1, "TAU_PET_Session", ["MR_Session"])
    idx2 = collapse(df2, "MR_Session", ["ID"] + FIELDS2)
    idx3 = collapse(df3, "ID", FIELDS3)

    keep = []
    for subj in sorted(p.name for p in DATASET.iterdir() if p.is_dir()):
        row1 = idx1.loc[norm(subj)] if norm(subj) in idx1.index else None
        mr_session = row1["MR_Session"] if row1 is not None else None
        row2 = idx2.loc[norm(mr_session)] if norm(mr_session) in idx2.index else None
        subj_id = row2["ID"] if row2 is not None else None
        row3 = idx3.loc[norm(subj_id)] if norm(subj_id) in idx3.index else None

        ok = row1 is not None and row2 is not None and row3 is not None
        ok = ok and all(has_value(row2[col]) for col in FIELDS2)
        ok = ok and all(has_value(row3[col]) for col in FIELDS3)
        if ok:
            keep.append(subj)

    TARGET.mkdir(parents=True, exist_ok=True)
    copied = 0
    for subj in keep:
        shutil.copytree(DATASET / subj, TARGET / subj, dirs_exist_ok=True)
        print(subj)
        copied += 1

    print(f"Copied {copied} subjects to {TARGET}")


if __name__ == "__main__":
    main()
