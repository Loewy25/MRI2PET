#!/usr/bin/env python3
import argparse
import csv
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/scratch/l.peiwang/kari_flair_all"
DEFAULT_META_CSV = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
TAU_COL = "TAU_PET_Session"
BRAAK_COLS = ["Braak1_2", "Braak3_4", "Braak5_6"]
BRAAK_THR = 1.2
STAGE_ORDER = {"0": 0, "I/II": 1, "III/IV": 2, "V/VI": 3}


def norm_key(x) -> str:
    return str(x).strip().lower()


def list_subject_dirs(root: str) -> List[str]:
    required = {
        "T1_masked.nii.gz",
        "FLAIR_in_T1_masked.nii.gz",
        "PET_in_T1_masked.nii.gz",
        "aseg_brainmask.nii.gz",
        "mask_cortex.nii.gz",
    }
    sids = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        files = set(os.listdir(entry.path))
        if required.issubset(files):
            sids.append(entry.name)
    if not sids:
        raise RuntimeError(f"No valid subjects found under {root}")
    return sids


def stage_from_row(row: pd.Series) -> str:
    b12 = float(pd.to_numeric(row["Braak1_2"], errors="coerce"))
    b34 = float(pd.to_numeric(row["Braak3_4"], errors="coerce"))
    b56 = float(pd.to_numeric(row["Braak5_6"], errors="coerce"))
    if np.isfinite(b56) and b56 >= BRAAK_THR:
        return "V/VI"
    if np.isfinite(b34) and b34 >= BRAAK_THR:
        return "III/IV"
    if np.isfinite(b12) and b12 >= BRAAK_THR:
        return "I/II"
    return "0"


def build_label_maps(meta_csv: str, subjects: List[str]):
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    need = [TAU_COL] + BRAAK_COLS
    missing = [col for col in need if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {meta_csv}: {missing}")

    df = df[need].copy()
    df["_key"] = df[TAU_COL].map(norm_key)
    df["stage"] = df.apply(stage_from_row, axis=1)

    stage_map: Dict[str, str] = {}
    label_map: Dict[str, int] = {}
    for sid in subjects:
        rows = df[df["_key"] == norm_key(sid)]
        if rows.empty:
            raise RuntimeError(f"No metadata row found for subject {sid}")
        stage = max(rows["stage"].tolist(), key=lambda x: STAGE_ORDER[x])
        stage_map[sid] = stage
        label_map[sid] = STAGE_ORDER[stage]
    return stage_map, label_map


def stratified_folds(subjects: List[str], stage_map: Dict[str, str], k: int, seed: int):
    rng = random.Random(seed)
    by_stage = {stage: [] for stage in STAGE_ORDER}
    for sid in subjects:
        by_stage[stage_map[sid]].append(sid)

    folds = {i: [] for i in range(k)}
    for stage in ["V/VI", "III/IV", "I/II", "0"]:
        group = sorted(by_stage[stage])
        rng.shuffle(group)
        for i, sid in enumerate(group):
            folds[i % k].append(sid)
    for i in range(k):
        folds[i] = sorted(folds[i])
    return folds


def write_fold_csv(path: str, train: List[str], val: List[str], test: List[str], label_map: Dict[str, int]):
    rows = max(len(train), len(val), len(test))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train", "validation", "test", "label"])
        for i in range(rows):
            train_sid = train[i] if i < len(train) else ""
            writer.writerow(
                [
                    train_sid,
                    val[i] if i < len(val) else "",
                    test[i] if i < len(test) else "",
                    label_map[train_sid] if train_sid else "",
                ]
            )


def main():
    parser = argparse.ArgumentParser(description="Create Braak-stratified 5-fold CSVs with train labels.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--meta_csv", default=DEFAULT_META_CSV)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1999)
    parser.add_argument("--val_offset", type=int, default=1)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir or os.path.join(root, "CV5_braak_strat"))
    os.makedirs(outdir, exist_ok=True)

    subjects = list_subject_dirs(root)
    stage_map, label_map = build_label_maps(args.meta_csv, subjects)
    folds = stratified_folds(subjects, stage_map, args.k, args.seed)

    for test_fold in range(args.k):
        val_fold = (test_fold + args.val_offset) % args.k
        test = folds[test_fold]
        val = folds[val_fold]
        train = sorted(
            sid
            for fold_idx, sids in folds.items()
            if fold_idx not in {test_fold, val_fold}
            for sid in sids
        )
        out_csv = os.path.join(outdir, f"fold{test_fold + 1}.csv")
        write_fold_csv(out_csv, train, val, test, label_map)
        print(
            f"[INFO] Wrote {out_csv} | train={len(train)} val={len(val)} test={len(test)}"
        )


if __name__ == "__main__":
    main()
