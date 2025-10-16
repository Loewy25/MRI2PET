#!/usr/bin/env python3
"""
Generate five CSVs for 5-fold cross-validation.

- Discovers subject folders like your KariAV1451Dataset:
  subject dir must contain:
    T1_masked.nii.gz
    PET_in_T1_masked.nii.gz
    (and by default) aseg_brainmask.nii.gz
- Deterministic shuffle (seed=1999), round-robin fold assignment.
- For fold f (1..5):
    test = subjects assigned to fold f
    val  = subjects assigned to fold f+1 (wrap-around)
    train = the remaining subjects
- Writes:
    <root>/cv_folds/fold1.csv
    ...
    <root>/cv_folds/fold5.csv
Each CSV has columns: train,validation,test (one subject per cell; rows padded with blanks).
"""

import os, glob, csv, argparse, random
from typing import List, Tuple

def find_subjects(root: str, require_mask: bool = True) -> List[str]:
    # Mirror your dataset patterns
    patterns = [os.path.join(root, "*T807*"), os.path.join(root, "*t807*")]
    cand_dirs = []
    for p in patterns:
        cand_dirs.extend(glob.glob(p))
    cand_dirs = sorted([d for d in cand_dirs if os.path.isdir(d)])

    sids = []
    for d in cand_dirs:
        t1  = os.path.join(d, "T1_masked.nii.gz")
        pet = os.path.join(d, "PET_in_T1_masked.nii.gz")
        msk = os.path.join(d, "aseg_brainmask.nii.gz")
        if os.path.exists(t1) and os.path.exists(pet):
            if (not require_mask) or os.path.exists(msk):
                sids.append(os.path.basename(d))
    return sids

def assign_folds(sids: List[str], k: int, seed: int) -> dict:
    rng = random.Random(seed)
    idxs = list(range(len(sids)))
    rng.shuffle(idxs)
    sid2fold = {}
    for rank, i in enumerate(idxs):
        sid2fold[sids[i]] = (rank % k)  # 0..k-1
    return sid2fold

def write_fold_csv(out_csv: str, train: List[str], val: List[str], test: List[str]):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    L = max(len(train), len(val), len(test))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train", "validation", "test"])
        for i in range(L):
            row = [
                train[i] if i < len(train) else "",
                val[i]   if i < len(val)   else "",
                test[i]  if i < len(test)  else "",
            ]
            w.writerow(row)

def main():
    # Defaults: prefer your “top300” root, fall back to your original if needed
    default_root = "/scratch/l.peiwang/kari_brain_top300"
    if not os.path.isdir(default_root):
        default_root = "/scratch/l.peiwang/kari_brainv33"

    ap = argparse.ArgumentParser("Make 5 CSVs for 5-fold CV (train/val/test columns).")
    ap.add_argument("--root", default=default_root, help="Dataset root containing subject folders.")
    ap.add_argument("--k", type=int, default=5, help="Number of folds.")
    ap.add_argument("--seed", type=int, default=1999, help="Deterministic shuffle seed.")
    ap.add_argument("--val_offset", type=int, default=1, help="Validation = (test_fold + val_offset) % k.")
    ap.add_argument("--allow-missing-mask", action="store_true",
                    help="Include subjects without aseg_brainmask.nii.gz (loader usually needs it).")
    ap.add_argument("--outdir", default=None,
                    help="Directory for CSVs (default: <root>/cv_folds).")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = args.outdir or os.path.join(root, "cv_folds")

    sids = find_subjects(root, require_mask=(not args.allow_missing_mask))
    if len(sids) == 0:
        raise SystemExit(f"No valid subjects found under {root}")

    sid2fold = assign_folds(sids, k=args.k, seed=args.seed)

    # Precompute per-fold membership
    folds = [[] for _ in range(args.k)]
    for sid in sids:
        folds[sid2fold[sid]].append(sid)

    # Write CSVs
    for f in range(args.k):
        test_s = sorted(folds[f])
        val_s  = sorted(folds[(f + args.val_offset) % args.k])
        train_s = sorted([sid for sid in sids if sid not in set(test_s) | set(val_s)])

        out_csv = os.path.join(outdir, f"fold{f+1}.csv")  # 1-based naming
        write_fold_csv(out_csv, train_s, val_s, test_s)

    # Print a quick summary
    print(f"Found {len(sids)} subjects under: {root}")
    print(f"CSV folder: {outdir}")
    for f in range(args.k):
        te = len(folds[f])
        va = len(folds[(f + args.val_offset) % args.k])
        tr = len(sids) - te - va
        print(f"fold{f+1}: train={tr}  val={va}  test={te}")

if __name__ == "__main__":
    main()
