#!/usr/bin/env python3
# This code was used to modificate the csv file to add RAW PET column 
import os, glob, pandas as pd, argparse

def resolve_strict(subject, pup_root):
    base = os.path.join(pup_root, subject)
    if not os.path.isdir(base):
        return None
    patterns = ["*msum_SUVR.nii.gz"]
    for pat in patterns:
        hits = glob.glob(os.path.join(base, "**", pat), recursive=True)
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pup_root", default="/ceph/chpc/mapped/benz04_kari/pup")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["pet_raw"] = [resolve_strict(s, args.pup_root) for s in df["subject"]]

    ok = df["pet_raw"].apply(lambda p: isinstance(p, str) and os.path.exists(p)).sum()
    print(f"[done] resolved {ok}/{len(df)}")

    out = args.out or args.csv.replace(".csv", "") + "_with_pet_raw.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
