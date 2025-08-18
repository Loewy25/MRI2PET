#!/usr/bin/env python3
import os, glob, shutil

# --- EDIT THESE ---
PUP_ROOT = "/ceph/chpc/mapped/benz04_kari/pup"
OUT_ROOT = "/scratch/l.peiwang/pup_collected"
# ------------------

def pick_best(paths):
    """Pick the deepest path; tie-break by name (last alphabetically)."""
    if not paths: return None
    return sorted(paths, key=lambda p: (p.count(os.sep), p))[-1]

def find_last(subject_root, patterns):
    hits = []
    for pat in patterns:
        hits += glob.glob(os.path.join(subject_root, "**", pat), recursive=True)
    return pick_best(hits)

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    subjects = [d for d in os.listdir(PUP_ROOT)
                if os.path.isdir(os.path.join(PUP_ROOT, d))
                and ("av1451" in d.lower() or "fdg" in d.lower())]
    subjects.sort()

    total = ok = skipped = 0
    for subj in subjects:
        total += 1
        subj_root = os.path.join(PUP_ROOT, subj)

        suvr = find_last(subj_root, ["*_msum_SUVR.nii.gz"])
        t1001 = find_last(subj_root, ["T1001.nii.gz"])
        bmask = find_last(subj_root, ["BrainMask.nii.gz")

        if not (suvr and t1001 and bmask):
            missing = ["PET" if not suvr else None,
                       "T1001" if not t1001 else None,
                       "brainmask" if not bmask else None]
            missing = ", ".join([m for m in missing if m])
            print(f"[SKIP] {subj} (missing: {missing})")
            skipped += 1
            continue

        out_dir = os.path.join(OUT_ROOT, subj)
        os.makedirs(out_dir, exist_ok=True)

        for src in (suvr, t1001, bmask):
            dst = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dst)

        print(f"[OK]   {subj} -> {out_dir}")
        ok += 1

    print("\n=== SUMMARY ===")
    print(f"Total matched subjects : {total}")
    print(f"Processed OK           : {ok}")
    print(f"Skipped (missing files): {skipped}")
    print(f"Output root            : {OUT_ROOT}")

if __name__ == "__main__":
    main()
