#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import pandas as pd

META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
COL  = "MR_Session"
ROOT = "/ceph/chpc/mapped/benz04_kari/scans"

# Ignore PET plumbing + attenuation correction + reports + mosaics etc.
IGNORE = re.compile(
    r"(mrac|petacquisition|\bpet\b|_ac\b|\bac\b|\bnac\b|ac_images|nac_images|prr|"
    r"umap|ute|uteflex|dixon|waterweighted|fatweighted|"
    r"phoenixzipreport|localizer|scout|mip|mosaics?|moco|mocoserie|mocoseries)",
    re.I
)

MODALITIES = ["FLAIR", "T2STAR_SWI", "DIFFUSION", "RSFMRI_BOLD", "ASL", "TOF"]

def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())  # drop leading "12-" etc
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

def classify_series(series_name: str):
    """
    Return a list of modality labels this series qualifies for.
    A series can match more than one label in principle, though usually it won't.
    """
    if IGNORE.search(series_name):
        return []

    t = toks(series_name)
    hits = []

    # FLAIR: require explicit token
    if "flair" in t:
        hits.append("FLAIR")

    # T2*/SWI: require explicit t2+star OR t2star OR swi
    if (("t2" in t and "star" in t) or ("t2star" in t) or ("swi" in t)):
        hits.append("T2STAR_SWI")

    # Diffusion: require strong diffusion markers
    if ("dwi" in t) or ("dti" in t) or ("dbsi" in t) or ("mddw" in t) or (("ep2d" in t) and ("diff" in t)) or ("diff" in t):
        hits.append("DIFFUSION")

    # rsfMRI/BOLD
    if ("rsfmri" in t) or ("bold" in t) or (("ep2d" in t) and ("bold" in t)) or (("resting" in t) and ("state" in t) and ("fmri" in t)):
        hits.append("RSFMRI_BOLD")

    # ASL
    if ("asl" in t) or ("pasl" in t) or ("pcasl" in t) or ("relcbf" in t):
        hits.append("ASL")

    # TOF angiography
    if "tof" in t:
        hits.append("TOF")

    return hits

def main():
    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    # index session directories (case-insensitive, handle _/- mismatch)
    dirs = {d.lower(): d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))}

    def resolve(s):
        k = s.lower().replace(".zip", "")
        return dirs.get(k) or dirs.get(k.replace("-", "_")) or dirs.get(k.replace("_", "-"))

    cnt = {k: 0 for k in MODALITIES}
    found = 0
    missing = 0

    # modality -> unique series name -> how many sessions it appeared in
    matched_names = {k: defaultdict(int) for k in MODALITIES}

    # optional: store per-session matched series if you want later debugging
    # session_hits = {}

    for s in sessions:
        d = resolve(s)
        if not d:
            missing += 1
            continue

        found += 1
        p = os.path.join(ROOT, d)
        series = [x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))]

        has = {k: False for k in MODALITIES}

        # to avoid counting the same series name twice within one session
        seen_this_session = {k: set() for k in MODALITIES}

        for series_name in series:
            hits = classify_series(series_name)
            for mod in hits:
                has[mod] = True
                if series_name not in seen_this_session[mod]:
                    matched_names[mod][series_name] += 1
                    seen_this_session[mod].add(series_name)

        for k, v in has.items():
            if v:
                cnt[k] += 1

        # session_hits[s] = {k: sorted(seen_this_session[k]) for k in MODALITIES if seen_this_session[k]}

    print("=== Extra modality availability (per MR_Session folder) ===")
    print(f"Total MR_Session in CSV: {len(sessions)}")
    print(f"Found session folders:   {found}")
    print(f"Missing folders:         {missing}\n")

    for k in MODALITIES:
        pct = (cnt[k] / found * 100.0) if found else 0.0
        print(f"{k:12s}: {cnt[k]:6d} / {found}  ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("=== Detailed matched series names by modality ===")
    print("=" * 80)

    for mod in MODALITIES:
        print(f"\n--- {mod} ---")
        if not matched_names[mod]:
            print("  No matched series names found.")
            continue

        # sort by frequency desc, then name asc
        for name, n_sess in sorted(matched_names[mod].items(), key=lambda x: (-x[1], x[0].lower())):
            print(f"  [{n_sess:4d} sessions] {name}")

if __name__ == "__main__":
    main()
