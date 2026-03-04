#!/usr/bin/env python3
import os, re
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

def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())  # drop leading "12-" etc
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

def detect(series_names):
    # Only EXTRA modalities (no T1)
    has = {k: False for k in ["FLAIR","T2STAR_SWI","DIFFUSION","RSFMRI_BOLD","ASL","TOF"]}
    for s in series_names:
        if IGNORE.search(s): 
            continue
        t = toks(s)

        # FLAIR: require explicit token
        if "flair" in t:
            has["FLAIR"] = True

        # T2*/SWI: require explicit t2+star OR t2star OR swi
        if (("t2" in t and "star" in t) or ("t2star" in t) or ("swi" in t)):
            has["T2STAR_SWI"] = True

        # Diffusion: require strong diffusion markers (avoid only FA/ADC, bc UTE has FA3/FA12)
        if ("dwi" in t) or ("dti" in t) or ("dbsi" in t) or ("mddw" in t) or (("ep2d" in t) and ("diff" in t)) or ("diff" in t):
            has["DIFFUSION"] = True

        # rsfMRI/BOLD: require rsfmri/bold or ep2d+bold, or resting+state+fmri
        if ("rsfmri" in t) or ("bold" in t) or (("ep2d" in t) and ("bold" in t)) or (("resting" in t) and ("state" in t) and ("fmri" in t)):
            has["RSFMRI_BOLD"] = True

        # ASL: require explicit asl/pasl/pcasl/relcbf (NOT generic "perfusion weighted")
        if ("asl" in t) or ("pasl" in t) or ("pcasl" in t) or ("relcbf" in t):
            has["ASL"] = True

        # TOF angiography
        if "tof" in t:
            has["TOF"] = True

    return has

def main():
    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    # index session directories (case-insensitive, handle _/- mismatch)
    dirs = {d.lower(): d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))}
    def resolve(s):
        k = s.lower().replace(".zip","")
        return dirs.get(k) or dirs.get(k.replace("-","_")) or dirs.get(k.replace("_","-"))

    cnt = {k: 0 for k in ["FLAIR","T2STAR_SWI","DIFFUSION","RSFMRI_BOLD","ASL","TOF"]}
    found = 0
    missing = 0

    for s in sessions:
        d = resolve(s)
        if not d:
            missing += 1
            continue
        found += 1
        p = os.path.join(ROOT, d)
        series = [x for x in os.listdir(p) if os.path.isdir(os.path.join(p, x))]
        has = detect(series)
        for k,v in has.items():
            if v: cnt[k] += 1

    print("=== Extra modality availability (per MR_Session folder) ===")
    print(f"Total MR_Session in CSV: {len(sessions)}")
    print(f"Found session folders:   {found}")
    print(f"Missing folders:         {missing}\n")
    for k in ["FLAIR","T2STAR_SWI","DIFFUSION","RSFMRI_BOLD","ASL","TOF"]:
        pct = (cnt[k]/found*100.0) if found else 0.0
        print(f"{k:12s}: {cnt[k]:6d} / {found}  ({pct:5.1f}%)")

if __name__ == "__main__":
    main()
