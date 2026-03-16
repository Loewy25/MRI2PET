#!/usr/bin/env python3
import os
import re
from collections import Counter
import pandas as pd

META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
COL  = "MR_Session"
ROOT = "/ceph/chpc/mapped/benz04_kari/scans"

# ------------------------------------------------------------
# Ignore obvious non-target junk
# ------------------------------------------------------------
IGNORE = re.compile(
    r"(mrac|petacquisition|\bpet\b|_ac\b|\bac\b|\bnac\b|ac_images|nac_images|prr|"
    r"umap|ute|uteflex|dixon|waterweighted|fatweighted|"
    r"phoenixzipreport|localizer|scout|mip|mosaics?|moco|mocoserie|mocoseries|"
    r"\bref\b|\btest\b|\breport\b)",
    re.I
)

def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())  # drop leading "12-" etc
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

# ============================================================
# FLAIR subtype classifier
# ============================================================
def flair_subtypes(series_name: str):
    """
    Return a set of FLAIR subtype labels for one series.
    Goal: group by processing-ready family, not just modality presence.
    """
    if IGNORE.search(series_name):
        return set()

    t = toks(series_name)
    s = series_name.lower()

    if "flair" not in t:
        return set()

    out = set()

    # 3D FLAIR family
    # examples:
    #   Sagittal_3D_FLAIR
    #   3D_FLAIR_MS_P_new
    #   SPACE_FLAIR / CUBE_FLAIR / VISTA_FLAIR if they exist later
    if (
        "3d" in t or
        "space" in t or
        "cube" in t or
        "vista" in t or
        "sagittal" in t
    ):
        out.add("FLAIR_3D")

    # 2D axial family
    # examples:
    #   Axial_T2_FLAIR
    #   Head_Axial_T2_FLAIR
    #   Axial_FLAIR
    if "axial" in t or "tra" in t or "transverse" in t:
        out.add("FLAIR_2D_AXIAL")

    # other 2D orientations
    if "coronal" in t or "cor" in t:
        out.add("FLAIR_2D_CORONAL")

    # fallback ambiguous FLAIR:
    # examples:
    #   FLAIR
    #   T2_FLAIR
    if not out:
        out.add("FLAIR_AMBIG")

    return out

# ============================================================
# T2*/SWI subtype classifier
# ============================================================
def t2star_subtypes(series_name: str):
    """
    Return a set of T2*/SWI subtype labels for one series.
    Goal: distinguish different susceptibility-family acquisitions.
    """
    if IGNORE.search(series_name):
        return set()

    t = toks(series_name)
    s = series_name.lower()

    # need explicit susceptibility-family evidence
    is_t2star = (
        ("t2" in t and "star" in t) or
        ("t2star" in t) or
        ("swi" in t)
    )
    if not is_t2star:
        return set()

    out = set()

    # SWI proper
    if "swi" in t:
        out.add("T2STAR_SWI")

    # multi-echo / 3TE type
    if "3te" in t or "multi" in t:
        out.add("T2STAR_MULTI_ECHO")

    # EPI-based T2*
    if "epi" in t or "ep2d" in t:
        out.add("T2STAR_EPI")

    # conventional axial T2*
    if "axial" in t or "tra" in t or "transverse" in t:
        out.add("T2STAR_AXIAL")

    # fallback ambiguous T2*
    if not out:
        out.add("T2STAR_AMBIG")

    return out

# ============================================================
# Helpers
# ============================================================
def summarize_combination(counter: Counter, total_found: int, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not counter:
        print("No subjects found.")
        return

    for combo, n in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        pct = (n / total_found * 100.0) if total_found else 0.0
        label = combo if combo else "NONE"
        print(f"{label:40s}: {n:6d} / {total_found}  ({pct:5.1f}%)")

def summarize_presence(counter: Counter, total_found: int, all_labels, title: str):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

    for lab in all_labels:
        n = counter.get(lab, 0)
        pct = (n / total_found * 100.0) if total_found else 0.0
        print(f"{lab:40s}: {n:6d} / {total_found}  ({pct:5.1f}%)")

def main():
    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    # index session directories (case-insensitive, handle _/- mismatch)
    dirs = {d.lower(): d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))}

    def resolve(s):
        k = s.lower().replace(".zip", "")
        return dirs.get(k) or dirs.get(k.replace("-", "_")) or dirs.get(k.replace("_", "-"))

    found = 0
    missing = 0

    # subject-level counters
    flair_combo_counter = Counter()
    flair_presence_counter = Counter()

    t2_combo_counter = Counter()
    t2_presence_counter = Counter()

    for sess in sessions:
        d = resolve(sess)
        if not d:
            missing += 1
            continue

        found += 1
        p = os.path.join(ROOT, d)

        series = [
            x for x in os.listdir(p)
            if os.path.isdir(os.path.join(p, x))
        ]

        flair_hits = set()
        t2_hits = set()

        for s in series:
            flair_hits |= flair_subtypes(s)
            t2_hits |= t2star_subtypes(s)

        # -------- subject-level FLAIR combo --------
        flair_combo = "|".join(sorted(flair_hits)) if flair_hits else "NONE"
        flair_combo_counter[flair_combo] += 1
        for lab in flair_hits:
            flair_presence_counter[lab] += 1

        # -------- subject-level T2* combo --------
        t2_combo = "|".join(sorted(t2_hits)) if t2_hits else "NONE"
        t2_combo_counter[t2_combo] += 1
        for lab in t2_hits:
            t2_presence_counter[lab] += 1

    print("=== Subject-level subtype summary ===")
    print(f"Total MR_Session in CSV: {len(sessions)}")
    print(f"Found session folders:   {found}")
    print(f"Missing folders:         {missing}")

    # ------------------------------------------------------------
    # FLAIR summary
    # ------------------------------------------------------------
    summarize_presence(
        flair_presence_counter,
        found,
        [
            "FLAIR_3D",
            "FLAIR_2D_AXIAL",
            "FLAIR_2D_CORONAL",
            "FLAIR_AMBIG",
        ],
        "FLAIR subtype presence (subject-level)"
    )

    summarize_combination(
        flair_combo_counter,
        found,
        "FLAIR subtype combinations (subject-level)"
    )

    # handy policy-style summary
    flair_usable = found - flair_combo_counter.get("NONE", 0)
    flair_has_3d = flair_presence_counter.get("FLAIR_3D", 0)
    flair_2d_fallback_only = flair_combo_counter.get("FLAIR_2D_AXIAL", 0) + flair_combo_counter.get("FLAIR_2D_CORONAL", 0)

    print("\n" + "-" * 80)
    print("FLAIR policy-oriented summary")
    print("-" * 80)
    print(f"Usable FLAIR (any subtype)               : {flair_usable:6d} / {found}  ({(flair_usable/found*100 if found else 0):5.1f}%)")
    print(f"Native 3D FLAIR available                : {flair_has_3d:6d} / {found}  ({(flair_has_3d/found*100 if found else 0):5.1f}%)")
    print(f"Likely 2D-only fallback subjects         : {flair_2d_fallback_only:6d} / {found}  ({(flair_2d_fallback_only/found*100 if found else 0):5.1f}%)")

    # ------------------------------------------------------------
    # T2*/SWI summary
    # ------------------------------------------------------------
    summarize_presence(
        t2_presence_counter,
        found,
        [
            "T2STAR_SWI",
            "T2STAR_AXIAL",
            "T2STAR_MULTI_ECHO",
            "T2STAR_EPI",
            "T2STAR_AMBIG",
        ],
        "T2*/SWI subtype presence (subject-level)"
    )

    summarize_combination(
        t2_combo_counter,
        found,
        "T2*/SWI subtype combinations (subject-level)"
    )

if __name__ == "__main__":
    main()
