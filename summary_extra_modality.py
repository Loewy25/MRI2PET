#!/usr/bin/env python3
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import SimpleITK as sitk

META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
COL  = "MR_Session"
ROOT = "/ceph/chpc/mapped/benz04_kari/scans"

IGNORE = re.compile(
    r"(mrac|petacquisition|\bpet\b|_ac\b|\bac\b|\bnac\b|ac_images|nac_images|prr|"
    r"umap|ute|uteflex|dixon|waterweighted|fatweighted|"
    r"phoenixzipreport|localizer|scout|mip|mosaics?|moco|mocoserie|mocoseries|"
    r"\bref\b|\btest\b|\breport\b)",
    re.I
)

def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

def flair_subtypes(series_name: str):
    if IGNORE.search(series_name):
        return set()

    t = toks(series_name)
    if "flair" not in t:
        return set()

    out = set()

    # likely 3D family by naming
    if (
        "3d" in t or
        "space" in t or
        "cube" in t or
        "vista" in t or
        "sagittal" in t
    ):
        out.add("FLAIR_3D_NAME")

    # likely 2D axial family by naming
    if "axial" in t or "tra" in t or "transverse" in t:
        out.add("FLAIR_2D_AXIAL_NAME")

    if "coronal" in t or "cor" in t:
        out.add("FLAIR_2D_CORONAL_NAME")

    if not out:
        out.add("FLAIR_AMBIG_NAME")

    return out

def resolve_session_dir(root, session_name):
    dirs = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    k = session_name.lower().replace(".zip", "")
    return dirs.get(k) or dirs.get(k.replace("-", "_")) or dirs.get(k.replace("_", "-"))

def list_series_dirs(session_dir):
    return [
        x for x in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, x))
    ]

def read_series_as_image(series_dir):
    """
    Try to read one DICOM series folder into a volume.
    Returns (img, error_message).
    """
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(series_dir)
        if not series_ids:
            return None, "no_dicom_series_id"

        # usually one series per folder; if multiple exist, take the largest
        best_files = None
        best_sid = None
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(series_dir, sid)
            if best_files is None or len(files) > len(best_files):
                best_files = files
                best_sid = sid

        if not best_files:
            return None, "no_dicom_files"

        reader.SetFileNames(best_files)
        img = reader.Execute()
        return img, None

    except Exception as e:
        return None, str(e)

def geom_class_from_spacing_and_size(spacing, size):
    """
    spacing: (sx, sy, sz)
    size:    (nx, ny, nz)

    Heuristic classification:
    - near-isotropic volumetric => likely native 3D acquisition
    - thick slice anisotropic   => likely 2D acquisition stored as 3D volume
    """
    sx, sy, sz = spacing
    nx, ny, nz = size

    # protect against nonsense
    if min(sx, sy, sz) <= 0:
        return "BAD_GEOM"

    # ratio of thickest to thinnest voxel dimension
    ratio = max(spacing) / min(spacing)

    # very small z count is suspicious
    if nz < 8:
        return "TOO_FEW_SLICES"

    # near isotropic volumetric
    if ratio <= 1.5 and sz <= 2.0:
        return "LIKELY_3D_VOLUME"

    # classic thick-slice stack
    if sz >= 3.0 or ratio >= 2.5:
        return "LIKELY_2D_ACQ_STORED_AS_3D"

    return "INTERMEDIATE_MIXED"

def choose_one_flair_series(series_names):
    """
    Pick one preferred FLAIR per session for inspection.
    Preference:
      1) named 3D flair
      2) named 2D axial flair
      3) any other flair
    """
    scored = []
    for s in series_names:
        labs = flair_subtypes(s)
        if not labs:
            continue

        if "FLAIR_3D_NAME" in labs:
            score = 0
        elif "FLAIR_2D_AXIAL_NAME" in labs:
            score = 1
        else:
            score = 2

        scored.append((score, s))

    if not scored:
        return None
    scored.sort()
    return scored[0][1]

def main():
    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    found = 0
    missing = 0
    no_flair = 0
    unreadable = 0

    geom_counter = Counter()
    name_counter = Counter()
    cross_counter = Counter()

    detailed_rows = []

    for sess in sessions:
        d = resolve_session_dir(ROOT, sess)
        if not d:
            missing += 1
            continue

        found += 1
        sess_path = os.path.join(ROOT, d)
        series_names = list_series_dirs(sess_path)

        chosen = choose_one_flair_series(series_names)
        if chosen is None:
            no_flair += 1
            continue

        name_labels = flair_subtypes(chosen)
        name_label = "|".join(sorted(name_labels))
        name_counter[name_label] += 1

        chosen_path = os.path.join(sess_path, chosen)
        img, err = read_series_as_image(chosen_path)

        if img is None:
            unreadable += 1
            geom_label = f"UNREADABLE:{err}"
            geom_counter[geom_label] += 1
            cross_counter[(name_label, geom_label)] += 1
            detailed_rows.append({
                "MR_Session": sess,
                "series_name": chosen,
                "name_label": name_label,
                "geom_label": geom_label,
                "size_x": None,
                "size_y": None,
                "size_z": None,
                "spacing_x": None,
                "spacing_y": None,
                "spacing_z": None,
            })
            continue

        size = img.GetSize()        # (x, y, z)
        spacing = img.GetSpacing()  # (sx, sy, sz)

        geom_label = geom_class_from_spacing_and_size(spacing, size)
        geom_counter[geom_label] += 1
        cross_counter[(name_label, geom_label)] += 1

        detailed_rows.append({
            "MR_Session": sess,
            "series_name": chosen,
            "name_label": name_label,
            "geom_label": geom_label,
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
            "spacing_x": spacing[0],
            "spacing_y": spacing[1],
            "spacing_z": spacing[2],
        })

    total = found

    print("=== FLAIR geometry sanity check ===")
    print(f"Total sessions in CSV: {len(sessions)}")
    print(f"Found session folders : {found}")
    print(f"Missing folders       : {missing}")
    print(f"No FLAIR found        : {no_flair}")
    print(f"Unreadable FLAIR      : {unreadable}")

    print("\n=== Name-based chosen FLAIR subtype ===")
    for k, n in sorted(name_counter.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * n / total if total else 0.0
        print(f"{k:30s}: {n:6d} / {total} ({pct:5.1f}%)")

    print("\n=== Geometry-based classification of chosen FLAIR ===")
    for k, n in sorted(geom_counter.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * n / total if total else 0.0
        print(f"{k:30s}: {n:6d} / {total} ({pct:5.1f}%)")

    print("\n=== Cross-tab: name subtype vs actual geometry ===")
    grouped = defaultdict(int)
    for (name_label, geom_label), n in cross_counter.items():
        grouped[name_label] += n

    for name_label in sorted(grouped):
        print(f"\n{name_label}")
        pairs = [(g, n) for (nm, g), n in cross_counter.items() if nm == name_label]
        for geom_label, n in sorted(pairs, key=lambda x: (-x[1], x[0])):
            pct = 100.0 * n / grouped[name_label] if grouped[name_label] else 0.0
            print(f"  {geom_label:28s}: {n:6d} ({pct:5.1f}% within this name group)")

    out_csv = "flair_geometry_check.csv"
    pd.DataFrame(detailed_rows).to_csv(out_csv, index=False)
    print(f"\nSaved detailed per-session table to: {out_csv}")

if __name__ == "__main__":
    main()
