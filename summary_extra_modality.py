#!/usr/bin/env python3
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import SimpleITK as sitk

META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
COL  = "MR_Session"
ROOT = "/ceph/chpc/mapped/benz04_kari/scans"

# ---------------------------------------------------------------------
# Ignore obvious non-target junk
# ---------------------------------------------------------------------
IGNORE = re.compile(
    r"(mrac|petacquisition|\bpet\b|_ac\b|\bac\b|\bnac\b|ac_images|nac_images|prr|"
    r"umap|ute|uteflex|dixon|waterweighted|fatweighted|"
    r"phoenixzipreport|localizer|scout|mip|mosaics?|moco|mocoserie|mocoseries|"
    r"\bref\b|\btest\b|\breport\b)",
    re.I
)

VOLUME_EXTS = (
    ".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".mgz"
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

def is_volume_file(fname: str):
    fl = fname.lower()
    return fl.endswith(VOLUME_EXTS)

def resolve_session_dir(root, session_name, dirs_map=None):
    if dirs_map is None:
        dirs_map = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    k = session_name.lower().replace(".zip", "")
    return dirs_map.get(k) or dirs_map.get(k.replace("-", "_")) or dirs_map.get(k.replace("_", "-"))

def list_series_dirs(session_dir):
    return sorted([
        x for x in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, x))
    ])

# ---------------------------------------------------------------------
# FLAIR name-based subtype classifier
# ---------------------------------------------------------------------
def flair_subtypes(series_name: str):
    """
    Name-based FLAIR family labels.
    These are still useful even before geometry inspection.
    """
    if IGNORE.search(series_name):
        return set()

    t = toks(series_name)
    if "flair" not in t:
        return set()

    out = set()

    # Likely 3D FLAIR family
    if (
        "3d" in t or
        "space" in t or
        "cube" in t or
        "vista" in t or
        "sagittal" in t
    ):
        out.add("FLAIR_3D_NAME")

    # Likely 2D axial family
    if "axial" in t or "tra" in t or "transverse" in t:
        out.add("FLAIR_2D_AXIAL_NAME")

    # Other 2D orientation
    if "coronal" in t or "cor" in t:
        out.add("FLAIR_2D_CORONAL_NAME")

    # Fallback ambiguous FLAIR
    if not out:
        out.add("FLAIR_AMBIG_NAME")

    return out

def choose_one_flair_series(series_names):
    """
    Pick one preferred FLAIR series per session.
    Preference:
      1) named 3D FLAIR
      2) named 2D axial FLAIR
      3) any other FLAIR
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

    scored.sort(key=lambda x: (x[0], x[1].lower()))
    return scored[0][1]

# ---------------------------------------------------------------------
# Recursive DICOM / volume search
# ---------------------------------------------------------------------
def find_dicoms_recursively(series_dir):
    """
    Search series_dir and descendants for readable DICOM series.
    Returns a list of tuples:
        (root, series_id, n_files)
    """
    out = []
    reader = sitk.ImageSeriesReader()

    for root, dirs, files in os.walk(series_dir):
        try:
            sids = reader.GetGDCMSeriesIDs(root)
        except Exception:
            sids = None

        if not sids:
            continue

        for sid in sids:
            try:
                fns = reader.GetGDCMSeriesFileNames(root, sid)
                if fns:
                    out.append((root, sid, len(fns)))
            except Exception:
                pass

    return out

def find_volume_files_recursively(series_dir):
    """
    Search series_dir and descendants for common volumetric files.
    Returns a list of tuples:
        (filepath, file_size_bytes)
    """
    out = []
    for root, dirs, files in os.walk(series_dir):
        for f in files:
            if is_volume_file(f):
                fp = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fp)
                except Exception:
                    sz = -1
                out.append((fp, sz))
    return out

def read_image_recursive(series_dir):
    """
    Try recursively:
      1) nested DICOM series
      2) common volumetric files (.nii/.nii.gz/.mha/.nrrd/.mgz)
    Returns:
      img, err, source_type, chosen_location, aux_info
    """
    # ----- DICOM first -----
    try:
        dicom_candidates = find_dicoms_recursively(series_dir)
        if dicom_candidates:
            chosen_root, chosen_sid, n_files = sorted(
                dicom_candidates, key=lambda x: x[2], reverse=True
            )[0]

            reader = sitk.ImageSeriesReader()
            files = reader.GetGDCMSeriesFileNames(chosen_root, chosen_sid)
            reader.SetFileNames(files)
            img = reader.Execute()
            return img, None, "DICOM_RECURSIVE", chosen_root, {"n_dicom_files": n_files, "series_id": chosen_sid}
    except Exception as e:
        # keep going to volume fallback
        dicom_err = str(e)
    else:
        dicom_err = None

    # ----- Volume file fallback -----
    try:
        vol_candidates = find_volume_files_recursively(series_dir)
        if vol_candidates:
            # choose the largest file
            chosen_fp, chosen_size = sorted(
                vol_candidates, key=lambda x: x[1], reverse=True
            )[0]
            img = sitk.ReadImage(chosen_fp)
            return img, None, "VOLUME_FILE_RECURSIVE", chosen_fp, {"file_size_bytes": chosen_size}
    except Exception as e:
        vol_err = str(e)
    else:
        vol_err = None

    # ----- Nothing worked -----
    if dicom_err and vol_err:
        err = f"no_readable_dicom_or_volume | dicom_err={dicom_err} | vol_err={vol_err}"
    elif dicom_err:
        err = f"no_readable_dicom_or_volume | dicom_err={dicom_err}"
    elif vol_err:
        err = f"no_readable_dicom_or_volume | vol_err={vol_err}"
    else:
        err = "no_dicom_series_or_volume_found_recursive"

    return None, err, None, None, {}

# ---------------------------------------------------------------------
# Geometry classification
# ---------------------------------------------------------------------
def geom_class_from_spacing_and_size(spacing, size, dim):
    """
    spacing: tuple from SimpleITK, usually (sx, sy, sz)
    size:    tuple from SimpleITK, usually (nx, ny, nz)
    dim:     image dimension

    Heuristic labels:
      - LIKELY_3D_VOLUME
      - LIKELY_2D_ACQ_STORED_AS_3D
      - INTERMEDIATE_MIXED
    """
    if dim != 3:
        return f"NON3D_DIM_{dim}"

    sx, sy, sz = spacing
    nx, ny, nz = size

    if min(sx, sy, sz) <= 0:
        return "BAD_GEOM"

    ratio = max(spacing) / min(spacing)

    # too few slices often means bad/partial/nonstandard read
    if nz < 8:
        return "TOO_FEW_SLICES"

    # near-isotropic / proper volumetric-ish
    if ratio <= 1.5 and sz <= 2.0:
        return "LIKELY_3D_VOLUME"

    # thick slice stack / strongly anisotropic
    if sz >= 3.0 or ratio >= 2.5:
        return "LIKELY_2D_ACQ_STORED_AS_3D"

    return "INTERMEDIATE_MIXED"

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    dirs_map = {d.lower(): d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))}

    found = 0
    missing = 0
    no_flair = 0
    unreadable = 0

    name_counter = Counter()
    geom_counter = Counter()
    source_counter = Counter()
    cross_counter = Counter()

    detailed_rows = []

    for sess in sessions:
        d = resolve_session_dir(ROOT, sess, dirs_map=dirs_map)
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

        chosen_path = os.path.join(sess_path, chosen)

        name_labels = flair_subtypes(chosen)
        name_label = "|".join(sorted(name_labels)) if name_labels else "UNKNOWN_NAME"
        name_counter[name_label] += 1

        img, err, source_type, chosen_location, aux_info = read_image_recursive(chosen_path)

        if img is None:
            unreadable += 1
            geom_label = f"UNREADABLE:{err}"
            geom_counter[geom_label] += 1
            source_counter["UNREADABLE"] += 1
            cross_counter[(name_label, geom_label)] += 1

            detailed_rows.append({
                "MR_Session": sess,
                "session_dir": d,
                "series_name": chosen,
                "series_path": chosen_path,
                "name_label": name_label,
                "read_source": "UNREADABLE",
                "chosen_location": chosen_location,
                "geom_label": geom_label,
                "dimension": None,
                "size_x": None,
                "size_y": None,
                "size_z": None,
                "spacing_x": None,
                "spacing_y": None,
                "spacing_z": None,
                "aux_info": str(aux_info),
            })
            continue

        dim = img.GetDimension()
        size = img.GetSize()
        spacing = img.GetSpacing()

        # pad missing slots for dim < 3 or > 3
        size_pad = list(size) + [None] * max(0, 3 - len(size))
        spacing_pad = list(spacing) + [None] * max(0, 3 - len(spacing))

        geom_label = geom_class_from_spacing_and_size(
            spacing=tuple(spacing[:3]) if len(spacing) >= 3 else tuple(spacing),
            size=tuple(size[:3]) if len(size) >= 3 else tuple(size),
            dim=dim
        )

        geom_counter[geom_label] += 1
        source_counter[source_type] += 1
        cross_counter[(name_label, geom_label)] += 1

        detailed_rows.append({
            "MR_Session": sess,
            "session_dir": d,
            "series_name": chosen,
            "series_path": chosen_path,
            "name_label": name_label,
            "read_source": source_type,
            "chosen_location": chosen_location,
            "geom_label": geom_label,
            "dimension": dim,
            "size_x": size_pad[0],
            "size_y": size_pad[1],
            "size_z": size_pad[2],
            "spacing_x": spacing_pad[0],
            "spacing_y": spacing_pad[1],
            "spacing_z": spacing_pad[2],
            "aux_info": str(aux_info),
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

    print("\n=== Read source type ===")
    for k, n in sorted(source_counter.items(), key=lambda x: (-x[1], x[0])):
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

    out_csv = "flair_geometry_check_recursive.csv"
    pd.DataFrame(detailed_rows).to_csv(out_csv, index=False)
    print(f"\nSaved detailed per-session table to: {out_csv}")

if __name__ == "__main__":
    main()
