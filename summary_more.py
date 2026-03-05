#!/usr/bin/env python3
import os
import re
import csv
import hashlib
from collections import Counter, defaultdict

import pandas as pd

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    raise SystemExit("Please install pydicom: python3 -m pip install --user pydicom")


# ============================================================
# Config
# ============================================================
META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
COL = "MR_Session"
ROOT = "/ceph/chpc/mapped/benz04_kari/scans"

OUT_DIR = "./modality_dicom_audit"
SAMPLE_N = 16   # read up to this many DICOM headers per DICOMS folder


# ============================================================
# Naming rules (kept aligned with your previous script)
# ============================================================
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


def classify_series_name(series_name: str):
    """
    Return a list of modalities matched by the folder name.
    Uses the same logic as your previous code, but per series folder.
    """
    if IGNORE.search(series_name):
        return []

    t = toks(series_name)
    mods = []

    if "flair" in t:
        mods.append("FLAIR")

    if (("t2" in t and "star" in t) or ("t2star" in t) or ("swi" in t)):
        mods.append("T2STAR_SWI")

    if ("dwi" in t) or ("dti" in t) or ("dbsi" in t) or ("mddw" in t) or (("ep2d" in t) and ("diff" in t)) or ("diff" in t):
        mods.append("DIFFUSION")

    if ("rsfmri" in t) or ("bold" in t) or (("ep2d" in t) and ("bold" in t)) or (("resting" in t) and ("state" in t) and ("fmri" in t)):
        mods.append("RSFMRI_BOLD")

    if ("asl" in t) or ("pasl" in t) or ("pcasl" in t) or ("relcbf" in t):
        mods.append("ASL")

    if "tof" in t:
        mods.append("TOF")

    return mods


# ============================================================
# Helpers
# ============================================================
def norm_session_name(s: str):
    s = str(s).strip().lower()
    s = s.replace(".zip", "")
    s = re.sub(r"[\s\-_]+", "_", s)
    return s


def resolve_session_folder(session_name: str, dir_map: dict):
    return dir_map.get(norm_session_name(session_name))


def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            x = x.decode(errors="ignore")
        except Exception:
            x = str(x)
    elif not isinstance(x, str):
        try:
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
                return "|".join(clean_text(v) for v in x)
        except Exception:
            pass
        x = str(x)
    return re.sub(r"\s+", " ", x.strip())


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def fmt_float(x, ndigits=3):
    v = to_float(x)
    if v is None:
        return ""
    return str(round(v, ndigits))


def list_files(root):
    out = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if fn.startswith("."):
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def evenly_spaced(items, k):
    if len(items) <= k:
        return items
    if k <= 1:
        return [items[0]]
    last = len(items) - 1
    idxs = sorted(set(int(round(i * last / (k - 1))) for i in range(k)))
    return [items[i] for i in idxs]


def read_dicom_header(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=False)
    except InvalidDicomError:
        try:
            return pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            return None
    except Exception:
        return None


def find_dicoms_dirs(series_dir):
    hits = []
    for dirpath, _, _ in os.walk(series_dir, followlinks=True):
        if os.path.basename(dirpath).upper() == "DICOMS":
            hits.append(dirpath)
    return sorted(hits)


def get_pixel_spacing(ds):
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return "", ""
    try:
        if len(ps) >= 2:
            return fmt_float(ps[0]), fmt_float(ps[1])
    except Exception:
        pass
    return "", ""


def build_signature(row):
    """
    Practical grouping key for 'same-ish protocol/series'.
    Not perfect truth, but very useful.
    """
    parts = [
        row["modality"],
        row["manufacturer"],
        row["model"],
        row["field_strength"],
        row["protocol_name"],
        row["series_description"],
        row["sequence_name"],
        row["scanning_sequence"],
        row["sequence_variant"],
        row["rows"],
        row["cols"],
        row["pixel_spacing_x"],
        row["pixel_spacing_y"],
        row["slice_thickness"],
        row["tr"],
        row["te"],
        row["flip_angle"],
        row["bvals_detected"],
    ]
    txt = " | ".join(str(x).strip().lower() for x in parts)
    sig = hashlib.md5(txt.encode("utf-8")).hexdigest()[:10]
    return sig, txt


def summarize_dicoms_folder(session, session_dir_name, session_path, series_name, modality, dicoms_dir, sample_n):
    all_files = list_files(dicoms_dir)
    n_files = len(all_files)

    if n_files == 0:
        return {
            "session": session,
            "session_dir": session_dir_name,
            "series_folder": series_name,
            "modality": modality,
            "series_path": os.path.join(session_path, series_name),
            "dicoms_dir": dicoms_dir,
            "status": "empty_dicoms",
            "basic_usable": 0,
            "n_files_in_dicoms": 0,
            "n_sampled_headers": 0,
            "n_series_uid_in_sample": 0,
            "representative_dicom": "",
            "protocol_name": "",
            "series_description": "",
            "sequence_name": "",
            "scanning_sequence": "",
            "sequence_variant": "",
            "image_type": "",
            "manufacturer": "",
            "model": "",
            "field_strength": "",
            "rows": "",
            "cols": "",
            "pixel_spacing_x": "",
            "pixel_spacing_y": "",
            "slice_thickness": "",
            "tr": "",
            "te": "",
            "flip_angle": "",
            "bvals_detected": "",
            "signature_id": "",
            "signature_text": "",
        }

    sample_files = evenly_spaced(all_files, min(sample_n, n_files))
    headers = []
    for f in sample_files:
        ds = read_dicom_header(f)
        if ds is not None:
            headers.append((f, ds))

    if not headers:
        return {
            "session": session,
            "session_dir": session_dir_name,
            "series_folder": series_name,
            "modality": modality,
            "series_path": os.path.join(session_path, series_name),
            "dicoms_dir": dicoms_dir,
            "status": "unreadable_dicoms",
            "basic_usable": 0,
            "n_files_in_dicoms": n_files,
            "n_sampled_headers": 0,
            "n_series_uid_in_sample": 0,
            "representative_dicom": "",
            "protocol_name": "",
            "series_description": "",
            "sequence_name": "",
            "scanning_sequence": "",
            "sequence_variant": "",
            "image_type": "",
            "manufacturer": "",
            "model": "",
            "field_strength": "",
            "rows": "",
            "cols": "",
            "pixel_spacing_x": "",
            "pixel_spacing_y": "",
            "slice_thickness": "",
            "tr": "",
            "te": "",
            "flip_angle": "",
            "bvals_detected": "",
            "signature_id": "",
            "signature_text": "",
        }

    rep_path, rep = headers[0]

    series_uids = set()
    bvals = set()
    for _, ds in headers:
        uid = clean_text(getattr(ds, "SeriesInstanceUID", ""))
        if uid:
            series_uids.add(uid)
        b = to_float(getattr(ds, "DiffusionBValue", None))
        if b is not None:
            bvals.add(int(round(b)))

    px, py = get_pixel_spacing(rep)

    row = {
        "session": session,
        "session_dir": session_dir_name,
        "series_folder": series_name,
        "modality": modality,
        "series_path": os.path.join(session_path, series_name),
        "dicoms_dir": dicoms_dir,
        "n_files_in_dicoms": n_files,
        "n_sampled_headers": len(headers),
        "n_series_uid_in_sample": len(series_uids),
        "representative_dicom": rep_path,
        "protocol_name": clean_text(getattr(rep, "ProtocolName", "")),
        "series_description": clean_text(getattr(rep, "SeriesDescription", "")),
        "sequence_name": clean_text(getattr(rep, "SequenceName", "")),
        "scanning_sequence": clean_text(getattr(rep, "ScanningSequence", "")),
        "sequence_variant": clean_text(getattr(rep, "SequenceVariant", "")),
        "image_type": clean_text(getattr(rep, "ImageType", "")),
        "manufacturer": clean_text(getattr(rep, "Manufacturer", "")),
        "model": clean_text(getattr(rep, "ManufacturerModelName", "")),
        "field_strength": fmt_float(getattr(rep, "MagneticFieldStrength", ""), 1),
        "rows": clean_text(getattr(rep, "Rows", "")),
        "cols": clean_text(getattr(rep, "Columns", "")),
        "pixel_spacing_x": px,
        "pixel_spacing_y": py,
        "slice_thickness": fmt_float(getattr(rep, "SliceThickness", ""), 3),
        "tr": fmt_float(getattr(rep, "RepetitionTime", ""), 3),
        "te": fmt_float(getattr(rep, "EchoTime", ""), 3),
        "flip_angle": fmt_float(getattr(rep, "FlipAngle", ""), 3),
        "bvals_detected": ",".join(str(x) for x in sorted(bvals)),
    }

    # simple usability rule:
    # - readable
    # - sampled headers are from one series only (or UID missing everywhere)
    if len(series_uids) > 1:
        row["status"] = "mixed_series_in_sample"
        row["basic_usable"] = 0
    else:
        row["status"] = "ok"
        row["basic_usable"] = 1

    sig, sig_txt = build_signature(row)
    row["signature_id"] = sig
    row["signature_text"] = sig_txt

    return row


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    details_csv = os.path.join(OUT_DIR, "modality_dicom_details.csv")
    protocol_csv = os.path.join(OUT_DIR, "modality_protocol_summary.csv")
    availability_csv = os.path.join(OUT_DIR, "modality_availability_summary.csv")

    df = pd.read_csv(META)
    sessions = sorted(set(df[COL].dropna().astype(str).str.strip()))

    # index session directories
    dir_map = {}
    for d in os.listdir(ROOT):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):
            dir_map[norm_session_name(d)] = d

    print(f"[INFO] META: {META}")
    print(f"[INFO] ROOT: {ROOT}")
    print(f"[INFO] total MR_Session in CSV: {len(sessions)}")

    found_sessions = 0
    missing_sessions = 0

    rows = []

    # per-modality session bookkeeping
    session_named = defaultdict(set)       # session has at least one series folder name matching modality
    session_dicoms = defaultdict(set)      # session has at least one DICOMS folder for that modality
    session_readable = defaultdict(set)    # session has at least one readable DICOMS folder
    session_usable = defaultdict(set)      # session has at least one basically usable scan

    for idx, sess in enumerate(sessions, 1):
        actual_dir = resolve_session_folder(sess, dir_map)
        if not actual_dir:
            missing_sessions += 1
            continue

        found_sessions += 1
        session_path = os.path.join(ROOT, actual_dir)

        try:
            series_folders = [
                x for x in os.listdir(session_path)
                if os.path.isdir(os.path.join(session_path, x))
            ]
        except Exception:
            continue

        for series_name in series_folders:
            mods = classify_series_name(series_name)
            if not mods:
                continue

            series_dir = os.path.join(session_path, series_name)
            dicoms_dirs = find_dicoms_dirs(series_dir)

            for mod in mods:
                session_named[mod].add(sess)

                if not dicoms_dirs:
                    rows.append({
                        "session": sess,
                        "session_dir": actual_dir,
                        "series_folder": series_name,
                        "modality": mod,
                        "series_path": series_dir,
                        "dicoms_dir": "",
                        "status": "no_dicoms_dir",
                        "basic_usable": 0,
                        "n_files_in_dicoms": 0,
                        "n_sampled_headers": 0,
                        "n_series_uid_in_sample": 0,
                        "representative_dicom": "",
                        "protocol_name": "",
                        "series_description": "",
                        "sequence_name": "",
                        "scanning_sequence": "",
                        "sequence_variant": "",
                        "image_type": "",
                        "manufacturer": "",
                        "model": "",
                        "field_strength": "",
                        "rows": "",
                        "cols": "",
                        "pixel_spacing_x": "",
                        "pixel_spacing_y": "",
                        "slice_thickness": "",
                        "tr": "",
                        "te": "",
                        "flip_angle": "",
                        "bvals_detected": "",
                        "signature_id": "",
                        "signature_text": "",
                    })
                    continue

                for dicoms_dir in dicoms_dirs:
                    session_dicoms[mod].add(sess)

                    row = summarize_dicoms_folder(
                        session=sess,
                        session_dir_name=actual_dir,
                        session_path=session_path,
                        series_name=series_name,
                        modality=mod,
                        dicoms_dir=dicoms_dir,
                        sample_n=SAMPLE_N,
                    )
                    rows.append(row)

                    if row["status"] in ("ok", "mixed_series_in_sample"):
                        session_readable[mod].add(sess)
                    if int(row["basic_usable"]) == 1:
                        session_usable[mod].add(sess)

        if idx % 100 == 0:
            print(f"[INFO] processed {idx} / {len(sessions)} sessions...")

    if not rows:
        print("[WARN] no modality-matched series found.")
        return

    # write details CSV
    fieldnames = list(rows[0].keys())
    with open(details_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary tables
    readable_rows = [r for r in rows if r["status"] in ("ok", "mixed_series_in_sample")]
    usable_rows = [r for r in rows if int(r["basic_usable"]) == 1]

    # protocol summary per modality + signature
    sig_counts = Counter()
    sig_sessions = defaultdict(set)
    sig_example = {}
    sig_first_row = {}

    for r in usable_rows:
        key = (r["modality"], r["signature_id"])
        sig_counts[key] += 1
        sig_sessions[key].add(r["session"])
        if key not in sig_example:
            sig_example[key] = r["series_path"]
            sig_first_row[key] = r

    protocol_rows = []
    for (mod, sig), n in sig_counts.most_common():
        r = sig_first_row[(mod, sig)]
        protocol_rows.append({
            "modality": mod,
            "signature_id": sig,
            "n_scans": n,
            "n_sessions": len(sig_sessions[(mod, sig)]),
            "protocol_name": r["protocol_name"],
            "series_description": r["series_description"],
            "sequence_name": r["sequence_name"],
            "manufacturer": r["manufacturer"],
            "model": r["model"],
            "field_strength": r["field_strength"],
            "rows": r["rows"],
            "cols": r["cols"],
            "pixel_spacing_x": r["pixel_spacing_x"],
            "pixel_spacing_y": r["pixel_spacing_y"],
            "slice_thickness": r["slice_thickness"],
            "tr": r["tr"],
            "te": r["te"],
            "flip_angle": r["flip_angle"],
            "bvals_detected": r["bvals_detected"],
            "example_series_path": sig_example[(mod, sig)],
            "signature_text": r["signature_text"],
        })

    if protocol_rows:
        with open(protocol_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(protocol_rows[0].keys()))
            w.writeheader()
            w.writerows(protocol_rows)

    # availability summary per modality
    availability_rows = []
    for mod in MODALITIES:
        mod_rows = [r for r in rows if r["modality"] == mod]
        mod_readable = [r for r in readable_rows if r["modality"] == mod]
        mod_usable = [r for r in usable_rows if r["modality"] == mod]

        mod_sig_counts = Counter(r["signature_id"] for r in mod_usable if r["signature_id"])
        top_sig = ""
        top_sig_n = 0
        top_sig_pct = 0.0
        if mod_sig_counts:
            top_sig, top_sig_n = mod_sig_counts.most_common(1)[0]
            top_sig_pct = 100.0 * top_sig_n / len(mod_usable)

        availability_rows.append({
            "modality": mod,
            "sessions_named_match": len(session_named[mod]),
            "sessions_with_dicoms": len(session_dicoms[mod]),
            "sessions_with_readable_dicoms": len(session_readable[mod]),
            "sessions_with_basic_usable_scan": len(session_usable[mod]),
            "scan_rows_total": len(mod_rows),
            "scan_rows_readable": len(mod_readable),
            "scan_rows_basic_usable": len(mod_usable),
            "unique_signatures_among_usable": len(mod_sig_counts),
            "top_signature_id": top_sig,
            "top_signature_count": top_sig_n,
            "top_signature_pct_among_usable": round(top_sig_pct, 2),
        })

    with open(availability_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(availability_rows[0].keys()))
        w.writeheader()
        w.writerows(availability_rows)

    # terminal summary
    print("\n=== Modality availability from folder names + DICOM verification ===")
    print(f"Total MR_Session in CSV: {len(sessions)}")
    print(f"Found session folders:   {found_sessions}")
    print(f"Missing folders:         {missing_sessions}\n")

    for row in availability_rows:
        mod = row["modality"]
        named = row["sessions_named_match"]
        dicoms = row["sessions_with_dicoms"]
        readable = row["sessions_with_readable_dicoms"]
        usable = row["sessions_with_basic_usable_scan"]
        nsig = row["unique_signatures_among_usable"]
        top_pct = row["top_signature_pct_among_usable"]

        print(f"{mod:12s}: "
              f"name-match={named:4d} | "
              f"dicoms={dicoms:4d} | "
              f"readable={readable:4d} | "
              f"usable={usable:4d} | "
              f"unique_sig={nsig:3d} | "
              f"top_sig_pct={top_pct:5.1f}%")

    print("\n[OK] wrote:")
    print(f"  {details_csv}")
    if protocol_rows:
        print(f"  {protocol_csv}")
    else:
        print(f"  {protocol_csv}  (not written; no usable protocol rows)")
    print(f"  {availability_csv}")


if __name__ == "__main__":
    main()
