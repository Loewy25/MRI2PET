#!/usr/bin/env python3
import os
import re
import csv
import math
import argparse
import hashlib
from collections import Counter, defaultdict

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    raise SystemExit(
        "pydicom is required.\n"
        "Install with: python3 -m pip install --user pydicom"
    )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
DIFF_RE = re.compile(r"(diff|diffusion|dwi|dti|hardi|noddi|ivim|dki)", re.IGNORECASE)
ADC_RE = re.compile(r"\badc\b", re.IGNORECASE)
TRACE_RE = re.compile(r"trace", re.IGNORECASE)

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
            # handles pydicom MultiValue too
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
                return "|".join(clean_text(v) for v in x)
        except Exception:
            pass
        x = str(x)
    x = re.sub(r"\s+", " ", x.strip())
    return x

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def fmt_float(x, ndigits=3):
    if x is None:
        return ""
    try:
        return str(round(float(x), ndigits))
    except Exception:
        return ""

def get_attr(ds, name, default=""):
    try:
        return getattr(ds, name, default)
    except Exception:
        return default

def first_nonempty(*vals):
    for v in vals:
        s = clean_text(v)
        if s:
            return s
    return ""

def evenly_spaced(items, k):
    if len(items) <= k:
        return items
    if k <= 1:
        return [items[0]]
    idxs = []
    last = len(items) - 1
    for i in range(k):
        idx = int(round(i * last / (k - 1)))
        idxs.append(idx)
    idxs = sorted(set(idxs))
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

def list_all_files(root):
    out = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in sorted(filenames):
            if fn.startswith("."):
                continue
            out.append(os.path.join(dirpath, fn))
    return out

def find_dicoms_dirs(in_root):
    for dirpath, _, _ in os.walk(in_root, followlinks=True):
        if os.path.basename(dirpath).upper() == "DICOMS":
            yield dirpath

def extract_subject_and_scan_rel(in_root, dicoms_dir):
    # dicoms_dir = IN_ROOT/subject/scan/.../DICOMS
    parent = os.path.dirname(dicoms_dir)  # scan folder
    rel = os.path.relpath(parent, in_root)
    parts = rel.split(os.sep)
    subject = parts[0] if parts else ""
    return subject, rel

def get_pixel_spacing(ds):
    ps = get_attr(ds, "PixelSpacing", None)
    if ps is None:
        return "", ""
    try:
        if len(ps) >= 2:
            return fmt_float(ps[0]), fmt_float(ps[1])
    except Exception:
        pass
    return "", ""

def detect_diffusion(label_text, bvals_found):
    text = label_text.lower()

    if ADC_RE.search(text):
        return True, "ADC"
    if TRACE_RE.search(text):
        return True, "TRACE"
    if DIFF_RE.search(text):
        return True, "DWI/DTI"
    if bvals_found:
        return True, "DWI/DTI"

    return False, ""

def build_signature(row):
    # A practical "protocol fingerprint" for grouping
    label = row["protocol_label"].lower()
    parts = [
        row["manufacturer"].lower(),
        row["model"].lower(),
        row["field_strength"],
        label,
        row["sequence_name"].lower(),
        row["scanning_sequence"].lower(),
        row["sequence_variant"].lower(),
        row["rows"],
        row["cols"],
        row["pixel_spacing_x"],
        row["pixel_spacing_y"],
        row["slice_thickness"],
        row["tr"],
        row["te"],
        row["phase_encode_dir"],
        row["bvals_detected"],
    ]
    text = " | ".join(str(p) for p in parts)
    sig_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return sig_id, text

def summarize_dicoms_folder(in_root, dicoms_dir, sample_n=24):
    all_files = list_all_files(dicoms_dir)
    n_files = len(all_files)
    if n_files == 0:
        return None

    sample_files = evenly_spaced(all_files, min(sample_n, n_files))
    sample_headers = []
    for f in sample_files:
        ds = read_dicom_header(f)
        if ds is not None:
            sample_headers.append((f, ds))

    if not sample_headers:
        return None

    rep_path, rep = sample_headers[0]

    subject, scan_rel = extract_subject_and_scan_rel(in_root, dicoms_dir)

    modality = clean_text(get_attr(rep, "Modality", ""))
    manufacturer = clean_text(get_attr(rep, "Manufacturer", ""))
    model = clean_text(get_attr(rep, "ManufacturerModelName", ""))
    field_strength = fmt_float(get_attr(rep, "MagneticFieldStrength", ""), 1)

    series_desc = clean_text(get_attr(rep, "SeriesDescription", ""))
    protocol_name = clean_text(get_attr(rep, "ProtocolName", ""))
    sequence_name = clean_text(get_attr(rep, "SequenceName", ""))
    scanning_sequence = clean_text(get_attr(rep, "ScanningSequence", ""))
    sequence_variant = clean_text(get_attr(rep, "SequenceVariant", ""))
    image_type = clean_text(get_attr(rep, "ImageType", ""))
    study_desc = clean_text(get_attr(rep, "StudyDescription", ""))
    series_number = clean_text(get_attr(rep, "SeriesNumber", ""))

    rows = clean_text(get_attr(rep, "Rows", ""))
    cols = clean_text(get_attr(rep, "Columns", ""))
    px, py = get_pixel_spacing(rep)
    slice_thickness = fmt_float(get_attr(rep, "SliceThickness", ""), 3)
    tr = fmt_float(get_attr(rep, "RepetitionTime", ""), 3)
    te = fmt_float(get_attr(rep, "EchoTime", ""), 3)
    flip_angle = fmt_float(get_attr(rep, "FlipAngle", ""), 3)
    phase_encode_dir = clean_text(get_attr(rep, "InPlanePhaseEncodingDirection", ""))

    series_uids = set()
    bvals = set()
    protocol_names = set()
    series_descs = set()

    for _, ds in sample_headers:
        uid = clean_text(get_attr(ds, "SeriesInstanceUID", ""))
        if uid:
            series_uids.add(uid)

        p = clean_text(get_attr(ds, "ProtocolName", ""))
        s = clean_text(get_attr(ds, "SeriesDescription", ""))
        if p:
            protocol_names.add(p)
        if s:
            series_descs.add(s)

        b = get_attr(ds, "DiffusionBValue", None)
        b = to_float(b)
        if b is not None:
            bvals.add(int(round(b)))

    bvals_sorted = sorted(bvals)
    bvals_text = ",".join(str(b) for b in bvals_sorted)

    protocol_label = first_nonempty(protocol_name, series_desc, sequence_name, scanning_sequence, os.path.basename(scan_rel))

    label_text = " ".join([
        protocol_label,
        protocol_name,
        series_desc,
        sequence_name,
        scanning_sequence,
        sequence_variant,
        image_type,
        scan_rel,
    ])

    is_diffusion, diff_class = detect_diffusion(label_text, bvals_sorted)

    row = {
        "subject": subject,
        "scan_rel": scan_rel,
        "dicoms_dir": dicoms_dir,
        "representative_dicom": rep_path,
        "n_files_in_dicoms": n_files,
        "n_sampled_headers": len(sample_headers),
        "n_series_uid_in_sample": len(series_uids),
        "modality": modality,
        "study_description": study_desc,
        "series_number": series_number,
        "protocol_name": protocol_name,
        "series_description": series_desc,
        "protocol_label": protocol_label,
        "sequence_name": sequence_name,
        "scanning_sequence": scanning_sequence,
        "sequence_variant": sequence_variant,
        "image_type": image_type,
        "manufacturer": manufacturer,
        "model": model,
        "field_strength": field_strength,
        "rows": rows,
        "cols": cols,
        "pixel_spacing_x": px,
        "pixel_spacing_y": py,
        "slice_thickness": slice_thickness,
        "tr": tr,
        "te": te,
        "flip_angle": flip_angle,
        "phase_encode_dir": phase_encode_dir,
        "bvals_detected": bvals_text,
        "is_diffusion": int(is_diffusion),
        "diffusion_class": diff_class,
    }

    if is_diffusion:
        sig_id, sig_text = build_signature(row)
    else:
        sig_id, sig_text = "", ""

    row["signature_id"] = sig_id
    row["signature_text"] = sig_text

    return row

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audit DICOM scan folders and summarize diffusion protocols.")
    parser.add_argument("--in_root", required=True, help="Root folder containing top-level subject folders")
    parser.add_argument("--out_dir", required=True, help="Where CSV summaries will be written")
    parser.add_argument("--sample_n", type=int, default=24, help="How many DICOM files to sample per DICOMS folder")
    args = parser.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    details_csv = os.path.join(out_dir, "dicom_scan_details.csv")
    protocol_csv = os.path.join(out_dir, "diffusion_protocol_summary.csv")

    subject_dirs = [
        d for d in sorted(os.listdir(in_root))
        if os.path.isdir(os.path.join(in_root, d))
    ]
    total_subjects = len(subject_dirs)

    print(f"[INFO] IN_ROOT: {in_root}")
    print(f"[INFO] OUT_DIR: {out_dir}")
    print(f"[INFO] top-level subjects: {total_subjects}")

    all_rows = []
    n_dicoms_dirs = 0
    unreadable = 0

    for dicoms_dir in find_dicoms_dirs(in_root):
        n_dicoms_dirs += 1
        row = summarize_dicoms_folder(in_root, dicoms_dir, sample_n=args.sample_n)
        if row is None:
            unreadable += 1
            continue
        all_rows.append(row)

        if n_dicoms_dirs % 200 == 0:
            print(f"[INFO] scanned {n_dicoms_dirs} DICOMS folders...")

    if not all_rows:
        print("[WARN] No readable DICOM scan folders found.")
        return

    # Write per-scan details CSV
    fieldnames = list(all_rows[0].keys())
    with open(details_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Diffusion-only summary
    diff_rows = [r for r in all_rows if int(r["is_diffusion"]) == 1]

    sig_scan_count = Counter()
    sig_subjects = defaultdict(set)
    sig_example = {}
    sig_rows = {}

    label_scan_count = Counter()
    label_subjects = defaultdict(set)

    manufacturer_count = Counter()
    diff_class_count = Counter()

    for r in diff_rows:
        sig = r["signature_id"]
        lbl = r["protocol_label"]
        subj = r["subject"]

        sig_scan_count[sig] += 1
        sig_subjects[sig].add(subj)
        if sig not in sig_example:
            sig_example[sig] = r["scan_rel"]
            sig_rows[sig] = r

        label_scan_count[lbl] += 1
        label_subjects[lbl].add(subj)

        manufacturer_count[r["manufacturer"]] += 1
        diff_class_count[r["diffusion_class"]] += 1

    # Write protocol summary CSV
    protocol_rows = []
    for sig, count in sig_scan_count.most_common():
        r = sig_rows[sig]
        protocol_rows.append({
            "signature_id": sig,
            "n_scans": count,
            "n_subjects": len(sig_subjects[sig]),
            "protocol_label": r["protocol_label"],
            "diffusion_class": r["diffusion_class"],
            "manufacturer": r["manufacturer"],
            "model": r["model"],
            "field_strength": r["field_strength"],
            "sequence_name": r["sequence_name"],
            "scanning_sequence": r["scanning_sequence"],
            "sequence_variant": r["sequence_variant"],
            "rows": r["rows"],
            "cols": r["cols"],
            "pixel_spacing_x": r["pixel_spacing_x"],
            "pixel_spacing_y": r["pixel_spacing_y"],
            "slice_thickness": r["slice_thickness"],
            "tr": r["tr"],
            "te": r["te"],
            "phase_encode_dir": r["phase_encode_dir"],
            "bvals_detected": r["bvals_detected"],
            "example_scan_rel": sig_example[sig],
            "signature_text": r["signature_text"],
        })

    if protocol_rows:
        with open(protocol_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(protocol_rows[0].keys()))
            writer.writeheader()
            writer.writerows(protocol_rows)

    # Terminal summary
    subjects_with_diff = len(set(r["subject"] for r in diff_rows))
    mr_rows = [r for r in all_rows if r["modality"].upper() == "MR"]

    print()
    print("==================== END-TO-END SUMMARY ====================")
    print(f"[INFO] top-level subjects                  : {total_subjects}")
    print(f"[INFO] DICOMS folders found                : {n_dicoms_dirs}")
    print(f"[INFO] readable scan folders               : {len(all_rows)}")
    print(f"[INFO] unreadable/empty DICOMS folders     : {unreadable}")
    print(f"[INFO] MR scan folders                     : {len(mr_rows)}")
    print(f"[INFO] diffusion-related scan folders      : {len(diff_rows)}")
    print(f"[INFO] subjects with >=1 diffusion scan    : {subjects_with_diff} / {total_subjects}")

    if diff_rows:
        print(f"[INFO] unique diffusion labels            : {len(label_scan_count)}")
        print(f"[INFO] unique diffusion signatures        : {len(sig_scan_count)}")

        print()
        print("[INFO] diffusion classes:")
        for k, v in diff_class_count.most_common():
            print(f"  {k:12s}: {v}")

        print()
        print("[INFO] manufacturers for diffusion scans:")
        for k, v in manufacturer_count.most_common():
            print(f"  {k or '<missing>':20s}: {v}")

        print()
        print("[INFO] top diffusion labels:")
        for lbl, count in label_scan_count.most_common(15):
            print(f"  {lbl[:70]:70s}  scans={count:4d}  subjects={len(label_subjects[lbl]):4d}")

        print()
        print("[INFO] top diffusion protocol signatures:")
        for sig, count in sig_scan_count.most_common(15):
            r = sig_rows[sig]
            print(f"  signature={sig}  scans={count:4d}  subjects={len(sig_subjects[sig]):4d}")
            print(f"    label      : {r['protocol_label']}")
            print(f"    scanner    : {r['manufacturer']} | {r['model']} | {r['field_strength']}T")
            print(f"    size       : {r['rows']}x{r['cols']} | "
                  f"{r['pixel_spacing_x']}x{r['pixel_spacing_y']} | thk={r['slice_thickness']}")
            print(f"    TR/TE      : {r['tr']} / {r['te']}")
            print(f"    PE dir     : {r['phase_encode_dir']}")
            print(f"    bvals      : {r['bvals_detected'] or '<not found in sampled headers>'}")
            print(f"    example    : {sig_example[sig]}")

    print()
    print(f"[OK] wrote: {details_csv}")
    if protocol_rows:
        print(f"[OK] wrote: {protocol_csv}")
    else:
        print("[WARN] no diffusion protocols found, so diffusion_protocol_summary.csv was not written")
    print("============================================================")

if __name__ == "__main__":
    main()
