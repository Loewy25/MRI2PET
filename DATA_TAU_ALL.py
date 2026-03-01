#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
All-TAU PET processing pipeline.

Features:
- Copy T1.nii.gz
- Compute T1001->T1 (FLIRT) and apply to PET -> PET_in_T1.nii.gz
- Make aseg_brainmask.nii.gz from aseg.mgz, with geometry checks:
  * if aseg/aparc match T1 shape+affine: use as-is
  * otherwise: compute FS->T1 rigid transform (mri_robust_register)
    and apply to labels with nearest-neighbor (mri_vol2vol --lta)
- Make ROI_* from aparc+aseg (and aseg for hippocampus) in T1 space
- Make mask_basalganglia.nii.gz and mask_parenchyma_noBG.nii.gz in T1 space
- Make mask_cortex.nii.gz in T1 space
- If anything is missing/mismatched, print explicit debug lines and skip that subject.
- No backups; re-runs overwrite.

CLI:
  --part N
  --num-parts M

Example:
  python DATA_ALL_DATA.py --part 10 --num-parts 10

Dedup rule:
  If CSV has multiple rows with the same TAU_PET_Session, keep only one:
    - choose higher CDR first
    - if tie, higher Centiloid/Amyloid numeric
    - if tie, higher Braak
    - if tie, deterministic pick

This dedup happens BEFORE partitioning, so concurrent runs do not clash on
the same TAU_PET_Session output.
"""

import os
import re
import glob
import csv
import shutil
import subprocess
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib


# =================== CONFIG ===================
BASE_ROOT = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT  = os.path.join(BASE_ROOT, "pup")
FS_ROOT   = os.path.join(BASE_ROOT, "freesurfers")

OUT_ROOT  = "/scratch/l.peiwang/kari_all"
LUT_PATH  = "/scratch/l.peiwang/FreeSurferColorLUT.txt"
CSV_PATH  = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"

RUN_LIMIT   = None
ATOL_AFFINE = 1e-4
DEBUG       = True

# Labels to keep for a clean brain parenchyma mask (GM+WM, cerebellum,
# subcortical, brainstem, VentralDC)
KEEP_LABELS = {
    2, 41, 3, 42, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51,
    13, 52, 17, 53, 18, 54, 26, 58, 28, 60, 16
}

# Basal ganglia (strict): Caudate, Putamen, Pallidum, Accumbens (L/R)
BG_IDS_STRICT = {11, 50, 12, 51, 13, 52, 26, 58}
INCLUDE_VENTRAL_DC = False
BG_IDS = (BG_IDS_STRICT | {28, 60}) if INCLUDE_VENTRAL_DC else BG_IDS_STRICT

# aseg cortex labels: Left/Right-Cerebral-Cortex
CORTEX_LABELS = {3, 42}

# ROI definitions (Desikan–Killiany names)
TEMPORAL_BASE = [
    "superiortemporal", "middletemporal", "inferiortemporal",
    "fusiform", "transversetemporal", "temporalpole", "bankssts"
]
LIMBIC_BASE = [
    "posteriorcingulate", "isthmuscingulate",
    "caudalanteriorcingulate", "rostralanteriorcingulate",
    "parahippocampal", "entorhinal"
]

random.seed(42)


# =================== LOGGING ===================
def log(msg, level="INFO"):
    if DEBUG or level in ("WARN", "ERROR"):
        print(f"[{level}] {msg}", flush=True)


# =================== HELPERS ===================
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())


def _parse_cnda_timestamp_from_name(name):
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def sniff_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        try:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)


def ci_get(row, key, default=""):
    """Case-insensitive dict get for CSV rows."""
    kl = key.lower()
    for k in row.keys():
        if k.lower() == kl:
            v = row.get(k)
            return v if v is not None else default
    return default


# ---------- PUP discovery ----------
def find_pup_subject_dir_from_tau_session(pup_root, tau_session):
    """Find a PUP subject directory whose name contains the TAU_PET_Session."""
    if not tau_session:
        return None
    cands = []
    ts_l = tau_session.lower()
    for d in glob.glob(os.path.join(pup_root, "*")):
        if not os.path.isdir(d):
            continue
        if ts_l in os.path.basename(d).lower():
            cands.append(d)
    cands.sort()
    return cands[-1] if cands else None


def find_pup_subject_dir_from_pupid(pup_root, pup_id):
    """
    Fallback: search for a CNDA folder named like TAU_PUP_ID under any subject,
    then return its subject dir.
    """
    if not pup_id:
        return None
    hits = glob.glob(os.path.join(pup_root, "*", pup_id))
    hits += glob.glob(os.path.join(pup_root, "*", pup_id + "*"))
    if hits:
        return os.path.dirname(os.path.dirname(os.path.join(hits[-1], "")))
    return None


def find_pup_nifti_dir(pup_dir):
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ")) if pup_dir else []
    if not hits:
        return None
    hits.sort()
    return hits[-1]


def choose_native_t1(nifti_dir):
    p = os.path.join(nifti_dir, "T1.nii.gz")
    return p if os.path.exists(p) else None


def choose_t1001_strict(nifti_dir):
    p = os.path.join(nifti_dir, "T1001.nii.gz")
    return p if os.path.exists(p) else None


def choose_pet_strict(nifti_dir):
    matches = glob.glob(os.path.join(nifti_dir, "*_msum_SUVR.nii.gz"))
    if not matches:
        return None
    matches.sort()
    return matches[0]


# ---------- FreeSurfer discovery by MR_Session ----------
def find_fs_subject_dir_by_mrsession(fs_root, mr_session):
    """Find FS subject dir whose name/path contains MR_Session."""
    if not mr_session:
        return None

    mr_l = mr_session.lower()
    cands = []

    # pass 1: immediate children
    for d in glob.glob(os.path.join(fs_root, "*")):
        if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
            cands.append(d)

    # pass 2: search deeper
    if not cands:
        for d in glob.glob(os.path.join(fs_root, "*", "*")):
            if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
                cands.append(d)

    # pass 3: recursive
    if not cands:
        for root, dirs, _ in os.walk(fs_root):
            for dd in dirs:
                if mr_l in dd.lower():
                    cands.append(os.path.join(root, dd))

    if not cands:
        return None

    def rank(name):
        n = name.lower()
        if "mri" in n:
            return 0
        if "mmr" in n:
            return 1
        return 2

    cands.sort(key=lambda p: rank(os.path.basename(p)))
    return cands[0]


def _run_dirs(fs_subject_dir):
    return glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))


def _find_label_in_run(run_dir, fname):
    hits = glob.glob(os.path.join(run_dir, "DATA", "*", "mri", fname))
    hits.sort()
    return hits[-1] if hits else None


def find_fs_labels_closest(fs_subject_dir, target_dt):
    runs = _run_dirs(fs_subject_dir)
    best, best_diff = None, float("inf")

    for rd in runs:
        dt = _parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if diff < best_diff:
            best, best_diff = rd, diff

    if best:
        aseg = _find_label_in_run(best, "aseg.mgz")
        aparc = _find_label_in_run(best, "aparc+aseg.mgz")
        if aseg and aparc:
            return aseg, aparc, best

    # fallback search across subject tree
    found_aseg, found_aparc = None, None
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files and not found_aseg:
            found_aseg = os.path.join(root, "aseg.mgz")
        if "aparc+aseg.mgz" in files and not found_aparc:
            found_aparc = os.path.join(root, "aparc+aseg.mgz")

    return found_aseg, found_aparc, None


# ---------- Utils ----------
def _ensure_3d_nifti(in_path, out_dir, tag):
    try:
        img = nib.load(in_path)
        if img.ndim == 4:
            out = os.path.join(out_dir, f"_{tag}_3d_tmp.nii.gz")
            data = np.asanyarray(img.dataobj)[..., 0]
            nib.save(nib.Nifti1Image(data, img.affine, img.header), out)
            return out, True
    except Exception:
        pass
    return in_path, False


def make_aseg_mask_nifti(aseg_path, out_path):
    """Create NIfTI mask from aseg (labels in KEEP_LABELS)."""
    aseg_img = nib.load(aseg_path)
    lab = np.asanyarray(aseg_img.dataobj)
    mask = np.isin(lab, list(KEEP_LABELS)).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, aseg_img.affine), out_path)
    return out_path


def shapes_affines_match(a_path, b_path, atol=ATOL_AFFINE):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and aff_ok, shape_ok, aff_ok, ia.shape, ib.shape, ia.affine, ib.affine


def resample_label_to_target(label_path, target_path, out_path):
    """
    Use FreeSurfer mri_vol2vol (nearest) with --regheader to map label -> target grid.
    """
    if not shutil.which("mri_vol2vol"):
        raise RuntimeError("mri_vol2vol not found. Source your FreeSurfer env.")

    cmd = [
        "mri_vol2vol",
        "--mov", label_path,
        "--targ", target_path,
        "--regheader",
        "--interp", "nearest",
        "--o", out_path
    ]
    log("RUN: " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-20:])
        log("mri_vol2vol stderr (tail):\n" + tail, level="ERROR")
        raise RuntimeError(f"mri_vol2vol failed: {r.returncode}")
    return out_path


def read_fs_lut(lut_path=LUT_PATH):
    lut = {}
    with open(lut_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[1]
                lut[idx] = name
                lut[name] = idx
    return lut


def ids_for_hemi_names(base_names, lut):
    ids = []
    for bn in base_names:
        for hemi in ("lh", "rh"):
            key = f"ctx-{hemi}-{bn}"
            if key in lut:
                ids.append(lut[key])
            else:
                log(f"LUT missing: {key}", level="WARN")
    return ids


def write_mask_from_labels(label_path, id_list, out_path, dtype=np.uint8):
    img = nib.load(label_path)
    data = np.asanyarray(img.dataobj)
    mask = np.isin(data, list(id_list)).astype(dtype)
    nib.save(nib.Nifti1Image(mask, img.affine), out_path)
    return int(mask.sum())


# ---------------- Registration helpers ----------------
def pick_fs_brain(fs_subject_dir, used_run):
    """Prefer brain.mgz from the chosen FS run; fallback to subject's mri/."""
    cand_patterns = []

    if used_run:
        cand_patterns += [
            os.path.join(used_run, "DATA", "*", "mri", "brain.mgz"),
            os.path.join(used_run, "DATA", "*", "mri", "T1.mgz"),
            os.path.join(used_run, "DATA", "*", "mri", "orig.mgz"),
        ]

    cand_patterns += [
        os.path.join(fs_subject_dir, "mri", "brain.mgz"),
        os.path.join(fs_subject_dir, "mri", "T1.mgz"),
        os.path.join(fs_subject_dir, "mri", "orig.mgz"),
    ]

    for pat in cand_patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None


def compute_fs2t1_lta(fs_ref, t1_path, out_lta):
    """Compute a rigid FS(conformed)->T1 transform."""
    if not shutil.which("mri_robust_register"):
        raise RuntimeError("mri_robust_register not found in PATH.")

    cmd = [
        "mri_robust_register",
        "--mov", fs_ref,
        "--dst", t1_path,
        "--lta", out_lta,
        "--satit",
        "--iscale",
        "--maxit", "200"
    ]
    log("RUN: " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-30:])
        log("mri_robust_register stderr (tail):\n" + tail, level="ERROR")
        raise RuntimeError(f"mri_robust_register failed ({r.returncode})")
    return out_lta


def resample_label_with_lta(label_path, target_path, lta_path, out_path):
    """Apply FS->T1 LTA to a discrete label map using nearest neighbor."""
    if not shutil.which("mri_vol2vol"):
        raise RuntimeError("mri_vol2vol not found in PATH.")

    cmd = [
        "mri_vol2vol",
        "--mov", label_path,
        "--targ", target_path,
        "--lta", lta_path,
        "--o", out_path,
        "--interp", "nearest"
    ]
    log("RUN: " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-30:])
        log("mri_vol2vol stderr (tail):\n" + tail, level="ERROR")
        raise RuntimeError(f"mri_vol2vol failed ({r.returncode})")
    return out_path


# ---------------- Dedup helpers ----------------
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}


def _parse_braak(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    sl = s.lower()

    try:
        return float(s)
    except Exception:
        pass

    if sl in _ROMAN:
        return float(_ROMAN[sl])

    m = re.search(r"(i{1,3}|iv|v|vi)", sl)
    if m and m.group(1) in _ROMAN:
        return float(_ROMAN[m.group(1)])

    return None


def _parse_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None


def _get_cdr(row):
    keys_exact = ["cdr", "cdr_global", "cdr global", "cdrglobal", "cdr (global)"]

    for k in row.keys():
        kl = k.lower()
        if kl in [ke.lower() for ke in keys_exact]:
            val = _parse_float(ci_get(row, k))
            if val is not None:
                return val

    for k in row.keys():
        kl = k.lower()
        if "cdr" in kl and "sb" not in kl:
            val = _parse_float(row[k])
            if val is not None:
                return val

    return None


def _get_centiloid(row):
    best = None
    for k in row.keys():
        kl = k.lower()
        if ("centiloid" in kl) or ("centloid" in kl) or (kl.startswith("amyloid") and "status" not in kl):
            val = _parse_float(row[k])
            if val is not None:
                best = val if (best is None or val > best) else best
    return best


def _get_braak_tuple(row):
    """Return Braak severity tuple, higher is worse."""
    def nz(x):
        return x if x is not None else float("-inf")

    b12 = _parse_float(ci_get(row, "Braak1_2", None))
    b34 = _parse_float(ci_get(row, "Braak3_4", None))
    b56 = _parse_float(ci_get(row, "Braak5_6", None))
    return (nz(b56), nz(b34), nz(b12))


def _severity_tuple(row):
    """Priority: CDR > Centiloid > Braak (5_6, 3_4, 1_2)."""
    def nz(x):
        return x if x is not None else float("-inf")

    cdr = _get_cdr(row)
    amy = _get_centiloid(row)
    b56, b34, b12 = _get_braak_tuple(row)
    return (nz(cdr), nz(amy), b56, b34, b12)


def dedup_by_tau_session(rows):
    """Keep exactly one row per TAU_PET_Session following severity priority."""
    groups = {}
    for r in rows:
        ts = ci_get(r, "TAU_PET_Session", "").strip()
        if ts == "":
            continue
        groups.setdefault(ts, []).append(r)

    deduped = []
    dropped = 0

    for ts, lst in groups.items():
        if len(lst) == 1:
            deduped.append(lst[0])
            continue

        scored = [
            (
                _severity_tuple(r),
                f"{ci_get(r, 'MR_Session', '')}_{ci_get(r, 'TAU_PUP_ID', '')}",
                r
            )
            for r in lst
        ]

        scored.sort(
            key=lambda t: (t[0][0], t[0][1], t[0][2], t[0][3], t[0][4], t[1]),
            reverse=True
        )

        keep = scored[0][2]
        deduped.append(keep)
        dropped += (len(lst) - 1)
        log(f"[DEDUP] {ts}: kept 1 of {len(lst)} (priority CDR>Centiloid>Braak).")

    log(f"Deduplicated TAU_PET_Session: kept {len(deduped)} of {sum(len(v) for v in groups.values())} (dropped {dropped}).")
    return deduped


# =================== CLI ===================
def parse_args():
    ap = argparse.ArgumentParser(
        description="All-TAU pipeline with flexible workload splitting."
    )
    ap.add_argument(
        "--part",
        type=int,
        default=None,
        help="1-based chunk index to process, e.g. 3"
    )
    ap.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="Total number of chunks, e.g. 10"
    )
    return ap.parse_args()


# =================== MAIN ===================
def main():
    args = parse_args()

    # Path sanity
    for p, isdir in [(OUT_ROOT, True), (PUP_ROOT, True), (FS_ROOT, True)]:
        if isdir and not os.path.isdir(p):
            log(f"Missing required dir: {p}", level="ERROR")
            sys.exit(1)

    if not os.path.isfile(LUT_PATH):
        log(f"LUT not found: {LUT_PATH}", level="ERROR")
        sys.exit(1)

    if not os.path.isfile(CSV_PATH):
        log(f"CSV not found: {CSV_PATH}", level="ERROR")
        sys.exit(1)

    # Load all rows with non-empty TAU_PET_Session
    all_rows = sniff_csv_rows(CSV_PATH)
    rows = []
    dropped_empty_session = 0

    for r in all_rows:
        if not any(k.lower() == "tau_pet_session" for k in r.keys()):
            continue

        tau_session = ci_get(r, "TAU_PET_Session", "").strip()
        if tau_session:
            rows.append(r)
        else:
            dropped_empty_session += 1

    log(f"Rows with non-empty TAU_PET_Session: {len(rows)} (dropped empty: {dropped_empty_session})")

    if RUN_LIMIT:
        rows = rows[:RUN_LIMIT]

    # Deduplicate by TAU_PET_Session using CDR > Centiloid > Braak priority
    before = len(rows)
    rows = dedup_by_tau_session(rows)
    after = len(rows)
    log(f"Eligible TAU rows (all TAU_PET_Session): {before} -> {after} after deduplication")

    # Stable ordering so chunk assignment is reproducible
    rows.sort(key=lambda r: (
        ci_get(r, "TAU_PET_Session", "").strip(),
        ci_get(r, "MR_Session", "").strip(),
        ci_get(r, "TAU_PUP_ID", "").strip(),
    ))

    # Partition into N equal parts
    total = len(rows)
    if args.part is not None:
        if args.num_parts < 1:
            log("--num-parts must be >= 1", level="ERROR")
            sys.exit(1)

        if not (1 <= args.part <= args.num_parts):
            log(f"--part must be between 1 and {args.num_parts}", level="ERROR")
            sys.exit(1)

        part = args.part
        num_parts = args.num_parts

        start = (part - 1) * total // num_parts
        end   = part * total // num_parts
        rows = rows[start:end]

        log(f"Processing part {part}/{num_parts} -> rows[{start}:{end}] = {len(rows)} (of {total})")
    else:
        log(f"Processing all rows: {total}")

    # Counters
    ok = 0
    skip_no_pupsubj = 0
    skip_no_nifti = 0
    skip_no_t1 = 0
    skip_no_t1001 = 0
    skip_no_pet = 0
    skip_no_fs = 0
    skip_no_aseg = 0
    skip_no_aparc = 0
    flirt_fail = 0
    roi_done = 0
    cortex_done = 0
    cortex_empty = 0
    space_warn_aseg = 0
    space_warn_aparc = 0
    vol2vol_fail = 0
    unexpected_fail = 0

    lut = read_fs_lut(LUT_PATH)
    summary = []

    for i, row in enumerate(rows, 1):
        tau_session = ci_get(row, "TAU_PET_Session", "").strip()
        mr_session  = ci_get(row, "MR_Session", "").strip()
        tau_pupid   = ci_get(row, "TAU_PUP_ID", "").strip()

        log(f"\n[{i}/{len(rows)}] MR_Session={mr_session}  TAU_PET_Session={tau_session}")

        try:
            # --- PUP subject via TAU_PET_Session; fallback: TAU_PUP_ID ---
            pup_dir = find_pup_subject_dir_from_tau_session(PUP_ROOT, tau_session)
            if not pup_dir and tau_pupid:
                pup_dir = find_pup_subject_dir_from_pupid(PUP_ROOT, tau_pupid)

            if not pup_dir:
                log(f"[SKIP:PUP_SUBJECT] {tau_session}  (no PUP subject dir found)", level="ERROR")
                skip_no_pupsubj += 1
                continue

            nifti_dir = find_pup_nifti_dir(pup_dir)
            if not nifti_dir:
                log(f"[SKIP:NIFTI] {tau_session}  (no NIFTI_GZ under {pup_dir})", level="ERROR")
                skip_no_nifti += 1
                continue

            # CNDA time for closest FS run selection
            pup_cnda_name = os.path.basename(os.path.dirname(nifti_dir))
            pup_dt = _parse_cnda_timestamp_from_name(pup_cnda_name)

            # Pick files (strict)
            t1_native = choose_native_t1(nifti_dir)
            t1_1001   = choose_t1001_strict(nifti_dir)
            pet_path  = choose_pet_strict(nifti_dir)

            if not t1_native:
                log(f"[SKIP:T1] {tau_session}  (missing T1.nii.gz)", level="ERROR")
                skip_no_t1 += 1
                continue

            if not t1_1001:
                log(f"[SKIP:T1001] {tau_session}  (missing T1001.nii.gz)", level="ERROR")
                skip_no_t1001 += 1
                continue

            if not pet_path:
                log(f"[SKIP:PET] {tau_session}  (missing *_msum_SUVR.nii.gz)", level="ERROR")
                skip_no_pet += 1
                continue

            # --- FreeSurfer by MR_Session ---
            fs_dir = find_fs_subject_dir_by_mrsession(FS_ROOT, mr_session)
            if not fs_dir:
                log(f"[SKIP:FS] {tau_session}  (no FreeSurfer subject matching MR_Session='{mr_session}')", level="ERROR")
                skip_no_fs += 1
                continue

            log(f"  FS subject dir: {fs_dir}")

            aseg_path, aparc_path, used_run = find_fs_labels_closest(fs_dir, pup_dt)
            log(f"  FS run used: {used_run if used_run else 'fallback-search'}")
            log(f"  aseg: {aseg_path}")
            log(f"  aparc+aseg: {aparc_path}")

            if not aseg_path:
                log(f"[SKIP:ASEG] {tau_session}  (aseg.mgz not found)", level="ERROR")
                skip_no_aseg += 1
                continue

            if not aparc_path:
                log(f"[SKIP:APARC] {tau_session}  (aparc+aseg.mgz not found)", level="ERROR")
                skip_no_aparc += 1
                continue

            # --- Output dir (unique per TAU session) ---
            subj_folder = os.path.basename(pup_dir)
            out_dir_name = f"{safe_name(subj_folder)}__{safe_name(tau_session)}"
            out_dir = os.path.join(OUT_ROOT, out_dir_name)
            os.makedirs(out_dir, exist_ok=True)

            # --- Copy T1 (native) ---
            dst_t1 = os.path.join(out_dir, os.path.basename(t1_native))
            shutil.copy2(t1_native, dst_t1)

            # --- T1001 -> T1 (FLIRT), then apply to PET -> PET_in_T1 ---
            flirt = shutil.which("flirt")
            if not flirt:
                log(f"[FAIL:FLIRT] {tau_session}  (flirt not found on PATH)", level="ERROR")
                flirt_fail += 1
                continue

            t1_for_cli, squeezed = _ensure_3d_nifti(dst_t1, out_dir, "t1")
            t1001_for_cli, t1001_sqz = _ensure_3d_nifti(t1_1001, out_dir, "t1001")
            pet_for_cli, pet_sqz = _ensure_3d_nifti(pet_path, out_dir, "pet")

            mat_path = os.path.join(out_dir, "T1001_to_T1.mat")
            dst_pet  = os.path.join(out_dir, "PET_in_T1.nii.gz")

            env = os.environ.copy()
            env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

            try:
                subprocess.run(
                    [flirt, "-in", t1001_for_cli, "-ref", t1_for_cli, "-omat", mat_path,
                     "-dof", "6", "-cost", "normmi"],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
                )
            except subprocess.CalledProcessError as e:
                log(f"[FAIL:FLIRT] {tau_session}  (estimate exit {e.returncode})", level="ERROR")
                msg = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else str(e.stderr)
                log("---- FLIRT stderr (estimate, last 20 lines) ----\n" + "\n".join(msg.splitlines()[-20:]), level="ERROR")
                for pth, flg in [(t1_for_cli, squeezed), (t1001_for_cli, t1001_sqz), (pet_for_cli, pet_sqz)]:
                    if flg and os.path.exists(pth):
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                flirt_fail += 1
                continue

            try:
                subprocess.run(
                    [flirt, "-in", pet_for_cli, "-ref", t1_for_cli, "-applyxfm", "-init", mat_path,
                     "-interp", "trilinear", "-out", dst_pet],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
                )
            except subprocess.CalledProcessError as e:
                log(f"[FAIL:FLIRT] {tau_session}  (apply exit {e.returncode})", level="ERROR")
                msg = e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else str(e.stderr)
                log("---- FLIRT stderr (apply, last 20 lines) ----\n" + "\n".join(msg.splitlines()[-20:]), level="ERROR")
                for pth, flg in [(t1_for_cli, squeezed), (t1001_for_cli, t1001_sqz), (pet_for_cli, pet_sqz)]:
                    if flg and os.path.exists(pth):
                        try:
                            os.remove(pth)
                        except Exception:
                            pass
                flirt_fail += 1
                continue

            for pth, flg in [(t1_for_cli, squeezed), (t1001_for_cli, t1001_sqz), (pet_for_cli, pet_sqz)]:
                if flg and os.path.exists(pth):
                    try:
                        os.remove(pth)
                    except Exception:
                        pass

            # --- Ensure labels are in T1 space ---
            sameA, shapeA, affA, a_shape, t_shapeA, a_aff, t_affA = shapes_affines_match(
                aseg_path, dst_t1, atol=ATOL_AFFINE
            )
            sameP, shapeP, affP, p_shape, t_shapeP, p_aff, t_affP = shapes_affines_match(
                aparc_path, dst_t1, atol=ATOL_AFFINE
            )

            aseg_inT1 = None
            aparc_inT1 = None

            if not sameA or not sameP:
                if not sameA:
                    space_warn_aseg += 1
                    log(f"[MISMATCH] aseg vs T1 (shape_ok={shapeA}, affine_ok={affA})")
                    log(f"  aseg.shape={a_shape}  T1.shape={t_shapeA}")
                    log("  aseg.affine:\n" + np.array2string(a_aff, precision=5))
                    log("  T1.affine:\n" + np.array2string(t_affA, precision=5))

                if not sameP:
                    space_warn_aparc += 1
                    log(f"[MISMATCH] aparc vs T1 (shape_ok={shapeP}, affine_ok={affP})")
                    log(f"  aparc.shape={p_shape}  T1.shape={t_shapeP}")
                    log("  aparc.affine:\n" + np.array2string(p_aff, precision=5))
                    log("  T1.affine:\n" + np.array2string(t_affP, precision=5))

                try:
                    fs_ref = pick_fs_brain(fs_dir, used_run)
                    if not fs_ref:
                        raise RuntimeError("FS reference (brain/T1/orig) not found")

                    lta_path = os.path.join(out_dir, "fs_to_T1.lta")
                    compute_fs2t1_lta(fs_ref, dst_t1, lta_path)

                    aseg_inT1 = os.path.join(out_dir, "aseg_inT1.nii.gz")
                    aparc_inT1 = os.path.join(out_dir, "aparc_inT1.nii.gz")

                    resample_label_with_lta(aseg_path, dst_t1, lta_path, aseg_inT1)
                    resample_label_with_lta(aparc_path, dst_t1, lta_path, aparc_inT1)

                    log(f"[REGISTER] FS->T1 LTA computed: {lta_path}")
                    log(f"  aseg_inT1:  {aseg_inT1}")
                    log(f"  aparc_inT1: {aparc_inT1}")

                except Exception as e:
                    log(f"[FALLBACK --regheader] {e}", level="WARN")
                    try:
                        if not sameA:
                            aseg_inT1 = os.path.join(out_dir, "aseg_inT1.nii.gz")
                            resample_label_to_target(aseg_path, dst_t1, aseg_inT1)
                        else:
                            aseg_inT1 = aseg_path

                        if not sameP:
                            aparc_inT1 = os.path.join(out_dir, "aparc_inT1.nii.gz")
                            resample_label_to_target(aparc_path, dst_t1, aparc_inT1)
                        else:
                            aparc_inT1 = aparc_path

                    except Exception as e2:
                        log(f"[SKIP:VOL2VOL] {tau_session}  ({e2})", level="ERROR")
                        vol2vol_fail += 1
                        continue
            else:
                aseg_inT1 = aseg_path
                aparc_inT1 = aparc_path

            # --- Parenchyma mask ---
            paren_path = os.path.join(out_dir, "aseg_brainmask.nii.gz")
            make_aseg_mask_nifti(aseg_inT1, paren_path)

            # --- Basal ganglia & no-BG composite ---
            bg_path = os.path.join(out_dir, "mask_basalganglia.nii.gz")
            vox_bg = write_mask_from_labels(aseg_inT1, BG_IDS, bg_path)
            log(f"  BG mask written: {bg_path} (voxels={vox_bg})")

            paren_img = nib.load(paren_path)
            bg_img = nib.load(bg_path)
            paren_np = np.asanyarray(paren_img.dataobj).astype(bool)
            bg_np = np.asanyarray(bg_img.dataobj).astype(bool)

            if paren_np.shape != bg_np.shape or not np.allclose(paren_img.affine, bg_img.affine, atol=ATOL_AFFINE):
                log("  ERROR: Parenchyma vs BG mask grid mismatch; cannot form noBG mask.", level="ERROR")
                log(f"    parenchyma.shape={paren_np.shape}  BG.shape={bg_np.shape}")
                log("    parenchyma.affine:\n" + np.array2string(paren_img.affine, precision=5))
                log("    BG.affine:\n" + np.array2string(bg_img.affine, precision=5))
                nobg_exists = False
            else:
                nobg_np = (paren_np & ~bg_np).astype(np.uint8)
                nobg_path = os.path.join(out_dir, "mask_parenchyma_noBG.nii.gz")
                nib.save(nib.Nifti1Image(nobg_np, paren_img.affine), nobg_path)
                log(f"  no-BG mask written: {nobg_path} (voxels={int(nobg_np.sum())})")
                nobg_exists = True

            # --- Cortex mask ---
            cortex_path = os.path.join(out_dir, "mask_cortex.nii.gz")
            vox_cortex = write_mask_from_labels(aseg_inT1, CORTEX_LABELS, cortex_path)
            if vox_cortex > 0:
                cortex_done += 1
                log(f"  cortex mask written: {cortex_path} (voxels={vox_cortex})")
            else:
                cortex_empty += 1
                log(f"  WARN: cortex mask is empty at {cortex_path}", level="WARN")

            # --- ROI masks ---
            created = {}

            hip_ids = [lut.get("Left-Hippocampus", 17), lut.get("Right-Hippocampus", 53)]
            hip_path = os.path.join(out_dir, "ROI_Hippocampus.nii.gz")
            vox_hip = write_mask_from_labels(aseg_inT1, hip_ids, hip_path)
            created[Path(hip_path).name] = vox_hip

            pcc_ids = ids_for_hemi_names(["posteriorcingulate"], lut)
            if pcc_ids:
                pcc_path = os.path.join(out_dir, "ROI_PosteriorCingulate.nii.gz")
                vox_pcc = write_mask_from_labels(aparc_inT1, pcc_ids, pcc_path)
                created[Path(pcc_path).name] = vox_pcc
            else:
                log("  ERROR: PCC labels not in LUT; skipping PCC", level="ERROR")

            pcun_ids = ids_for_hemi_names(["precuneus"], lut)
            if pcun_ids:
                pcun_path = os.path.join(out_dir, "ROI_Precuneus.nii.gz")
                vox_pcun = write_mask_from_labels(aparc_inT1, pcun_ids, pcun_path)
                created[Path(pcun_path).name] = vox_pcun
            else:
                log("  ERROR: Precuneus labels not in LUT; skipping Precuneus", level="ERROR")

            temp_ids = ids_for_hemi_names(TEMPORAL_BASE, lut)
            if temp_ids:
                temp_path = os.path.join(out_dir, "ROI_TemporalLobe.nii.gz")
                vox_temp = write_mask_from_labels(aparc_inT1, temp_ids, temp_path)
                created[Path(temp_path).name] = vox_temp
            else:
                log("  ERROR: Temporal lobe labels not in LUT; skipping Temporal", level="ERROR")

            limb_ids = ids_for_hemi_names(LIMBIC_BASE, lut)
            if limb_ids:
                limb_path = os.path.join(out_dir, "ROI_LimbicCortex.nii.gz")
                vox_limb = write_mask_from_labels(aparc_inT1, limb_ids, limb_path)
                created[Path(limb_path).name] = vox_limb
            else:
                log("  ERROR: Limbic labels not in LUT; skipping Limbic", level="ERROR")

            log("  ROI voxel counts:\n" + json.dumps(created, indent=2))

            print(f"[OK] {out_dir_name}")
            print(f"     T1        -> {dst_t1}")
            print(f"     PET(new)  -> {dst_pet}")
            print(f"     MASK(new) -> {paren_path}")

            ok += 1
            roi_done += 1

            summary.append({
                "subject": out_dir_name,
                "MR_Session": mr_session,
                "TAU_PET_Session": tau_session,
                "created": created,
                "bg_voxels": int(vox_bg),
                "cortex_voxels": int(vox_cortex),
                "noBG_exists": bool(nobg_exists)
            })

        except Exception as e:
            log(f"[SKIP:UNEXPECTED] {tau_session}  ({type(e).__name__}: {e})", level="ERROR")
            unexpected_fail += 1
            continue

    # =================== SUMMARY ===================
    print("\n=== SUMMARY ===")
    print(f"Processed OK                     : {ok}")
    print(f"Skipped (no PUP subject)         : {skip_no_pupsubj}")
    print(f"Skipped (no NIFTI_GZ)            : {skip_no_nifti}")
    print(f"Skipped (no T1)                  : {skip_no_t1}")
    print(f"Skipped (no T1001)               : {skip_no_t1001}")
    print(f"Skipped (no PET SUVR)            : {skip_no_pet}")
    print(f"Skipped (no FS match)            : {skip_no_fs}")
    print(f"Skipped (no ASEG)                : {skip_no_aseg}")
    print(f"Skipped (no APARC+ASEG)          : {skip_no_aparc}")
    print(f"FLIRT failures                   : {flirt_fail}")
    print(f"VOL2VOL failures                 : {vol2vol_fail}")
    print(f"Unexpected subject failures      : {unexpected_fail}")
    print(f"ROI sets created                 : {roi_done}")
    print(f"Cortex masks created             : {cortex_done}")
    print(f"Cortex masks empty               : {cortex_empty}")
    print(f"DEBUG: resample triggers (aseg)  : {space_warn_aseg}")
    print(f"DEBUG: resample triggers (aparc) : {space_warn_aparc}")
    print(f"Output root                      : {OUT_ROOT}")

    try:
        if args.part is not None:
            part_tag = f"part{args.part:02d}_of_{args.num_parts:02d}"
        else:
            part_tag = "all"

        summ_path = os.path.join(OUT_ROOT, f"roi_creation_summary_{part_tag}.json")
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Summary written: {summ_path}")
    except Exception as e:
        log(f"Could not write summary JSON: {e}", level="WARN")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Unhandled error: {e}", level="ERROR")
        sys.exit(2)
