#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, csv, shutil, subprocess, sys, argparse
from datetime import datetime
import numpy as np
import nibabel as nib

# ========= defaults (edit if you want) =========
BASE_ROOT    = "/ceph/chpc/mapped/benz04_kari"
PUP_ROOT_DFT = os.path.join(BASE_ROOT, "pup")
FS_ROOT_DFT  = os.path.join(BASE_ROOT, "freesurfers")

DATASET_ROOT_DFT = "/scratch/l.peiwang/kari_brainv33_top300"
CSV_PATH_DFT     = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"

OUT_NAME_DFT     = "mask_cortex.nii.gz"
ATOL_AFFINE_DFT  = 1e-4

CORTEX_ASEG_LABELS = {3, 42}  # Left/Right-Cerebral-Cortex in aseg
# ==============================================

def log(msg, level="INFO", debug=True):
    if debug or level in ("WARN", "ERROR"):
        print(f"[{level}] {msg}", flush=True)

def die(msg):
    print(f"[FATAL] {msg}", flush=True)
    raise SystemExit(1)

def ci_get(row, key, default=""):
    kl = key.lower()
    for k in row.keys():
        if k.lower() == kl:
            v = row.get(k)
            return v if v is not None else default
    return default

def sniff_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        try:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))

def _parse_cnda_timestamp_from_name(name):
    m = re.search(r"(\d{14})$", name) or re.search(r"(\d{14})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None

# ---------- PUP discovery ----------
def find_pup_subject_dir_from_tau_session(pup_root, tau_session):
    if not tau_session:
        return None
    ts_l = tau_session.lower()
    cands = []
    for d in glob.glob(os.path.join(pup_root, "*")):
        if os.path.isdir(d) and ts_l in os.path.basename(d).lower():
            cands.append(d)
    cands.sort()
    return cands[-1] if cands else None

def find_pup_subject_dir_from_pupid(pup_root, pup_id):
    if not pup_id:
        return None
    hits = glob.glob(os.path.join(pup_root, "*", pup_id)) + glob.glob(os.path.join(pup_root, "*", pup_id + "*"))
    if hits:
        return os.path.dirname(os.path.dirname(os.path.join(hits[-1], "")))
    return None

def find_pup_nifti_dir(pup_dir):
    hits = glob.glob(os.path.join(pup_dir, "*", "NIFTI_GZ")) if pup_dir else []
    if not hits:
        return None
    hits.sort()
    return hits[-1]

# ---------- FreeSurfer discovery by MR_Session ----------
def find_fs_subject_dir_by_mrsession(fs_root, mr_session):
    if not mr_session:
        return None
    mr_l = mr_session.lower()
    cands = []

    # pass 1
    for d in glob.glob(os.path.join(fs_root, "*")):
        if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
            cands.append(d)

    # pass 2
    if not cands:
        for d in glob.glob(os.path.join(fs_root, "*", "*")):
            if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
                cands.append(d)

    # pass 3
    if not cands:
        for root, dirs, _ in os.walk(fs_root):
            for dd in dirs:
                if mr_l in dd.lower():
                    cands.append(os.path.join(root, dd))

    if not cands:
        return None

    def rank(p):
        n = os.path.basename(p).lower()
        if "mri" in n: return 0
        if "mmr" in n: return 1
        return 2

    cands.sort(key=rank)
    return cands[0]

def _run_dirs(fs_subject_dir):
    return glob.glob(os.path.join(fs_subject_dir, "CNDA*_*_freesurfer_*"))

def _find_label_in_run(run_dir, fname):
    hits = glob.glob(os.path.join(run_dir, "DATA", "*", "mri", fname))
    hits.sort()
    return hits[-1] if hits else None

def find_aseg_closest(fs_subject_dir, target_dt):
    runs = _run_dirs(fs_subject_dir)
    best_run, best_diff = None, float("inf")

    for rd in runs:
        dt = _parse_cnda_timestamp_from_name(os.path.basename(rd))
        diff = abs((dt - target_dt).total_seconds()) if (dt and target_dt) else float("inf")
        if diff < best_diff:
            best_run, best_diff = rd, diff

    if best_run:
        aseg = _find_label_in_run(best_run, "aseg.mgz")
        if aseg:
            return aseg, best_run

    # fallback: any aseg
    for root, _, files in os.walk(fs_subject_dir):
        if "aseg.mgz" in files:
            return os.path.join(root, "aseg.mgz"), None

    return None, None

# ---------- Geometry + registration ----------
def shapes_affines_match(a_path, b_path, atol):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok = np.allclose(ia.affine, ib.affine, atol=atol)
    return (shape_ok and aff_ok), shape_ok, aff_ok, ia.shape, ib.shape, ia.affine, ib.affine

def pick_fs_ref(fs_subject_dir, used_run):
    pats = []
    if used_run:
        pats += [
            os.path.join(used_run, "DATA", "*", "mri", "brain.mgz"),
            os.path.join(used_run, "DATA", "*", "mri", "T1.mgz"),
            os.path.join(used_run, "DATA", "*", "mri", "orig.mgz"),
        ]
    pats += [
        os.path.join(fs_subject_dir, "mri", "brain.mgz"),
        os.path.join(fs_subject_dir, "mri", "T1.mgz"),
        os.path.join(fs_subject_dir, "mri", "orig.mgz"),
    ]
    for pat in pats:
        hits = glob.glob(pat)
        if hits:
            hits.sort()
            return hits[-1]
    return None

def compute_fs2t1_lta(fs_ref, t1_path, out_lta, debug=True):
    if not shutil.which("mri_robust_register"):
        die("mri_robust_register not found in PATH. Source FreeSurfer env.")
    cmd = ["mri_robust_register", "--mov", fs_ref, "--dst", t1_path, "--lta", out_lta,
           "--satit", "--iscale", "--maxit", "200"]
    log("RUN: " + " ".join(cmd), debug=debug)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-40:])
        die("mri_robust_register failed (tail):\n" + tail)
    return out_lta

def resample_label_with_lta(label_path, t1_path, lta_path, out_path, debug=True):
    if not shutil.which("mri_vol2vol"):
        die("mri_vol2vol not found in PATH. Source FreeSurfer env.")
    cmd = ["mri_vol2vol", "--mov", label_path, "--targ", t1_path,
           "--lta", lta_path, "--o", out_path, "--interp", "nearest"]
    log("RUN: " + " ".join(cmd), debug=debug)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-40:])
        die("mri_vol2vol failed (tail):\n" + tail)
    return out_path

def write_cortex_mask_from_aseg(aseg_like_path, t1_path, out_mask_path):
    t1 = nib.load(t1_path)
    lab_img = nib.load(aseg_like_path)
    lab = np.asanyarray(lab_img.dataobj)

    if lab.shape != t1.shape:
        die(f"Label->T1 shape mismatch: label {lab.shape} vs T1 {t1.shape}")

    mask = np.isin(lab, list(CORTEX_ASEG_LABELS)).astype(np.uint8)
    vox = int(mask.sum())
    if vox == 0:
        die(f"Cortex mask EMPTY using labels {sorted(CORTEX_ASEG_LABELS)} from {aseg_like_path}")

    out = nib.Nifti1Image(mask, t1.affine)
    out.set_data_dtype(np.uint8)
    nib.save(out, out_mask_path)
    return vox

def parse_args():
    ap = argparse.ArgumentParser(description="Generate cortex-only mask (aseg labels 3/42) into existing T807 dataset folders.")
    ap.add_argument("--csv", default=CSV_PATH_DFT)
    ap.add_argument("--dataset_root", default=DATASET_ROOT_DFT)
    ap.add_argument("--pup_root", default=PUP_ROOT_DFT)
    ap.add_argument("--fs_root", default=FS_ROOT_DFT)
    ap.add_argument("--out_name", default=OUT_NAME_DFT)
    ap.add_argument("--atol_affine", type=float, default=ATOL_AFFINE_DFT)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N unique sessions (debug).")
    ap.add_argument("--part", type=int, choices=[1,2,3,4], default=None, help="Process 1/4 of sessions (like your pipeline).")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logging.")
    return ap.parse_args()

def main():
    args = parse_args()
    DEBUG = bool(args.debug)

    for p in (args.pup_root, args.fs_root, args.dataset_root):
        if not os.path.isdir(p):
            die(f"Missing directory: {p}")
    if not os.path.isfile(args.csv):
        die(f"Missing CSV: {args.csv}")

    rows0 = sniff_csv_rows(args.csv)
    if not rows0:
        die("CSV has 0 rows.")

    # Filter to T807_v1 rows (same idea as your v33 script)
    eligible = []
    for r in rows0:
        ts = ci_get(r, "TAU_PET_Session", "").strip()
        if "t807" in ts.lower() and "_v1" in ts.lower():
            eligible.append(r)
    if not eligible:
        die("No eligible T807_v1 rows found in CSV (TAU_PET_Session filter).")

    # Dedup by TAU_PET_Session (strict consistency check on MR_Session)
    by_ts = {}
    dup_counts = 0
    for r in eligible:
        ts = ci_get(r, "TAU_PET_Session", "").strip()
        if not ts:
            die("Found eligible row with empty TAU_PET_Session.")
        mr = ci_get(r, "MR_Session", "").strip()
        if not mr:
            die(f"{ts}: MR_Session is empty (CSV issue).")

        if ts in by_ts:
            dup_counts += 1
            mr0 = ci_get(by_ts[ts], "MR_Session", "").strip()
            if mr0 != mr:
                die(f"Duplicate TAU_PET_Session '{ts}' has conflicting MR_Session: '{mr0}' vs '{mr}'")
            continue
        by_ts[ts] = r

    rows = list(by_ts.values())
    rows.sort(key=lambda r: ci_get(r, "TAU_PET_Session", ""))
    log(f"Eligible sessions: {len(eligible)} rows -> {len(rows)} unique TAU_PET_Session (dups ignored={dup_counts})", debug=True)

    if args.limit is not None:
        rows = rows[:args.limit]
        log(f"LIMIT active: processing first {len(rows)} sessions", debug=True)

    # Partition like your pipeline
    total = len(rows)
    if args.part:
        start = (args.part - 1) * total // 4
        end   = (args.part)     * total // 4
        log(f"PART {args.part}/4: sessions[{start}:{end}] of {total}", debug=True)
        rows = rows[start:end]

    log(f"Will process {len(rows)} sessions.", debug=True)

    for i, row in enumerate(rows, 1):
        tau_session = ci_get(row, "TAU_PET_Session", "").strip()
        mr_session  = ci_get(row, "MR_Session", "").strip()
        tau_pupid   = ci_get(row, "TAU_PUP_ID", "").strip()

        log(f"\n[{i}/{len(rows)}] TAU_PET_Session={tau_session}", debug=True)
        log(f"  MR_Session={mr_session}", debug=True)
        log(f"  TAU_PUP_ID={tau_pupid}", debug=True)

        # 1) PUP subject dir (defines subj_folder naming)
        pup_dir = find_pup_subject_dir_from_tau_session(args.pup_root, tau_session)
        if not pup_dir and tau_pupid:
            pup_dir = find_pup_subject_dir_from_pupid(args.pup_root, tau_pupid)
        if not pup_dir:
            die(f"{tau_session}: cannot find PUP subject dir (by TAU_PET_Session/TAU_PUP_ID).")

        subj_folder = os.path.basename(pup_dir)
        log(f"  PUP subject dir: {pup_dir}", debug=True)
        log(f"  subj_folder (dataset key): {subj_folder}", debug=True)

        # 2) dataset folder + T1
        out_dir = os.path.join(args.dataset_root, subj_folder)
        if not os.path.isdir(out_dir):
            die(f"{tau_session}: dataset folder missing: {out_dir}")
        t1_path = os.path.join(out_dir, "T1.nii.gz")
        if not os.path.exists(t1_path):
            die(f"{tau_session}: missing T1.nii.gz in dataset folder: {t1_path}")
        log(f"  Dataset folder: {out_dir}", debug=True)
        log(f"  T1: {t1_path}", debug=True)

        # 3) PUP timestamp (for closest FS run)
        nifti_dir = find_pup_nifti_dir(pup_dir)
        if not nifti_dir:
            die(f"{tau_session}: no NIFTI_GZ under PUP dir: {pup_dir}")
        pup_cnda_name = os.path.basename(os.path.dirname(nifti_dir))
        pup_dt = _parse_cnda_timestamp_from_name(pup_cnda_name)
        log(f"  PUP NIFTI_GZ: {nifti_dir}", debug=True)
        log(f"  PUP CNDA folder: {pup_cnda_name}", debug=True)
        log(f"  PUP timestamp parsed: {pup_dt}", debug=True)

        # 4) FS dir by MR_Session
        fs_dir = find_fs_subject_dir_by_mrsession(args.fs_root, mr_session)
        if not fs_dir:
            die(f"{tau_session}: cannot locate FS subject dir for MR_Session='{mr_session}' under {args.fs_root}")
        log(f"  FS subject dir: {fs_dir}", debug=True)

        # 5) aseg closest
        aseg_path, used_run = find_aseg_closest(fs_dir, pup_dt)
        if not aseg_path or not os.path.exists(aseg_path):
            die(f"{tau_session}: aseg.mgz not found under {fs_dir}")
        log(f"  FS run used: {used_run if used_run else 'fallback-search'}", debug=True)
        log(f"  aseg.mgz: {aseg_path}", debug=True)

        # 6) geometry check; if mismatch -> robust register + resample labels into T1
        same, shape_ok, aff_ok, a_shape, t_shape, a_aff, t_aff = shapes_affines_match(aseg_path, t1_path, atol=args.atol_affine)
        log(f"  Geometry match (aseg vs T1): {same} (shape_ok={shape_ok}, affine_ok={aff_ok}, atol={args.atol_affine})", debug=True)

        tmp_lta = None
        tmp_aseg_inT1 = None
        aseg_for_mask = aseg_path

        if not same:
            log("  [MISMATCH] aseg vs T1 geometry — will compute FS->T1 LTA and resample labels (nearest).", level="WARN", debug=True)
            log(f"    aseg.shape={a_shape}  T1.shape={t_shape}", level="WARN", debug=True)
            log("    aseg.affine:\n" + np.array2string(a_aff, precision=6), level="WARN", debug=True)
            log("    T1.affine:\n"   + np.array2string(t_aff, precision=6), level="WARN", debug=True)

            fs_ref = pick_fs_ref(fs_dir, used_run)
            if not fs_ref:
                die(f"{tau_session}: cannot find FS ref (brain/T1/orig) for mri_robust_register.")
            log(f"  FS ref for registration: {fs_ref}", debug=True)

            tmp_lta = os.path.join(out_dir, "._tmp_fs_to_T1.lta")
            tmp_aseg_inT1 = os.path.join(out_dir, "._tmp_aseg_inT1.nii.gz")

            compute_fs2t1_lta(fs_ref, t1_path, tmp_lta, debug=DEBUG)
            resample_label_with_lta(aseg_path, t1_path, tmp_lta, tmp_aseg_inT1, debug=DEBUG)
            aseg_for_mask = tmp_aseg_inT1

        # 7) build cortex mask (in T1 grid)
        out_mask = os.path.join(out_dir, args.out_name)
        vox = write_cortex_mask_from_aseg(aseg_for_mask, t1_path, out_mask)
        log(f"  WROTE: {out_mask} (voxels={vox})", debug=True)

        # cleanup temps
        for p in (tmp_aseg_inT1, tmp_lta):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    log(f"  cleaned temp: {p}", debug=True)
                except Exception:
                    log(f"  could not remove temp: {p}", level="WARN", debug=True)

        print(f"[OK] {subj_folder} -> {out_mask} (voxels={vox})", flush=True)

    print("[done] all cortex masks generated.", flush=True)

if __name__ == "__main__":
    main()
