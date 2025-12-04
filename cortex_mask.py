#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, glob, sys, subprocess
import numpy as np
import nibabel as nib

# ===== EDIT THESE =====
DATASET_ROOT = "/scratch/l.peiwang/kari_brainv33_top300"
CSV_PATH     = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
FS_ROOT      = "/ceph/chpc/mapped/benz04_kari/freesurfers"

T1_NAME      = "T1.nii.gz"
OUT_MASK     = "mask_cortex.nii.gz"

ATOL_AFFINE  = 1e-4
CORTEX_LABELS = {3, 42}  # aseg: Left/Right-Cerebral-Cortex
# =====================

def die(msg):
    print(f"[FATAL] {msg}", flush=True)
    sys.exit(1)

def info(msg):
    print(f"[info] {msg}", flush=True)

def warn(msg):
    print(f"[warn] {msg}", flush=True)

def ci_get(row, key, default=""):
    kl = key.lower()
    for k in row.keys():
        if k.lower() == kl:
            v = row.get(k)
            return v if v is not None else default
    return default

def which(cmd):
    for p in os.environ.get("PATH", "").split(os.pathsep):
        full = os.path.join(p, cmd)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    # common fallback: $FSLDIR/bin/cmd
    fsldir = os.environ.get("FSLDIR", "")
    if fsldir:
        full = os.path.join(fsldir, "bin", cmd)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return None

def run_or_die(cmd, tag):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.splitlines()[-60:])
        die(f"{tag} failed (exit={r.returncode}). Tail:\n{tail}")
    return r

def mr_rank(mr_session: str) -> int:
    """lower is better: mri > mmr > token mr > other"""
    s = (mr_session or "").strip().lower()
    if "mri" in s: return 0
    if "mmr" in s: return 1
    if re.search(r"(^|[^a-z0-9])mr([^a-z0-9]|$)", s): return 2
    return 9

def load_t807_groups(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        ts = ci_get(r, "TAU_PET_Session", "").strip()
        if not ts:
            continue
        tsl = ts.lower()
        if ("t807" in tsl) and ("_v1" in tsl):
            groups.setdefault(ts, []).append(r)
    return groups

def pick_row_for_session(rows):
    # prefer MR-like (mri/mmr/mr), then deterministic tie-break
    def key(r):
        mr  = ci_get(r, "MR_Session", "").strip()
        pup = ci_get(r, "TAU_PUP_ID", "").strip()
        return (mr_rank(mr), mr, pup)
    return sorted(rows, key=key)[0]

def match_tau_session(folder_name, sessions):
    if folder_name in sessions:
        return folder_name
    fn = folder_name.lower()
    best, best_len = None, -1
    for s in sessions:
        sl = s.lower()
        if sl in fn or fn in sl:
            if len(s) > best_len:
                best, best_len = s, len(s)
    return best

def find_fs_subject_dir(fs_root, mr_session):
    if not mr_session:
        return None
    mr_l = mr_session.lower()
    cands = []

    for d in glob.glob(os.path.join(fs_root, "*")):
        if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
            cands.append(d)
    if not cands:
        for d in glob.glob(os.path.join(fs_root, "*", "*")):
            if os.path.isdir(d) and mr_l in os.path.basename(d).lower():
                cands.append(d)
    if not cands:
        for root, dirs, _ in os.walk(fs_root):
            for dd in dirs:
                if mr_l in dd.lower():
                    cands.append(os.path.join(root, dd))

    if not cands:
        return None

    def rank_path(p):
        b = os.path.basename(p).lower()
        if "mri" in b: r = 0
        elif "mmr" in b: r = 1
        else: r = 2
        return (r, len(p), p)

    cands.sort(key=rank_path)
    return cands[0]

def find_aseg_mgz(fs_subject_dir):
    hits = glob.glob(os.path.join(fs_subject_dir, "**", "aseg.mgz"), recursive=True)
    if not hits:
        return None
    mri_hits = [h for h in hits if (os.sep + "mri" + os.sep + "aseg.mgz") in h]
    candidates = mri_hits if mri_hits else hits
    return max(candidates, key=os.path.getmtime)

def find_fs_ref_mgz(fs_subject_dir):
    # prefer brain.mgz, then T1.mgz, then orig.mgz
    for name in ("brain.mgz", "T1.mgz", "orig.mgz"):
        hits = glob.glob(os.path.join(fs_subject_dir, "**", name), recursive=True)
        hits = [h for h in hits if os.path.isfile(h)]
        if hits:
            # prefer .../mri/<name> and newest mtime
            mri_hits = [h for h in hits if (os.sep + "mri" + os.sep + name) in h]
            candidates = mri_hits if mri_hits else hits
            return max(candidates, key=os.path.getmtime)
    return None

def shapes_affines_match(a_path, b_path, atol):
    ia, ib = nib.load(a_path), nib.load(b_path)
    shape_ok = (ia.shape == ib.shape)
    aff_ok = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and aff_ok, shape_ok, aff_ok

def save_as_nifti(src_path, dst_path, dtype=None):
    img = nib.load(src_path)
    data = np.asanyarray(img.dataobj)
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    nib.save(nib.Nifti1Image(data, img.affine), dst_path)

def ensure_aseg_in_t1_grid_flirt(fs_dir, aseg_path, t1_path, tmp_out, work_dir):
    same, shape_ok, aff_ok = shapes_affines_match(aseg_path, t1_path, ATOL_AFFINE)
    if same:
        return aseg_path, "direct"

    flirt = which("flirt")
    if not flirt:
        die("flirt not found on PATH. In your job script do: module load fsl (or source FSLDIR).")

    fs_ref = find_fs_ref_mgz(fs_dir)
    if not fs_ref:
        die(f"FS ref not found under {fs_dir} (expected brain.mgz/T1.mgz/orig.mgz)")

    tmp_ref  = os.path.join(work_dir, "._tmp_fsref.nii.gz")
    tmp_aseg = os.path.join(work_dir, "._tmp_aseg.nii.gz")
    tmp_mat  = os.path.join(work_dir, "._tmp_fs2t1.mat")

    # Convert mgz -> nifti for FLIRT
    save_as_nifti(fs_ref, tmp_ref, dtype=np.float32)
    save_as_nifti(aseg_path, tmp_aseg, dtype=np.int16)

    info(f"{os.path.basename(work_dir)}: mismatch (shape_ok={shape_ok}, affine_ok={aff_ok}) -> using FLIRT")
    info(f"{os.path.basename(work_dir)}: fs_ref={fs_ref}")

    # 1) rigid FSref -> T1
    run_or_die([flirt, "-in", tmp_ref, "-ref", t1_path,
                "-omat", tmp_mat, "-dof", "6", "-cost", "normmi"],
               "flirt(fsref->T1)")

    # 2) apply to aseg labels (nearest)
    run_or_die([flirt, "-in", tmp_aseg, "-ref", t1_path,
                "-applyxfm", "-init", tmp_mat,
                "-interp", "nearestneighbour",
                "-out", tmp_out],
               "flirt(apply aseg)")

    # cleanup
    for p in (tmp_ref, tmp_aseg, tmp_mat):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    return tmp_out, f"flirt(dof6,normmi,nearest) (shape_ok={shape_ok}, affine_ok={aff_ok})"

def main():
    if not os.path.isdir(DATASET_ROOT): die(f"DATASET_ROOT missing: {DATASET_ROOT}")
    if not os.path.isfile(CSV_PATH): die(f"CSV missing: {CSV_PATH}")
    if not os.path.isdir(FS_ROOT): die(f"FS_ROOT missing: {FS_ROOT}")

    groups = load_t807_groups(CSV_PATH)
    sessions = set(groups.keys())
    info(f"Loaded {len(sessions)} unique T807_v1 TAU_PET_Session from CSV.")

    subj_folders = sorted([d for d in os.listdir(DATASET_ROOT)
                           if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    info(f"Found {len(subj_folders)} subject folders under dataset root.")

    for subj in subj_folders:
        subj_dir = os.path.join(DATASET_ROOT, subj)
        t1_path = os.path.join(subj_dir, T1_NAME)
        if not os.path.exists(t1_path):
            die(f"{subj}: missing {T1_NAME} at {t1_path}")

        tau_session = match_tau_session(subj, sessions)
        if not tau_session:
            die(f"{subj}: cannot match folder name to any TAU_PET_Session in CSV.")

        rows = groups[tau_session]
        if len(rows) > 1:
            mr_list = [ci_get(r, "MR_Session", "").strip() for r in rows]
            warn(f"{subj}: duplicate TAU_PET_Session={tau_session} with MR_Session candidates={mr_list}")

        row = pick_row_for_session(rows)
        mr_session = ci_get(row, "MR_Session", "").strip()
        if not mr_session:
            die(f"{subj}: MR_Session empty in CSV for TAU_PET_Session={tau_session}")

        info(f"{subj}: TAU_PET_Session={tau_session} | chosen MR_Session={mr_session}")

        fs_dir = find_fs_subject_dir(FS_ROOT, mr_session)
        if not fs_dir:
            die(f"{subj}: cannot find FS subject folder for MR_Session={mr_session}")

        aseg_path = find_aseg_mgz(fs_dir)
        if not aseg_path:
            die(f"{subj}: no aseg.mgz found under FS subject folder: {fs_dir}")

        info(f"{subj}: FS={fs_dir}")
        info(f"{subj}: aseg={aseg_path}")

        tmp_aseg_out = os.path.join(subj_dir, "._tmp_aseg_inT1.nii.gz")
        aseg_in_t1, how = ensure_aseg_in_t1_grid_flirt(fs_dir, aseg_path, t1_path, tmp_aseg_out, subj_dir)
        info(f"{subj}: aseg->T1 method: {how}")

        t1 = nib.load(t1_path)
        aseg_img = nib.load(aseg_in_t1)
        lab = np.asanyarray(aseg_img.dataobj)

        # FLIRT output might be float but still integer-like; make it safe:
        lab = np.rint(lab).astype(np.int32, copy=False)

        if lab.shape != t1.shape:
            die(f"{subj}: after mapping, aseg shape {lab.shape} != T1 shape {t1.shape}")

        mask = np.isin(lab, list(CORTEX_LABELS)).astype(np.uint8)
        vox = int(mask.sum())
        if vox == 0:
            die(f"{subj}: cortex mask is EMPTY (labels {sorted(CORTEX_LABELS)})")

        out_path = os.path.join(subj_dir, OUT_MASK)
        nib.save(nib.Nifti1Image(mask, t1.affine), out_path)

        if aseg_in_t1 == tmp_aseg_out and os.path.exists(tmp_aseg_out):
            try: os.remove(tmp_aseg_out)
            except Exception: pass

        info(f"{subj}: wrote {out_path} (voxels={vox})")

    info("done.")

if __name__ == "__main__":
    main()
