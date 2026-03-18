#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DIAN PET preprocessing pipeline.

Flow:
- PET folders live under PET_ROOT; folder name is PET session_label and contains pet.nii.gz
- PET CSV maps PET session_label -> subject_label, visit_num
- MR CSV maps (subject_label, visit_num) -> MR session_label
- FreeSurfer exports live under FS_ROOT/<mr_session>/... and contain:
  T1_fs_orig.nii.gz, aseg.nii.gz, aparc_aseg.nii.gz, optionally brainmask.nii.gz
- PET is rigidly registered to the subject T1
- aseg/aparc are checked against T1; if needed they are resampled into T1 space
- Standard outputs are written under OUT_ROOT/<subject_label>

Because the requested output layout is subject-level, multiple visits for the same
subject would overwrite each other. This script keeps only one visit per subject:
the highest visit_num, with explicit logging for skipped visits.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


PET_ROOT = "/scratch/l.peiwang/DIAN_PET"
FS_ROOT = "/scratch/l.peiwang/DIAN_fs"
PET_CSV = "/scratch/l.peiwang/DIAN_spreadsheet/DIANDF18_PET_session_details.csv"
MR_CSV = "/scratch/l.peiwang/DIAN_spreadsheet/DIANDF18_MR_session_details.csv"
OUT_ROOT = "/scratch/l.peiwang/DIAN_ALL_FINISHED"
DEFAULT_LUT_PATH = "/scratch/l.peiwang/FreeSurferColorLUT.txt"

ATOL_AFFINE = 1e-4
DEBUG = True

KEEP_LABELS = {
    2, 41, 3, 42, 7, 46, 8, 47, 10, 49, 11, 50, 12, 51,
    13, 52, 17, 53, 18, 54, 26, 58, 28, 60, 16,
}
BG_IDS_STRICT = {11, 50, 12, 51, 13, 52, 26, 58}
BG_IDS = BG_IDS_STRICT
CORTEX_LABELS = {3, 42}

TEMPORAL_BASE = [
    "superiortemporal", "middletemporal", "inferiortemporal",
    "fusiform", "transversetemporal", "temporalpole", "bankssts",
]
LIMBIC_BASE = [
    "posteriorcingulate", "isthmuscingulate",
    "caudalanteriorcingulate", "rostralanteriorcingulate",
    "parahippocampal", "entorhinal",
]


def log(msg, level="INFO"):
    if DEBUG or level in {"WARN", "ERROR"}:
        print(f"[{level}] {msg}", flush=True)


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args():
    ap = argparse.ArgumentParser(
        description="Preprocess DIAN PET sessions into old-dataset-style T1/PET/mask outputs."
    )
    ap.add_argument("--pet-root", default=PET_ROOT)
    ap.add_argument("--fs-root", default=FS_ROOT)
    ap.add_argument("--pet-csv", default=PET_CSV)
    ap.add_argument("--mr-csv", default=MR_CSV)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--lut-path", default=DEFAULT_LUT_PATH)
    ap.add_argument("--num-tasks", type=int, default=env_int("SLURM_ARRAY_TASK_COUNT", 1))
    ap.add_argument("--task-id", type=int, default=env_int("SLURM_ARRAY_TASK_ID", 0))
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def sniff_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        try:
            sample = f.read(4096)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))


def ci_get(row, key, default=""):
    want = key.lower()
    for k in row.keys():
        if k.lower() == want:
            val = row.get(k)
            return val if val is not None else default
    return default


def norm_text(value):
    return str(value).strip().lower()


def normalize_visit(value):
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s.lower()


def visit_rank(value):
    s = normalize_visit(value)
    try:
        return (1, float(s))
    except Exception:
        return (0, s)


def resolve_lut_path(user_path):
    candidates = [user_path]
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if fs_home:
        candidates.append(os.path.join(fs_home, "FreeSurferColorLUT.txt"))
    candidates.append(DEFAULT_LUT_PATH)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return user_path


def read_fs_lut(lut_path):
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
    for base_name in base_names:
        for hemi in ("lh", "rh"):
            key = f"ctx-{hemi}-{base_name}"
            if key in lut:
                ids.append(lut[key])
            else:
                log(f"LUT missing label: {key}", level="WARN")
    return ids


def _ensure_3d_nifti(in_path, out_dir, tag):
    try:
        img = nib.load(in_path)
        if len(img.shape) == 4:
            out_path = os.path.join(out_dir, f"_{tag}_3d_tmp.nii.gz")
            data = np.asanyarray(img.dataobj)[..., 0]
            nib.save(nib.Nifti1Image(data, img.affine, img.header), out_path)
            return out_path, True
    except Exception as exc:
        log(f"Failed to inspect {in_path}: {exc}", level="WARN")
    return in_path, False


def make_aseg_mask_nifti(aseg_path, out_path):
    aseg_img = nib.load(aseg_path)
    data = np.asanyarray(aseg_img.dataobj)
    mask = np.isin(data, list(KEEP_LABELS)).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, aseg_img.affine), out_path)
    return int(mask.sum())


def shapes_affines_match(a_path, b_path, atol=ATOL_AFFINE):
    ia = nib.load(a_path)
    ib = nib.load(b_path)
    shape_ok = ia.shape == ib.shape
    affine_ok = np.allclose(ia.affine, ib.affine, atol=atol)
    return shape_ok and affine_ok, shape_ok, affine_ok, ia.shape, ib.shape, ia.affine, ib.affine


def run_command(cmd, env=None, tail_lines=30):
    log("RUN: " + " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if res.returncode != 0:
        tail = "\n".join(res.stderr.splitlines()[-tail_lines:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{tail}")
    return res


def compute_fs2t1_lta(fs_ref, t1_path, out_lta):
    if not shutil.which("mri_robust_register"):
        raise RuntimeError("mri_robust_register not found in PATH")
    run_command(
        [
            "mri_robust_register",
            "--mov", fs_ref,
            "--dst", t1_path,
            "--lta", out_lta,
            "--satit",
            "--iscale",
            "--maxit", "200",
        ]
    )
    return out_lta


def resample_label_with_lta(label_path, target_path, lta_path, out_path):
    if not shutil.which("mri_vol2vol"):
        raise RuntimeError("mri_vol2vol not found in PATH")
    run_command(
        [
            "mri_vol2vol",
            "--mov", label_path,
            "--targ", target_path,
            "--lta", lta_path,
            "--o", out_path,
            "--interp", "nearest",
        ]
    )
    return out_path


def write_mask_from_labels(label_path, id_list, out_path, dtype=np.uint8):
    img = nib.load(label_path)
    data = np.asanyarray(img.dataobj)
    mask = np.isin(data, list(id_list)).astype(dtype)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), out_path)
    return int(mask.sum())


def apply_mask_to_image(image_path, mask_bool, mask_affine, out_path):
    img = nib.load(image_path)
    data = np.asanyarray(img.dataobj)
    if data.shape != mask_bool.shape or not np.allclose(img.affine, mask_affine, atol=ATOL_AFFINE):
        raise RuntimeError(
            f"Mask grid mismatch for {image_path}: image_shape={data.shape}, mask_shape={mask_bool.shape}"
        )
    masked = data * mask_bool.astype(data.dtype)
    nib.save(nib.Nifti1Image(masked, img.affine, img.header), out_path)
    return int(mask_bool.sum())


def discover_pet_dirs(pet_root):
    entries = []
    for entry in os.scandir(pet_root):
        if entry.is_dir():
            entries.append(entry.path)
    entries.sort()
    return entries


def build_pet_index(rows):
    out = {}
    for row in rows:
        key = norm_text(ci_get(row, "session_label", ""))
        if key:
            out.setdefault(key, []).append(row)
    return out


def build_mr_index(rows):
    out = {}
    for row in rows:
        subject = norm_text(ci_get(row, "subject_label", ""))
        visit = normalize_visit(ci_get(row, "visit_num", ""))
        if subject and visit:
            out.setdefault((subject, visit), []).append(row)
    return out


def choose_subject_jobs(pet_dirs, pet_index, mr_index):
    groups = {}
    rejected = 0
    for pet_dir in pet_dirs:
        pet_session = os.path.basename(pet_dir)
        pet_path = os.path.join(pet_dir, "pet.nii.gz")
        if not os.path.isfile(pet_path):
            log(f"[SKIP:NO_PET_FILE] {pet_session} missing pet.nii.gz", level="WARN")
            rejected += 1
            continue

        pet_matches = pet_index.get(norm_text(pet_session), [])
        if not pet_matches:
            log(f"[SKIP:PET_CSV] {pet_session} not found in PET CSV", level="WARN")
            rejected += 1
            continue
        if len(pet_matches) > 1:
            log(f"[SKIP:PET_CSV_DUP] {pet_session} has {len(pet_matches)} PET CSV matches", level="WARN")
            rejected += 1
            continue

        pet_row = pet_matches[0]
        subject_label = str(ci_get(pet_row, "subject_label", "")).strip()
        visit_num = str(ci_get(pet_row, "visit_num", "")).strip()
        if not subject_label or not visit_num:
            log(f"[SKIP:PET_META] {pet_session} missing subject_label or visit_num in PET CSV", level="WARN")
            rejected += 1
            continue

        mr_matches = mr_index.get((norm_text(subject_label), normalize_visit(visit_num)), [])
        if not mr_matches:
            log(
                f"[SKIP:MR_JOIN] {pet_session} subject={subject_label} visit={visit_num} "
                "has no MR CSV match",
                level="WARN",
            )
            rejected += 1
            continue
        if len(mr_matches) > 1:
            log(
                f"[SKIP:MR_JOIN_DUP] {pet_session} subject={subject_label} visit={visit_num} "
                f"has {len(mr_matches)} MR CSV matches",
                level="WARN",
            )
            rejected += 1
            continue

        mr_row = mr_matches[0]
        mr_session = str(ci_get(mr_row, "session_label", "")).strip()
        if not mr_session:
            log(f"[SKIP:MR_SESSION] {pet_session} matched MR row with empty session_label", level="WARN")
            rejected += 1
            continue

        job = {
            "subject_label": subject_label,
            "visit_num": visit_num,
            "pet_session": pet_session,
            "mr_session": mr_session,
            "pet_path": pet_path,
        }
        groups.setdefault(norm_text(subject_label), []).append(job)

    chosen = []
    for subject_key in sorted(groups):
        jobs = groups[subject_key]
        jobs.sort(key=lambda item: (visit_rank(item["visit_num"]), item["pet_session"]), reverse=True)
        keep = jobs[0]
        chosen.append(keep)
        if len(jobs) > 1:
            skipped = [f"{j['pet_session']}(visit={j['visit_num']})" for j in jobs[1:]]
            log(
                f"[SUBJECT_DUP] {keep['subject_label']} keeping {keep['pet_session']}(visit={keep['visit_num']}) "
                f"and skipping {', '.join(skipped)}",
                level="WARN",
            )

    return chosen, rejected


def split_for_task(items, num_tasks, task_id):
    total = len(items)
    start = task_id * total // num_tasks
    end = (task_id + 1) * total // num_tasks
    return items[start:end], start, end


def find_fs_subject_dir(fs_root, mr_session):
    exact = os.path.join(fs_root, mr_session)
    if os.path.isdir(exact):
        return exact

    candidates = []
    try:
        for name in os.listdir(fs_root):
            path = os.path.join(fs_root, name)
            if os.path.isdir(path) and norm_text(name) == norm_text(mr_session):
                candidates.append(path)
        if not candidates:
            for name in os.listdir(fs_root):
                path = os.path.join(fs_root, name)
                if os.path.isdir(path) and norm_text(mr_session) in norm_text(name):
                    candidates.append(path)
    except FileNotFoundError:
        return None

    candidates.sort()
    return candidates[0] if candidates else None


def find_fs_leaf(fs_subject_dir):
    required = {"T1_fs_orig.nii.gz", "aseg.nii.gz", "aparc_aseg.nii.gz"}
    hits = []
    for root, _, files in os.walk(fs_subject_dir):
        if required.issubset(set(files)):
            hits.append(root)
    hits.sort(key=lambda path: (-path.count(os.sep), path))
    return hits[0] if hits else None


def choose_fs_ref(fs_leaf):
    for name in ("brainmask.nii.gz", "T1_fs_orig.nii.gz"):
        path = os.path.join(fs_leaf, name)
        if os.path.isfile(path):
            return path
    return None


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_pet_to_t1_registration(pet_path, t1_path, out_pet, out_mat, work_dir):
    if not shutil.which("flirt"):
        raise RuntimeError("flirt not found in PATH")

    pet_for_cli, pet_tmp = _ensure_3d_nifti(pet_path, work_dir, "pet")
    t1_for_cli, t1_tmp = _ensure_3d_nifti(t1_path, work_dir, "t1")
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    try:
        run_command(
            [
                "flirt",
                "-in", pet_for_cli,
                "-ref", t1_for_cli,
                "-out", out_pet,
                "-omat", out_mat,
                "-dof", "6",
                "-cost", "normmi",
                "-interp", "trilinear",
            ],
            env=env,
        )
    finally:
        for tmp_path, is_tmp in ((pet_for_cli, pet_tmp), (t1_for_cli, t1_tmp)):
            if is_tmp and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def process_job(job, args, lut):
    subject_label = job["subject_label"]
    out_dir = os.path.join(args.out_root, subject_label)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "source_sessions.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        existing_pet = str(existing.get("pet_session", "")).strip()
        if existing_pet and existing_pet != job["pet_session"]:
            msg = (
                f"{subject_label} already has output from pet_session={existing_pet}; "
                f"current={job['pet_session']}"
            )
            log(f"[SKIP:OUTPUT_COLLISION] {msg}", level="WARN")
            return {
                "subject_label": subject_label,
                "visit_num": job["visit_num"],
                "pet_session": job["pet_session"],
                "mr_session": job["mr_session"],
                "status": "skip_output_collision",
                "note": msg,
            }

    fs_subject_dir = find_fs_subject_dir(args.fs_root, job["mr_session"])
    if not fs_subject_dir:
        msg = f"MR session folder not found under {args.fs_root}"
        log(f"[SKIP:FS_SUBJECT] {job['mr_session']} {msg}", level="WARN")
        return {
            "subject_label": subject_label,
            "visit_num": job["visit_num"],
            "pet_session": job["pet_session"],
            "mr_session": job["mr_session"],
            "status": "skip_fs_subject",
            "note": msg,
        }

    fs_leaf = find_fs_leaf(fs_subject_dir)
    if not fs_leaf:
        msg = "No nested FreeSurfer export leaf with T1_fs_orig.nii.gz, aseg.nii.gz, aparc_aseg.nii.gz"
        log(f"[SKIP:FS_LEAF] {job['mr_session']} {msg}", level="WARN")
        return {
            "subject_label": subject_label,
            "visit_num": job["visit_num"],
            "pet_session": job["pet_session"],
            "mr_session": job["mr_session"],
            "status": "skip_fs_leaf",
            "note": msg,
        }

    t1_src = os.path.join(fs_leaf, "T1_fs_orig.nii.gz")
    aseg_src = os.path.join(fs_leaf, "aseg.nii.gz")
    aparc_src = os.path.join(fs_leaf, "aparc_aseg.nii.gz")
    fs_ref = choose_fs_ref(fs_leaf)
    for path in (t1_src, aseg_src, aparc_src):
        if not os.path.isfile(path):
            msg = f"Missing required FreeSurfer export file: {path}"
            log(f"[SKIP:FS_FILE] {msg}", level="WARN")
            return {
                "subject_label": subject_label,
                "visit_num": job["visit_num"],
                "pet_session": job["pet_session"],
                "mr_session": job["mr_session"],
                "status": "skip_fs_file",
                "note": msg,
            }

    dst_t1 = os.path.join(out_dir, "T1.nii.gz")
    dst_pet = os.path.join(out_dir, "PET_in_T1.nii.gz")
    pet_mat = os.path.join(out_dir, "PET_to_T1.mat")
    shutil.copy2(t1_src, dst_t1)

    try:
        run_pet_to_t1_registration(job["pet_path"], dst_t1, dst_pet, pet_mat, out_dir)
    except Exception as exc:
        log(f"[FAIL:FLIRT] {subject_label} {exc}", level="ERROR")
        return {
            "subject_label": subject_label,
            "visit_num": job["visit_num"],
            "pet_session": job["pet_session"],
            "mr_session": job["mr_session"],
            "status": "fail_flirt",
            "note": str(exc),
        }

    aseg_in_t1 = aseg_src
    aparc_in_t1 = aparc_src
    same_aseg, shape_aseg, aff_aseg, aseg_shape, t1_shape, aseg_aff, t1_aff = shapes_affines_match(
        aseg_src, dst_t1
    )
    same_aparc, shape_aparc, aff_aparc, aparc_shape, _, aparc_aff, _ = shapes_affines_match(
        aparc_src, dst_t1
    )

    if not same_aseg or not same_aparc:
        log(
            f"[MISMATCH] {subject_label} aseg(shape_ok={shape_aseg}, affine_ok={aff_aseg}) "
            f"aparc(shape_ok={shape_aparc}, affine_ok={aff_aparc})",
            level="WARN",
        )
        log(f"  aseg.shape={aseg_shape} t1.shape={t1_shape}", level="WARN")
        log(f"  aparc.shape={aparc_shape} t1.shape={t1_shape}", level="WARN")
        log("  aseg.affine:\n" + np.array2string(aseg_aff, precision=5), level="WARN")
        log("  aparc.affine:\n" + np.array2string(aparc_aff, precision=5), level="WARN")
        log("  t1.affine:\n" + np.array2string(t1_aff, precision=5), level="WARN")

        if not fs_ref:
            msg = "Could not find brainmask.nii.gz or T1_fs_orig.nii.gz for FS->T1 registration"
            log(f"[FAIL:FS_REF] {subject_label} {msg}", level="ERROR")
            return {
                "subject_label": subject_label,
                "visit_num": job["visit_num"],
                "pet_session": job["pet_session"],
                "mr_session": job["mr_session"],
                "status": "fail_fs_ref",
                "note": msg,
            }

        lta_path = os.path.join(out_dir, "fs_to_T1.lta")
        aseg_in_t1 = os.path.join(out_dir, "aseg_inT1.nii.gz")
        aparc_in_t1 = os.path.join(out_dir, "aparc_inT1.nii.gz")
        try:
            compute_fs2t1_lta(fs_ref, dst_t1, lta_path)
            resample_label_with_lta(aseg_src, dst_t1, lta_path, aseg_in_t1)
            resample_label_with_lta(aparc_src, dst_t1, lta_path, aparc_in_t1)
        except Exception as exc:
            log(f"[FAIL:LABEL_REG] {subject_label} {exc}", level="ERROR")
            return {
                "subject_label": subject_label,
                "visit_num": job["visit_num"],
                "pet_session": job["pet_session"],
                "mr_session": job["mr_session"],
                "status": "fail_label_registration",
                "note": str(exc),
            }

    paren_path = os.path.join(out_dir, "aseg_brainmask.nii.gz")
    bg_path = os.path.join(out_dir, "mask_basalganglia.nii.gz")
    nobg_path = os.path.join(out_dir, "mask_parenchyma_noBG.nii.gz")
    cortex_path = os.path.join(out_dir, "mask_cortex.nii.gz")

    make_aseg_mask_nifti(aseg_in_t1, paren_path)
    vox_bg = write_mask_from_labels(aseg_in_t1, BG_IDS, bg_path)
    vox_cortex = write_mask_from_labels(aseg_in_t1, CORTEX_LABELS, cortex_path)

    paren_img = nib.load(paren_path)
    bg_img = nib.load(bg_path)
    paren_np = np.asanyarray(paren_img.dataobj).astype(bool)
    bg_np = np.asanyarray(bg_img.dataobj).astype(bool)
    if paren_np.shape != bg_np.shape or not np.allclose(paren_img.affine, bg_img.affine, atol=ATOL_AFFINE):
        msg = "Parenchyma and BG masks are not on the same grid"
        log(f"[FAIL:MASK_GRID] {subject_label} {msg}", level="ERROR")
        return {
            "subject_label": subject_label,
            "visit_num": job["visit_num"],
            "pet_session": job["pet_session"],
            "mr_session": job["mr_session"],
            "status": "fail_mask_grid",
            "note": msg,
        }

    nobg_np = (paren_np & ~bg_np).astype(np.uint8)
    nib.save(nib.Nifti1Image(nobg_np, paren_img.affine, paren_img.header), nobg_path)

    roi_counts = {}
    hip_ids = [lut.get("Left-Hippocampus", 17), lut.get("Right-Hippocampus", 53)]
    hip_path = os.path.join(out_dir, "ROI_Hippocampus.nii.gz")
    roi_counts[Path(hip_path).name] = write_mask_from_labels(aseg_in_t1, hip_ids, hip_path)

    pcc_ids = ids_for_hemi_names(["posteriorcingulate"], lut)
    if pcc_ids:
        pcc_path = os.path.join(out_dir, "ROI_PosteriorCingulate.nii.gz")
        roi_counts[Path(pcc_path).name] = write_mask_from_labels(aparc_in_t1, pcc_ids, pcc_path)

    pcun_ids = ids_for_hemi_names(["precuneus"], lut)
    if pcun_ids:
        pcun_path = os.path.join(out_dir, "ROI_Precuneus.nii.gz")
        roi_counts[Path(pcun_path).name] = write_mask_from_labels(aparc_in_t1, pcun_ids, pcun_path)

    temp_ids = ids_for_hemi_names(TEMPORAL_BASE, lut)
    if temp_ids:
        temp_path = os.path.join(out_dir, "ROI_TemporalLobe.nii.gz")
        roi_counts[Path(temp_path).name] = write_mask_from_labels(aparc_in_t1, temp_ids, temp_path)

    limb_ids = ids_for_hemi_names(LIMBIC_BASE, lut)
    if limb_ids:
        limb_path = os.path.join(out_dir, "ROI_LimbicCortex.nii.gz")
        roi_counts[Path(limb_path).name] = write_mask_from_labels(aparc_in_t1, limb_ids, limb_path)

    t1_masked = os.path.join(out_dir, "T1_masked.nii.gz")
    pet_masked = os.path.join(out_dir, "PET_in_T1_masked.nii.gz")
    try:
        apply_mask_to_image(dst_t1, nobg_np.astype(bool), paren_img.affine, t1_masked)
        apply_mask_to_image(dst_pet, nobg_np.astype(bool), paren_img.affine, pet_masked)
    except Exception as exc:
        log(f"[FAIL:MASK_APPLY] {subject_label} {exc}", level="ERROR")
        return {
            "subject_label": subject_label,
            "visit_num": job["visit_num"],
            "pet_session": job["pet_session"],
            "mr_session": job["mr_session"],
            "status": "fail_mask_apply",
            "note": str(exc),
        }

    meta = {
        "subject_label": subject_label,
        "visit_num": job["visit_num"],
        "pet_session": job["pet_session"],
        "mr_session": job["mr_session"],
        "pet_source": job["pet_path"],
        "fs_subject_dir": fs_subject_dir,
        "fs_leaf": fs_leaf,
        "t1_source": t1_src,
        "aseg_source": aseg_src,
        "aparc_source": aparc_src,
        "roi_counts": roi_counts,
        "bg_voxels": int(vox_bg),
        "cortex_voxels": int(vox_cortex),
    }
    write_json(meta_path, meta)

    log(
        f"[OK] {subject_label} pet_session={job['pet_session']} mr_session={job['mr_session']} "
        f"fs_leaf={fs_leaf}"
    )
    log("  ROI voxel counts:\n" + json.dumps(roi_counts, indent=2))
    return {
        "subject_label": subject_label,
        "visit_num": job["visit_num"],
        "pet_session": job["pet_session"],
        "mr_session": job["mr_session"],
        "status": "ok",
        "note": fs_leaf,
    }


def write_summary(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_label", "visit_num", "pet_session", "mr_session", "status", "note"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    if args.num_tasks < 1:
        raise SystemExit("--num-tasks must be >= 1")
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise SystemExit("--task-id must satisfy 0 <= task-id < num-tasks")

    for path in (args.pet_root, args.fs_root, args.out_root):
        if path == args.out_root:
            os.makedirs(path, exist_ok=True)
        elif not os.path.isdir(path):
            raise SystemExit(f"Missing required directory: {path}")
    for path in (args.pet_csv, args.mr_csv):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing required CSV: {path}")

    lut_path = resolve_lut_path(args.lut_path)
    if not os.path.isfile(lut_path):
        raise SystemExit(f"Could not find FreeSurfer LUT: {args.lut_path}")

    pet_rows = sniff_csv_rows(args.pet_csv)
    mr_rows = sniff_csv_rows(args.mr_csv)
    pet_index = build_pet_index(pet_rows)
    mr_index = build_mr_index(mr_rows)
    lut = read_fs_lut(lut_path)

    pet_dirs = discover_pet_dirs(args.pet_root)
    log(f"Discovered PET session folders: {len(pet_dirs)}")
    chosen_jobs, rejected = choose_subject_jobs(pet_dirs, pet_index, mr_index)
    log(f"Usable subject-level jobs after joins/dedup: {len(chosen_jobs)} (rejected={rejected})")

    if args.limit is not None:
        chosen_jobs = chosen_jobs[:args.limit]
        log(f"Applying --limit={args.limit}: remaining jobs={len(chosen_jobs)}")

    selected_jobs, start, end = split_for_task(chosen_jobs, args.num_tasks, args.task_id)
    log(
        f"Task selection task_id={args.task_id} num_tasks={args.num_tasks} "
        f"rows[{start}:{end}] -> {len(selected_jobs)} subjects"
    )

    summary_rows = []
    ok = 0
    for idx, job in enumerate(selected_jobs, 1):
        log(
            f"\n[{idx}/{len(selected_jobs)}] subject={job['subject_label']} "
            f"visit={job['visit_num']} pet_session={job['pet_session']} mr_session={job['mr_session']}"
        )
        result = process_job(job, args, lut)
        summary_rows.append(result)
        if result["status"] == "ok":
            ok += 1

    summary_path = os.path.join(
        args.out_root,
        f"dian_processing_summary_task{args.task_id:02d}of{args.num_tasks:02d}.tsv",
    )
    write_summary(summary_path, summary_rows)
    log(f"Summary written to {summary_path}")
    log(f"Done. ok={ok} total_task_subjects={len(selected_jobs)}")


if __name__ == "__main__":
    main()
