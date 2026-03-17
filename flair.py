#!/usr/bin/env python3
import argparse
import os
import re
import json
import socket
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import SimpleITK as sitk

# ============================================================
# USER SETTINGS
# ============================================================
META = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"

TAU_COL = "TAU_PET_Session"
MR_COL  = "MR_Session"

RAW_MR_ROOT   = "/ceph/chpc/mapped/benz04_kari/scans"
KARI_ALL_ROOT = "/scratch/l.peiwang/kari_all"

# Put your true mask convention first.
# If PET_in_T1_masked / T1_masked were built from another mask,
# move that mask filename to the front.
MASK_CANDIDATES = [
    "aseg_brainmask.nii.gz",
    "mask_parenchyma_noBG.nii.gz",
    "mask_cortex.nii.gz",
]

OVERWRITE = False
SAVE_QC = True
SAVE_NATIVE_COPY = False
LIMIT = None  # e.g. 20 for a quick test, or None for all

# N4 settings
N4_SHRINK_FACTOR = 4
N4_ITERS = [50, 50, 30, 20]

# Registration settings
MI_BINS = 50
REG_SAMPLING_PCT = 0.20
REG_SHRINK_FACTORS = [4, 2, 1]
REG_SMOOTHING_SIGMAS = [2, 1, 0]
REG_LEARNING_RATE = 2.0
REG_MIN_STEP = 1e-4
REG_ITERS = 200
REG_GRAD_TOL = 1e-8

# ============================================================
# OPTIONAL QC PLOTTING
# ============================================================
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ============================================================
# IGNORE RULES / FILE TYPES
# ============================================================
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

# ============================================================
# BASIC HELPERS
# ============================================================
def norm_id(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(".zip", "")
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "", s)
    return s

def toks(name: str):
    name = re.sub(r"^\d+[-_ ]*", "", name.lower())
    return set(t for t in re.split(r"[^a-z0-9]+", name) if t)

def is_volume_file(fname: str) -> bool:
    fl = fname.lower()
    return fl.endswith(VOLUME_EXTS)

def write_image(img, path):
    sitk.WriteImage(img, path, useCompression=True)

def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def identity_tx(dim: int):
    return sitk.Transform(dim, sitk.sitkIdentity)

def safe_tuple(x):
    return tuple(float(v) if isinstance(v, (np.floating,)) else int(v) if isinstance(v, (np.integer,)) else v for v in x)

# ============================================================
# CSV / DIRECTORY MAPPING
# ============================================================
def build_pet_to_mr_map(df: pd.DataFrame):
    tmp = df[[TAU_COL, MR_COL]].dropna().copy()
    tmp[TAU_COL] = tmp[TAU_COL].astype(str).str.strip()
    tmp[MR_COL] = tmp[MR_COL].astype(str).str.strip()

    pet_to_mr_candidates = defaultdict(set)
    for _, row in tmp.iterrows():
        pet_key = norm_id(row[TAU_COL])
        pet_to_mr_candidates[pet_key].add(row[MR_COL])

    pet_to_mr = {}
    ambiguous = {}
    for pet_key, mr_set in pet_to_mr_candidates.items():
        if len(mr_set) == 1:
            pet_to_mr[pet_key] = list(mr_set)[0]
        else:
            ambiguous[pet_key] = sorted(mr_set)

    return pet_to_mr, ambiguous

def build_normalized_dir_map(root: str):
    out = {}
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out[norm_id(d)] = d
    return out

def resolve_dir_from_map(name: str, dir_map: dict):
    return dir_map.get(norm_id(name))

def list_kari_subject_dirs(root: str):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        # Only treat folders with a T1 as actual subject folders
        if os.path.exists(os.path.join(p, "T1.nii.gz")):
            out.append(d)
    return out

def list_series_dirs(session_dir: str):
    return sorted([
        x for x in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, x))
    ])

# ============================================================
# FLAIR NAME-BASED CLASSIFICATION
# ============================================================
def flair_subtypes(series_name: str):
    """
    Name-based FLAIR family labels.
    """
    if IGNORE.search(series_name):
        return set()

    t = toks(series_name)
    if "flair" not in t:
        return set()

    out = set()

    # likely 3D FLAIR family
    if (
        "3d" in t or
        "space" in t or
        "cube" in t or
        "vista" in t or
        "sagittal" in t
    ):
        out.add("FLAIR_3D_NAME")

    # likely 2D axial family
    if "axial" in t or "tra" in t or "transverse" in t:
        out.add("FLAIR_2D_AXIAL_NAME")

    # other 2D orientation
    if "coronal" in t or "cor" in t:
        out.add("FLAIR_2D_CORONAL_NAME")

    # fallback ambiguous
    if not out:
        out.add("FLAIR_AMBIG_NAME")

    return out

# ============================================================
# READ RAW SERIES RECURSIVELY
# ============================================================
def find_dicoms_recursively(series_dir: str):
    """
    Search series_dir and descendants for readable DICOM series.
    Returns list of tuples:
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

def find_volume_files_recursively(series_dir: str):
    """
    Search series_dir and descendants for common volumetric files.
    Returns list of tuples:
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

def read_image_recursive(series_dir: str):
    """
    Try recursively:
      1) nested DICOM series
      2) common volumetric files
    Returns:
      img, err, source_type, chosen_location, aux_info
    """
    # --- DICOM first
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
            return img, None, "DICOM_RECURSIVE", chosen_root, {
                "n_dicom_files": int(n_files),
                "series_id": str(chosen_sid),
            }
    except Exception as e:
        dicom_err = str(e)
    else:
        dicom_err = None

    # --- volume file fallback
    try:
        vol_candidates = find_volume_files_recursively(series_dir)
        if vol_candidates:
            chosen_fp, chosen_size = sorted(
                vol_candidates, key=lambda x: x[1], reverse=True
            )[0]
            img = sitk.ReadImage(chosen_fp)
            return img, None, "VOLUME_FILE_RECURSIVE", chosen_fp, {
                "file_size_bytes": int(chosen_size)
            }
    except Exception as e:
        vol_err = str(e)
    else:
        vol_err = None

    if dicom_err and vol_err:
        err = f"no_readable_dicom_or_volume | dicom_err={dicom_err} | vol_err={vol_err}"
    elif dicom_err:
        err = f"no_readable_dicom_or_volume | dicom_err={dicom_err}"
    elif vol_err:
        err = f"no_readable_dicom_or_volume | vol_err={vol_err}"
    else:
        err = "no_dicom_series_or_volume_found_recursive"

    return None, err, None, None, {}

# ============================================================
# GEOMETRY / POLICY
# ============================================================
def geom_class_from_spacing_and_size(spacing, size, dim):
    """
    Heuristic labels:
      - LIKELY_3D_VOLUME
      - LIKELY_2D_ACQ_STORED_AS_3D
      - INTERMEDIATE_MIXED
      - NON3D_DIM_4, etc.
    """
    if dim != 3:
        return f"NON3D_DIM_{dim}"

    sx, sy, sz = spacing
    nx, ny, nz = size

    if min(sx, sy, sz) <= 0:
        return "BAD_GEOM"

    ratio = max(spacing) / min(spacing)

    if nz < 8:
        return "TOO_FEW_SLICES"

    if ratio <= 1.5 and sz <= 2.0:
        return "LIKELY_3D_VOLUME"

    if sz >= 3.0 or ratio >= 2.5:
        return "LIKELY_2D_ACQ_STORED_AS_3D"

    return "INTERMEDIATE_MIXED"

def classify_flair_candidate_for_v1(name_labels, dim, geom_label):
    """
    Version-1 policy:
      - reject non-3D (including 4D)
      - prefer usable 3D FLAIR
      - else fallback to usable 2D axial FLAIR
    """
    nl = set(name_labels)

    if dim != 3:
        return False, None, 99, f"reject_dim_{dim}"

    # first choice
    if ("FLAIR_3D_NAME" in nl) and (geom_label == "LIKELY_3D_VOLUME"):
        return True, "USE_3D", 0, None

    # second choice
    if ("FLAIR_2D_AXIAL_NAME" in nl) and (geom_label == "LIKELY_2D_ACQ_STORED_AS_3D"):
        return True, "USE_2D_AXIAL", 1, None

    return False, None, 99, f"reject_policy(name={sorted(nl)},geom={geom_label})"

def candidate_rank(c):
    """
    Lower is better.
    """
    spacing = c["spacing"]
    size = c["size"]
    ratio = max(spacing) / min(spacing) if min(spacing) > 0 else 999.0
    vox = int(np.prod(size))

    return (
        c["priority"],          # USE_3D first, then USE_2D_AXIAL
        max(spacing),           # smaller spacing is better
        ratio,                  # more isotropic is better
        spacing[2],             # thinner through-plane is better
        -size[2],               # more slices is better
        -vox,                   # more voxels is better
        c["series_name"].lower()
    )

# ============================================================
# MRI PREPROCESSING
# ============================================================
def n4_bias_correct(img):
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    # Otsu mask in native flair space
    mask = sitk.OtsuThreshold(img_f, 0, 1, 200)

    if N4_SHRINK_FACTOR > 1:
        shrink = [N4_SHRINK_FACTOR] * img_f.GetDimension()
        img_s = sitk.Shrink(img_f, shrink)
        mask_s = sitk.Shrink(mask, shrink)
    else:
        img_s = img_f
        mask_s = mask

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(N4_ITERS)

    _ = corrector.Execute(img_s, mask_s)
    log_bias = corrector.GetLogBiasFieldAsImage(img_f)
    corrected = img_f / sitk.Exp(log_bias)

    return sitk.Cast(corrected, sitk.sitkFloat32)

def rigid_register_3d(moving_img, fixed_img):
    fixed = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving = sitk.Cast(moving_img, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=MI_BINS)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(REG_SAMPLING_PCT, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=REG_LEARNING_RATE,
        minStep=REG_MIN_STEP,
        numberOfIterations=REG_ITERS,
        gradientMagnitudeTolerance=REG_GRAD_TOL
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel(REG_SHRINK_FACTORS)
    reg.SetSmoothingSigmasPerLevel(REG_SMOOTHING_SIGMAS)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)

    return final_transform, float(reg.GetMetricValue()), reg.GetOptimizerStopConditionDescription()

def resample_to_reference(moving_img, reference_img, transform=None, is_mask=False):
    if transform is None:
        transform = identity_tx(reference_img.GetDimension())

    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    out_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32

    return sitk.Resample(
        moving_img,
        reference_img,
        transform,
        interp,
        0.0,
        out_type
    )

def load_mask_in_t1_space(subject_path, t1_img):
    """
    Returns:
      mask_img_in_t1_space, mask_source
    """
    # preferred explicit mask files
    for mask_name in MASK_CANDIDATES:
        p = os.path.join(subject_path, mask_name)
        if os.path.exists(p):
            m = sitk.ReadImage(p)
            m = sitk.Cast(m > 0, sitk.sitkUInt8)
            m = resample_to_reference(m, t1_img, transform=identity_tx(t1_img.GetDimension()), is_mask=True)
            return m, mask_name

    # fallback: derive from T1_masked if needed
    t1_masked_path = os.path.join(subject_path, "T1_masked.nii.gz")
    if os.path.exists(t1_masked_path):
        tm = sitk.ReadImage(t1_masked_path)
        m = sitk.Cast(tm != 0, sitk.sitkUInt8)
        m = resample_to_reference(m, t1_img, transform=identity_tx(t1_img.GetDimension()), is_mask=True)
        return m, "DERIVED_FROM_T1_MASKED"

    return None, None

def apply_mask(img, mask):
    mask_bin = sitk.Cast(mask > 0, sitk.sitkUInt8)
    return sitk.Mask(sitk.Cast(img, sitk.sitkFloat32), mask_bin)

# ============================================================
# QC
# ============================================================
def robust_norm(arr):
    arr = arr.astype(np.float32)
    vals = arr[arr > 0]
    if vals.size < 20:
        vals = arr.reshape(-1)

    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out.astype(np.float32)

def get_mid_slices(arr_zyx):
    z = arr_zyx.shape[0] // 2
    y = arr_zyx.shape[1] // 2
    x = arr_zyx.shape[2] // 2

    axial = np.rot90(arr_zyx[z, :, :])
    coronal = np.rot90(arr_zyx[:, y, :])
    sagittal = np.rot90(arr_zyx[:, :, x])
    return axial, coronal, sagittal

def save_qc_png(t1_img, flair_img, flair_masked_img, out_png, title_text=""):
    if not HAS_MPL:
        return

    t1 = sitk.GetArrayFromImage(t1_img)
    fl = sitk.GetArrayFromImage(flair_img)
    flm = sitk.GetArrayFromImage(flair_masked_img) if flair_masked_img is not None else fl

    t1n = robust_norm(t1)
    fln = robust_norm(fl)
    flmn = robust_norm(flm)

    t1_s = get_mid_slices(t1n)
    fl_s = get_mid_slices(fln)
    flm_s = get_mid_slices(flmn)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    names = ["Axial", "Coronal", "Sagittal"]

    for i in range(3):
        axes[0, i].imshow(t1_s[i], cmap="gray")
        axes[0, i].imshow(fl_s[i], cmap="autumn", alpha=0.35)
        axes[0, i].set_title(f"{names[i]} overlay")
        axes[0, i].axis("off")

        axes[1, i].imshow(t1_s[i], cmap="gray")
        axes[1, i].imshow(flm_s[i], cmap="autumn", alpha=0.35)
        axes[1, i].set_title(f"{names[i]} overlay (masked)")
        axes[1, i].axis("off")

    if title_text:
        fig.suptitle(title_text, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# CANDIDATE COLLECTION
# ============================================================
def candidate_jsonable(c):
    out = {}
    for k, v in c.items():
        if k == "img":
            continue
        if isinstance(v, set):
            out[k] = sorted(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out

def collect_flair_candidates(raw_mr_session_path: str):
    candidates = []

    for series_name in list_series_dirs(raw_mr_session_path):
        name_labels = flair_subtypes(series_name)
        if not name_labels:
            continue

        series_path = os.path.join(raw_mr_session_path, series_name)
        img, err, source_type, chosen_location, aux_info = read_image_recursive(series_path)

        cand = {
            "series_name": series_name,
            "series_path": series_path,
            "name_labels": sorted(name_labels),
            "read_source": source_type,
            "chosen_location": chosen_location,
            "aux_info": aux_info,
            "img": None,
            "dim": None,
            "size": None,
            "spacing": None,
            "geom_label": None,
            "usable": False,
            "selection_class": None,
            "priority": 99,
            "reject_reason": None,
        }

        if img is None:
            cand["reject_reason"] = f"unreadable:{err}"
            candidates.append(cand)
            continue

        dim = img.GetDimension()
        size = tuple(int(x) for x in img.GetSize())
        spacing = tuple(float(x) for x in img.GetSpacing())

        geom_label = geom_class_from_spacing_and_size(
            spacing=tuple(spacing[:3]) if len(spacing) >= 3 else tuple(spacing),
            size=tuple(size[:3]) if len(size) >= 3 else tuple(size),
            dim=dim
        )

        usable, selection_class, priority, reject_reason = classify_flair_candidate_for_v1(
            name_labels=name_labels,
            dim=dim,
            geom_label=geom_label
        )

        cand.update({
            "img": img if usable else None,
            "dim": int(dim),
            "size": size,
            "spacing": spacing,
            "geom_label": geom_label,
            "usable": bool(usable),
            "selection_class": selection_class,
            "priority": int(priority),
            "reject_reason": reject_reason,
        })
        candidates.append(cand)

    return candidates

# ============================================================
# CLI / SHARD HELPERS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FLAIR_in_T1 outputs with optional task sharding."
    )
    parser.add_argument(
        "--raw-mr-root",
        type=str,
        default=os.environ.get("RAW_MR_ROOT", RAW_MR_ROOT),
        help="Raw MR root directory. Can also be set via RAW_MR_ROOT env var.",
    )
    parser.add_argument(
        "--wait-for-raw-root-sec",
        type=int,
        default=int(os.environ.get("RAW_MR_WAIT_SEC", "90")),
        help="Seconds to wait/retry if raw MR root is temporarily unavailable.",
    )
    parser.add_argument(
        "--raw-root-poll-sec",
        type=int,
        default=5,
        help="Polling interval (seconds) while waiting for raw MR root.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total parallel tasks. Defaults to SLURM_ARRAY_TASK_COUNT or 1.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Zero-based task index. Defaults to SLURM_ARRAY_TASK_ID or 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional subject cap (applied before sharding).",
    )
    return parser.parse_args()

def resolve_tasking(args):
    env_num_tasks = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if args.num_tasks is not None:
        num_tasks = int(args.num_tasks)
    elif env_num_tasks is not None:
        num_tasks = int(env_num_tasks)
    else:
        num_tasks = 1

    if args.task_id is not None:
        task_id = int(args.task_id)
    elif env_task_id is not None:
        task_id = int(env_task_id)
    else:
        task_id = 0

    if num_tasks < 1:
        raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError(f"task_id must be in [0, {num_tasks - 1}], got {task_id}")

    return task_id, num_tasks

def shard_subjects(subject_dirs, task_id, num_tasks):
    if num_tasks == 1:
        return subject_dirs
    return subject_dirs[task_id::num_tasks]

def manifest_path_for_task(task_id, num_tasks):
    if num_tasks == 1:
        return os.path.join(KARI_ALL_ROOT, "flair_processing_manifest_v1.csv")
    return os.path.join(
        KARI_ALL_ROOT,
        f"flair_processing_manifest_v1_part{task_id:02d}of{num_tasks:02d}.csv"
    )

def wait_for_directory(path, wait_sec=0, poll_sec=5):
    """
    Wait for a directory to exist/access in case of delayed mount visibility.
    Returns:
      (ok: bool, waited_sec: int, last_reason: str|None)
    """
    wait_sec = max(0, int(wait_sec))
    poll_sec = max(1, int(poll_sec))
    start = time.time()
    deadline = start + wait_sec
    last_reason = None

    while True:
        try:
            if os.path.isdir(path):
                return True, int(time.time() - start), None
            last_reason = "not_a_directory_or_not_found"
        except Exception as e:
            last_reason = str(e)

        if time.time() >= deadline:
            return False, int(time.time() - start), last_reason

        time.sleep(poll_sec)

# ============================================================
# MAIN SUBJECT PROCESSING
# ============================================================
def process_one_subject(subject_dir, pet_to_mr, ambiguous_pet_map, raw_dir_map, raw_mr_root):
    subject_path = os.path.join(KARI_ALL_ROOT, subject_dir)

    out_flair = os.path.join(subject_path, "FLAIR_in_T1.nii.gz")
    out_flair_masked = os.path.join(subject_path, "FLAIR_in_T1_masked.nii.gz")
    out_tfm = os.path.join(subject_path, "FLAIR_to_T1.tfm")
    out_meta = os.path.join(subject_path, "FLAIR_processing.json")
    out_qc = os.path.join(subject_path, "FLAIR_in_T1_qc.png")
    out_native = os.path.join(subject_path, "FLAIR_native.nii.gz")

    result = {
        "TAU_PET_Session_folder": subject_dir,
        "status": None,
        "MR_Session": None,
        "raw_MR_session_dir": None,
        "n_flair_candidates_total": 0,
        "n_flair_candidates_usable": 0,
        "chosen_series_name": None,
        "chosen_series_path": None,
        "chosen_name_labels": None,
        "chosen_geom_label": None,
        "chosen_selection_class": None,
        "chosen_read_source": None,
        "chosen_size": None,
        "chosen_spacing": None,
        "mask_source": None,
        "reg_metric": None,
        "reg_stop": None,
        "notes": None,
        "candidate_summary": None,
    }

    if (not OVERWRITE) and os.path.exists(out_flair):
        result["status"] = "SKIP_OUTPUT_EXISTS"
        return result

    pet_key = norm_id(subject_dir)

    if pet_key in ambiguous_pet_map:
        result["status"] = "AMBIGUOUS_PET_TO_MR_MAPPING"
        result["notes"] = ";".join(ambiguous_pet_map[pet_key])
        return result

    if pet_key not in pet_to_mr:
        result["status"] = "NO_CSV_MATCH_FOR_TAU_PET_SESSION"
        return result

    mr_session = pet_to_mr[pet_key]
    result["MR_Session"] = mr_session

    raw_mr_dir = resolve_dir_from_map(mr_session, raw_dir_map)
    if raw_mr_dir is None:
        result["status"] = "RAW_MR_SESSION_FOLDER_NOT_FOUND"
        return result

    result["raw_MR_session_dir"] = raw_mr_dir
    raw_mr_path = os.path.join(raw_mr_root, raw_mr_dir)

    t1_path = os.path.join(subject_path, "T1.nii.gz")
    if not os.path.exists(t1_path):
        result["status"] = "MISSING_T1_IN_KARI_ALL"
        return result

    candidates = collect_flair_candidates(raw_mr_path)
    result["n_flair_candidates_total"] = len(candidates)

    candidate_summary_bits = []
    for c in candidates:
        nm = "|".join(c["name_labels"]) if c["name_labels"] else "NONE"
        dm = f"dim={c['dim']}" if c["dim"] is not None else "dim=None"
        gm = f"geom={c['geom_label']}" if c["geom_label"] is not None else "geom=None"
        tag = c["selection_class"] if c["usable"] else f"REJ:{c['reject_reason']}"
        candidate_summary_bits.append(f"{c['series_name']}[{nm};{dm};{gm};{tag}]")
    result["candidate_summary"] = " || ".join(candidate_summary_bits)

    if len(candidates) == 0:
        result["status"] = "NO_FLAIR_SERIES_FOUND"
        return result

    usable = [c for c in candidates if c["usable"]]
    result["n_flair_candidates_usable"] = len(usable)

    if len(usable) == 0:
        result["status"] = "NO_USABLE_FLAIR_AFTER_FILTER"
        return result

    chosen = sorted(usable, key=candidate_rank)[0]

    result["chosen_series_name"] = chosen["series_name"]
    result["chosen_series_path"] = chosen["series_path"]
    result["chosen_name_labels"] = "|".join(chosen["name_labels"])
    result["chosen_geom_label"] = chosen["geom_label"]
    result["chosen_selection_class"] = chosen["selection_class"]
    result["chosen_read_source"] = chosen["read_source"]
    result["chosen_size"] = str(list(chosen["size"])) if chosen["size"] is not None else None
    result["chosen_spacing"] = str([round(float(x), 5) for x in chosen["spacing"]]) if chosen["spacing"] is not None else None

    # ------------------------------------------------------------
    # Actual preprocessing
    # ------------------------------------------------------------
    t1_img = sitk.ReadImage(t1_path)
    flair_native = chosen["img"]

    # Optional native copy
    if SAVE_NATIVE_COPY:
        write_image(sitk.Cast(flair_native, sitk.sitkFloat32), out_native)

    # Bias correction
    try:
        flair_bc = n4_bias_correct(flair_native)
    except Exception as e:
        flair_bc = sitk.Cast(flair_native, sitk.sitkFloat32)
        result["notes"] = f"N4_failed_fallback_to_native:{e}"

    # Register FLAIR -> T1
    final_tfm, metric_value, stop_desc = rigid_register_3d(flair_bc, t1_img)
    result["reg_metric"] = metric_value
    result["reg_stop"] = stop_desc

    flair_in_t1 = resample_to_reference(
        moving_img=flair_bc,
        reference_img=t1_img,
        transform=final_tfm,
        is_mask=False
    )
    flair_in_t1 = sitk.Cast(flair_in_t1, sitk.sitkFloat32)

    # Mask in T1 space
    mask_img, mask_source = load_mask_in_t1_space(subject_path, t1_img)
    result["mask_source"] = mask_source

    flair_in_t1_masked = None
    if mask_img is not None:
        flair_in_t1_masked = apply_mask(flair_in_t1, mask_img)

    # Save outputs
    write_image(flair_in_t1, out_flair)
    sitk.WriteTransform(final_tfm, out_tfm)

    if flair_in_t1_masked is not None:
        write_image(flair_in_t1_masked, out_flair_masked)

    # QC
    if SAVE_QC and HAS_MPL:
        title_text = (
            f"{subject_dir}\n"
            f"MR={mr_session} | series={chosen['series_name']} | "
            f"class={chosen['selection_class']} | geom={chosen['geom_label']}"
        )
        save_qc_png(
            t1_img=t1_img,
            flair_img=flair_in_t1,
            flair_masked_img=flair_in_t1_masked,
            out_png=out_qc,
            title_text=title_text
        )

    # Metadata JSON
    meta = {
        "TAU_PET_Session_folder": subject_dir,
        "MR_Session": mr_session,
        "raw_MR_session_dir": raw_mr_dir,
        "chosen_series_name": chosen["series_name"],
        "chosen_series_path": chosen["series_path"],
        "chosen_name_labels": chosen["name_labels"],
        "chosen_geom_label": chosen["geom_label"],
        "chosen_selection_class": chosen["selection_class"],
        "chosen_read_source": chosen["read_source"],
        "chosen_location": chosen["chosen_location"],
        "chosen_aux_info": chosen["aux_info"],
        "chosen_dim": chosen["dim"],
        "chosen_size": list(chosen["size"]),
        "chosen_spacing": list(chosen["spacing"]),
        "mask_source": mask_source,
        "reg_metric": metric_value,
        "reg_stop": stop_desc,
        "outputs": {
            "FLAIR_in_T1": os.path.basename(out_flair),
            "FLAIR_in_T1_masked": os.path.basename(out_flair_masked) if flair_in_t1_masked is not None else None,
            "FLAIR_to_T1": os.path.basename(out_tfm),
            "FLAIR_qc_png": os.path.basename(out_qc) if SAVE_QC and HAS_MPL else None,
            "FLAIR_native": os.path.basename(out_native) if SAVE_NATIVE_COPY else None,
        },
        "policy": {
            "reject_4d": True,
            "prefer_usable_3d": True,
            "fallback_usable_2d_axial": True,
        },
        "all_candidates": [candidate_jsonable(c) for c in candidates],
    }
    write_json(meta, out_meta)

    if flair_in_t1_masked is not None:
        result["status"] = "SUCCESS"
    else:
        result["status"] = "SUCCESS_NO_MASK"

    return result

# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    task_id, num_tasks = resolve_tasking(args)
    host = socket.gethostname()

    raw_mr_root = args.raw_mr_root
    ok, waited_sec, last_reason = wait_for_directory(
        raw_mr_root,
        wait_sec=args.wait_for_raw_root_sec,
        poll_sec=args.raw_root_poll_sec
    )
    if not ok:
        raise FileNotFoundError(
            "RAW_MR_ROOT not accessible "
            f"(host={host}, path={raw_mr_root}, waited_sec={waited_sec}, "
            f"last_reason={last_reason})"
        )

    df = pd.read_csv(META)
    pet_to_mr, ambiguous_pet_map = build_pet_to_mr_map(df)
    raw_dir_map = build_normalized_dir_map(raw_mr_root)

    all_subject_dirs = list_kari_subject_dirs(KARI_ALL_ROOT)
    effective_limit = LIMIT if args.limit is None else args.limit
    if effective_limit is not None:
        all_subject_dirs = all_subject_dirs[:effective_limit]
    subject_dirs = shard_subjects(all_subject_dirs, task_id, num_tasks)

    print("=== FLAIR build pipeline ===")
    print(f"Host          : {host}")
    print(f"KARI_ALL_ROOT : {KARI_ALL_ROOT}")
    print(f"RAW_MR_ROOT   : {raw_mr_root}")
    print(f"CSV           : {META}")
    print(f"Subjects with T1 in kari_all (total): {len(all_subject_dirs)}")
    print(f"Tasking       : task_id={task_id} / num_tasks={num_tasks}")
    print(f"Subjects in this task: {len(subject_dirs)}")
    print(f"Overwrite     : {OVERWRITE}")
    print(f"Save QC       : {SAVE_QC and HAS_MPL}")
    print("")

    manifest = []
    status_counter = Counter()
    chosen_counter = Counter()

    for i, subject_dir in enumerate(subject_dirs, 1):
        print(f"[{i:4d}/{len(subject_dirs)}] {subject_dir}")
        try:
            row = process_one_subject(
                subject_dir=subject_dir,
                pet_to_mr=pet_to_mr,
                ambiguous_pet_map=ambiguous_pet_map,
                raw_dir_map=raw_dir_map,
                raw_mr_root=raw_mr_root
            )
        except Exception as e:
            row = {
                "TAU_PET_Session_folder": subject_dir,
                "status": "EXCEPTION",
                "MR_Session": None,
                "raw_MR_session_dir": None,
                "n_flair_candidates_total": None,
                "n_flair_candidates_usable": None,
                "chosen_series_name": None,
                "chosen_series_path": None,
                "chosen_name_labels": None,
                "chosen_geom_label": None,
                "chosen_selection_class": None,
                "chosen_read_source": None,
                "chosen_size": None,
                "chosen_spacing": None,
                "mask_source": None,
                "reg_metric": None,
                "reg_stop": None,
                "notes": f"{e} || {traceback.format_exc(limit=1)}",
                "candidate_summary": None,
            }

        manifest.append(row)
        status_counter[row["status"]] += 1
        if row.get("chosen_selection_class"):
            chosen_counter[row["chosen_selection_class"]] += 1

        print(f"    -> {row['status']}")

    manifest_df = pd.DataFrame(manifest)
    manifest_path = manifest_path_for_task(task_id, num_tasks)
    manifest_df.to_csv(manifest_path, index=False)

    print("\n=== Done ===")
    print(f"Manifest saved to: {manifest_path}")

    print("\n=== Status counts ===")
    for k, v in sorted(status_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k:32s}: {v}")

    print("\n=== Chosen FLAIR class counts ===")
    for k, v in sorted(chosen_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{k:32s}: {v}")

if __name__ == "__main__":
    main()
