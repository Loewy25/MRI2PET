#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib


ROOT = Path("/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids")
OUT = Path("/scratch/l.peiwang/DIAN_PET")
TAU = {"m62", "t80"}


def log(msg: str) -> None:
    print(msg, flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert DIAN tau PET DICOM to static pet.nii.gz")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out-root", default=str(OUT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-tasks", type=int, default=env_int("SLURM_ARRAY_TASK_COUNT", 1))
    ap.add_argument("--task-id", type=int, default=env_int("SLURM_ARRAY_TASK_ID", 0))
    return ap.parse_args()


def tracer_from_name(name: str) -> str | None:
    lower = name.lower()
    for tracer in TAU:
        if lower.endswith(f"_{tracer}"):
            return tracer
    return None


def split_for_task(items: list[Path], num_tasks: int, task_id: int) -> list[Path]:
    if num_tasks < 1:
        raise SystemExit("--num-tasks must be >= 1")
    if task_id < 0 or task_id >= num_tasks:
        raise SystemExit("--task-id must satisfy 0 <= task-id < num-tasks")
    n = len(items)
    start = task_id * n // num_tasks
    end = (task_id + 1) * n // num_tasks
    return items[start:end]


def is_bad_series(name: str) -> bool:
    n = name.lower()
    bad = [
        "topogram",
        "scout",
        "localizer",
        "protocol",
        "statistics",
        "fusion",
        "att_map",
        "ac_ct",
        "ct_att",
        "ct_brain",
        "ct1",
        "dose_report",
        "application_data",
        "ot1",
    ]
    return any(x in n for x in bad)


def n_dcm_files(dicom_dir: Path) -> int:
    return sum(1 for p in dicom_dir.iterdir() if p.is_file() and p.name.lower().endswith(".dcm"))


def score_series(name: str, tracer: str, n_dcm: int) -> int | None:
    n = name.lower()

    if is_bad_series(n):
        return None
    if n_dcm == 0:
        return None

    score = 0

    if tracer == "t80":
        if not any(x in n for x in ["av1451", "tau", "pet"]):
            return None

        if "av1451_static_no_filter" in n:
            score += 300
        if "static_no_filter" in n:
            score += 260
        if "short_no_filter" in n:
            score += 255
        if "75_105_min" in n:
            score += 250
        if "dian_obs_av1451_pet_ac" in n:
            score += 245
        if "iterative_all_pass" in n:
            score += 235
        if "static_gaussian" in n:
            score += 180
        if "short_gaussian" in n:
            score += 175

    elif tracer == "m62":
        if not any(x in n for x in ["mk6240", "tau", "pet", "short"]):
            return None

        if "pet_ctac_sum" in n:
            score += 320
        if "pet_ctac_dyn_sum" in n:
            score += 310
        if "mk6240_short" in n:
            score += 280
        if "short_no_filter" in n:
            score += 270

    if "pet" in n:
        score += 40
    if "sum" in n:
        score += 80
    if "static" in n:
        score += 70
    if "short" in n:
        score += 60
    if "brain" in n:
        score += 5
    if "ac" in n:
        score += 5
    if "no_filter" in n:
        score += 20

    if "dynamic" in n:
        score -= 140
    if "gaussian" in n:
        score -= 10
    if "fbp" in n:
        score -= 15
    if "truex" in n:
        score -= 12

    if n_dcm > 2000:
        score -= 40
    elif n_dcm >= 400 and n_dcm <= 1300:
        score += 10
    elif n_dcm >= 40 and n_dcm <= 200:
        score += 8

    return score


def find_candidates(session_dir: Path, tracer: str) -> list[tuple[int, str, Path, int]]:
    out: list[tuple[int, str, Path, int]] = []
    for sub in sorted(session_dir.iterdir()):
        if not sub.is_dir():
            continue
        dicom_dir = sub / "DICOM"
        if not dicom_dir.is_dir():
            continue
        count = n_dcm_files(dicom_dir)
        score = score_series(sub.name, tracer, count)
        if score is None:
            continue
        out.append((score, sub.name, dicom_dir, count))
    out.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return out


def run(cmd: list[str]) -> None:
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def convert_one(session_dir: Path, out_root: Path, overwrite: bool) -> str:
    tracer = tracer_from_name(session_dir.name)
    if tracer is None:
        return "skip"

    out_dir = out_root / session_dir.name
    final_pet = out_dir / "pet.nii.gz"
    if final_pet.exists() and not overwrite:
        log(f"[SKIP] {session_dir.name}: {final_pet} already exists")
        return "skip"

    candidates = find_candidates(session_dir, tracer)
    if not candidates:
        log(f"[SKIP] {session_dir.name}: no usable tau PET series found")
        return "skip"

    log("")
    log(f"=== {session_dir.name} ===")
    for score, name, _, count in candidates[:5]:
        log(f"  cand score={score:>4}  n_dcm={count:>4}  {name}")

    best_score, best_name, best_dicom, best_count = candidates[0]
    log(f"[USE] {best_name}  (score={best_score}, n_dcm={best_count})")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_pet_", dir=str(out_dir)))

    try:
        run([
            "dcm2niix",
            "-z", "y",
            "-b", "y",
            "-f", "pet",
            "-o", str(tmp_dir),
            str(best_dicom),
        ])

        nii_files = sorted([p for p in tmp_dir.iterdir() if p.is_file() and p.name.endswith((".nii", ".nii.gz"))])
        if len(nii_files) != 1:
            log(f"[FAIL] {session_dir.name}: expected 1 NIfTI, got {len(nii_files)}")
            for p in nii_files:
                log(f"       {p.name}")
            return "fail"

        src = nii_files[0]
        img = nib.load(str(src))
        log(f"[INFO] converted shape = {img.shape}")

        if len(img.shape) != 3:
            log(f"[SKIP] {session_dir.name}: chosen series converted to {img.shape}, not a 3D static PET")
            return "skip"

        nib.save(img, str(final_pet))
        log(f"[OK] {session_dir.name}: wrote {final_pet}")
        return "ok"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()

    if shutil.which("dcm2niix") is None:
        raise SystemExit("dcm2niix not found in PATH")

    root = Path(args.root)
    out_root = Path(args.out_root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    sessions = sorted([p for p in root.iterdir() if p.is_dir() and tracer_from_name(p.name) is not None])
    if args.limit is not None:
        sessions = sessions[:args.limit]
    sessions = split_for_task(sessions, args.num_tasks, args.task_id)

    out_root.mkdir(parents=True, exist_ok=True)

    log(f"ROOT     = {root}")
    log(f"OUT_ROOT = {out_root}")
    log(f"TASK     = {args.task_id}/{args.num_tasks}")
    log(f"N_SESS   = {len(sessions)}")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for session_dir in sessions:
        status = convert_one(session_dir, out_root, args.overwrite)
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_fail += 1

    log("")
    log("===== SUMMARY =====")
    log(f"OK   {n_ok}")
    log(f"SKIP {n_skip}")
    log(f"FAIL {n_fail}")


if __name__ == "__main__":
    main()
