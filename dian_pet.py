#!/usr/bin/env python3

"""
DIAN raw PET preprocessing.

This is a pragmatic PUP-like flow for the DIAN raw PET series:
1. Pick the PET DICOM series for each DIAN tau session.
2. Convert DICOM -> NIfTI with dcm2niix.
3. Preserve the raw dynamic PET if the series is 4D.
4. Motion-correct the dynamic frames with MCFLIRT when available.
5. Sum/average a selected frame window into a single 3D static PET.

The final usable image for downstream registration is always:
  /scratch/l.peiwang/DIAN_PET/<session_label>/pet.nii.gz

If the source series is dynamic, the folder also keeps:
  pet_dynamic.nii.gz
  pet_dynamic.json
  pet_dynamic_moco.nii.gz
  pet_preproc.json

This is intentionally PUP-like, not an exact clone of the WashU 4dfp toolchain.
It mirrors the same core idea: convert, motion-correct, then frame-sum to a
usable static PET volume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np


CSV_PATH = Path("/scratch/l.peiwang/DIAN_spreadsheet/DIANDF18_PET_session_details.csv")
INPUT_ROOT = Path("/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids")
OUTPUT_ROOT = Path("/scratch/l.peiwang/DIAN_PET")

TAU_TRACERS = {"t80", "m62"}


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}", flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Convert DIAN tau PET DICOMs and build a PUP-like static PET by "
            "motion-correcting and summing the dynamic frames."
        )
    )
    ap.add_argument("--csv-path", default=str(CSV_PATH))
    ap.add_argument("--input-root", default=str(INPUT_ROOT))
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-tasks", type=int, default=env_int("SLURM_ARRAY_TASK_COUNT", 1))
    ap.add_argument("--task-id", type=int, default=env_int("SLURM_ARRAY_TASK_ID", 0))
    ap.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="1-based first frame to include in the static PET. Default: use all frames.",
    )
    ap.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="1-based last frame to include in the static PET. Default: use all frames.",
    )
    ap.add_argument(
        "--skip-motion-correction",
        action="store_true",
        help="Skip MCFLIRT even when the PET is dynamic.",
    )
    return ap.parse_args()


def read_tau_sessions(csv_path: Path) -> list[dict[str, str]]:
    sessions = []
    seen: set[tuple[str, str]] = set()

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found in {csv_path}")

        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        if "session_label" not in field_map or "tracer_shortname" not in field_map:
            raise RuntimeError(
                "CSV must contain session_label and tracer_shortname columns. "
                f"Found: {reader.fieldnames}"
            )

        session_col = field_map["session_label"]
        tracer_col = field_map["tracer_shortname"]

        for row in reader:
            session_label = (row.get(session_col) or "").strip()
            tracer = (row.get(tracer_col) or "").strip().lower()
            if not session_label or tracer not in TAU_TRACERS:
                continue

            key = (session_label, tracer)
            if key in seen:
                continue
            seen.add(key)
            sessions.append({"session_label": session_label, "tracer": tracer})

    sessions.sort(key=lambda item: item["session_label"])
    return sessions


def split_for_task(items: list[dict[str, str]], num_tasks: int, task_id: int) -> list[dict[str, str]]:
    total = len(items)
    start = task_id * total // num_tasks
    end = (task_id + 1) * total // num_tasks
    return items[start:end]


def find_session_dir(input_root: Path, session_label: str) -> Path | None:
    exact = input_root / session_label
    if exact.is_dir():
        return exact

    candidates = sorted([p for p in input_root.glob(f"{session_label}*") if p.is_dir()])
    if len(candidates) == 1:
        log(f"{session_label}: exact folder not found, using close match {candidates[0].name}", level="WARN")
        return candidates[0]
    if len(candidates) > 1:
        log(f"{session_label}: multiple possible session folders:", level="WARN")
        for candidate in candidates:
            log(f"  - {candidate.name}", level="WARN")
        log(f"{session_label}: using {candidates[0].name}", level="WARN")
        return candidates[0]
    return None


def score_series_name(name: str, tracer: str) -> int:
    n = name.lower()
    score = 0

    if "topogram" in n or "localizer" in n:
        score -= 100
    if "ct" in n:
        score -= 80

    if "pet" in n:
        score += 20
    if "brain" in n:
        score += 2
    if "tau" in n:
        score += 25

    if tracer == "t80":
        if "av1451" in n:
            score += 40
        if "t807" in n:
            score += 35
        if "flortaucipir" in n:
            score += 35
        if "t80" in n:
            score += 10
    elif tracer == "m62":
        if "mk6240" in n:
            score += 40
        if "m62" in n:
            score += 15
        if "tau" in n:
            score += 15

    if "static" in n:
        score += 15
    if "short" in n:
        score += 10
    if "dynamic" in n:
        score += 5

    if "__ac" in n or "_ac_" in n or " pet_ac" in n:
        score += 4
    if "nac" in n:
        score -= 8

    if "gaussian" in n:
        score -= 1
    if "no_filter" in n:
        score += 1

    return score


def find_best_pet_dicom(session_dir: Path, tracer: str) -> tuple[Path | None, list[tuple[int, str, Path]]]:
    candidates: list[tuple[int, str, Path]] = []

    for sub in sorted(session_dir.iterdir()):
        if not sub.is_dir():
            continue
        dicom_dir = sub / "DICOM"
        if not dicom_dir.is_dir():
            continue
        score = score_series_name(sub.name, tracer)
        candidates.append((score, sub.name, dicom_dir))

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_dicom = candidates[0]
    if best_score < -20:
        return None, candidates
    return best_dicom, candidates


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    log("RUN: " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_dcm2niix(dicom_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "y",
        "-f",
        "pet",
        "-o",
        str(out_dir),
        str(dicom_dir),
    ]
    run_command(cmd)


def json_sidecar_for_nifti(nii_path: Path) -> Path:
    if nii_path.name.endswith(".nii.gz"):
        return nii_path.with_name(nii_path.name[:-7] + ".json")
    return nii_path.with_suffix(".json")


def find_primary_converted_image(tmp_dir: Path) -> tuple[Path | None, Path | None]:
    images = [p for p in tmp_dir.iterdir() if p.is_file() and p.name.endswith((".nii", ".nii.gz"))]
    if not images:
        return None, None
    images.sort(key=lambda p: (p.stat().st_size, p.name), reverse=True)
    image = images[0]
    json_path = json_sidecar_for_nifti(image)
    return image, (json_path if json_path.exists() else None)


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_number_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                return []
        return out
    if isinstance(value, str):
        text = value.replace(",", " ").strip()
        if not text:
            return []
        out = []
        for piece in text.split():
            try:
                out.append(float(piece))
            except ValueError:
                return []
        return out
    return []


def frame_weights_from_json(meta: dict, n_frames: int) -> list[float]:
    for key in (
        "FrameDuration",
        "FrameDurations",
        "FrameDurationSeconds",
        "FrameDurationsSeconds",
    ):
        values = coerce_number_list(meta.get(key))
        if len(values) == n_frames:
            return values
        if len(values) == 1 and n_frames > 1:
            return values * n_frames
    return [1.0] * n_frames


def choose_frame_window(n_frames: int, start_frame: int | None, end_frame: int | None) -> tuple[int, int]:
    start = 1 if start_frame is None else start_frame
    end = n_frames if end_frame is None else end_frame
    if start < 1 or end < 1 or start > end or end > n_frames:
        raise RuntimeError(
            f"Invalid frame window start={start} end={end} for n_frames={n_frames}. "
            "Frames are 1-based."
        )
    return start, end


def save_nifti_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> None:
    header = ref_img.header.copy()
    header.set_data_shape(data.shape)
    nib.save(nib.Nifti1Image(data, ref_img.affine, header), str(out_path))


def inspect_pet(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return img, data


def summarize_frames(data: np.ndarray) -> tuple[list[float], list[float]]:
    if data.ndim != 4:
        return [], []
    means = [float(data[..., i].mean()) for i in range(data.shape[3])]
    maxes = [float(data[..., i].max()) for i in range(data.shape[3])]
    return means, maxes


def motion_correct_dynamic(dynamic_path: Path, out_path: Path, skip_motion_correction: bool) -> tuple[Path, bool, str]:
    if skip_motion_correction:
        shutil.copy2(dynamic_path, out_path)
        return out_path, False, "motion correction skipped by flag"

    mcflirt = shutil.which("mcflirt")
    if mcflirt is None:
        shutil.copy2(dynamic_path, out_path)
        return out_path, False, "mcflirt not found; copied raw dynamic PET without motion correction"

    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    run_command(
        [
            mcflirt,
            "-in",
            str(dynamic_path),
            "-out",
            str(out_path),
            "-plots",
            "-mats",
            "-report",
        ],
        env=env,
    )
    return out_path, True, "mcflirt"


def collapse_dynamic_to_static(
    dynamic_path: Path,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    weights: list[float],
) -> dict[str, object]:
    img, data = inspect_pet(dynamic_path)
    if data.ndim != 4:
        raise RuntimeError(f"Expected a 4D PET for collapsing, got shape={data.shape} at {dynamic_path}")

    start_idx = start_frame - 1
    end_idx = end_frame
    data_sel = data[..., start_idx:end_idx]
    weights_sel = np.asarray(weights[start_idx:end_idx], dtype=np.float64)
    if weights_sel.shape[0] != data_sel.shape[3]:
        raise RuntimeError(
            f"Frame weight count mismatch: weights={weights_sel.shape[0]} frames={data_sel.shape[3]}"
        )
    if not np.isfinite(weights_sel).all() or np.sum(weights_sel) <= 0:
        weights_sel = np.ones(data_sel.shape[3], dtype=np.float64)

    weights_norm = weights_sel / np.sum(weights_sel)
    static = np.tensordot(data_sel, weights_norm, axes=([-1], [0]))
    save_nifti_like(img, static.astype(np.float32), out_path)

    return {
        "selected_frames_1based": list(range(start_frame, end_frame + 1)),
        "selected_frame_weights": [float(x) for x in weights_sel.tolist()],
        "normalized_frame_weights": [float(x) for x in weights_norm.tolist()],
    }


def convert_and_prepare_session(
    session_label: str,
    tracer: str,
    dicom_dir: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_pet = out_dir / "pet.nii.gz"
    if final_pet.exists() and not args.overwrite:
        return "skip", "final pet.nii.gz already exists"

    tmp_dir = out_dir / "_dcm2niix_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_dcm2niix(dicom_dir, tmp_dir)
        raw_nii, raw_json = find_primary_converted_image(tmp_dir)
        if raw_nii is None:
            raise RuntimeError("dcm2niix completed but no NIfTI output was found")

        meta = read_json(raw_json)
        raw_img, raw_data = inspect_pet(raw_nii)
        frame_means, frame_maxes = summarize_frames(raw_data)

        preproc_info = {
            "session_label": session_label,
            "tracer": tracer,
            "source_dicom_dir": str(dicom_dir),
            "source_series_name": dicom_dir.parent.name,
            "converted_shape": list(raw_img.shape),
            "converted_dtype": str(raw_data.dtype),
            "frame_means": frame_means,
            "frame_maxes": frame_maxes,
        }

        if raw_data.ndim == 3:
            shutil.copy2(raw_nii, final_pet)
            if raw_json and raw_json.exists():
                shutil.copy2(raw_json, out_dir / "pet_source.json")
            preproc_info.update(
                {
                    "mode": "static_input",
                    "motion_correction": False,
                    "selected_frames_1based": [1],
                }
            )
        elif raw_data.ndim == 4:
            n_frames = raw_data.shape[3]
            start_frame, end_frame = choose_frame_window(n_frames, args.start_frame, args.end_frame)

            raw_dynamic = out_dir / "pet_dynamic.nii.gz"
            shutil.copy2(raw_nii, raw_dynamic)
            if raw_json and raw_json.exists():
                shutil.copy2(raw_json, out_dir / "pet_dynamic.json")

            dynamic_moco = out_dir / "pet_dynamic_moco.nii.gz"
            dynamic_for_sum, did_moco, moco_note = motion_correct_dynamic(
                raw_dynamic, dynamic_moco, args.skip_motion_correction
            )
            frame_weights = frame_weights_from_json(meta, n_frames)
            sum_info = collapse_dynamic_to_static(
                dynamic_for_sum,
                final_pet,
                start_frame,
                end_frame,
                frame_weights,
            )
            preproc_info.update(
                {
                    "mode": "dynamic_to_static",
                    "n_frames": int(n_frames),
                    "motion_correction": bool(did_moco),
                    "motion_correction_note": moco_note,
                    "frame_weights_all": [float(x) for x in frame_weights],
                    **sum_info,
                }
            )
        else:
            raise RuntimeError(f"Unsupported PET dimensionality shape={raw_img.shape}")

        with (out_dir / "pet_preproc.json").open("w", encoding="utf-8") as f:
            json.dump(preproc_info, f, indent=2, sort_keys=True)
        return "ok", preproc_info["mode"]
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if shutil.which("dcm2niix") is None:
        raise SystemExit("dcm2niix not found in PATH")
    if args.num_tasks < 1:
        raise SystemExit("--num-tasks must be >= 1")
    if args.task_id < 0 or args.task_id >= args.num_tasks:
        raise SystemExit("--task-id must satisfy 0 <= task-id < num-tasks")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not input_root.exists():
        raise SystemExit(f"INPUT_ROOT not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    sessions = read_tau_sessions(csv_path)
    if args.limit is not None:
        sessions = sessions[: args.limit]
    sessions = split_for_task(sessions, args.num_tasks, args.task_id)

    log(f"Sessions assigned to task {args.task_id}/{args.num_tasks}: {len(sessions)}")
    log(
        f"Frame window: start={args.start_frame or 'auto/all'} end={args.end_frame or 'auto/all'} "
        f"skip_motion_correction={args.skip_motion_correction}"
    )

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for idx, item in enumerate(sessions, start=1):
        session_label = item["session_label"]
        tracer = item["tracer"]
        log(f"\n[{idx}/{len(sessions)}] {session_label} tracer={tracer}")

        session_dir = find_session_dir(input_root, session_label)
        if session_dir is None:
            log(f"{session_label}: session folder not found", level="WARN")
            n_skip += 1
            continue

        dicom_dir, candidates = find_best_pet_dicom(session_dir, tracer)
        if dicom_dir is None:
            log(f"{session_label}: no clear PET DICOM folder found", level="WARN")
            if candidates:
                for score, name, _ in candidates:
                    log(f"  candidate score={score:>4} {name}", level="WARN")
            n_skip += 1
            continue

        log(f"IN  {session_dir}")
        log(f"SER {dicom_dir.parent.name}")
        log(f"DIC {dicom_dir}")
        log(f"OUT {output_root / session_label}")
        for score, name, _ in candidates[:4]:
            log(f"TOP score={score:>4} {name}")

        if args.dry_run:
            log("dry-run only; not converting")
            n_ok += 1
            continue

        try:
            status, note = convert_and_prepare_session(
                session_label,
                tracer,
                dicom_dir,
                output_root / session_label,
                args,
            )
            if status == "ok":
                log(f"{session_label}: prepared PET ({note})")
                n_ok += 1
            else:
                log(f"{session_label}: {note}", level="WARN")
                n_skip += 1
        except subprocess.CalledProcessError as exc:
            log(f"{session_label}: command failed: {exc}", level="ERROR")
            n_fail += 1
        except Exception as exc:
            log(f"{session_label}: unexpected error: {exc}", level="ERROR")
            n_fail += 1

    log("")
    log("========== SUMMARY ==========")
    log(f"OK   {n_ok}")
    log(f"SKIP {n_skip}")
    log(f"FAIL {n_fail}")


if __name__ == "__main__":
    main()
