#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np


ROOT = Path('/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids')
OUT = Path('/scratch/l.peiwang/DIAN_PET')
TAU = {'m62', 't80'}


def log(msg: str) -> None:
    print(msg, flush=True)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Convert DIAN tau PET DICOM to /subject/pet.nii.gz')
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--out-root', default=str(OUT))
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--num-tasks', type=int, default=env_int('SLURM_ARRAY_TASK_COUNT', 1))
    ap.add_argument('--task-id', type=int, default=env_int('SLURM_ARRAY_TASK_ID', 0))
    ap.add_argument('--max-4d-frames', type=int, default=8, help='If converted PET is 4D with <= this many frames, average to 3D.')
    return ap.parse_args()


def tracer_from_name(name: str) -> str | None:
    lower = name.lower()
    for tracer in TAU:
        if lower.endswith(f'_{tracer}'):
            return tracer
    return None


def split_for_task(items: list[Path], num_tasks: int, task_id: int) -> list[Path]:
    if num_tasks < 1:
        raise SystemExit('--num-tasks must be >= 1')
    if task_id < 0 or task_id >= num_tasks:
        raise SystemExit('--task-id must satisfy 0 <= task-id < num-tasks')
    n = len(items)
    start = task_id * n // num_tasks
    end = (task_id + 1) * n // num_tasks
    return items[start:end]


def is_bad_series(name: str) -> bool:
    n = name.lower()
    bad = [
        'topogram', 'scout', 'localizer', 'protocol', 'statistics', 'fusion',
        'att_map', 'ac_ct', 'ct_att', 'ct_brain', 'ct1', 'dose_report',
        'application_data', 'ot1', 'mip', 'screen', 'report'
    ]
    return any(x in n for x in bad)


def n_dcm_files(dicom_dir: Path) -> int:
    return sum(1 for p in dicom_dir.iterdir() if p.is_file() and p.name.lower().endswith('.dcm'))


def score_series(name: str, tracer: str, n_dcm: int) -> int | None:
    n = name.lower()

    if is_bad_series(n) or n_dcm == 0:
        return None

    score = 0

    if tracer == 't80':
        if not any(x in n for x in ['av1451', 'tau', 'pet']):
            return None
        if 'av1451_static_no_filter' in n:
            score += 300
        elif 'static_no_filter' in n:
            score += 285
        elif 'static' in n:
            score += 260
        elif 'short_no_filter' in n:
            score += 245
        elif 'short' in n:
            score += 225
        elif 'pet_ac' in n:
            score += 210
    else:
        if not any(x in n for x in ['mk6240', 'tau', 'pet', 'short', 'sum', 'ctac']):
            return None
        if 'pet_ctac_sum' in n or 'pet_ctac_dyn_sum' in n:
            score += 330
        elif 'short_no_filter' in n:
            score += 285
        elif 'mk6240_short' in n:
            score += 275
        elif 'short' in n:
            score += 240
        elif 'pet' in n:
            score += 190

    if 'pet' in n:
        score += 35
    if 'sum' in n:
        score += 70
    if 'static' in n:
        score += 60
    if 'short' in n:
        score += 50
    if 'no_filter' in n:
        score += 20
    if 'gaussian' in n:
        score -= 15
    if 'dynamic' in n:
        score -= 120
    if 'fbp' in n:
        score -= 15
    if 'truex' in n:
        score -= 10

    # Prefer likely single-volume static series when they exist.
    if 50 <= n_dcm <= 250:
        score += 30
    elif 400 <= n_dcm <= 1300:
        score += 5
    elif n_dcm > 3000:
        score -= 80

    return score


def find_candidates(session_dir: Path, tracer: str) -> list[tuple[int, str, Path, int]]:
    out: list[tuple[int, str, Path, int]] = []
    for sub in sorted(session_dir.iterdir()):
        if not sub.is_dir():
            continue
        dicom_dir = sub / 'DICOM'
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
    log('RUN: ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def write_like(ref_img: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_shape(data.shape)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_img.affine, hdr), str(out_path))


def positive_stats(data: np.ndarray) -> str:
    x = data[np.isfinite(data) & (data > 0)]
    if x.size == 0:
        return 'no positive voxels'
    p = np.percentile(x, [1, 50, 90, 95, 99, 99.5, 100])
    return (
        f'p01={p[0]:.4g} p50={p[1]:.4g} p90={p[2]:.4g} '
        f'p95={p[3]:.4g} p99={p[4]:.4g} p99.5={p[5]:.4g} max={p[6]:.4g}'
    )


def frame_means_text(data4d: np.ndarray) -> str:
    means = [float(np.mean(data4d[..., i])) for i in range(data4d.shape[3])]
    return ' '.join(f'{m:.4g}' for m in means)


def convert_candidate(dicom_dir: Path, out_dir: Path, max_4d_frames: int) -> tuple[str, str] | tuple[None, str]:
    tmp_dir = Path(tempfile.mkdtemp(prefix='tmp_pet_', dir=str(out_dir)))
    try:
        run([
            'dcm2niix',
            '-z', 'y',
            '-b', 'y',
            '-f', 'pet',
            '-o', str(tmp_dir),
            str(dicom_dir),
        ])

        nii_files = sorted([p for p in tmp_dir.iterdir() if p.is_file() and p.name.endswith(('.nii', '.nii.gz'))])
        if len(nii_files) != 1:
            return None, f'expected 1 NIfTI, got {len(nii_files)}'

        src = nii_files[0]
        img = nib.load(str(src))
        shape = img.shape
        log(f'[INFO] converted shape = {shape}')

        if len(shape) == 3:
            data = img.get_fdata(dtype=np.float32)
            return '3d', positive_stats(data)

        if len(shape) == 4 and shape[3] <= max_4d_frames:
            data4d = img.get_fdata(dtype=np.float32)
            log(f'[INFO] frame means = {frame_means_text(data4d)}')
            avg = np.mean(data4d, axis=3, dtype=np.float32)
            out_file = out_dir / 'pet.nii.gz'
            write_like(img, avg, out_file)
            return '4d_mean', positive_stats(avg)

        return None, f'converted to {shape}, too many frames to auto-average'

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def convert_one(session_dir: Path, out_root: Path, overwrite: bool, max_4d_frames: int) -> str:
    tracer = tracer_from_name(session_dir.name)
    if tracer is None:
        return 'skip'

    out_dir = out_root / session_dir.name
    final_pet = out_dir / 'pet.nii.gz'
    if final_pet.exists() and not overwrite:
        log(f'[SKIP] {session_dir.name}: {final_pet} already exists')
        return 'skip'

    candidates = find_candidates(session_dir, tracer)
    if not candidates:
        log(f'[SKIP] {session_dir.name}: no usable tau PET series found')
        return 'skip'

    log('')
    log(f'=== {session_dir.name} ===')
    for score, name, _, count in candidates[:6]:
        log(f'  cand score={score:>4}  n_dcm={count:>4}  {name}')

    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, (score, name, dicom_dir, count) in enumerate(candidates[:4], start=1):
        log(f'[TRY {rank}] {name}  (score={score}, n_dcm={count})')
        kind, note = convert_candidate(dicom_dir, out_dir, max_4d_frames)

        if kind == '3d':
            # reconvert once more and save directly, to keep code very simple
            tmp_dir = Path(tempfile.mkdtemp(prefix='tmp_pet_', dir=str(out_dir)))
            try:
                run([
                    'dcm2niix',
                    '-z', 'y',
                    '-b', 'y',
                    '-f', 'pet',
                    '-o', str(tmp_dir),
                    str(dicom_dir),
                ])
                nii_files = sorted([p for p in tmp_dir.iterdir() if p.is_file() and p.name.endswith(('.nii', '.nii.gz'))])
                if len(nii_files) != 1:
                    log(f'[FAIL] {session_dir.name}: expected 1 NIfTI on final save, got {len(nii_files)}')
                    return 'fail'
                img = nib.load(str(nii_files[0]))
                data = img.get_fdata(dtype=np.float32)
                write_like(img, data, final_pet)
                log(f'[OK] {session_dir.name}: wrote {final_pet} from {name}')
                log(f'[INFO] positive voxel stats = {note}')
                return 'ok'
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if kind == '4d_mean':
            log(f'[OK] {session_dir.name}: wrote {final_pet} as mean over small 4D series from {name}')
            log(f'[INFO] positive voxel stats = {note}')
            return 'ok'

        log(f'[NO]  {name}: {note}')

    log(f'[SKIP] {session_dir.name}: no candidate produced a usable 3D PET')
    return 'skip'


def main() -> None:
    args = parse_args()

    if shutil.which('dcm2niix') is None:
        raise SystemExit('dcm2niix not found in PATH')

    root = Path(args.root)
    out_root = Path(args.out_root)
    if not root.exists():
        raise SystemExit(f'root not found: {root}')

    sessions = sorted([p for p in root.iterdir() if p.is_dir() and tracer_from_name(p.name) is not None])
    if args.limit is not None:
        sessions = sessions[:args.limit]
    sessions = split_for_task(sessions, args.num_tasks, args.task_id)

    out_root.mkdir(parents=True, exist_ok=True)

    log(f'ROOT          = {root}')
    log(f'OUT_ROOT      = {out_root}')
    log(f'TASK          = {args.task_id}/{args.num_tasks}')
    log(f'N_SESS        = {len(sessions)}')
    log(f'MAX_4D_FRAMES = {args.max_4d_frames}')

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for session_dir in sessions:
        try:
            status = convert_one(session_dir, out_root, args.overwrite, args.max_4d_frames)
        except subprocess.CalledProcessError as exc:
            log(f'[FAIL] {session_dir.name}: command failed: {exc}')
            status = 'fail'
        except Exception as exc:
            log(f'[FAIL] {session_dir.name}: unexpected error: {exc}')
            status = 'fail'

        if status == 'ok':
            n_ok += 1
        elif status == 'skip':
            n_skip += 1
        else:
            n_fail += 1

    log('')
    log('===== SUMMARY =====')
    log(f'OK   {n_ok}')
    log(f'SKIP {n_skip}')
    log(f'FAIL {n_fail}')


if __name__ == '__main__':
    main()
