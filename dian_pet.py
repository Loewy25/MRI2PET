#!/usr/bin/env python3
from pathlib import Path
import csv
import shutil
import subprocess
import sys

CSV_PATH = Path('/scratch/l.peiwang/DIAN_spreadsheet/DIANDF18_PET_session_details.csv')
INPUT_ROOT = Path('/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids')
OUTPUT_ROOT = Path('/scratch/l.peiwang/DIAN_PET')

TAU_TRACERS = {'t80', 'm62'}

# First run with True to check folder selections only
DRY_RUN = True

# Set True if you want to reconvert even when pet.nii.gz already exists
OVERWRITE = False


def read_tau_sessions(csv_path: Path):
    sessions = []
    seen = set()

    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f'No header found in {csv_path}')

        field_map = {name.strip().lower(): name for name in reader.fieldnames}

        if 'session_label' not in field_map or 'tracer_shortname' not in field_map:
            raise RuntimeError(
                'CSV must contain session_label and tracer_shortname columns. '
                f'Found: {reader.fieldnames}'
            )

        session_col = field_map['session_label']
        tracer_col = field_map['tracer_shortname']

        for row in reader:
            session_label = (row.get(session_col) or '').strip()
            tracer = (row.get(tracer_col) or '').strip().lower()

            if not session_label:
                continue
            if tracer not in TAU_TRACERS:
                continue

            key = (session_label, tracer)
            if key in seen:
                continue
            seen.add(key)

            sessions.append({
                'session_label': session_label,
                'tracer': tracer,
            })

    return sessions


def find_session_dir(session_label: str):
    exact = INPUT_ROOT / session_label
    if exact.is_dir():
        return exact

    # fallback: sometimes weird suffix/prefix may exist
    candidates = sorted([p for p in INPUT_ROOT.glob(f'{session_label}*') if p.is_dir()])

    if len(candidates) == 1:
        print(f'  [WARN] exact folder not found, using close match: {candidates[0].name}')
        return candidates[0]

    if len(candidates) > 1:
        print(f'  [WARN] multiple possible session folders for {session_label}:')
        for p in candidates:
            print(f'         - {p.name}')
        print(f'         using: {candidates[0].name}')
        return candidates[0]

    return None


def score_series_name(name: str, tracer: str) -> int:
    n = name.lower()
    score = 0

    # positive PET hints
    if 'pet' in n:
        score += 20
    if 'brain' in n:
        score += 2
    if 'tau' in n:
        score += 15

    # tracer-specific hints
    if tracer == 't80':
        if 'av1451' in n:
            score += 20
        if 't807' in n:
            score += 20
        if 't80' in n:
            score += 10
    elif tracer == 'm62':
        if 'mk6240' in n:
            score += 20
        if 'm62' in n:
            score += 10
        if 'tau' in n:
            score += 5

    # negative hints
    if 'ct' in n:
        score -= 50
    if 'ac' in n:
        score -= 30
    if 'localizer' in n:
        score -= 20
    if 'cal' in n:
        score -= 10

    return score


def find_best_pet_dicom(session_dir: Path, tracer: str):
    candidates = []

    for sub in sorted(session_dir.iterdir()):
        if not sub.is_dir():
            continue

        dicom_dir = sub / 'DICOM'
        if not dicom_dir.is_dir():
            continue

        score = score_series_name(sub.name, tracer)
        candidates.append((score, sub.name, dicom_dir))

    if not candidates:
        return None, []

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_name, best_dicom = candidates[0]

    if best_score <= 0:
        return None, candidates

    return best_dicom, candidates


def run_dcm2niix(dicom_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'dcm2niix',
        '-z', 'y',
        '-b', 'y',
        '-f', 'pet',
        '-o', str(out_dir),
        str(dicom_dir),
    ]

    print('  [CMD]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if shutil.which('dcm2niix') is None:
        print('[ERROR] dcm2niix not found in PATH')
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f'[ERROR] CSV not found: {CSV_PATH}')
        sys.exit(1)

    if not INPUT_ROOT.exists():
        print(f'[ERROR] INPUT_ROOT not found: {INPUT_ROOT}')
        sys.exit(1)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sessions = read_tau_sessions(CSV_PATH)

    print(f'[INFO] tau sessions in CSV: {len(sessions)}')
    print(f'[INFO] DRY_RUN={DRY_RUN}')
    print(f'[INFO] OVERWRITE={OVERWRITE}')

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for i, item in enumerate(sessions, start=1):
        session_label = item['session_label']
        tracer = item['tracer']

        print(f'\n[{i}/{len(sessions)}] {session_label}   tracer={tracer}')

        session_dir = find_session_dir(session_label)
        if session_dir is None:
            print('  [SKIP] session folder not found')
            n_skip += 1
            continue

        dicom_dir, candidates = find_best_pet_dicom(session_dir, tracer)
        if dicom_dir is None:
            print('  [SKIP] no clear PET DICOM folder found')
            if candidates:
                print('  [INFO] candidate series:')
                for score, name, _ in candidates:
                    print(f'         score={score:>3}   {name}')
            n_skip += 1
            continue

        out_dir = OUTPUT_ROOT / session_label
        nii_gz = out_dir / 'pet.nii.gz'
        nii = out_dir / 'pet.nii'

        print(f'  [IN ] {session_dir}')
        print(f'  [SER] {dicom_dir.parent.name}')
        print(f'  [DIC] {dicom_dir}')
        print(f'  [OUT] {out_dir}')

        if not OVERWRITE and (nii_gz.exists() or nii.exists()):
            print('  [SKIP] output already exists')
            n_skip += 1
            continue

        if DRY_RUN:
            print('  [DRY] not converting')
            n_ok += 1
            continue

        try:
            run_dcm2niix(dicom_dir, out_dir)
        except subprocess.CalledProcessError as e:
            print(f'  [FAIL] dcm2niix failed: {e}')
            n_fail += 1
            continue
        except Exception as e:
            print(f'  [FAIL] unexpected error: {e}')
            n_fail += 1
            continue

        if nii_gz.exists() or nii.exists():
            print('  [OK] conversion done')
            n_ok += 1
        else:
            print('  [FAIL] conversion ran but pet.nii(.gz) not found')
            n_fail += 1

    print('\n========== SUMMARY ==========')
    print(f'[OK]   {n_ok}')
    print(f'[SKIP] {n_skip}')
    print(f'[FAIL] {n_fail}')


if __name__ == '__main__':
    main()
