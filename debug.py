#!/usr/bin/env python3
from pathlib import Path
import subprocess
import shutil

INPUT_ROOT = Path('/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids')
OUTPUT_ROOT = Path('/scratch/l.peiwang/DIAN_PET')

cases = {
    '2M4W5N_v02_m62': '12-PET_CTAC_DYNAMIC',
    'YFCJ4O_v02_m62': '12-PET_CTAC_DYNAMIC',
    'RYPQT3_v04_m62': '12-PET_CTAC_DYNAMIC',
    '4G4Z8E_v04_m62': '12-PET_CTAC_DYNAMIC',
    '1BVPA1_v11_m62': '12-PET_CTAC_DYNAMIC',
    '1EBBWV_v08_t80': '1-Dynamic_emission',
    'SADZMU_v06_t80': '1-Dynamic_emission',
    'TLPC33_v07_t80': '1-Dynamic_emission',
    '8CCNE3_v09_t80': '1-Dynamic_emission',
    'H9CLW8_v09_t80': '1-Dynamic_emission',
    'VS5VHS_v03_t80': '1-Dynamic_emission',
    'LC1BJ7_v00_t80': '1-Dynamic_emission',
    'KSVHOU_v09_t80': '1-Dynamic_emission',
    'ZL1BDX_v11_t80': '1-Dynamic_emission',
}

DRY_RUN = True   # change to False for real run
OVERWRITE = False

if shutil.which('dcm2niix') is None:
    raise RuntimeError('dcm2niix not found in PATH')

for sess, series in cases.items():
    sess_dir = INPUT_ROOT / sess
    series_dir = sess_dir / series
    dicom_dir = series_dir / 'DICOM'
    out_dir = OUTPUT_ROOT / sess
    out_file_gz = out_dir / 'pet.nii.gz'
    out_file = out_dir / 'pet.nii'

    print(f'\n=== {sess} ===')
    print(f'  series: {series}')

    if not sess_dir.is_dir():
        print('  [SKIP] missing session folder')
        continue

    if dicom_dir.is_dir():
        src = dicom_dir
    elif series_dir.is_dir():
        src = series_dir
    else:
        print('  [SKIP] missing series folder')
        continue

    if not OVERWRITE and (out_file_gz.exists() or out_file.exists()):
        print('  [SKIP] output exists')
        continue

    print(f'  in : {src}')
    print(f'  out: {out_dir}')

    cmd = [
        'dcm2niix',
        '-z', 'y',
        '-b', 'y',
        '-f', 'pet',
        '-o', str(out_dir),
        str(src),
    ]
    print('  cmd:', ' '.join(cmd))

    if DRY_RUN:
        print('  [DRY]')
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True)
        if out_file_gz.exists() or out_file.exists():
            print('  [OK]')
        else:
            print('  [FAIL] no pet.nii(.gz) found')
    except Exception as e:
        print(f'  [FAIL] {e}')
