from pathlib import Path

ROOT = Path('/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids')

targets = [
    '2M4W5N_v02_m62',
    'YFCJ4O_v02_m62',
    'RYPQT3_v04_m62',
    '4G4Z8E_v04_m62',
    '1BVPA1_v11_m62',
    '1EBBWV_v08_t80',
    'SADZMU_v06_t80',
    'TLPC33_v07_t80',
    '8CCNE3_v09_t80',
    'H9CLW8_v09_t80',
    'VS5VHS_v03_t80',
    'LC1BJ7_v00_t80',
    'KSVHOU_v09_t80',
    'ZL1BDX_v11_t80',
]

for name in targets:
    p = ROOT / name
    print(f'\n=== {name} ===')
    if not p.is_dir():
        print('  [missing folder]')
        continue

    subs = sorted([x for x in p.iterdir() if x.is_dir()])
    if not subs:
        print('  [no subfolders]')
        continue

    for s in subs:
        tag = ' [DICOM]' if (s / 'DICOM').is_dir() else ''
        print(f'  - {s.name}{tag}')
