#!/usr/bin/env python3

from pathlib import Path
import csv
import argparse


def count_dcms(dicom_dir: Path) -> int:
    return sum(
        1
        for p in dicom_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(".dcm")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/ceph/chpc/mapped/dian_obs_data_shared/obs_pet_scans_imagids",
        help="Root folder containing session folders like 2M4W5N_v04_m62",
    )
    ap.add_argument(
        "--out-csv",
        default="pet_series_inventory.csv",
        help="CSV file to save the inventory",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    rows = []

    session_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for session_dir in session_dirs:
        subdirs = sorted([p for p in session_dir.iterdir() if p.is_dir()])

        if not subdirs:
            rows.append(
                {
                    "session_label": session_dir.name,
                    "series_folder": "",
                    "has_dicom_dir": False,
                    "n_dcm_files": 0,
                }
            )
            continue

        for sub in subdirs:
            dicom_dir = sub / "DICOM"
            has_dicom = dicom_dir.is_dir()
            n_dcm = count_dcms(dicom_dir) if has_dicom else 0

            rows.append(
                {
                    "session_label": session_dir.name,
                    "series_folder": sub.name,
                    "has_dicom_dir": has_dicom,
                    "n_dcm_files": n_dcm,
                }
            )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session_label", "series_folder", "has_dicom_dir", "n_dcm_files"],
        )
        writer.writeheader()
        writer.writerows(rows)

    current = None
    for row in rows:
        if row["session_label"] != current:
            current = row["session_label"]
            print(f"\n{current}")
        print(
            f"  {row['series_folder']:<28} "
            f"DICOM={str(row['has_dicom_dir']):<5} "
            f"n_dcm={row['n_dcm_files']}"
        )

    print(f"\nSaved inventory to: {args.out_csv}")


if __name__ == "__main__":
    main()
