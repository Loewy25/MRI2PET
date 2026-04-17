#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("/scratch/l.peiwang/kari_flair_all")
DEFAULT_TARGET_ROOT = Path("/scratch/l.peiwang/kari_brainv33_top300")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each subject folder in the target dataset, copy any files that are "
            "missing there but present under the same subject folder in the source dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Source dataset root. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help=f"Target dataset root. Default: {DEFAULT_TARGET_ROOT}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without changing files.",
    )
    return parser.parse_args()


def subject_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def sync_subject(source_subject: Path, target_subject: Path, dry_run: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for source_path in sorted(source_subject.rglob("*")):
        if not source_path.is_file():
            continue

        rel_path = source_path.relative_to(source_subject)
        target_path = target_subject / rel_path

        if target_path.exists():
            skipped += 1
            continue

        print(f"[COPY] {target_subject.name}/{rel_path}")
        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        copied += 1

    return copied, skipped


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    target_root = args.target_root

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not target_root.is_dir():
        raise FileNotFoundError(f"Target root not found: {target_root}")

    targets = subject_dirs(target_root)
    if not targets:
        raise RuntimeError(f"No subject folders found in target root: {target_root}")

    print(f"Source root: {source_root}")
    print(f"Target root: {target_root}")
    print(f"Target subjects to scan: {len(targets)}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 70)

    total_copied = 0
    total_skipped = 0
    missing_subjects = []

    for i, target_subject in enumerate(targets, start=1):
        source_subject = source_root / target_subject.name
        print(f"[{i}/{len(targets)}] {target_subject.name}")

        if not source_subject.is_dir():
            print("  [WARN] Missing in source dataset, skipped subject.")
            missing_subjects.append(target_subject.name)
            continue

        copied, skipped = sync_subject(source_subject, target_subject, args.dry_run)
        total_copied += copied
        total_skipped += skipped
        print(f"  copied={copied} existing={skipped}")

    print("-" * 70)
    print(f"Done. Total copied: {total_copied}")
    print(f"Already present: {total_skipped}")
    print(f"Subjects missing in source: {len(missing_subjects)}")
    if missing_subjects:
        preview = ", ".join(missing_subjects[:10])
        suffix = " ..." if len(missing_subjects) > 10 else ""
        print(f"Missing subject examples: {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
