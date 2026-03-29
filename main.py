#!/usr/bin/env python3
from pathlib import Path
import shutil

SRC = Path("/scratch/l.peiwang/kari_all")
DST = Path("/scratch/l.peiwang/kari_all_falir")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    copied = 0
    for subj in sorted(p for p in SRC.iterdir() if p.is_dir()):
        if any("flair" in f.name.lower() for f in subj.iterdir()):
            shutil.copytree(subj, DST / subj.name, dirs_exist_ok=True)
            print(subj.name)
            copied += 1
    print(f"Copied {copied} subjects to {DST}")


if __name__ == "__main__":
    main()
