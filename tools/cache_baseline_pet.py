#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["OVERSAMPLE_ENABLE"] = "0"
os.environ["USE_BASELINE_CACHE"] = "0"

from mri2pet.config import BASE_PRETRAIN_CKPT, FOLD_CSV, FOLD_INDEX, RESIZE_TO, ROOT_DIR
from mri2pet.data import (
    KariAV1451Dataset,
    _compute_braak_stats,
    _compute_clinical_stats,
    _read_fold_csv_lists,
)
from mri2pet.models import Generator


def _parse_args():
    p = argparse.ArgumentParser(description="Cache baseline PET_base outputs in the model grid.")
    p.add_argument("--checkpoint", default=BASE_PRETRAIN_CKPT, help="Baseline Generator checkpoint path.")
    p.add_argument("--cache-dir", required=True, help="Output directory for *_pet_base.npy files.")
    p.add_argument("--root-dir", default=ROOT_DIR, help="Subject root directory.")
    p.add_argument("--fold-csv", default=FOLD_CSV, help="Fold CSV used to define train/val/test subjects.")
    p.add_argument("--fold-index", type=int, default=FOLD_INDEX, help="0-based fold index for manifest only.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.fold_csv):
        raise FileNotFoundError(f"Fold CSV not found: {args.fold_csv}")

    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[cache] root={args.root_dir}", flush=True)
    print(f"[cache] fold_csv={args.fold_csv}", flush=True)
    print(f"[cache] checkpoint={args.checkpoint}", flush=True)
    print(f"[cache] cache_dir={args.cache_dir}", flush=True)
    print(f"[cache] resize_to={RESIZE_TO} device={device}", flush=True)

    train_sids, val_sids, test_sids, train_sid_to_label = _read_fold_csv_lists(args.fold_csv)
    ds = KariAV1451Dataset(root_dir=args.root_dir, resize_to=RESIZE_TO, sid_to_label=train_sid_to_label)
    sid_to_index = {item["sid"]: i for i, item in enumerate(ds.items)}

    def _indices(sids):
        missing = [sid for sid in sids if sid not in sid_to_index]
        if missing:
            raise RuntimeError(f"{len(missing)} fold subjects not found on disk. Examples: {missing[:8]}")
        return [sid_to_index[sid] for sid in sids]

    idx_train = _indices(train_sids)
    idx_all = sorted(set(_indices(train_sids) + _indices(val_sids) + _indices(test_sids)))
    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))
    braak_mean, braak_std = _compute_braak_stats(ds, idx_train)
    ds.set_braak_stats(braak_mean, braak_std)

    G = Generator(in_ch=1, out_ch=1, use_checkpoint=False).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    G.load_state_dict(state, strict=True)
    G.eval()

    t0 = time.time()
    for n, idx in enumerate(idx_all, start=1):
        item_t0 = time.time()
        mri, _, meta = ds[idx]
        sid = meta["sid"]
        out_path = os.path.join(args.cache_dir, f"{sid}_pet_base.npy")
        with torch.no_grad():
            mri5 = mri.unsqueeze(0).to(device)
            pet_base = G(mri5).squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
        brain = meta["brain_mask"].astype(bool)
        pet_base[~brain] = 0.0
        np.save(out_path, pet_base)
        print(
            f"[cache] {n}/{len(idx_all)} {sid} saved {out_path} "
            f"shape={pet_base.shape} sec={time.time() - item_t0:.1f}",
            flush=True,
        )

    manifest = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fold_index": int(args.fold_index),
        "fold_csv": os.path.abspath(args.fold_csv),
        "checkpoint": os.path.abspath(args.checkpoint),
        "resize_to": list(RESIZE_TO) if RESIZE_TO is not None else None,
        "root_dir": os.path.abspath(args.root_dir),
        "num_subjects": len(idx_all),
        "train_subjects": len(train_sids),
        "validation_subjects": len(val_sids),
        "test_subjects": len(test_sids),
    }
    manifest_path = os.path.join(args.cache_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[cache] wrote manifest: {manifest_path}", flush=True)
    print(f"[cache] done total_sec={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
