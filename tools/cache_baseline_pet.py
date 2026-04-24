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
    p.add_argument(
        "--checkpoint",
        default=BASE_PRETRAIN_CKPT,
        help="Baseline Generator checkpoint, or residual model checkpoint containing base.* weights.",
    )
    p.add_argument("--cache-dir", required=True, help="Output directory for *_pet_base.npy files.")
    p.add_argument("--root-dir", default=ROOT_DIR, help="Subject root directory.")
    p.add_argument("--fold-csv", default=FOLD_CSV, help="Fold CSV used to define train/val/test subjects.")
    p.add_argument("--fold-index", type=int, default=FOLD_INDEX, help="0-based fold index for manifest only.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _state_dict_candidates(obj):
    candidates = [("raw", obj)]
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "G", "generator", "best_G"):
            val = obj.get(key)
            if isinstance(val, dict):
                candidates.append((key, val))
    return candidates


def _strip_module_prefix(state):
    if not isinstance(state, dict):
        return state
    if state and all(str(k).startswith("module.") for k in state.keys()):
        return {str(k)[len("module."):]: v for k, v in state.items()}
    return state


def _load_generator_checkpoint(G: Generator, checkpoint_path: str) -> str:
    raw = torch.load(checkpoint_path, map_location="cpu")
    expected = set(G.state_dict().keys())
    errors = []

    for source_name, state in _state_dict_candidates(raw):
        state = _strip_module_prefix(state)
        if not isinstance(state, dict):
            continue

        plain = {str(k): v for k, v in state.items()}
        if set(plain.keys()) == expected:
            G.load_state_dict(plain, strict=True)
            return f"{source_name}:plain_generator"

        base_state = {
            str(k)[len("base."):]: v
            for k, v in plain.items()
            if str(k).startswith("base.")
        }
        if base_state and set(base_state.keys()) == expected:
            G.load_state_dict(base_state, strict=True)
            return f"{source_name}:extracted_base_prefix"

        missing_plain = len(expected - set(plain.keys()))
        unexpected_plain = len(set(plain.keys()) - expected)
        missing_base = len(expected - set(base_state.keys())) if base_state else len(expected)
        errors.append(
            f"{source_name}: plain missing={missing_plain} unexpected={unexpected_plain}; "
            f"base missing={missing_base} base_keys={len(base_state)}"
        )

    preview = []
    if isinstance(raw, dict):
        preview = [str(k) for k in list(raw.keys())[:12]]
    raise RuntimeError(
        "Could not load checkpoint into baseline Generator. Expected either plain Generator keys "
        "or residual checkpoint keys prefixed with 'base.'. "
        f"Checkpoint={checkpoint_path}. Top-level preview={preview}. Attempts: {errors}"
    )


def main():
    args = _parse_args()
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
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
    checkpoint_load_mode = _load_generator_checkpoint(G, args.checkpoint)
    G.eval()
    print(f"[cache] checkpoint_load_mode={checkpoint_load_mode}", flush=True)

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
        "checkpoint_load_mode": checkpoint_load_mode,
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
