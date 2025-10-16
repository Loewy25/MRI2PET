import os, glob
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom as nd_zoom
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .config import (
    ROOT_DIR, RESIZE_TO, TRAIN_FRACTION, VAL_FRACTION, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY
)
from .utils import _pad_or_crop_to  # used by models.py; keep import path if needed

def _maybe_resize(vol: np.ndarray, target: Optional[Tuple[int,int,int]], order: int = 1) -> np.ndarray:
    if target is None:
        return vol.astype(np.float32)
    Dz, Hy, Wx = vol.shape
    td, th, tw = target
    if (Dz, Hy, Wx) == (td, th, tw):
        return vol.astype(np.float32)
    zoom_factors = (td / Dz, th / Hy, tw / Wx)
    return nd_zoom(vol, zoom_factors, order=order).astype(np.float32)

def norm_mri_to_01(vol: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    x = vol.astype(np.float32)
    if mask is None:
        raise TypeError("no mask")
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mean = float(vals.mean())
    std  = float(vals.std() + 1e-6)
    z = (x - mean) / std
    z[~mask] = 0.0
    return z.astype(np.float32)

def norm_pet_to_01(vol: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    x = vol.astype(np.float32)
    if mask is None:
        raise TypeError("No Mask")
    x_out = x.copy()
    x_out[~mask] = 0.0
    return x_out.astype(np.float32)

class KariAV1451Dataset(Dataset):
    """
    Loads pairs (T1_masked.nii.gz, PET_in_T1_masked.nii.gz) from AV1451 subject folders,
    normalizes, optional resize, returns (MRI, PET, meta) where MRI/PET are FloatTensors [1,D,H,W].
    Uses aseg_brainmask.nii.gz if available for masking; otherwise T1>0 as mask.
    """
    def __init__(self, root_dir: str = ROOT_DIR, resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO):
        self.root_dir = root_dir
        self.resize_to = resize_to

        patterns = [
            os.path.join(root_dir, "*T807*"),
            os.path.join(root_dir, "*t807*"),
        ]
        subjects: List[str] = []
        for p in patterns:
            subjects.extend(glob.glob(p))
        subjects = sorted([d for d in subjects if os.path.isdir(d)])

        self.items: List[Tuple[str,str,Optional[str]]] = []
        for d in subjects:
            t1p  = os.path.join(d, "T1_masked.nii.gz")
            petp = os.path.join(d, "PET_in_T1_masked.nii.gz")
            if os.path.exists(t1p) and os.path.exists(petp):
                maskp = os.path.join(d, "aseg_brainmask.nii.gz")
                self.items.append((t1p, petp, maskp if os.path.exists(maskp) else None))

        if len(self.items) == 0:
            raise RuntimeError(f"No subject folders with required files under {root_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        t1_path, pet_path, mask_path = self.items[idx]
        sid = os.path.basename(os.path.dirname(t1_path))

        t1_img  = nib.load(t1_path);  t1  = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        pet_img = nib.load(pet_path); pet = np.asarray(pet_img.get_fdata(), dtype=np.float32)

        if mask_path is not None:
            m_img = nib.load(mask_path); mask = (np.asarray(m_img.get_fdata()) > 0)
        else:
            raise TypeError("No Mask")

        orig_shape = tuple(t1.shape)
        t1_affine  = t1_img.affine
        pet_affine = pet_img.affine

        if t1.shape != pet.shape:
            raise TypeError("T1 and PET are not in the same grid")

        t1  = _maybe_resize(t1,  self.resize_to, order=1)
        pet = _maybe_resize(pet, self.resize_to, order=1)
        cur_shape = tuple(t1.shape)

        if self.resize_to is not None and mask is not None:
            Dz, Hy, Wx = mask.shape
            td, th, tw = self.resize_to
            if (Dz,Hy,Wx) != (td,th,tw):
                mask = nd_zoom(mask.astype(np.float32), (td/Dz, th/Hy, tw/Wx), order=0) > 0.5

        t1n  = norm_mri_to_01(t1,  mask)
        petn = norm_pet_to_01(pet, mask=mask)

        t1n_t  = torch.from_numpy(np.expand_dims(t1n,  axis=0))
        petn_t = torch.from_numpy(np.expand_dims(petn, axis=0))

        meta = {
            "sid": sid,
            "t1_path": t1_path,
            "pet_path": pet_path,
            "t1_affine": t1_affine,
            "pet_affine": pet_affine,
            "orig_shape": orig_shape,
            "cur_shape": cur_shape,
            "resized_to": self.resize_to,
            "brain_mask": mask.astype(np.uint8) if mask is not None else None,
        }
        return t1n_t, petn_t, meta

def _collate_keep_meta(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    if len(batch) == 1:
        return batch[0]
    mri = torch.stack([b[0] for b in batch], dim=0)
    pet = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return mri, pet, metas

def build_loaders(
    root: str = ROOT_DIR,
    resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = 1999,
):
    ds = KariAV1451Dataset(root_dir=root, resize_to=resize_to)
    N = len(ds)
    n_train = int(round(train_fraction * N))
    n_val   = int(round(val_fraction   * N))
    n_test  = N - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test], generator=gen)

    dl_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )
    dl_val = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )
    dl_test = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )
    return dl_train, dl_val, dl_test, N, n_train, n_val, n_test

# === NEW: CSV-driven fold readers ===
import csv

def _read_fold_csv_lists(path: str):
    """
    Read a foldX.csv with columns train,validation,test (one subject per cell).
    Returns three lists of subject IDs (folder names).
    """
    train, val, test = [], [], []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        # tolerant column matching
        cols = {c.strip().lower(): c for c in r.fieldnames}
        tcol = cols.get("train"); vcol = cols.get("validation"); ccol = cols.get("test")
        if not (tcol and vcol and ccol):
            raise ValueError(f"fold CSV missing required columns train/validation/test: {path}")
        for row in r:
            t = (row.get(tcol) or "").strip()
            v = (row.get(vcol) or "").strip()
            c = (row.get(ccol) or "").strip()
            if t: train.append(t)
            if v: val.append(v)
            if c: test.append(c)
    return train, val, test

def _sid_for_item(item_tuple):
    """Extract subject folder name from KariAV1451Dataset.items entry."""
    t1_path, _, _ = item_tuple
    return os.path.basename(os.path.dirname(t1_path))

def build_loaders_from_fold_csv(
    fold_csv_path: str,
    root: str = ROOT_DIR,
    resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    """
    Build train/val/test DataLoaders using explicit subject lists from a fold CSV.
    """
    # 1) Full dataset (no random split)
    ds = KariAV1451Dataset(root_dir=root, resize_to=resize_to)
    N = len(ds)

    # 2) Map subject IDs to dataset indices
    sid_list = [_sid_for_item(x) for x in ds.items]
    sid_to_index = {sid: i for i, sid in enumerate(sid_list)}

    train_sids, val_sids, test_sids = _read_fold_csv_lists(fold_csv_path)

    def _to_indices(sids):
        idxs = []
        missing = []
        for s in sids:
            if s in sid_to_index:
                idxs.append(sid_to_index[s])
            else:
                missing.append(s)
        if missing:
            print(f"[WARN] {len(missing)} subjects from CSV not found on disk. Examples: {missing[:8]}")
        return sorted(idxs)

    idx_train = _to_indices(train_sids)
    idx_val   = _to_indices(val_sids)
    idx_test  = _to_indices(test_sids)

    from torch.utils.data import Subset
    train_set = Subset(ds, idx_train)
    val_set   = Subset(ds, idx_val)
    test_set  = Subset(ds, idx_test)

    dl_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )
    dl_val = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )
    dl_test = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        collate_fn=_collate_keep_meta
    )

    return dl_train, dl_val, dl_test, N, len(idx_train), len(idx_val), len(idx_test)
