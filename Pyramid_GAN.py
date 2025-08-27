#!/usr/bin/env python3

import os, glob, time, csv
from math import log10
from typing import Iterable, Dict, Any, Tuple, Optional, Sequence, List, Union

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom as nd_zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Config (edit here if needed)
# ----------------------------
ROOT_DIR   = "/scratch/l.peiwang/kari_brainv11"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

RUN_NAME   = "baselinev1_fast"
OUT_RUN    = os.path.join(OUT_DIR, RUN_NAME)
CKPT_DIR   = os.path.join(OUT_RUN, "checkpoints")
VOL_DIR    = os.path.join(OUT_RUN, "volumes")
os.makedirs(OUT_RUN, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(VOL_DIR, exist_ok=True)


RESIZE_TO: Optional[Tuple[int,int,int]] = (128,128,128)
RESAMPLE_BACK_TO_T1 = True

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
BATCH_SIZE     = 1
NUM_WORKERS    = 4
PIN_MEMORY     = True

EPOCHS      = 150
LR_G        = 1e-4
LR_D        = 4e-4
GAMMA       = 1.0
LAMBDA_GAN  = 0.5

# <<< CHANGED: use a fixed dataset-wide SUVR range for PSNR/SSIM/MS-SSIM >>>
DATA_RANGE  = 3.5   # e.g., 0–3.5 SUVR; adjust to your dataset-wide fixed range

torch.backends.cudnn.benchmark = True


# Utility: shape alignment
def _pad_or_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center pad/crop x spatially to match ref's D,H,W."""
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape

    d_pad = max(0, Dr - D)
    h_pad = max(0, Hr - H)
    w_pad = max(0, Wr - W)

    if d_pad or h_pad or w_pad:
        pad = (w_pad // 2, w_pad - w_pad // 2,
               h_pad // 2, h_pad - h_pad // 2,
               d_pad // 2, d_pad - d_pad // 2)
        x = F.pad(x, pad, mode='constant', value=0.)

    _, _, D2, H2, W2 = x.shape
    d_start = max(0, (D2 - Dr) // 2)
    h_start = max(0, (H2 - Hr) // 2)
    w_start = max(0, (W2 - Wr) // 2)
    x = x[:, :, d_start:d_start+Dr, h_start:h_start+Hr, w_start:w_start+Wr]
    return x


# ----------------------------
# Dataset & Normalization
# ----------------------------

# <<< CHANGED: allow choosing interpolation order >>>
def _maybe_resize(vol: np.ndarray, target: Optional[Tuple[int,int,int]], order: int = 1) -> np.ndarray:
    if target is None:
        return vol.astype(np.float32)
    Dz, Hy, Wx = vol.shape
    td, th, tw = target
    if (Dz, Hy, Wx) == (td, th, tw):
        return vol.astype(np.float32)
    zoom_factors = (td / Dz, th / Hy, tw / Wx)
    return nd_zoom(vol, zoom_factors, order=order).astype(np.float32)

# <<< CHANGED: MRI normalization = per-subject z-score in brain mask (no min-max) >>>
def norm_mri_to_01(vol: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    """
    MRI: per-subject z-score within brain mask (no min-max). Outside-mask voxels set to 0.
    """
    x = vol.astype(np.float32)
    if mask is None:
        mask = (x != 0)
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    mean = float(vals.mean())
    std  = float(vals.std() + 1e-6)
    z = (x - mean) / std
    z[~mask] = 0.0
    return z.astype(np.float32)

# <<< CHANGED: PET kept in SUVR (identity); zero outside mask >>>
def norm_pet_to_01(vol: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    """
    PET: keep SUVR values (no per-subject scaling). Outside-mask voxels set to 0.
    """
    x = vol.astype(np.float32)
    if mask is None:
        # fallback: treat non-zero as brain for zeroing outside
        mask = (x != 0)
    x_out = x.copy()
    x_out[~mask] = 0.0
    return x_out.astype(np.float32)


class KariAV1451Dataset(Dataset):
    """
    Loads pairs (T1_masked.nii.gz, PET_in_T1_masked.nii.gz) from AV1451 subject folders,
    normalizes, optional resize, returns (MRI, PET, meta) where MRI/PET are FloatTensors [1,D,H,W].
    Uses aseg_brainmask.nii.gz if available for masking; otherwise T1>0 as mask.
    """
    def __init__(
        self,
        root_dir: str = ROOT_DIR,
        resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO,
    ):
        self.root_dir = root_dir
        self.resize_to = resize_to

        patterns = [
            os.path.join(root_dir, "*_av1451_*"),
            os.path.join(root_dir, "*_AV1451_*"),
        ]
        subjects = []
        for p in patterns:
            subjects.extend(glob.glob(p))
        subjects = sorted([d for d in subjects if os.path.isdir(d)])

        self.items = []
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
            mask = (t1 != 0)

        orig_shape = tuple(t1.shape)
        t1_affine  = t1_img.affine
        pet_affine = pet_img.affine

        if t1.shape != pet.shape:
            raise TypeError("T1 and PET are not in the same grid")

        # <<< CHANGED: MRI cubic (3), PET linear (1) >>>
        t1  = _maybe_resize(t1,  self.resize_to, order=3)
        pet = _maybe_resize(pet, self.resize_to, order=1)
        cur_shape = tuple(t1.shape)

        if self.resize_to is not None and mask is not None:
            Dz, Hy, Wx = mask.shape
            td, th, tw = self.resize_to
            if (Dz,Hy,Wx) != (td,th,tw):
                mask = nd_zoom(mask.astype(np.float32), (td/Dz, th/Hy, tw/Wx), order=0) > 0.5

        # <<< CHANGED: Normalize (MRI z-score; PET SUVR identity); keep zeros outside >>>
        t1n  = norm_mri_to_01(t1,  mask)
        petn = norm_pet_to_01(pet, mask=mask)

        # Channel-first [1,D,H,W]
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
            # <<< CHANGED: pass brain mask for masked metrics >>>
            "brain_mask": mask.astype(np.uint8) if mask is not None else None,
        }
        return t1n_t, petn_t, meta


# ----------------------------
# DataLoader collate (avoid meta collation for B=1)
# ----------------------------
def _collate_keep_meta(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    """
    For BATCH_SIZE=1 returns the single (mri, pet, meta) sample unchanged.
    For B>1, stacks mri/pet and returns meta as a list of dicts.
    """
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


# ----------------------------
# Model blocks
# ----------------------------
class SelfAttention3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.Wf = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.Wphi = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.Wv = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W
        f = self.Wf(x).view(B, C, N)
        phi = self.Wphi(x).view(B, C, N)
        eta = self.softmax(f)
        weighted_phi = eta * phi
        summed = weighted_phi.sum(dim=-1, keepdim=True)
        a = self.Wv(summed.view(B, C, 1, 1, 1))
        return self.sigmoid(a)

class PyramidConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch_each: int, kernel_sizes=(3, 5, 7)):
        super().__init__()
        paths = []
        for k in kernel_sizes:
            pad = k // 2
            paths.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch_each, kernel_size=k, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch_each, out_ch_each, kernel_size=k, padding=pad, bias=True),
                nn.ReLU(inplace=True),
            ))
        self.paths = nn.ModuleList(paths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [p(x) for p in self.paths]
        return torch.cat(outs, dim=1)

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)

class Generator(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down1 = PyramidConvBlock(in_ch, out_ch_each=64, kernel_sizes=(3, 5, 7))
        ch1 = 64 * 3

        self.down2 = PyramidConvBlock(ch1, out_ch_each=128, kernel_sizes=(3, 5))
        ch2 = 128 * 2

        self.down3 = _double_conv(ch2, 512)
        ch3 = 512

        self.bottleneck = _double_conv(ch3, ch3)
        self.bottleneck_res6 = nn.Sequential(
            ResidualBlock3D(ch3), ResidualBlock3D(ch3), ResidualBlock3D(ch3),
            ResidualBlock3D(ch3), ResidualBlock3D(ch3), ResidualBlock3D(ch3),
        )

        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(ch3 + ch3, 256)

        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + ch2, 128)

        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + ch1, 64)

        self.att = SelfAttention3D(64)
        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x); p1 = self.pool(x1)
        x2 = self.down2(p1); p2 = self.pool(x2)
        x3 = self.down3(p2); p3 = self.pool(x3)

        b = self.bottleneck(p3)
        b = self.bottleneck_res6(b)

        u1 = self.up1_ups(b); u1 = _pad_or_crop_to(u1, x3); u1 = torch.cat([u1, x3], dim=1); u1 = self.up1_conv(u1)
        u2 = self.up2_ups(u1); u2 = _pad_or_crop_to(u2, x2); u2 = torch.cat([u2, x2], dim=1); u2 = self.up2_conv(u2)
        u3 = self.up3_ups(u2); u3 = _pad_or_crop_to(u3, x1); u3 = torch.cat([u3, x1], dim=1); u3 = self.up3_conv(u3)

        gate = self.att(u3)
        return self.out_conv(gate * u3)

class StandardDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        feats = []
        prev = in_ch
        for c in [32, 64, 128, 256, 512]:
            feats += [
                nn.Conv3d(prev, c, kernel_size=3, stride=2, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = c
        self.features = nn.Sequential(*feats)
        self.head = nn.Conv3d(prev, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        return torch.sigmoid(x)


# ----------------------------
# Losses & Metrics
# ----------------------------
def ssim3d(x: torch.Tensor, y: torch.Tensor, ksize: int = 3,
           k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    pad = ksize // 2
    mu_x = F.avg_pool3d(x, ksize, stride=1, padding=pad)
    mu_y = F.avg_pool3d(y, ksize, stride=1, padding=pad)
    sigma_x = F.avg_pool3d(x * x, ksize, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, ksize, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, ksize, stride=1, padding=pad) - mu_x * mu_y

    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return ssim_map.mean()

# <<< NEW: 3D MS-SSIM (used for loss and eval) >>>
def ms_ssim3d(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
              ksize: int = 11, levels: int = 4,
              weights: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)) -> torch.Tensor:
    """
    3D Multi-Scale SSIM. x,y: [B,1,D,H,W], intensities in the same scale.
    """
    def _ssim3d_local(a, b):
        pad = ksize // 2
        mu_a = F.avg_pool3d(a, ksize, stride=1, padding=pad)
        mu_b = F.avg_pool3d(b, ksize, stride=1, padding=pad)
        sigma_a = F.avg_pool3d(a*a, ksize, stride=1, padding=pad) - mu_a**2
        sigma_b = F.avg_pool3d(b*b, ksize, stride=1, padding=pad) - mu_b**2
        sigma_ab = F.avg_pool3d(a*b, ksize, stride=1, padding=pad) - mu_a*mu_b
        k1, k2 = 0.01, 0.03
        C1 = (k1 * data_range) ** 2
        C2 = (k2 * data_range) ** 2
        ssim_map = ((2*mu_a*mu_b + C1)*(2*sigma_ab + C2)) / ((mu_a**2 + mu_b**2 + C1)*(sigma_a + sigma_b + C2) + 1e-8)
        return ssim_map.mean()

    ws = torch.tensor(weights[:levels], device=x.device, dtype=x.dtype)
    ws = ws / ws.sum()

    vals = []
    a, b = x, y
    for l in range(levels):
        vals.append(_ssim3d_local(a, b))
        if l < levels - 1:
            a = F.avg_pool3d(a, kernel_size=2, stride=2, padding=0)
            b = F.avg_pool3d(b, kernel_size=2, stride=2, padding=0)
    vals = torch.stack(vals)
    # product of powers (weighted by ws)
    ms = torch.prod(vals ** ws)
    return ms

def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x, y)

def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float('inf')
    return 10.0 * log10((data_range ** 2) / mse)

# <<< CHANGED: MMD supports optional mask; samples only brain voxels if mask provided >>>
@torch.no_grad()
def mmd_gaussian(real: torch.Tensor,
                 fake: torch.Tensor,
                 num_voxels: int = 2048,
                 sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
                 mask: Optional[torch.Tensor] = None) -> float:
    B = real.size(0)
    dev = real.device
    total = 0.0
    for i in range(B):
        r = real[i, 0].reshape(-1)
        f = fake[i, 0].reshape(-1)

        if mask is not None:
            m = (mask[i, 0].reshape(-1) > 0.5)
            idx_pool = torch.nonzero(m, as_tuple=False).reshape(-1)
            if idx_pool.numel() == 0:
                return 0.0
            S = min(num_voxels, idx_pool.numel())
            sel = idx_pool[torch.randint(0, idx_pool.numel(), (S,), device=dev)]
            r_s = r[sel].view(S, 1)
            f_s = f[sel].view(S, 1)
        else:
            S = min(num_voxels, r.numel(), f.numel())
            ridx = torch.randint(0, r.numel(), (S,), device=dev)
            fidx = torch.randint(0, f.numel(), (S,), device=dev)
            r_s = r[ridx].view(S, 1)
            f_s = f[fidx].view(S, 1)

        d_rr = (r_s - r_s.t()).pow(2)
        d_ff = (f_s - f_s.t()).pow(2)
        d_rf = (r_s - f_s.t()).pow(2)

        mmd = 0.0
        for s in sigmas:
            Krr = torch.exp(-d_rr / (2 * s * s))
            Kff = torch.exp(-d_ff / (2 * s * s))
            Krf = torch.exp(-d_rf / (2 * s * s))
            mmd += Krr.mean() + Kff.mean() - 2 * Krf.mean()
        mmd /= len(sigmas)
        total += mmd.item()
    return total / B


# ----------------------------
# Training & Evaluation
# ----------------------------
def train_paggan(
    G: nn.Module,
    D: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    device: torch.device,
    epochs: int = EPOCHS,
    gamma: float = GAMMA,
    lambda_gan: float = LAMBDA_GAN,
    data_range: float = DATA_RANGE,
    verbose: bool = True,
) -> Dict[str, Any]:
    G.to(device); D.to(device)
    G.train(); D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    bce = nn.BCELoss()

    best_val = float('inf')
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, _ = batch
            else:
                raise ValueError("There is something wrong happened when passing data")
            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0) if mri.dim() == 5 else 1
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # ---- Update D ----
            with torch.no_grad():
                fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))
            D.zero_grad(set_to_none=True)
            out_real = D(pet if pet.dim()==5 else pet.unsqueeze(0))
            out_fake = D(fake.detach())
            loss_D = bce(out_real, real_lbl) + bce(out_fake, fake_lbl)
            loss_D.backward()
            opt_D.step()

            # ---- Update G ----
            G.zero_grad(set_to_none=True)
            fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))
            out_fake_for_G = D(fake)
            loss_gan = bce(out_fake_for_G, real_lbl)
            loss_l1 = l1_loss(fake, pet if pet.dim()==5 else pet.unsqueeze(0))
            # <<< CHANGED: use MS-SSIM for the perceptual term >>>
            ms_val = ms_ssim3d(fake, pet if pet.dim()==5 else pet.unsqueeze(0),
                               data_range=data_range, ksize=11, levels=4)
            loss_G = gamma * (loss_l1 + (1.0 - ms_val)) + lambda_gan * loss_gan
            loss_G.backward()
            opt_G.step()

            g_running += loss_G.item()
            d_running += loss_D.item()
            n_batches += 1

        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)
        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)

        # ---- Validation ----
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon, v_batches = 0.0, 0
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri, pet, _ = batch
                    else:
                        mri, pet = batch
                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))
                    loss_l1 = l1_loss(fake, pet if pet.dim()==5 else pet.unsqueeze(0))
                    ms_v  = ms_ssim3d(fake, pet if pet.dim()==5 else pet.unsqueeze(0),
                                      data_range=data_range, ksize=11, levels=4)
                    val_recon += (loss_l1 + (1.0 - ms_v)).item()
                    v_batches += 1
            val_recon /= max(1, v_batches)
            hist["val_recon"].append(val_recon)

            # Save best
            if val_recon < best_val:
                best_val = val_recon
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))

            if verbose:
                dt = time.time() - t0
                print(f"Epoch [{epoch:03d}/{epochs}]  "
                      f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                      f"ValRecon(L1 + 1-MS-SSIM): {val_recon:.4f}  "
                      f"| best {best_val:.4f}  | {dt:.1f}s")
            G.train()
        elif verbose:
            dt = time.time() - t0
            print(f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  | {dt:.1f}s")

    # restore best weights
    if best_G is not None:
        G.load_state_dict(best_G)
    if best_D is not None:
        D.load_state_dict(best_D)

    return {"history": hist, "best_G": best_G, "best_D": best_D}


@torch.no_grad()
def evaluate_paggan(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    data_range: float = DATA_RANGE,
    mmd_voxels: int = 2048,
) -> Dict[str, float]:
    G.to(device)
    G.eval()

    msssim_sum = 0.0
    psnr_sum = 0.0
    mse_sum = 0.0
    mmd_sum = 0.0
    n = 0

    for batch in test_loader:
        # support (mri, pet, meta) or (mri, pet)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {}
        mri = mri.to(device, non_blocking=True)
        pet = pet.to(device, non_blocking=True)
        fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))

        pet_for_metric  = pet if pet.dim()==5 else pet.unsqueeze(0)

        # <<< CHANGED: masked metrics (use aseg mask if available) >>>
        brain_mask_np = meta.get("brain_mask", None) if isinstance(meta, dict) else None
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake * brain
        pet_m  = pet_for_metric * brain

        msssim_sum += ms_ssim3d(fake_m, pet_m, data_range=data_range, ksize=11, levels=4).item()
        psnr_sum   += psnr(fake_m,  pet_m, data_range=data_range)
        mse_sum    += F.mse_loss(fake_m, pet_m).item()
        mmd_sum    += mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)
        n += 1

    return {
        "MS-SSIM": msssim_sum / max(1, n),
        "PSNR":    psnr_sum   / max(1, n),
        "MSE":     mse_sum    / max(1, n),
        "MMD":     mmd_sum    / max(1, n),
    }


def _safe_name(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

def _save_nifti(vol: np.ndarray, affine: np.ndarray, path: str):
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, path)

def _as_int_tuple3(x: Union[Tuple[int,int,int], List[Any], np.ndarray, torch.Tensor]) -> Tuple[int,int,int]:
    """
    Normalize various collated representations to a plain (D,H,W) of ints.
    Handles: (D,H,W), [D,H,W], np arrays, torch tensors, and
    collate artifacts like (tensor([D]), tensor([H]), tensor([W])) or tensor([[D,H,W]]).
    """
    # Unwrap lists/tuples of length 1 that hold tensors/arrays
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    # If it's a list/tuple of scalars or 0/1-d tensors
    if isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            if isinstance(v, torch.Tensor):
                vals.append(int(v.detach().cpu().reshape(-1)[0].item()))
            else:
                vals.append(int(v))
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])
        raise ValueError(f"orig/cur shape has unexpected length: {vals}")
    # If it's a tensor/ndarray
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        # Single scalar?
        try:
            return tuple(int(t) for t in x)  # may raise
        except Exception:
            raise ValueError(f"Cannot parse shape from type: {type(x)}")
    flat = np.array(arr).astype(np.int64).reshape(-1)
    if flat.size < 3:
        raise ValueError(f"Shape vector too small: {flat}")
    return (int(flat[0]), int(flat[1]), int(flat[2]))

def _meta_unbatch(meta: Any) -> Dict[str, Any]:
    """
    Convert meta produced by DataLoader (dict-of-lists/tuples/tensors) back to a plain dict for B=1.
    If already a plain dict, returns it. If list of dicts, returns the first.
    """
    if isinstance(meta, list):
        if len(meta) == 0:
            return {}
        if isinstance(meta[0], dict):
            return meta[0]
        # else fallthrough
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        # unwrap 1-length containers
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        # convert tensors
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        out[k] = v
    # normalize shapes if present
    if "orig_shape" in out:
        out["orig_shape"] = _as_int_tuple3(out["orig_shape"])
    if "cur_shape" in out:
        out["cur_shape"]  = _as_int_tuple3(out["cur_shape"])
    return out


@torch.no_grad()
def save_test_volumes(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    out_dir: str,
    resample_back_to_t1: bool = RESAMPLE_BACK_TO_T1,
):
    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    print(f"Saving test volumes to: {out_dir}  (resample_back_to_t1={resample_back_to_t1})")

    for i, batch in enumerate(test_loader):
        # Unpack and unbatch meta robustly
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {"sid": f"sample_{i:04d}", "t1_affine": np.eye(4), "orig_shape": tuple(mri.shape[2:]), "cur_shape": tuple(mri.shape[2:]), "resized_to": None}
        meta = _meta_unbatch(meta)

        sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
        subdir = os.path.join(out_dir, sid)
        os.makedirs(subdir, exist_ok=True)

        # Move through model
        mri_t  = mri.to(device, non_blocking=True)
        fake_t = G(mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0))

        # to numpy [D,H,W]
        mri_np  = (mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        pet_np  = (pet   if pet.dim()==5   else pet.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        fake_np = fake_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        err_np  = np.abs(fake_np - pet_np)

        cur_shape  = tuple(mri_np.shape)
        orig_shape = tuple(meta.get("orig_shape", cur_shape))

        if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
            # compute zoom factors as pure floats
            zf = (float(orig_shape[0]) / float(cur_shape[0]),
                  float(orig_shape[1]) / float(cur_shape[1]),
                  float(orig_shape[2]) / float(cur_shape[2]))
            mri_np  = nd_zoom(mri_np,  zf, order=1)
            pet_np  = nd_zoom(pet_np,  zf, order=1)
            fake_np = nd_zoom(fake_np, zf, order=1)
            err_np  = nd_zoom(err_np,  zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            # If we didn't resize during dataset, keep original affine; otherwise identity
            resized_to = meta.get("resized_to", None)
            if resized_to is None:
                affine_to_use = meta.get("t1_affine", np.eye(4))
            else:
                affine_to_use = np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

        print(f"  saved {sid}: MRI/PET_gt/PET_fake/PET_abs_error")


# ----------------------------
# NEW: Combined evaluate + save (single pass)
# ----------------------------
@torch.no_grad()
def evaluate_and_save(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    out_dir: str,
    data_range: float = DATA_RANGE,
    mmd_voxels: int = 2048,
    resample_back_to_t1: bool = RESAMPLE_BACK_TO_T1,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    msssim_sum = 0.0
    psnr_sum = 0.0
    mse_sum  = 0.0
    mmd_sum  = 0.0
    n = 0

    for i, batch in enumerate(test_loader):
        # Unpack meta
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {"sid": f"sample_{i:04d}", "t1_affine": np.eye(4), "orig_shape": tuple(mri.shape[2:]), "cur_shape": tuple(mri.shape[2:]), "resized_to": None}
        meta = _meta_unbatch(meta)

        # Move to device and forward once
        mri_t = mri.to(device, non_blocking=True)
        pet_t = pet.to(device, non_blocking=True)
        fake_t = G(mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0))

        pet_for_metric  = pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)

        # <<< CHANGED: masked metrics >>>
        brain_mask_np = meta.get("brain_mask", None)
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake_t * brain
        pet_m  = pet_for_metric * brain

        # --- Metrics (masked) ---
        msssim_sum += ms_ssim3d(fake_m, pet_m, data_range=data_range, ksize=11, levels=4).item()
        psnr_sum   += psnr(fake_m,  pet_m, data_range=data_range)
        mse_sum    += F.mse_loss(fake_m, pet_m).item()
        mmd_sum    += mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)
        n += 1

        # --- Saving volumes (reuse same forward) ---
        sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
        subdir = os.path.join(out_dir, sid)
        os.makedirs(subdir, exist_ok=True)

        mri_np  = (mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        pet_np  = (pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        fake_np =  fake_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        err_np  = np.abs(fake_np - pet_np)

        cur_shape  = tuple(mri_np.shape)
        orig_shape = tuple(meta.get("orig_shape", cur_shape))

        if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
            zf = (float(orig_shape[0]) / float(cur_shape[0]),
                  float(orig_shape[1]) / float(cur_shape[1]),
                  float(orig_shape[2]) / float(cur_shape[2]))
            mri_np  = nd_zoom(mri_np,  zf, order=1)
            pet_np  = nd_zoom(pet_np,  zf, order=1)
            fake_np = nd_zoom(fake_np, zf, order=1)
            err_np  = nd_zoom(err_np,  zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            resized_to = meta.get("resized_to", None)
            if resized_to is None:
                affine_to_use = meta.get("t1_affine", np.eye(4))
            else:
                affine_to_use = np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

    return {
        "MS-SSIM": msssim_sum / max(1, n),
        "PSNR":    psnr_sum   / max(1, n),
        "MSE":     mse_sum    / max(1, n),
        "MMD":     mmd_sum    / max(1, n),
    }


# ----------------------------
# Plotting & Logging helpers
# ----------------------------
def save_loss_curves(history: Dict[str, Sequence[float]], out_path: str):
    plt.figure(figsize=(7,5))
    if "train_G" in history:
        plt.plot(history["train_G"], label="Train G")
    if "train_D" in history:
        plt.plot(history["train_D"], label="Train D")
    if "val_recon" in history and len(history["val_recon"]) > 0:
        plt.plot(history["val_recon"], label="Val (L1 + 1-MS-SSIM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_history_csv(history: Dict[str, Sequence[float]], out_csv: str):
    L = max(len(history.get("train_G", [])),
            len(history.get("train_D", [])),
            len(history.get("val_recon", [])))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_G", "train_D", "val_recon"])
        for i in range(L):
            row = [i+1,
                   history.get("train_G",  [None]*L)[i] if i < len(history.get("train_G",[])) else "",
                   history.get("train_D",  [None]*L)[i] if i < len(history.get("train_D",[])) else "",
                   history.get("val_recon",[None]*L)[i] if i < len(history.get("val_recon",[])) else ""]
            w.writerow(row)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"Data root: {ROOT_DIR}")
    print(f"Output root: {OUT_DIR}")
    print(f"Run name: {RUN_NAME}")
    print(f"Run dir: {OUT_RUN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build loaders
    train_loader, val_loader, test_loader, N, ntr, nva, nte = build_loaders()
    print(f"Subjects: total={N}, train={ntr}, val={nva}, test={nte}")
    with torch.no_grad():
        sample = next(iter(train_loader))
        if isinstance(sample, (list, tuple)) and len(sample) == 3:
            mri0, pet0, meta0 = sample
            sid0 = meta0.get("sid", "NA") if isinstance(meta0, dict) else "NA"
        else:
            mri0, pet0 = sample
            sid0 = "NA"
        print(f"Sample tensor shapes: MRI {tuple(mri0.shape)}, PET {tuple(pet0.shape)}, SID {sid0}")

    # Instantiate models
    G = Generator(in_ch=1, out_ch=1)
    D = StandardDiscriminator(in_ch=1)

    # Train
    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=EPOCHS, gamma=GAMMA, lambda_gan=LAMBDA_GAN, data_range=DATA_RANGE, verbose=True
    )

    # Save curves & CSV
    curves_path = os.path.join(OUT_RUN, "loss_curves.png")
    save_loss_curves(out["history"], curves_path)
    print(f"Saved loss curves to: {curves_path}")

    csv_path = os.path.join(OUT_RUN, "training_log.csv")
    save_history_csv(out["history"], csv_path)
    print(f"Saved training log CSV to: {csv_path}")

    # Evaluate + Save (single pass)
    metrics = evaluate_and_save(
        G, test_loader, device=device,
        out_dir=VOL_DIR, data_range=DATA_RANGE,
        mmd_voxels=2048, resample_back_to_t1=RESAMPLE_BACK_TO_T1
    )
    print("Test metrics:", metrics)

    metrics_txt = os.path.join(OUT_RUN, "test_metrics.txt")
    with open(metrics_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics to: {metrics_txt}")
