#!/usr/bin/env python3
"""
PA-GAN (Pyramid + Attention GAN) without Task-Induced Discriminator
-------------------------------------------------------------------
- Generator: U-Net with two pyramid conv blocks (kernels {3,5,7} then {3,5}),
  three decoder up-blocks (Upsample x2 + Conv3D(3x3) + ReLU + Conv3D(3x3) + ReLU),
  self-attention module applied at the last up-block, and final 1-channel output.
- Bottleneck: 2×(3×3×3 conv) followed by a residual stack ×6 (each: 3×3×3, ReLU, 3×3×3, +skip, ReLU).
- Discriminator (Dstd): 3×3×3 stride-2 conv stack (channels 32,64,128,256,512) + 1ch head + GAP + sigmoid.
- Loss: L_G = gamma*(L1 + (1-SSIM)) + lambda_gan * BCE(D(G(x)), 1).  D uses BCE on real/fake.
- Optim: Adam, LR(G)=1e-4, LR(D)=4e-4.
- Metrics: SSIM, PSNR, MSE, MMD (Gaussian-kernel, voxel subsampling).

Data assumptions:
- Each subject directory under ROOT contains:
    T1_masked.nii.gz
    PET_in_T1_masked.nii.gz
  (both in FreeSurfer native space, brain-masked and aligned)

Outputs (all under /home/l.peiwang/MRI2PET/<RUN_NAME>/):
- loss_curves.png
- training_log.csv
- checkpoints/best_G.pth, best_D.pth
- test_metrics.txt
"""

import os, glob, time, csv, math
from math import log10
from typing import Iterable, Dict, Any, Tuple, Optional, Sequence

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom as nd_zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")  # no X server needed
import matplotlib.pyplot as plt


# ----------------------------
# Config (edit here if needed)
# ----------------------------
ROOT_DIR   = "/scratch/l.peiwang/kari_brainv11"
OUT_DIR    = "/home/l.peiwang/MRI2PET"

# >>> Set this per run <<<
RUN_NAME   = "test"   # e.g., "test", "baseline_256", "2025-08-21_0930"
OUT_RUN    = os.path.join(OUT_DIR, RUN_NAME)
CKPT_DIR   = os.path.join(OUT_RUN, "checkpoints")
os.makedirs(OUT_RUN, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Keep native FreeSurfer size by default; set to (128,128,128) if you need to save memory.
RESIZE_TO: Optional[Tuple[int,int,int]] = (128,128,128)  # e.g., (128,128,128) or (76,94,76)

# Splits & loader
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15  # TEST will be the rest
BATCH_SIZE     = 1     # 256^3 is heavy; increase to 2 only if memory allows
NUM_WORKERS    = 4     # adjust per node
PIN_MEMORY     = True

# Training hyper-params
EPOCHS      = 150
LR_G        = 1e-4
LR_D        = 4e-4
GAMMA       = 1.0    # weight for (L1 + (1-SSIM))
LAMBDA_GAN  = 0.5
DATA_RANGE  = 1.0    # assumed data range for SSIM/PSNR
SEED        = 2024

# ----------------------------
# Utilities: reproducibility
# ----------------------------
def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# ----------------------------
# Utility: shape alignment
# ----------------------------
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
def norm_mri_to_01(vol: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    """
    MRI: z-score within mask (no clipping), then per-volume min-max to [0,1] within mask.
    """
    x = vol.astype(np.float32)
    if mask is None:
        mask = (x != 0)
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    mean, std = float(vals.mean()), float(vals.std() + 1e-6)
    z = (x - mean) / std

    z_mask = z[mask]
    zmin, zmax = float(z_mask.min()), float(z_mask.max())
    x01 = (z - zmin) / (zmax - zmin + 1e-6)
    x01[~mask] = 0.0
    return x01.astype(np.float32)

def norm_pet_to_01(vol: np.ndarray, pet_max: float = PET_MAX, mask: Optional[np.ndarray]=None) -> np.ndarray:
    """
    PET: per-volume min–max to [0,1] within mask (no clipping, no PET_MAX scaling).
    """
    x = vol.astype(np.float32)
    if mask is None:
        mask = (x != 0)
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    vmin, vmax = float(vals.min()), float(vals.max())
    x01 = (x - vmin) / (vmax - vmin + 1e-6)
    x01[~mask] = 0.0
    return x01.astype(np.float32)

def _maybe_resize(vol: np.ndarray, target: Optional[Tuple[int,int,int]]) -> np.ndarray:
    if target is None:
        return vol
    Dz, Hy, Wx = vol.shape
    td, th, tw = target
    if (Dz, Hy, Wx) == (td, th, tw):
        return vol
    zoom_factors = (td / Dz, th / Hy, tw / Wx)
    return nd_zoom(vol, zoom_factors, order=1).astype(np.float32)  # trilinear

class KariAV1451Dataset(Dataset):
    """
    Loads pairs (T1_masked.nii.gz, PET_in_T1_masked.nii.gz) from AV1451 subject folders,
    normalizes, optional resize, returns (MRI, PET) as FloatTensors [1,D,H,W].
    Uses aseg_brainmask.nii.gz if available for masking; otherwise T1>0 as mask.
    """
    def __init__(
        self,
        root_dir: str = ROOT_DIR,
        resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO,
        pet_max: float = PET_MAX
    ):
        self.root_dir = root_dir
        self.resize_to = resize_to
        self.pet_max = pet_max

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
        t1_img  = nib.load(t1_path);  t1  = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        pet_img = nib.load(pet_path); pet = np.asarray(pet_img.get_fdata(), dtype=np.float32)

        if mask_path is not None:
            m_img = nib.load(mask_path); mask = (np.asarray(m_img.get_fdata()) > 0)
        else:
            mask = (t1 != 0)

        # Assure PET on T1 grid; if not, resample PET to T1
        if t1.shape != pet.shape:
            pet = _maybe_resize(pet, t1.shape)

        # Optional global resize for the model
        t1  = _maybe_resize(t1,  self.resize_to)
        pet = _maybe_resize(pet, self.resize_to)
        if self.resize_to is not None and mask is not None:
            Dz, Hy, Wx = mask.shape
            td, th, tw = self.resize_to
            if (Dz,Hy,Wx) != (td,th,tw):
                mask = nd_zoom(mask.astype(np.float32), (td/Dz, th/Hy, tw/Wx), order=0) > 0.5

        # Normalize (MRI: z->minmax; PET: minmax)
        t1n  = norm_mri_to_01(t1,  mask)
        petn = norm_pet_to_01(pet, pet_max=self.pet_max, mask=mask)

        # Channel-first [1,D,H,W]
        t1n  = np.expand_dims(t1n,  axis=0)
        petn = np.expand_dims(petn, axis=0)

        return torch.from_numpy(t1n), torch.from_numpy(petn)

def build_loaders(
    root: str = ROOT_DIR,
    resize_to: Optional[Tuple[int,int,int]] = RESIZE_TO,
    pet_max: float = PET_MAX,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = SEED,
):
    ds = KariAV1451Dataset(root_dir=root, resize_to=resize_to, pet_max=pet_max)
    N = len(ds)
    n_train = int(round(train_fraction * N))
    n_val   = int(round(val_fraction   * N))
    n_test  = N - n_train - n_val
    # Deterministic split
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test], generator=gen)

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    dl_val   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    dl_test  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return dl_train, dl_val, dl_test, N, n_train, n_val, n_test


# ----------------------------
# Model blocks
# ----------------------------
class SelfAttention3D(nn.Module):
    """Channel-wise gating from global spatial attention (Eqs. (1)-(2))."""
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
        f = self.Wf(x).view(B, C, N)          # B x C x N
        phi = self.Wphi(x).view(B, C, N)      # B x C x N
        eta = self.softmax(f)                  # softmax over spatial locations
        weighted_phi = eta * phi               # B x C x N
        summed = weighted_phi.sum(dim=-1, keepdim=True)  # B x C x 1
        a = self.Wv(summed.view(B, C, 1, 1, 1))          # B x C x 1 x 1 x 1
        return self.sigmoid(a)                 # channel gate in [0,1]

class PyramidConvBlock(nn.Module):
    """Parallel conv paths with specified kernel sizes; each path: Conv -> ReLU -> Conv -> ReLU."""
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
        return torch.cat(outs, dim=1)  # concat along channel dim

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )

class ResidualBlock3D(nn.Module):
    """(3x3x3 -> ReLU -> 3x3x3) + residual add -> ReLU."""
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
    """
    Pyramid & Attention Generator (U-Net)
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.down1 = PyramidConvBlock(in_ch, out_ch_each=64, kernel_sizes=(3, 5, 7))  # -> 64*3
        ch1 = 64 * 3

        self.down2 = PyramidConvBlock(ch1, out_ch_each=128, kernel_sizes=(3, 5))      # -> 128*2
        ch2 = 128 * 2

        self.down3 = _double_conv(ch2, 512)                                           # -> 512
        ch3 = 512

        # Bottleneck (2x conv) + residual ×6
        self.bottleneck = _double_conv(ch3, ch3)
        self.bottleneck_res6 = nn.Sequential(
            ResidualBlock3D(ch3),
            ResidualBlock3D(ch3),
            ResidualBlock3D(ch3),
            ResidualBlock3D(ch3),
            ResidualBlock3D(ch3),
            ResidualBlock3D(ch3),
        )

        # Decoder
        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(ch3 + ch3, 256)

        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + ch2, 128)

        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + ch1, 64)

        # Attention on last up-block
        self.att = SelfAttention3D(64)
        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)          # B x ch1 x D x H x W
        p1 = self.pool(x1)

        x2 = self.down2(p1)         # B x ch2 x D/2 x H/2 x W/2
        p2 = self.pool(x2)

        x3 = self.down3(p2)         # B x 512 x D/4 x H/4 x W/4
        p3 = self.pool(x3)          # B x 512 x D/8 x H/8 x W/8

        b = self.bottleneck(p3)
        b = self.bottleneck_res6(b)

        u1 = self.up1_ups(b)
        u1 = _pad_or_crop_to(u1, x3)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up1_conv(u1)

        u2 = self.up2_ups(u1)
        u2 = _pad_or_crop_to(u2, x2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2_conv(u2)

        u3 = self.up3_ups(u2)
        u3 = _pad_or_crop_to(u3, x1)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.up3_conv(u3)

        gate = self.att(u3)
        y = gate * u3
        return self.out_conv(y)

class StandardDiscriminator(nn.Module):
    """
    Standard discriminator (6 conv layers total): 5 stride-2 3x3x3 conv blocks [32,64,128,256,512],
    followed by a 1-channel conv head, global avg pool to 1, then sigmoid.
    """
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
        x = self.head(x)                  # [B,1,D',H',W']
        x = F.adaptive_avg_pool3d(x, 1)   # [B,1,1,1,1]
        x = x.view(x.size(0), -1)         # [B,1]
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

@torch.no_grad()
def mmd_gaussian(real: torch.Tensor,
                 fake: torch.Tensor,
                 num_voxels: int = 2048,
                 sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)) -> float:
    B = real.size(0)
    dev = real.device
    total = 0.0
    for i in range(B):
        r = real[i, 0].reshape(-1)
        f = fake[i, 0].reshape(-1)
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

        for mri, pet in train_loader:
            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # ---- Update D ----
            with torch.no_grad():
                fake = G(mri)
            D.zero_grad(set_to_none=True)
            out_real = D(pet)
            out_fake = D(fake.detach())
            loss_D = bce(out_real, real_lbl) + bce(out_fake, fake_lbl)
            loss_D.backward()
            opt_D.step()

            # ---- Update G ----
            G.zero_grad(set_to_none=True)
            fake = G(mri)
            out_fake_for_G = D(fake)
            loss_gan = bce(out_fake_for_G, real_lbl)
            loss_l1 = l1_loss(fake, pet)
            ssim_val = ssim3d(fake, pet, data_range=data_range)
            loss_G = gamma * (loss_l1 + (1.0 - ssim_val)) + lambda_gan * loss_gan
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
                for mri, pet in val_loader:
                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    fake = G(mri)
                    loss_l1 = l1_loss(fake, pet)
                    ssim_v  = ssim3d(fake, pet, data_range=data_range)
                    val_recon += (loss_l1 + (1.0 - ssim_v)).item()
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
                      f"ValRecon(L1+1-SSIM): {val_recon:.4f}  "
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

    ssim_sum = 0.0
    psnr_sum = 0.0
    mse_sum = 0.0
    mmd_sum = 0.0
    n = 0

    for mri, pet in test_loader:
        mri = mri.to(device, non_blocking=True)
        pet = pet.to(device, non_blocking=True)
        fake = G(mri)

        ssim_sum += ssim3d(fake, pet, data_range=data_range).item()
        psnr_sum += psnr(fake, pet, data_range=data_range)
        mse_sum  += F.mse_loss(fake, pet).item()
        mmd_sum  += mmd_gaussian(pet, fake, num_voxels=mmd_voxels)
        n += 1

    return {
        "SSIM": ssim_sum / max(1, n),
        "PSNR": psnr_sum / max(1, n),
        "MSE":  mse_sum  / max(1, n),
        "MMD":  mmd_sum  / max(1, n),
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
        plt.plot(history["val_recon"], label="Val (L1 + 1-SSIM)")
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
    # Peek one shape
    with torch.no_grad():
        mri0, pet0 = next(iter(train_loader))
        print(f"Sample tensor shapes: MRI {tuple(mri0.shape)}, PET {tuple(pet0.shape)}")

    # Instantiate models
    G = Generator(in_ch=1, out_ch=1)
    D = StandardDiscriminator(in_ch=1)

    # Train
    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=EPOCHS, gamma=GAMMA, lambda_gan=LAMBDA_GAN, data_range=DATA_RANGE, verbose=True
    )

    # Save curves & CSV (in run folder)
    curves_path = os.path.join(OUT_RUN, "loss_curves.png")
    save_loss_curves(out["history"], curves_path)
    print(f"Saved loss curves to: {curves_path}")

    csv_path = os.path.join(OUT_RUN, "training_log.csv")
    save_history_csv(out["history"], csv_path)
    print(f"Saved training log CSV to: {csv_path}")

    # Evaluate
    metrics = evaluate_paggan(G, test_loader, device=device, data_range=DATA_RANGE, mmd_voxels=2048)
    print("Test metrics:", metrics)

    metrics_txt = os.path.join(OUT_RUN, "test_metrics.txt")
    with open(metrics_txt, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved test metrics to: {metrics_txt}")


