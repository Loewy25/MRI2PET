"""
PA-GAN (Pyramid + Attention GAN) without Task-Induced Discriminator
-------------------------------------------------------------------
- Generator: U-Net with two pyramid conv blocks (kernels {3,5,7} then {3,5}),
  three decoder up-blocks (Upsample x2 + Conv3D(3x3) + ReLU + Conv3D(3x3) + ReLU),
  self-attention module applied at the last up-block, and final 1-channel output.
- Discriminator (Dstd): 3x3x3 stride-2 conv stack (channels 32,64,128,256,512) + 1ch head + GAP + sigmoid.
- Loss: L = gamma*(L1 + L_SSIM) + lambda_*L_GAN  (standard logistic GAN).
- Optim: Adam, LR(G)=1e-4, LR(D)=4e-4, batch size 2 (set by your DataLoader).
- Metrics: SSIM, PSNR, MSE, MMD (Gaussian-kernel, voxel subsampling for memory safety).

NOTE: This file intentionally omits dataset construction and any file I/O.
Provide train/val/test DataLoaders that yield (mri, pet) pairs with shape [B,1,D,H,W].

Reference: Gao et al., "Task-Induced Pyramid and Attention GAN..." IEEE JBHI, 2022.
"""

from math import log10
from typing import Iterable, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utility: shape alignment
# ----------------------------
def _pad_or_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center pad/crop x spatially to match ref's D,H,W."""
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape

    # Pad or crop depth
    d_pad = max(0, Dr - D)
    h_pad = max(0, Hr - H)
    w_pad = max(0, Wr - W)

    if d_pad or h_pad or w_pad:
        pad = (w_pad // 2, w_pad - w_pad // 2,
               h_pad // 2, h_pad - h_pad // 2,
               d_pad // 2, d_pad - d_pad // 2)
        x = F.pad(x, pad, mode='constant', value=0.)

    # After padding, crop center if needed
    _, _, D2, H2, W2 = x.shape
    d_start = max(0, (D2 - Dr) // 2)
    h_start = max(0, (H2 - Hr) // 2)
    w_start = max(0, (W2 - Wr) // 2)
    x = x[:, :, d_start:d_start+Dr, h_start:h_start+Hr, w_start:w_start+Wr]
    return x


# ----------------------------
# Building blocks
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


class Generator(nn.Module):
    """
    Pyramid & Attention Generator (U-Net)
    - Encoder: PyramidConvBlock({3,5,7}) -> pool -> PyramidConvBlock({3,5}) -> pool -> 2x Conv(3x3x3)
    - Decoder: three up-blocks (Upsample x2 + double 3x3x3 conv + ReLU)
    - Self-attention: applied to the last up-block output, used as a gate (multiplication).
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

        # Bottleneck (two 3x3 convs as in text)
        self.bottleneck = _double_conv(ch3, ch3)

        # Decoder (upsample then concat with skip, then double conv)
        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(ch3 + ch3, 256)

        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + ch2, 128)

        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + ch1, 64)

        # Attention on last up-block
        self.att = SelfAttention3D(64)

        # Output head
        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.down1(x)          # B x ch1 x D x H x W
        p1 = self.pool(x1)

        x2 = self.down2(p1)         # B x ch2 x D/2 x H/2 x W/2
        p2 = self.pool(x2)

        x3 = self.down3(p2)         # B x 512 x D/4 x H/4 x W/4
        p3 = self.pool(x3)          # B x 512 x D/8 x H/8 x W/8

        # Bottleneck
        b = self.bottleneck(p3)     # B x 512 x D/8 x H/8 x W/8

        # Decoder
        u1 = self.up1_ups(b)
        u1 = _pad_or_crop_to(u1, x3)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.up1_conv(u1)      # -> 256

        u2 = self.up2_ups(u1)
        u2 = _pad_or_crop_to(u2, x2)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2_conv(u2)      # -> 128

        u3 = self.up3_ups(u2)
        u3 = _pad_or_crop_to(u3, x1)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.up3_conv(u3)      # -> 64

        # Attention gate on last up-block
        gate = self.att(u3)
        y = gate * u3

        return self.out_conv(y)     # linear output (no tanh/sigmoid)


class StandardDiscriminator(nn.Module):
    """
    Standard discriminator (Sec. III-B.2): 6 conv layers total.
    Implementation: 5 stride-2 3x3x3 conv blocks with channels [32,64,128,256,512],
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
        x = self.head(x)
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        return torch.sigmoid(x)  # B x 1


# ----------------------------
# Losses & Metrics
# ----------------------------
def ssim3d(x: torch.Tensor, y: torch.Tensor, ksize: int = 3,
           k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    """
    3D SSIM (mean over volume), consistent with Eq. (7)-(9).
    x,y: [B,1,D,H,W] in same range; data_range is L in the paper.
    Returns mean SSIM over batch (scalar tensor).
    """
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
    """
    Gaussian-kernel MMD between real/fake voxel distributions (per-batch, averaged).
    To keep memory reasonable, we uniformly subsample `num_voxels` voxels per volume.
    real,fake: [B,1,D,H,W]
    """
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
# GAN training (PA-GAN, no Dtask)
# ----------------------------
def train_paggan(
    G: nn.Module,
    D: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    device: torch.device,
    epochs: int = 150,
    gamma: float = 1.0,
    lambda_gan: float = 0.5,
    data_range: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train PA-GAN (Generator + Standard Discriminator).
    - train_loader / val_loader yield (mri, pet) tensors in [B,1,D,H,W].
    - data_range: dynamic range (L) used in SSIM/PSNR.
    Returns history dict with losses and best states.
    """
    G.to(device); D.to(device)
    G.train(); D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=4e-4)
    bce = nn.BCELoss()

    best_val = float('inf')
    best_G: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

    for epoch in range(1, epochs + 1):
        g_running, d_running, n_batches = 0.0, 0.0, 0
        for mri, pet in train_loader:
            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # ----------------------
            # Update D (real vs fake)
            # ----------------------
            with torch.no_grad():
                fake = G(mri)

            D.zero_grad(set_to_none=True)
            out_real = D(pet)
            out_fake = D(fake.detach())
            loss_D = bce(out_real, real_lbl) + bce(out_fake, fake_lbl)
            loss_D.backward()
            opt_D.step()

            # ----------------------
            # Update G
            # ----------------------
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

        hist["train_G"].append(g_running / max(1, n_batches))
        hist["train_D"].append(d_running / max(1, n_batches))

        # ------------
        # Validation
        # ------------
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon, v_batches = 0.0, 0
                for mri, pet in val_loader:
                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    fake = G(mri)
                    loss_l1 = l1_loss(fake, pet)
                    ssim_val = ssim3d(fake, pet, data_range=data_range)
                    val_recon += (loss_l1 + (1.0 - ssim_val)).item()
                    v_batches += 1
            val_recon /= max(1, v_batches)
            hist["val_recon"].append(val_recon)
            if val_recon < best_val:
                best_val = val_recon
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
            if verbose:
                print(f"Epoch [{epoch}/{epochs}]  "
                      f"G: {hist['train_G'][-1]:.4f}  D: {hist['train_D'][-1]:.4f}  "
                      f"ValRecon(L1+1-SSIM): {val_recon:.4f}")
            G.train()
        elif verbose:
            print(f"Epoch [{epoch}/{epochs}]  G: {hist['train_G'][-1]:.4f}  D: {hist['train_D'][-1]:.4f}")

    if best_G is not None:
        G.load_state_dict(best_G)

    return {"history": hist, "best_G": best_G}


@torch.no_grad()
def evaluate_paggan(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    data_range: float = 1.0,
    mmd_voxels: int = 2048,
) -> Dict[str, float]:
    """
    Evaluate SSIM, PSNR, MSE, MMD on the test set. Returns averaged metrics.
    """
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
        mse_sum += F.mse_loss(fake, pet).item()
        mmd_sum += mmd_gaussian(pet, fake, num_voxels=mmd_voxels)
        n += 1

    return {
        "SSIM": ssim_sum / max(1, n),
        "PSNR": psnr_sum / max(1, n),
        "MSE": mse_sum / max(1, n),
        "MMD": mmd_sum / max(1, n),
    }


# ----------------------------
# Example (skeleton) usage
# ----------------------------
if __name__ == "__main__":
    """
    Plug in your own DataLoaders:
        - train_loader, val_loader, test_loader must yield (mri, pet) tensors, shape [B,1,D,H,W].
        - Keep batch_size=2 for parity with the paper (or adjust as resources allow).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: replace with your actual loaders
    train_loader = None
    val_loader = None
    test_loader = None

    if any(ld is None for ld in [train_loader, val_loader, test_loader]):
        raise RuntimeError("Please provide train/val/test DataLoaders yielding (mri, pet).")

    G = Generator(in_ch=1, out_ch=1)
    D = StandardDiscriminator(in_ch=1)

    out = train_paggan(
        G, D, train_loader, val_loader,
        device=device, epochs=150, gamma=1.0, lambda_gan=0.5, data_range=1.0, verbose=True
    )

    metrics = evaluate_paggan(G, test_loader, device=device, data_range=1.0, mmd_voxels=2048)
    print("Test metrics:", metrics)
