# model.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _pad_or_crop_to


class SelfAttention3D(nn.Module):
    """
    NOTE: Despite the name, this is effectively a per-channel global gating
    (spatial softmax -> weighted sum -> 1x1x1 conv -> sigmoid), not Transformer-style
    non-local self-attention.
    """
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
        f = self.Wf(x).view(B, C, N)        # [B, C, N]
        phi = self.Wphi(x).view(B, C, N)    # [B, C, N]

        eta = self.softmax(f)               # [B, C, N]
        weighted_phi = eta * phi            # [B, C, N]
        summed = weighted_phi.sum(dim=-1, keepdim=True)  # [B, C, 1]

        a = self.Wv(summed.view(B, C, 1, 1, 1))          # [B, C, 1, 1, 1]
        return self.sigmoid(a)                            # (0,1)


class SpatialAttention3D(nn.Module):
    """
    Lightweight 3D spatial attention (CBAM-style):
      - compress channels with (mean, max) -> [B, 2, D, H, W]
      - conv 2->1 -> sigmoid -> [B, 1, D, H, W]

    Identity-preserving init:
      - conv weights = 0, bias = 0 => conv output = 0
      - sigmoid(0)=0.5; we return (2 * mask) so initial gate == 1 everywhere.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("SpatialAttention3D kernel_size must be odd (e.g., 3, 5).")
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Identity-preserving init (no-op at step 0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        avg_map = x.mean(dim=1, keepdim=True)          # [B, 1, D, H, W]
        max_map, _ = x.max(dim=1, keepdim=True)        # [B, 1, D, H, W]
        u = torch.cat([avg_map, max_map], dim=1)       # [B, 2, D, H, W]
        m = self.sigmoid(self.conv(u))                 # [B, 1, D, H, W] in (0,1)
        return 2.0 * m                                 # identity at init


class SkipGate3D(nn.Module):
    """
    Conditioned spatial gate for U-Net skip connections.

    mask = sigmoid(Conv([skip, gate])) -> [B, 1, D, H, W]
    return skip * (2*mask) so init is identity (mask starts at 0.5 => 2*0.5=1).
    """
    def __init__(self, skip_ch: int, gate_ch: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("SkipGate3D kernel_size must be odd (e.g., 1, 3, 5).")
        pad = kernel_size // 2

        self.conv = nn.Conv3d(
            skip_ch + gate_ch,
            1,
            kernel_size=kernel_size,
            padding=pad,
            bias=True
        )
        self.sigmoid = nn.Sigmoid()

        # Identity-preserving init
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Safety: enforce spatial match (especially helpful with odd sizes)
        if gate.shape[2:] != skip.shape[2:]:
            gate = _pad_or_crop_to(gate, skip)

        m = self.sigmoid(self.conv(torch.cat([skip, gate], dim=1)))  # [B,1,D,H,W]
        return skip * (2.0 * m)


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
    """
    Deeper-by-1 version (minimal-risk):
      - Adds down4 at 512 channels
      - Bottleneck now runs one level deeper
      - Adds up0 that outputs 512
      - Adds gate_x4 for the new skip
    Everything else is unchanged.
    """
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.down1 = PyramidConvBlock(in_ch, out_ch_each=64, kernel_sizes=(3, 5, 7))
        ch1 = 64 * 3  # 192

        self.down2 = PyramidConvBlock(ch1, out_ch_each=128, kernel_sizes=(3, 5))
        ch2 = 128 * 2  # 256

        self.down3 = _double_conv(ch2, 512)
        ch3 = 512

        # ---- NEW: one more deeper encoder block (keep 512 channels) ----
        self.down4 = _double_conv(ch3, 512)
        ch4 = 512
        # ----------------------------------------------------------------

        # Bottleneck (same channel width, now applied after one more pool)
        self.bottleneck = _double_conv(ch4, ch4)
        self.bottleneck_res6 = nn.Sequential(
            ResidualBlock3D(ch4), ResidualBlock3D(ch4), ResidualBlock3D(ch4),
            ResidualBlock3D(ch4), ResidualBlock3D(ch4), ResidualBlock3D(ch4),
        )

        # Decoder
        # ---- NEW: first up stage to fuse with x4 (skip at depth 4) ----
        self.up0_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up0_conv = _double_conv(ch4 + ch4, ch4)   # cat([u0_up(512), x4(512)]) -> 1024 -> 512
        # ----------------------------------------------------------------

        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(ch3 + ch3, 256)   # cat([u1_up(512), x3(512)]) -> 1024 -> 256

        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + ch2, 128)   # cat([u2_up(256), x2(256)]) -> 512 -> 128

        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + ch1, 64)    # cat([u3_up(128), x1(192)]) -> 320 -> 64

        # Skip gates (conditioned spatial)
        self.gate_x4 = SkipGate3D(skip_ch=ch4, gate_ch=ch4)    # x4 gated by u0_up (512+512)
        self.gate_x3 = SkipGate3D(skip_ch=ch3, gate_ch=ch4)    # x3 gated by u1_up (512+512)
        self.gate_x2 = SkipGate3D(skip_ch=ch2, gate_ch=256)    # x2 gated by u2_up (256+256)
        self.gate_x1 = SkipGate3D(skip_ch=ch1, gate_ch=128)    # x1 gated by u3_up (192+128)

        # Existing last-stage channel gate
        self.att = SelfAttention3D(64)

        # Existing last-stage spatial gate
        self.satt = SpatialAttention3D(kernel_size=3)

        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.down1(x);  p1 = self.pool(x1)
        x2 = self.down2(p1); p2 = self.pool(x2)
        x3 = self.down3(p2); p3 = self.pool(x3)

        # ---- NEW deeper encoder stage ----
        x4 = self.down4(p3); p4 = self.pool(x4)
        # ---------------------------------

        # Bottleneck (now applied at p4)
        b = self.bottleneck(p4)
        b = self.bottleneck_res6(b)

        # ---- NEW Decoder stage 0 (gate x4) ----
        u0_up = self.up0_ups(b)
        u0_up = _pad_or_crop_to(u0_up, x4)
        x4_g  = self.gate_x4(x4, u0_up)
        u0 = torch.cat([u0_up, x4_g], dim=1)
        u0 = self.up0_conv(u0)          # -> [B, 512, ...]
        # --------------------------------------

        # Decoder stage 1 (gate x3)
        u1_up = self.up1_ups(u0)
        u1_up = _pad_or_crop_to(u1_up, x3)
        x3_g  = self.gate_x3(x3, u1_up)
        u1 = torch.cat([u1_up, x3_g], dim=1)
        u1 = self.up1_conv(u1)          # -> [B, 256, ...]

        # Decoder stage 2 (gate x2)
        u2_up = self.up2_ups(u1)
        u2_up = _pad_or_crop_to(u2_up, x2)
        x2_g  = self.gate_x2(x2, u2_up)
        u2 = torch.cat([u2_up, x2_g], dim=1)
        u2 = self.up2_conv(u2)          # -> [B, 128, ...]

        # Decoder stage 3 (gate x1)
        u3_up = self.up3_ups(u2)
        u3_up = _pad_or_crop_to(u3_up, x1)
        x1_g  = self.gate_x1(x1, u3_up)
        u3 = torch.cat([u3_up, x1_g], dim=1)
        u3 = self.up3_conv(u3)          # -> [B, 64, ...]

        # Last decoder attention: channel + spatial (unchanged)
        gate_c = self.att(u3)           # [B, 64, 1, 1, 1]
        u3 = gate_c * u3

        gate_s = self.satt(u3)          # [B, 1, D, H, W] (starts ~1 everywhere)
        u3 = gate_s * u3

        return self.out_conv(u3)


class CondPatchDiscriminator3D(nn.Module):
    """
    Conditional 3D PatchGAN: input is [B, 2, D, H, W] = concat(MRI, PET).
    Outputs patch logits [B, 1, d, h, w] (no global pooling).
    """
    def __init__(self, in_ch: int = 2):  # MRI+PET
        super().__init__()
        layers = []
        prev = in_ch

        # PatchGAN-style: 3 downsampling layers (stride=2), then 2 local layers (stride=1)
        channels = [32, 64, 128, 256, 512]
        strides  = [2, 2, 2, 1, 1]

        for c, s in zip(channels, strides):
            layers += [
                nn.Conv3d(prev, c, kernel_size=3, stride=s, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = c

        self.features = nn.Sequential(*layers)
        self.head = nn.Conv3d(prev, 1, kernel_size=3, padding=1, bias=True)  # patch logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)     # [B, C, d, h, w]
        s = self.head(f)         # [B, 1, d, h, w]
        return s                 # logits (no sigmoid)

