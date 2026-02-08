from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _pad_or_crop_to


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

        # Identity-preserving init (no-op at step 0):
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        avg_map = x.mean(dim=1, keepdim=True)          # [B, 1, D, H, W]
        max_map, _ = x.max(dim=1, keepdim=True)        # [B, 1, D, H, W]
        u = torch.cat([avg_map, max_map], dim=1)       # [B, 2, D, H, W]
        m = self.sigmoid(self.conv(u))                 # [B, 1, D, H, W] in (0,1)

        # Scale so that at init: m=0.5 -> 2*m = 1.0 (identity)
        return 2.0 * m


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

        # Existing channel-wise gate (unchanged)
        self.att = SelfAttention3D(64)

        # New spatial gate (identity-preserving init)
        self.satt = SpatialAttention3D(kernel_size=3)

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

        # ---- Last decoder attention: channel + spatial ----
        gate_c = self.att(u3)         # [B, 64, 1, 1, 1] in (0,1)
        u3 = gate_c * u3

        gate_s = self.satt(u3)        # [B, 1, D, H, W], starts as 1 everywhere
        u3 = gate_s * u3
        # -----------------------------------------------

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
        # x: [B, 2, D, H, W]
        f = self.features(x)          # [B, C, d, h, w]
        s = self.head(f)              # [B, 1, d, h, w]
        return s                      # logits (no sigmoid)


