# models.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import (
    DETACH_BASE_LATENT_FOR_PRIOR,
    PRIOR_GAIN_INIT_B,
    PRIOR_GAIN_INIT_X3,
    PRIOR_GAIN_INIT_X4,
    SPATIAL_PRIOR_K,
    USE_BRAAK_HEAD,
    USE_CLINICAL,
    USE_FLAIR,
    USE_SPATIAL_PRIOR,
)
from .utils import _pad_or_crop_to


# =========================================================================
# Shared building blocks (unchanged from baseline)
# =========================================================================
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
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        u = torch.cat([avg_map, max_map], dim=1)
        m = self.sigmoid(self.conv(u))
        return 2.0 * m


class SkipGate3D(nn.Module):
    def __init__(self, skip_ch: int, gate_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(skip_ch + gate_ch, 1, kernel_size=kernel_size, padding=pad, bias=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[2:] != skip.shape[2:]:
            gate = _pad_or_crop_to(gate, skip)
        m = self.sigmoid(self.conv(torch.cat([skip, gate], dim=1)))
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
        return torch.cat([p(x) for p in self.paths], dim=1)


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
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


def _apply_film(feat: torch.Tensor, film: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    gamma, beta = film
    return feat * (1.0 + gamma[..., None, None, None]) + beta[..., None, None, None]


# =========================================================================
# Baseline Generator (T1-only) — unchanged architecture, added
# use_checkpoint and return_features support
# =========================================================================
class Generator(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        self.down1 = PyramidConvBlock(in_ch, out_ch_each=64, kernel_sizes=(3, 5, 7))
        ch1 = 64 * 3  # 192
        self.down2 = PyramidConvBlock(ch1, out_ch_each=128, kernel_sizes=(3, 5))
        ch2 = 128 * 2  # 256
        self.down3 = _double_conv(ch2, 512)
        ch3 = 512
        self.down4 = _double_conv(ch3, 512)
        ch4 = 512

        # Bottleneck
        self.bottleneck = _double_conv(ch4, ch4)
        self.bottleneck_res6 = nn.Sequential(
            ResidualBlock3D(ch4), ResidualBlock3D(ch4), ResidualBlock3D(ch4),
            ResidualBlock3D(ch4), ResidualBlock3D(ch4), ResidualBlock3D(ch4),
        )

        # Decoder
        self.up0_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up0_conv = _double_conv(ch4 + ch4, ch4)
        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(ch3 + ch3, 256)
        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + ch2, 128)
        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + ch1, 64)

        # Skip gates
        self.gate_x4 = SkipGate3D(skip_ch=ch4, gate_ch=ch4)
        self.gate_x3 = SkipGate3D(skip_ch=ch3, gate_ch=ch4)
        self.gate_x2 = SkipGate3D(skip_ch=ch2, gate_ch=256)
        self.gate_x1 = SkipGate3D(skip_ch=ch1, gate_ch=128)

        self.att = SelfAttention3D(64)
        self.satt = SpatialAttention3D(kernel_size=3)
        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        _ckpt = (lambda fn, t: grad_checkpoint(fn, t, use_reentrant=False)) if self.use_checkpoint else (lambda fn, t: fn(t))

        def _ups(module, t):
            return module(t.float()).to(t.dtype)

        # Encoder
        x1 = _ckpt(self.down1, x);  p1 = self.pool(x1)
        x2 = _ckpt(self.down2, p1); p2 = self.pool(x2)
        x3 = _ckpt(self.down3, p2); p3 = self.pool(x3)
        x4 = _ckpt(self.down4, p3); p4 = self.pool(x4)

        # Bottleneck
        b = _ckpt(self.bottleneck, p4)
        b = _ckpt(self.bottleneck_res6, b)

        # Decoder
        u0_up = _ups(self.up0_ups, b)
        u0_up = _pad_or_crop_to(u0_up, x4)
        x4_g = self.gate_x4(x4, u0_up)
        u0 = self.up0_conv(torch.cat([u0_up, x4_g], dim=1))

        u1_up = _ups(self.up1_ups, u0)
        u1_up = _pad_or_crop_to(u1_up, x3)
        x3_g = self.gate_x3(x3, u1_up)
        u1 = self.up1_conv(torch.cat([u1_up, x3_g], dim=1))

        u2_up = _ups(self.up2_ups, u1)
        u2_up = _pad_or_crop_to(u2_up, x2)
        x2_g = self.gate_x2(x2, u2_up)
        u2 = self.up2_conv(torch.cat([u2_up, x2_g], dim=1))

        u3_up = _ups(self.up3_ups, u2)
        u3_up = _pad_or_crop_to(u3_up, x1)
        x1_g = self.gate_x1(x1, u3_up)
        u3 = self.up3_conv(torch.cat([u3_up, x1_g], dim=1))

        gate_c = self.att(u3)
        u3 = gate_c * u3
        gate_s = self.satt(u3)
        u3 = gate_s * u3

        out = self.out_conv(u3)

        if not return_features:
            return out

        feats = {
            "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            "b": b,
            "u0": u0, "u1": u1, "u2": u2, "u3": u3,
        }
        return out, feats


# =========================================================================
# Discriminator (unchanged from baseline)
# =========================================================================
class CondPatchDiscriminator3D(nn.Module):
    def __init__(self, in_ch: int = 2):
        super().__init__()
        layers = []
        prev = in_ch
        channels = [32, 64, 128, 256, 512]
        strides = [2, 2, 2, 1, 1]
        for c, s in zip(channels, strides):
            layers += [
                nn.Conv3d(prev, c, kernel_size=3, stride=s, padding=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv3d(prev, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# =========================================================================
# NEW: Residual-spatial-prior modules
# =========================================================================

# --- A. FlairPromptEncoder3D ---
class FlairPromptEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 1, z_dim: int = 128):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.enc1 = _double_conv(in_ch, 16)   # p1
        self.enc2 = _double_conv(16, 32)       # p2
        self.enc3 = _double_conv(32, 64)       # p3
        self.enc4 = _double_conv(64, 64)       # p4
        self.enc_b = _double_conv(64, 64)      # pb
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(64, z_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        p1 = self.enc1(x)
        p2 = self.enc2(self.pool(p1))
        p3 = self.enc3(self.pool(p2))
        p4 = self.enc4(self.pool(p3))
        pb = self.enc_b(self.pool(p4))
        z_flair = self.proj(self.gap(pb).flatten(1))
        return {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "pb": pb, "z_flair": z_flair}

    def zero_prompts(self, B: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        """Return None spatial prompts (ablation: no FLAIR). Skips proj_prompt conv entirely."""
        return {
            "p1": None,
            "p2": None,
            "p3": None,
            "p4": None,
            "pb": None,
            "z_flair": torch.zeros(B, self.proj.out_features, device=device, dtype=dtype),
        }


# --- B. ClinicalFiLMConditioner ---
class ClinicalFiLMConditioner(nn.Module):
    def __init__(self, clinical_dim: int = 10, z_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, z_dim),
            nn.ReLU(inplace=True),
        )
        # FiLM heads: (gamma, beta) pairs for each decoder scale
        self.film_b = nn.Linear(z_dim, 128 * 2)
        self.film_d1 = nn.Linear(z_dim, 128 * 2)
        self.film_d2 = nn.Linear(z_dim, 64 * 2)
        self.film_d3 = nn.Linear(z_dim, 32 * 2)
        self.film_d4 = nn.Linear(z_dim, 16 * 2)

        # Zero-init all FiLM heads → identity modulation at start
        for head in [self.film_b, self.film_d1, self.film_d2, self.film_d3, self.film_d4]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, clinical: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        z = self.trunk(clinical)

        def _split(head, ch):
            out = head(z)
            return out[:, :ch], out[:, ch:]

        film = {
            "b": _split(self.film_b, 128),
            "d1": _split(self.film_d1, 128),
            "d2": _split(self.film_d2, 64),
            "d3": _split(self.film_d3, 32),
            "d4": _split(self.film_d4, 16),
        }
        return z, film

    def identity(self, B: int, device: torch.device, dtype: torch.dtype
                 ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """Return identity FiLM (gamma=0, beta=0) and zero z_clin (ablation: no clinical)."""
        z = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
        film = {
            "b":  (torch.zeros(B, 128, device=device, dtype=dtype), torch.zeros(B, 128, device=device, dtype=dtype)),
            "d1": (torch.zeros(B, 128, device=device, dtype=dtype), torch.zeros(B, 128, device=device, dtype=dtype)),
            "d2": (torch.zeros(B, 64,  device=device, dtype=dtype), torch.zeros(B, 64,  device=device, dtype=dtype)),
            "d3": (torch.zeros(B, 32,  device=device, dtype=dtype), torch.zeros(B, 32,  device=device, dtype=dtype)),
            "d4": (torch.zeros(B, 16,  device=device, dtype=dtype), torch.zeros(B, 16,  device=device, dtype=dtype)),
        }
        return z, film


class DirectClinicalConditioner(nn.Module):
    def __init__(self, clinical_dim: int = 10, z_dim: int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, z_dim),
            nn.ReLU(inplace=True),
        )
        self.film_b = nn.Linear(z_dim, 512 * 2)
        self.film_d1 = nn.Linear(z_dim, 512 * 2)
        self.film_d2 = nn.Linear(z_dim, 256 * 2)
        self.film_d3 = nn.Linear(z_dim, 128 * 2)
        self.film_d4 = nn.Linear(z_dim, 64 * 2)

        for head in [self.film_b, self.film_d1, self.film_d2, self.film_d3, self.film_d4]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, clinical: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        z = self.trunk(clinical)

        def _split(head, ch):
            out = head(z)
            return out[:, :ch], out[:, ch:]

        film = {
            "b": _split(self.film_b, 512),
            "d1": _split(self.film_d1, 512),
            "d2": _split(self.film_d2, 256),
            "d3": _split(self.film_d3, 128),
            "d4": _split(self.film_d4, 64),
        }
        return z, film

    def identity(
        self,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        z = torch.zeros(B, self.z_dim, device=device, dtype=dtype)
        film = {
            "b": (torch.zeros(B, 512, device=device, dtype=dtype), torch.zeros(B, 512, device=device, dtype=dtype)),
            "d1": (torch.zeros(B, 512, device=device, dtype=dtype), torch.zeros(B, 512, device=device, dtype=dtype)),
            "d2": (torch.zeros(B, 256, device=device, dtype=dtype), torch.zeros(B, 256, device=device, dtype=dtype)),
            "d3": (torch.zeros(B, 128, device=device, dtype=dtype), torch.zeros(B, 128, device=device, dtype=dtype)),
            "d4": (torch.zeros(B, 64, device=device, dtype=dtype), torch.zeros(B, 64, device=device, dtype=dtype)),
        }
        return z, film


# --- C. BraakHead ---
class BraakHead(nn.Module):
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.head = nn.Linear(in_dim, 3)

    def forward(self, z_fuse: torch.Tensor) -> torch.Tensor:
        return self.head(z_fuse)


# --- D. CortexSpatialPrior ---
class CortexSpatialPrior(nn.Module):
    def __init__(self, z_dim: int = 128, num_basis: int = SPATIAL_PRIOR_K):
        super().__init__()
        self.router = nn.Linear(z_dim, num_basis)
        self.basis_low = nn.Parameter(torch.randn(num_basis, 1, 8, 8, 8) * 0.01)
        self.proj_b = nn.Conv3d(1, 128, kernel_size=1, bias=True)
        self.proj_x4 = nn.Conv3d(1, 128, kernel_size=1, bias=True)
        self.proj_x3 = nn.Conv3d(1, 64, kernel_size=1, bias=True)
        self.gain_b = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_B)))
        self.gain_x4 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X4)))
        self.gain_x3 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X3)))

        for proj in [self.proj_b, self.proj_x4, self.proj_x3]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(proj.bias)

    def zero_prior_maps(self, feats: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        dev = feats["b"].device
        dtype = feats["b"].dtype
        B = feats["b"].size(0)
        prior_maps = {
            "b": torch.zeros(B, 128, *feats["b"].shape[2:], device=dev, dtype=dtype),
            "x4": torch.zeros(B, 128, *feats["x4"].shape[2:], device=dev, dtype=dtype),
            "x3": torch.zeros(B, 64, *feats["x3"].shape[2:], device=dev, dtype=dtype),
        }
        zero = torch.zeros((), device=dev, dtype=torch.float32)
        prior_stats = {
            "in_cortex_mag": zero,
            "out_cortex_mag": zero,
            "router_entropy": zero,
            "router_top1_mean": zero,
        }
        return prior_maps, prior_stats


class ShallowImageStem(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.down1 = PyramidConvBlock(in_ch, out_ch_each=64, kernel_sizes=(3, 5, 7))
        self.down2 = PyramidConvBlock(64 * 3, out_ch_each=128, kernel_sizes=(3, 5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        return x1, x2


class FusionBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        return self.block(torch.cat([x_a, x_b], dim=1))


class RegionalModulator(nn.Module):
    def __init__(self, z_dim: int = 128, num_basis: int = SPATIAL_PRIOR_K):
        super().__init__()
        self.router = nn.Linear(z_dim, num_basis)
        self.basis_low = nn.Parameter(torch.randn(num_basis, 1, 8, 8, 8) * 0.01)
        self.proj_b = nn.Conv3d(1, 512, kernel_size=1, bias=True)
        self.proj_d1 = nn.Conv3d(1, 512, kernel_size=1, bias=True)
        self.proj_d2 = nn.Conv3d(1, 256, kernel_size=1, bias=True)
        self.proj_d3 = nn.Conv3d(1, 128, kernel_size=1, bias=True)
        self.proj_d4 = nn.Conv3d(1, 64, kernel_size=1, bias=True)
        self.gain_b = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_B)))
        self.gain_d1 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X4)))
        self.gain_d2 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X3)))
        self.gain_d3 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X3)))
        self.gain_d4 = nn.Parameter(torch.tensor(float(PRIOR_GAIN_INIT_X3)))

        for proj in [self.proj_b, self.proj_d1, self.proj_d2, self.proj_d3, self.proj_d4]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(proj.bias)

    def zero_maps(
        self,
        ref_feats: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device = ref_feats["b"].device
        dtype = ref_feats["b"].dtype
        B = ref_feats["b"].size(0)
        maps = {
            "b": torch.zeros(B, 512, *ref_feats["b"].shape[2:], device=device, dtype=dtype),
            "d1": torch.zeros(B, 512, *ref_feats["x4"].shape[2:], device=device, dtype=dtype),
            "d2": torch.zeros(B, 256, *ref_feats["x3"].shape[2:], device=device, dtype=dtype),
            "d3": torch.zeros(B, 128, *ref_feats["x2"].shape[2:], device=device, dtype=dtype),
            "d4": torch.zeros(B, 64, *ref_feats["x1"].shape[2:], device=device, dtype=dtype),
        }
        zero = torch.zeros((), device=device, dtype=torch.float32)
        stats = {
            "in_cortex_mag": zero,
            "out_cortex_mag": zero,
            "router_entropy": zero,
            "router_top1_mean": zero,
        }
        return maps, stats

    def _resize_mask(
        self,
        mask: Optional[torch.Tensor],
        size: Tuple[int, int, int],
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        fill: float,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full((B, 1, *size), fill, device=device, dtype=dtype)
        return F.interpolate(mask.float(), size=size, mode="nearest").to(device=device, dtype=dtype)

    def _mix_basis(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bk,kcdhw->bcdhw", weights, self.basis_low.to(dtype=weights.dtype))

    def _build_map(
        self,
        prior_low: torch.Tensor,
        proj: nn.Conv3d,
        gain: nn.Parameter,
        ref: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = ref.size(0)
        size = tuple(ref.shape[2:])
        device = ref.device
        dtype = ref.dtype
        prior = prior_low
        if prior.shape[2:] != size:
            prior = F.interpolate(prior.float(), size=size, mode="trilinear", align_corners=False).to(dtype=dtype)
        brain_s = self._resize_mask(brain_mask, size, B, device, dtype, fill=1.0)
        cortex_s = self._resize_mask(cortex_mask, size, B, device, dtype, fill=0.0)
        gate = (0.2 * brain_s) + (0.8 * cortex_s)
        mod = proj(prior * gate)
        mod = gain.to(device=device, dtype=mod.dtype) * mod
        outside = torch.clamp(brain_s - cortex_s, min=0.0)
        return mod, cortex_s, outside

    def forward(
        self,
        z_fuse: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
        ref_feats: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights = torch.softmax(self.router(z_fuse.float()), dim=1).to(dtype=ref_feats["b"].dtype)
        prior_low = self._mix_basis(weights)

        refs = {
            "b": (self.proj_b, self.gain_b, ref_feats["b"]),
            "d1": (self.proj_d1, self.gain_d1, ref_feats["x4"]),
            "d2": (self.proj_d2, self.gain_d2, ref_feats["x3"]),
            "d3": (self.proj_d3, self.gain_d3, ref_feats["x2"]),
            "d4": (self.proj_d4, self.gain_d4, ref_feats["x1"]),
        }
        mod_maps = {}
        cortex_masks = {}
        outside_masks = {}
        for key, (proj, gain, ref) in refs.items():
            mod_maps[key], cortex_masks[key], outside_masks[key] = self._build_map(
                prior_low, proj, gain, ref, brain_mask, cortex_mask
            )

        def _masked_mag(mod: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = (mask.sum() * mod.size(1)).clamp(min=1.0)
            return (mod.abs() * mask).sum() / denom

        in_mag = torch.stack([
            _masked_mag(mod_maps[key], cortex_masks[key]) for key in mod_maps
        ]).mean()
        out_mag = torch.stack([
            _masked_mag(mod_maps[key], outside_masks[key]) for key in mod_maps
        ]).mean()
        router_entropy = -(weights.float() * torch.log(weights.float() + 1e-8)).sum(dim=1).mean()
        router_top1_mean = weights.float().max(dim=1).values.mean()
        stats = {
            "in_cortex_mag": in_mag.detach().float(),
            "out_cortex_mag": out_mag.detach().float(),
            "router_entropy": router_entropy.detach().float(),
            "router_top1_mean": router_top1_mean.detach().float(),
        }
        return mod_maps, stats


# --- E. PromptFusionBlock ---
class PromptFusionBlock(nn.Module):
    def __init__(self, c_base: int, c_prompt: int, c_out: int):
        super().__init__()
        self.proj_base = nn.Conv3d(c_base, c_out, 1) if c_base != c_out else nn.Identity()
        self.proj_prompt = nn.Conv3d(c_prompt, c_out, 1)
        self.gate_conv = nn.Conv3d(2 * c_out, c_out, 1)
        self.res_conv = nn.Sequential(
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(
        self,
        base_feat: torch.Tensor,
        flair_prompt: Optional[torch.Tensor],
        film: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        gamma, beta = film
        bp = self.proj_base(base_feat)
        if flair_prompt is None:
            fp = torch.zeros_like(bp)
        else:
            fp = self.proj_prompt(flair_prompt)
            if fp.shape[2:] != bp.shape[2:]:
                fp = _pad_or_crop_to(fp, bp)
        gate = torch.sigmoid(self.gate_conv(torch.cat([bp, fp], dim=1)))
        fused = bp + gate * fp
        fused = fused * (1.0 + gamma[..., None, None, None]) + beta[..., None, None, None]
        fused = fused + self.res_conv(fused)
        return F.relu(fused)


# --- F. ResidualDecoder3D ---
class ResidualDecoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse_b = PromptFusionBlock(512, 64, 128)
        self.fuse_s4 = PromptFusionBlock(512, 64, 128)
        self.fuse_s3 = PromptFusionBlock(512, 64, 64)
        self.fuse_s2 = PromptFusionBlock(256, 32, 32)
        self.fuse_s1 = PromptFusionBlock(192, 16, 16)

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv_d1 = _double_conv(128 + 128, 128)
        self.conv_d2 = _double_conv(128 + 64, 64)
        self.conv_d3 = _double_conv(64 + 32, 32)
        self.conv_d4 = _double_conv(32 + 16, 16)

        self.delta_out = nn.Conv3d(16, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        pf: Dict[str, torch.Tensor],
        film: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        prior_maps: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        def _up(t, ref):
            return _pad_or_crop_to(self.up(t.float()).to(t.dtype), ref)

        rb = self.fuse_b(feats["b"], pf["pb"], film["b"])
        rb = rb + _pad_or_crop_to(prior_maps["b"], rb)
        s4 = self.fuse_s4(feats["x4"], pf["p4"], film["d1"])
        s4 = s4 + _pad_or_crop_to(prior_maps["x4"], s4)
        d1 = self.conv_d1(torch.cat([_up(rb, s4), s4], dim=1))

        s3 = self.fuse_s3(feats["x3"], pf["p3"], film["d2"])
        s3 = s3 + _pad_or_crop_to(prior_maps["x3"], s3)
        d2 = self.conv_d2(torch.cat([_up(d1, s3), s3], dim=1))

        s2 = self.fuse_s2(feats["x2"], pf["p2"], film["d3"])
        d3 = self.conv_d3(torch.cat([_up(d2, s2), s2], dim=1))

        s1 = self.fuse_s1(feats["x1"], pf["p1"], film["d4"])
        d4 = self.conv_d4(torch.cat([_up(d3, s1), s1], dim=1))

        return self.delta_out(d4)


# =========================================================================
# G. ResidualSpatialPriorGenerator — top-level wrapper
# =========================================================================
class ResidualSpatialPriorGenerator(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, use_checkpoint: bool = False,
                 clinical_dim: int = 10, prompt_z_dim: int = 128):
        super().__init__()
        self.base = Generator(in_ch=in_ch, out_ch=out_ch, use_checkpoint=use_checkpoint)
        self.flair_encoder = FlairPromptEncoder3D(in_ch=1, z_dim=prompt_z_dim)
        self.clinical_conditioner = ClinicalFiLMConditioner(clinical_dim=clinical_dim, z_dim=prompt_z_dim)
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Fusion MLP: z_t1(512) + z_flair(128) + z_clin(128) → 128
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 + prompt_z_dim + prompt_z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, prompt_z_dim),
            nn.ReLU(inplace=True),
        )
        self.braak_head = BraakHead(in_dim=prompt_z_dim)
        self.spatial_prior = CortexSpatialPrior(z_dim=prompt_z_dim)
        self.residual_decoder = ResidualDecoder3D()

    def forward(
        self,
        t1: torch.Tensor,
        flair: torch.Tensor,
        clinical: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
        return_aux: bool = False,
    ):
        B = t1.size(0)
        dev, dtype = t1.device, t1.dtype

        # 1. Base T1-only generator
        pet_base, feats = self.base(t1, return_features=True)

        # 2. FLAIR prompt pyramid (or zeros if ablation step < 2)
        if USE_FLAIR:
            pf = self.flair_encoder(flair)
        else:
            pf = self.flair_encoder.zero_prompts(B, dev, dtype)

        # 3. Clinical FiLM (or identity if ablation step < 3)
        if USE_CLINICAL:
            z_clin, film = self.clinical_conditioner(clinical)
        else:
            z_clin, film = self.clinical_conditioner.identity(B, dev, dtype)

        # 4. Fuse latents for auxiliary heads
        z_t1 = self.gap(feats["b"]).flatten(1)  # [B, 512]
        z_t1_aux = z_t1.detach() if DETACH_BASE_LATENT_FOR_PRIOR else z_t1
        z_fuse = self.fusion_mlp(torch.cat([z_t1_aux, pf["z_flair"], z_clin], dim=1))

        # 5. Braak head + spatial prior
        if USE_BRAAK_HEAD:
            braak_pred = self.braak_head(z_fuse)
        else:
            braak_pred = torch.zeros(B, 3, device=dev, dtype=dtype)

        if USE_SPATIAL_PRIOR:
            prior_maps, prior_stats = self.spatial_prior(z_fuse, brain_mask, cortex_mask, feats)
        else:
            prior_maps, prior_stats = self.spatial_prior.zero_prior_maps(feats)

        # 6. Residual decoder
        delta_pet = self.residual_decoder(feats, pf, film, prior_maps)

        # 7. Combine
        pet_hat = pet_base + delta_pet

        if not return_aux:
            return pet_hat

        aux: Dict[str, Any] = {
            "pet_base": pet_base,
            "delta_pet": delta_pet,
            "braak_pred": braak_pred,
            "z_fuse": z_fuse,
            "prior_stats": prior_stats,
        }
        return pet_hat, aux


PromptResidualBraakGenerator = ResidualSpatialPriorGenerator


class DirectMMConditionalGenerator(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        flair_ch: int = 1,
        out_ch: int = 1,
        clinical_dim: int = 10,
        prompt_z_dim: int = 128,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.t1_stem = ShallowImageStem(in_ch=in_ch)
        self.flair_stem = ShallowImageStem(in_ch=flair_ch)
        self.fuse1 = FusionBlock3D((64 * 3) * 2, 64 * 3)
        self.fuse2 = FusionBlock3D((128 * 2) * 2, 128 * 2)

        self.down3 = _double_conv(128 * 2, 512)
        self.down4 = _double_conv(512, 512)
        self.bottleneck = _double_conv(512, 512)
        self.bottleneck_res6 = nn.Sequential(
            ResidualBlock3D(512), ResidualBlock3D(512), ResidualBlock3D(512),
            ResidualBlock3D(512), ResidualBlock3D(512), ResidualBlock3D(512),
        )

        self.up0_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up0_conv = _double_conv(512 + 512, 512)
        self.up1_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up1_conv = _double_conv(512 + 512, 256)
        self.up2_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up2_conv = _double_conv(256 + 256, 128)
        self.up3_ups = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up3_conv = _double_conv(128 + 192, 64)

        self.gate_x4 = SkipGate3D(skip_ch=512, gate_ch=512)
        self.gate_x3 = SkipGate3D(skip_ch=512, gate_ch=512)
        self.gate_x2 = SkipGate3D(skip_ch=256, gate_ch=256)
        self.gate_x1 = SkipGate3D(skip_ch=192, gate_ch=128)

        self.clinical_conditioner = DirectClinicalConditioner(clinical_dim=clinical_dim, z_dim=prompt_z_dim)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512 + prompt_z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, prompt_z_dim),
            nn.ReLU(inplace=True),
        )
        self.braak_head = BraakHead(in_dim=prompt_z_dim)
        self.regional_modulator = RegionalModulator(z_dim=prompt_z_dim)

        self.att = SelfAttention3D(64)
        self.satt = SpatialAttention3D(kernel_size=3)
        self.out_conv = nn.Conv3d(64, out_ch, kernel_size=3, padding=1, bias=True)

    def _condition(
        self,
        feat: torch.Tensor,
        film: Tuple[torch.Tensor, torch.Tensor],
        mod: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mod is not None:
            feat = feat + _pad_or_crop_to(mod, feat)
        return F.relu(_apply_film(feat, film))

    def forward(
        self,
        t1: torch.Tensor,
        flair: torch.Tensor,
        clinical: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
        return_aux: bool = False,
    ):
        B = t1.size(0)
        device, dtype = t1.device, t1.dtype

        x1_t1, x2_t1 = self.t1_stem(t1)
        if USE_FLAIR:
            x1_fl, x2_fl = self.flair_stem(flair)
        else:
            x1_fl = torch.zeros_like(x1_t1)
            x2_fl = torch.zeros_like(x2_t1)

        x1 = self.fuse1(x1_t1, x1_fl)
        x2 = self.fuse2(x2_t1, x2_fl)
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        b = self.bottleneck(self.pool(x4))
        b = self.bottleneck_res6(b)

        if USE_CLINICAL:
            z_clin, film = self.clinical_conditioner(clinical)
        else:
            z_clin, film = self.clinical_conditioner.identity(B, device, dtype)

        z_img = self.gap(b).flatten(1)
        z_fuse = self.fusion_mlp(torch.cat([z_img, z_clin], dim=1))

        if USE_BRAAK_HEAD:
            braak_pred = self.braak_head(z_fuse)
        else:
            braak_pred = torch.zeros(B, 3, device=device, dtype=dtype)

        ref_feats = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "b": b}
        if USE_SPATIAL_PRIOR:
            mod_maps, mod_stats = self.regional_modulator(z_fuse, brain_mask, cortex_mask, ref_feats)
        else:
            mod_maps, mod_stats = self.regional_modulator.zero_maps(ref_feats)

        def _upsample(up_module: nn.Module, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            return _pad_or_crop_to(up_module(src.float()).to(src.dtype), ref)

        b = self._condition(b, film["b"], mod_maps["b"])

        u0_up = _upsample(self.up0_ups, b, x4)
        x4_g = self.gate_x4(x4, u0_up)
        u0 = self.up0_conv(torch.cat([u0_up, x4_g], dim=1))
        u0 = self._condition(u0, film["d1"], mod_maps["d1"])

        u1_up = _upsample(self.up1_ups, u0, x3)
        x3_g = self.gate_x3(x3, u1_up)
        u1 = self.up1_conv(torch.cat([u1_up, x3_g], dim=1))
        u1 = self._condition(u1, film["d2"], mod_maps["d2"])

        u2_up = _upsample(self.up2_ups, u1, x2)
        x2_g = self.gate_x2(x2, u2_up)
        u2 = self.up2_conv(torch.cat([u2_up, x2_g], dim=1))
        u2 = self._condition(u2, film["d3"], mod_maps["d3"])

        u3_up = _upsample(self.up3_ups, u2, x1)
        x1_g = self.gate_x1(x1, u3_up)
        u3 = self.up3_conv(torch.cat([u3_up, x1_g], dim=1))
        u3 = self._condition(u3, film["d4"], mod_maps["d4"])

        u3 = self.att(u3) * u3
        u3 = self.satt(u3) * u3
        pet_hat = self.out_conv(u3)

        if not return_aux:
            return pet_hat

        aux: Dict[str, Any] = {
            "braak_pred": braak_pred,
            "z_fuse": z_fuse,
            "mod_stats": mod_stats,
        }
        return pet_hat, aux
