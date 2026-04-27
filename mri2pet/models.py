# models.py
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import (
    DETACH_BASE_LATENT_FOR_PRIOR,
    DIFF_EMB_DIM,
    DIFF_UNET_BASE_CH,
    PRIOR_GAIN_INIT_B,
    PRIOR_GAIN_INIT_X3,
    PRIOR_GAIN_INIT_X4,
    SPATIAL_PRIOR_K,
    USE_CHECKPOINT,
    USE_BRAAK_HEAD,
    USE_CLINICAL,
    USE_FLAIR,
    USE_SPATIAL_PRIOR,
)
from .residual_manifold import load_basis_arrays
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

    def _mix_basis(
        self,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("bk,kcdhw->bcdhw", weights, self.basis_low.to(dtype=weights.dtype))

    def _build_prior(
        self,
        prior_low: torch.Tensor,
        proj: nn.Conv3d,
        gain: nn.Parameter,
        feat: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = feat.size(0)
        size = tuple(feat.shape[2:])
        device = feat.device
        dtype = feat.dtype
        brain_s = self._resize_mask(brain_mask, size, B, device, dtype, fill=1.0)
        cortex_s = self._resize_mask(cortex_mask, size, B, device, dtype, fill=0.0)
        gate_s = (0.2 * brain_s) + (0.8 * cortex_s)
        prior_s = prior_low
        if prior_s.shape[2:] != size:
            prior_s = F.interpolate(prior_s.float(), size=size, mode="trilinear", align_corners=False).to(dtype=dtype)
        prior_s = proj(prior_s * gate_s)
        prior_s = gain.to(device=device, dtype=prior_s.dtype) * prior_s
        outside_s = torch.clamp(brain_s - cortex_s, min=0.0)
        return prior_s, cortex_s, outside_s

    def forward(
        self,
        z_fuse: torch.Tensor,
        brain_mask: Optional[torch.Tensor],
        cortex_mask: Optional[torch.Tensor],
        feats: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights = torch.softmax(self.router(z_fuse.float()), dim=1).to(dtype=feats["b"].dtype)
        prior_low = self._mix_basis(weights)

        prior_b, cortex_b, outside_b = self._build_prior(
            prior_low, self.proj_b, self.gain_b, feats["b"], brain_mask, cortex_mask
        )
        prior_x4, cortex_x4, outside_x4 = self._build_prior(
            prior_low, self.proj_x4, self.gain_x4, feats["x4"], brain_mask, cortex_mask
        )
        prior_x3, cortex_x3, outside_x3 = self._build_prior(
            prior_low, self.proj_x3, self.gain_x3, feats["x3"], brain_mask, cortex_mask
        )

        def _masked_mag(prior: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = (mask.sum() * prior.size(1)).clamp(min=1.0)
            return (prior.abs() * mask).sum() / denom

        in_mag = torch.stack([
            _masked_mag(prior_b, cortex_b),
            _masked_mag(prior_x4, cortex_x4),
            _masked_mag(prior_x3, cortex_x3),
        ]).mean()
        out_mag = torch.stack([
            _masked_mag(prior_b, outside_b),
            _masked_mag(prior_x4, outside_x4),
            _masked_mag(prior_x3, outside_x3),
        ]).mean()
        router_entropy = -(weights.float() * torch.log(weights.float() + 1e-8)).sum(dim=1).mean()
        router_top1_mean = weights.float().max(dim=1).values.mean()

        prior_maps = {"b": prior_b, "x4": prior_x4, "x3": prior_x3}
        prior_stats = {
            "in_cortex_mag": in_mag.detach().float(),
            "out_cortex_mag": out_mag.detach().float(),
            "router_entropy": router_entropy.detach().float(),
            "router_top1_mean": router_top1_mean.detach().float(),
        }
        return prior_maps, prior_stats


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


# =========================================================================
# Residual diffusion denoiser
# =========================================================================
def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return t.float().unsqueeze(1)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =========================================================================
# Residual manifold coefficient predictor
# =========================================================================
class CompactEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 1, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=True),
            _group_norm(16),
            nn.SiLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
            _group_norm(32),
            nn.SiLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            _group_norm(64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            _group_norm(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualManifoldNet(nn.Module):
    def __init__(
        self,
        basis_dir: str,
        clinical_dim: int = 10,
        stat_dim: int = 16,
        t1_freeze: bool = True,
        use_checkpoint: bool = USE_CHECKPOINT,
        z_dim: int = 128,
    ):
        super().__init__()
        B_cal, B_dis = load_basis_arrays(basis_dir)
        if B_cal.shape[1:] != B_dis.shape[1:]:
            raise ValueError(f"B_cal shape {B_cal.shape} and B_dis shape {B_dis.shape} are incompatible")
        if int(stat_dim) != 16:
            raise ValueError("ResidualManifoldNet currently uses exactly 16 PET_base/FLAIR statistics")

        self.k_cal = int(B_cal.shape[0])
        self.k_dis = int(B_dis.shape[0])
        self.t1_freeze = bool(t1_freeze)
        self.stat_dim = 16

        self.register_buffer("B_cal", torch.from_numpy(B_cal[:, None].astype(np.float32)))
        self.register_buffer("B_dis", torch.from_numpy(B_dis[:, None].astype(np.float32)))

        self.t1_backbone = Generator(in_ch=1, out_ch=1, use_checkpoint=use_checkpoint)
        self.t1_proj = nn.Sequential(
            nn.Linear(512, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=True),
        )
        if self.t1_freeze:
            self.t1_backbone.use_checkpoint = False
            for p in self.t1_backbone.parameters():
                p.requires_grad = False

        self.flair_encoder = CompactEncoder3D(in_ch=1, z_dim=z_dim)
        self.petbase_encoder = CompactEncoder3D(in_ch=1, z_dim=z_dim)
        self.clinical_encoder = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(64, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
        )
        self.stat_encoder = nn.Sequential(
            nn.LayerNorm(self.stat_dim),
            nn.Linear(self.stat_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(z_dim * 5, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
            nn.SiLU(inplace=True),
        )
        self.c_head = nn.Linear(z_dim, self.k_cal)
        self.a_head = nn.Linear(z_dim, self.k_dis)

    @staticmethod
    def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vals = x[mask]
        if vals.numel() == 0:
            z = torch.zeros((), device=x.device, dtype=x.dtype)
            return z, z
        mean = vals.mean()
        std = vals.std(unbiased=False) if vals.numel() > 1 else torch.zeros((), device=x.device, dtype=x.dtype)
        return mean, std

    @staticmethod
    def _masked_top_mean(x: torch.Tensor, mask: torch.Tensor, q: float = 0.85) -> torch.Tensor:
        vals = x[mask]
        if vals.numel() == 0:
            return torch.zeros((), device=x.device, dtype=x.dtype)
        if vals.numel() == 1:
            return vals.mean()
        thr = torch.quantile(vals.float(), float(q)).to(dtype=x.dtype)
        top = vals[vals >= thr]
        return top.mean() if top.numel() > 0 else vals.mean()

    def _stats_for_volume(self, x: torch.Tensor, brain_mask: torch.Tensor, cortex_mask: torch.Tensor) -> torch.Tensor:
        rows = []
        for b in range(x.size(0)):
            xb = x[b, 0]
            brain = brain_mask[b, 0] > 0.5
            cortex = (cortex_mask[b, 0] > 0.5) & brain
            nonctx = brain & (~cortex)
            brain_mean, brain_std = self._masked_mean_std(xb, brain)
            ctx_mean, ctx_std = self._masked_mean_std(xb, cortex)
            non_mean, non_std = self._masked_mean_std(xb, nonctx)
            ctx_minus_non = ctx_mean - non_mean
            top_ctx_mean = self._masked_top_mean(xb, cortex)
            rows.append(torch.stack([
                brain_mean, brain_std, ctx_mean, ctx_std,
                non_mean, non_std, ctx_minus_non, top_ctx_mean,
            ]))
        return torch.stack(rows, dim=0)

    def _stat_features(self, pet_base: torch.Tensor, flair: torch.Tensor,
                       brain_mask: torch.Tensor, cortex_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self._stats_for_volume(pet_base.float(), brain_mask.float(), cortex_mask.float()),
            self._stats_for_volume(flair.float(), brain_mask.float(), cortex_mask.float()),
        ], dim=1)

    def _t1_latent(self, t1: torch.Tensor) -> torch.Tensor:
        if self.t1_freeze:
            with torch.no_grad():
                _, feats = self.t1_backbone(t1, return_features=True)
        else:
            _, feats = self.t1_backbone(t1, return_features=True)
        z = F.adaptive_avg_pool3d(feats["b"].float(), 1).flatten(1)
        return self.t1_proj(z)

    def forward(
        self,
        t1: torch.Tensor,
        flair: torch.Tensor,
        pet_base: torch.Tensor,
        clinical: torch.Tensor,
        brain_mask: torch.Tensor,
        cortex_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        z_t1 = self._t1_latent(t1)
        z_flair = self.flair_encoder(flair)
        z_petbase = self.petbase_encoder(pet_base)
        z_clinical = self.clinical_encoder(clinical.float())
        stats = self._stat_features(pet_base, flair, brain_mask, cortex_mask).to(
            device=clinical.device,
            dtype=clinical.dtype,
        )
        z_stats = self.stat_encoder(stats.float())
        z = self.fusion(torch.cat([z_t1, z_flair, z_petbase, z_clinical, z_stats], dim=1))

        c_hat = self.c_head(z)
        a_hat = F.softplus(self.a_head(z))
        res_cal = torch.einsum("bk,kcdhw->bcdhw", c_hat.float(), self.B_cal.float()).to(dtype=pet_base.dtype)
        res_dis = torch.einsum("bk,kcdhw->bcdhw", a_hat.float(), self.B_dis.float()).to(dtype=pet_base.dtype)
        brain = brain_mask.to(device=pet_base.device, dtype=pet_base.dtype)
        pet_fake = (pet_base + res_cal + res_dis) * brain

        if not return_aux:
            return pet_fake
        aux: Dict[str, Any] = {
            "pet_base": pet_base,
            "res_cal": res_cal * brain,
            "res_dis": res_dis * brain,
            "res_total": (res_cal + res_dis) * brain,
            "c_hat": c_hat,
            "a_hat": a_hat,
            "stats": stats,
            "z_fuse": z,
        }
        return pet_fake, aux


class FiLMResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.norm2 = _group_norm(out_ch)
        self.film = nn.Linear(emb_dim, out_ch * 2)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.film(emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + gamma[..., None, None, None]) + beta[..., None, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x.float(), scale_factor=2, mode="trilinear", align_corners=False).to(dtype=ref.dtype)
        x = _pad_or_crop_to(x, ref)
        return self.conv(x)


class BottleneckAttention3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.k = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.v = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, D * H * W).transpose(1, 2)
        k = self.k(h).reshape(B, C, D * H * W)
        v = self.v(h).reshape(B, C, D * H * W).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q.float(), k.float()) / math.sqrt(float(C)), dim=-1).to(dtype=x.dtype)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(B, C, D, H, W)
        return x + self.proj(out)


class ResidualDiffusionUNet3D(nn.Module):
    def __init__(
        self,
        in_ch: int = 6,
        base_ch: int = DIFF_UNET_BASE_CH,
        emb_dim: int = DIFF_EMB_DIM,
        clinical_dim: int = 10,
        use_checkpoint: bool = USE_CHECKPOINT,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_checkpoint = use_checkpoint
        ch0 = int(base_ch)
        ch1 = ch0 * 2
        ch2 = ch0 * 4
        ch3 = ch0 * 8
        ch4 = ch0 * 16

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.clinical_mlp = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.in_proj = nn.Conv3d(in_ch, ch0, kernel_size=3, padding=1, bias=True)
        self.enc0 = FiLMResBlock3D(ch0, ch0, emb_dim)
        self.down1 = Downsample3D(ch0, ch1)
        self.enc1 = FiLMResBlock3D(ch1, ch1, emb_dim)
        self.down2 = Downsample3D(ch1, ch2)
        self.enc2 = FiLMResBlock3D(ch2, ch2, emb_dim)
        self.down3 = Downsample3D(ch2, ch3)
        self.enc3 = FiLMResBlock3D(ch3, ch3, emb_dim)
        self.down4 = Downsample3D(ch3, ch4)
        self.mid1 = FiLMResBlock3D(ch4, ch4, emb_dim)
        self.mid_attn = BottleneckAttention3D(ch4)
        self.mid2 = FiLMResBlock3D(ch4, ch4, emb_dim)

        self.up3 = Upsample3D(ch4, ch3)
        self.dec3 = FiLMResBlock3D(ch3 + ch3, ch3, emb_dim)
        self.up2 = Upsample3D(ch3, ch2)
        self.dec2 = FiLMResBlock3D(ch2 + ch2, ch2, emb_dim)
        self.up1 = Upsample3D(ch2, ch1)
        self.dec1 = FiLMResBlock3D(ch1 + ch1, ch1, emb_dim)
        self.up0 = Upsample3D(ch1, ch0)
        self.dec0 = FiLMResBlock3D(ch0 + ch0, ch0, emb_dim)

        self.out_norm = _group_norm(ch0)
        self.out_conv = nn.Conv3d(ch0, 1, kernel_size=3, padding=1, bias=True)
        self.braak_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch4, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 3),
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _run_block(self, block: nn.Module, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return grad_checkpoint(lambda a, b: block(a, b), x, emb, use_reentrant=False)
        return block(x, emb)

    def _condition(self, t: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        z_t = self.time_mlp(_sinusoidal_timestep_embedding(t, self.emb_dim).to(device=clinical.device))
        z_c = self.clinical_mlp(clinical.float()).to(dtype=z_t.dtype)
        return z_t + z_c

    def forward(
        self,
        noisy_residual: torch.Tensor,
        t: torch.Tensor,
        t1: torch.Tensor,
        flair: torch.Tensor,
        pet_base: torch.Tensor,
        brain_mask: torch.Tensor,
        cortex_mask: torch.Tensor,
        clinical: torch.Tensor,
        return_aux: bool = False,
    ):
        emb = self._condition(t, clinical)
        x = torch.cat([noisy_residual, t1, flair, pet_base, brain_mask, cortex_mask], dim=1)

        x0 = self.in_proj(x)
        e0 = self._run_block(self.enc0, x0, emb)
        e1 = self._run_block(self.enc1, self.down1(e0), emb)
        e2 = self._run_block(self.enc2, self.down2(e1), emb)
        e3 = self._run_block(self.enc3, self.down3(e2), emb)

        mid = self.down4(e3)
        mid = self._run_block(self.mid1, mid, emb)
        mid = self.mid_attn(mid)
        mid = self._run_block(self.mid2, mid, emb)

        d3 = self.up3(mid, e3)
        d3 = self._run_block(self.dec3, torch.cat([d3, e3], dim=1), emb)
        d2 = self.up2(d3, e2)
        d2 = self._run_block(self.dec2, torch.cat([d2, e2], dim=1), emb)
        d1 = self.up1(d2, e1)
        d1 = self._run_block(self.dec1, torch.cat([d1, e1], dim=1), emb)
        d0 = self.up0(d1, e0)
        d0 = self._run_block(self.dec0, torch.cat([d0, e0], dim=1), emb)

        eps = self.out_conv(F.silu(self.out_norm(d0)))
        if not return_aux:
            return eps
        aux = {"braak_pred": self.braak_head(mid)}
        return eps, aux
