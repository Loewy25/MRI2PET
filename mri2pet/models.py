# models.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import DETACH_BASE_LATENT_FOR_AUX, RESIDUAL_ALPHA_INIT
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
# NEW: Prompt-Residual-Braak modules
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


# --- B. ClinicalFiLMConditioner ---
class ClinicalFiLMConditioner(nn.Module):
    def __init__(self, clinical_dim: int = 10, z_dim: int = 128):
        super().__init__()
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


# --- C. OrdinalStageHead ---
class OrdinalStageHead(nn.Module):
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.stage_logits = nn.Linear(in_dim, 3)
        self.braak_head = nn.Linear(in_dim, 3)

    def forward(self, z_fuse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stage_logits(z_fuse), self.braak_head(z_fuse)


# --- D. StagePromptBank ---
class StagePromptBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.bank_b = nn.Parameter(torch.randn(4, 128) * 0.01)
        self.bank_d1 = nn.Parameter(torch.randn(4, 128) * 0.01)
        self.bank_d2 = nn.Parameter(torch.randn(4, 64) * 0.01)
        self.bank_d3 = nn.Parameter(torch.randn(4, 32) * 0.01)
        self.bank_d4 = nn.Parameter(torch.randn(4, 16) * 0.01)

    def forward(self, stage_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "b": stage_weights @ self.bank_b,
            "d1": stage_weights @ self.bank_d1,
            "d2": stage_weights @ self.bank_d2,
            "d3": stage_weights @ self.bank_d3,
            "d4": stage_weights @ self.bank_d4,
        }


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
        flair_prompt: torch.Tensor,
        film: Tuple[torch.Tensor, torch.Tensor],
        stage_prompt: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = film
        bp = self.proj_base(base_feat)
        fp = self.proj_prompt(flair_prompt)
        if fp.shape[2:] != bp.shape[2:]:
            fp = _pad_or_crop_to(fp, bp)
        gate = torch.sigmoid(self.gate_conv(torch.cat([bp, fp], dim=1)))
        fused = bp + gate * fp
        fused = fused * (1.0 + gamma[..., None, None, None]) + beta[..., None, None, None]
        fused = fused + stage_prompt[..., None, None, None]
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
        stage_prompts: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        def _up(t, ref):
            return _pad_or_crop_to(self.up(t.float()).to(t.dtype), ref)

        rb = self.fuse_b(feats["b"], pf["pb"], film["b"], stage_prompts["b"])
        s4 = self.fuse_s4(feats["x4"], pf["p4"], film["d1"], stage_prompts["d1"])
        d1 = self.conv_d1(torch.cat([_up(rb, s4), s4], dim=1))

        s3 = self.fuse_s3(feats["x3"], pf["p3"], film["d2"], stage_prompts["d2"])
        d2 = self.conv_d2(torch.cat([_up(d1, s3), s3], dim=1))

        s2 = self.fuse_s2(feats["x2"], pf["p2"], film["d3"], stage_prompts["d3"])
        d3 = self.conv_d3(torch.cat([_up(d2, s2), s2], dim=1))

        s1 = self.fuse_s1(feats["x1"], pf["p1"], film["d4"], stage_prompts["d4"])
        d4 = self.conv_d4(torch.cat([_up(d3, s1), s1], dim=1))

        return self.delta_out(d4)


# =========================================================================
# Utility: ordinal logits → stage probabilities
# =========================================================================
def ordinal_logits_to_stage_probs(logits: torch.Tensor) -> torch.Tensor:
    p_ge1 = torch.sigmoid(logits[:, 0])
    p_ge2 = torch.sigmoid(logits[:, 1])
    p_ge3 = torch.sigmoid(logits[:, 2])

    # enforce ordinal monotonicity: P(>=k+1) <= P(>=k)
    p_ge2 = torch.minimum(p_ge2, p_ge1)
    p_ge3 = torch.minimum(p_ge3, p_ge2)

    w0 = 1.0 - p_ge1
    w1 = p_ge1 - p_ge2
    w2 = p_ge2 - p_ge3
    w3 = p_ge3

    w = torch.stack([w0, w1, w2, w3], dim=1)
    w = torch.clamp(w, min=0.0)
    return w / (w.sum(dim=1, keepdim=True) + 1e-8)


def build_stage_onehot(stage_ord: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
    return F.one_hot(stage_ord.long(), num_classes).float()


# =========================================================================
# G. PromptResidualBraakGenerator — top-level wrapper
# =========================================================================
class PromptResidualBraakGenerator(nn.Module):
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
        self.stage_head = OrdinalStageHead(in_dim=prompt_z_dim)
        self.prompt_bank = StagePromptBank()
        self.residual_decoder = ResidualDecoder3D()

        self.alpha_logit = nn.Parameter(torch.tensor(float(RESIDUAL_ALPHA_INIT)))

    def forward(
        self,
        t1: torch.Tensor,
        flair: torch.Tensor,
        clinical: torch.Tensor,
        stage_prompt_weights: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        # 1. Base T1-only generator
        pet_base, feats = self.base(t1, return_features=True)

        # 2. FLAIR prompt pyramid
        pf = self.flair_encoder(flair)

        # 3. Fuse latents for auxiliary heads
        z_t1 = self.gap(feats["b"]).flatten(1)  # [B, 512]
        z_t1_aux = z_t1.detach() if DETACH_BASE_LATENT_FOR_AUX else z_t1

        z_clin, film = self.clinical_conditioner(clinical)

        z_fuse = self.fusion_mlp(torch.cat([z_t1_aux, pf["z_flair"], z_clin], dim=1))

        # 4. Auxiliary heads
        stage_logits, braak_pred = self.stage_head(z_fuse)
        stage_probs = ordinal_logits_to_stage_probs(stage_logits)

        # 5. Stage prompt weights
        weights = stage_prompt_weights if stage_prompt_weights is not None else stage_probs
        stage_prompts = self.prompt_bank(weights)

        # 6. Residual decoder
        delta_pet = self.residual_decoder(feats, pf, film, stage_prompts)

        # 7. Combine
        alpha = torch.sigmoid(self.alpha_logit)
        pet_hat = pet_base + alpha * delta_pet

        if not return_aux:
            return pet_hat

        aux: Dict[str, Any] = {
            "pet_base": pet_base,
            "delta_pet": delta_pet,
            "alpha": alpha,
            "stage_logits": stage_logits,
            "stage_probs": stage_probs,
            "braak_pred": braak_pred,
            "z_fuse": z_fuse,
        }
        return pet_hat, aux
