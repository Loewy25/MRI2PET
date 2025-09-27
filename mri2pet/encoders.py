# mri2pet/encoders.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.video import r3d_18
    _HAS_TV = True
except Exception:
    _HAS_TV = False

# ---- Projection head: Linear -> ReLU -> Linear -> L2 norm at use time ----
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---- Simple global pooling for 3D feature maps ----
def global_avgpool_3d(x: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool3d(x, output_size=1).flatten(1)

# ---- 3D encoder backbone builder ----
class ResNet3DEncoder(nn.Module):
    """
    Wrap torchvision r3d_18 if available; otherwise a minimal 3D CNN fallback.
    Outputs a feature vector (pre-projection) with dim 'feat_dim'.
    """
    def __init__(self, in_ch: int = 1, feat_dim: int = 512):
        super().__init__()
        if _HAS_TV:
            model = r3d_18(weights=None)        # we will pretrain via InfoNCE
            # Replace first conv to accept 1 channel
            model.stem[0] = nn.Conv3d(
                in_ch, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False
            )
            # Keep the full model for stage-aware freezing,
            # but build a feature trunk (no avgpool/fc) for forward()
            self.backbone = model
            self.features = nn.Sequential(
                model.stem, model.layer1, model.layer2, model.layer3, model.layer4
            )
            self.feat_dim = 512                  # r3d_18 final planes
            self.pool = global_avgpool_3d
        else:
            # Fallback: small 3D CNN
            layers = []
            chs = [in_ch, 32, 64, 128, 256, 512]
            for i in range(len(chs)-1):
                layers += [
                    nn.Conv3d(chs[i], chs[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(chs[i+1]),
                    nn.ReLU(inplace=True),
                ]
            self.backbone = nn.Sequential(*layers)
            self.features = self.backbone
            self.feat_dim = 512
            self.pool = global_avgpool_3d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)   # 5D feature map
        g = self.pool(f)       # [B, feat_dim]
        return g

# ---- utils: freezing / partial fine-tuning ----
def set_finetune_pct(module: nn.Module, pct: float):
    """
    Freeze lower layers and unfreeze the top pct of the encoder.
    For torchvision r3d_18, we use stage-level granularity: [stem, layer1, layer2, layer3, layer4].
    Fallback: last-k children.
    """
    pct = float(max(0.0, min(1.0, pct)))

    # Freeze everything first
    for p in module.parameters():
        p.requires_grad = False

    # Special path for ResNet3DEncoder with r3d_18 backbone
    bb = getattr(module, "backbone", None)
    if bb is not None and hasattr(bb, "layer4"):
        stages = []
        if hasattr(bb, "stem"):
            stages.append(bb.stem)
        if hasattr(bb, "layer1"): stages.append(bb.layer1)
        if hasattr(bb, "layer2"): stages.append(bb.layer2)
        if hasattr(bb, "layer3"): stages.append(bb.layer3)
        if hasattr(bb, "layer4"): stages.append(bb.layer4)

        L = len(stages)
        if L == 0:
            if pct > 0.0:
                for p in module.parameters(): p.requires_grad = True
            return

        k = max(1, int(round(pct * L))) if pct > 0.0 else 0
        for s in stages[-k:]:
            for p in s.parameters():
                p.requires_grad = True
        return

    # Generic fallback (non-r3d_18)
    children = list(module.children())
    if len(children) == 0:
        if pct > 0.0:
            for p in module.parameters(): p.requires_grad = True
    else:
        k = max(1, int(round(pct * len(children)))) if pct > 0.0 else 0
        for child in children[-k:]:
            for p in child.parameters():
                p.requires_grad = True

def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

# ---- Factory returning MRI and PET encoders + projections ----
def build_encoders_and_heads(in_ch_mri: int, in_ch_pet: int, proj_dim: int) -> Dict[str, nn.Module]:
    enc_M = ResNet3DEncoder(in_ch=in_ch_mri)
    enc_P = ResNet3DEncoder(in_ch=in_ch_pet)
    proj_M = ProjectionHead(enc_M.feat_dim, proj_dim)
    proj_P = ProjectionHead(enc_P.feat_dim, proj_dim)
    return {"enc_M": enc_M, "enc_P": enc_P, "proj_M": proj_M, "proj_P": proj_P}

def save_teachers(mods: Dict[str, nn.Module], path: str):
    torch.save({k: v.state_dict() for k, v in mods.items()}, path)

def load_teachers(mods: Dict[str, nn.Module], path: str, map_location=None):
    sd = torch.load(path, map_location=map_location)
    for k, v in mods.items():
        v.load_state_dict(sd[k])

def freeze_teachers(mods: Dict[str, nn.Module]):
    for m in mods.values():
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
