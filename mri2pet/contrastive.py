# mri2pet/contrastive.py
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from .encoders import l2_normalize
from .patches import sample_aligned_patches

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,d], b: [N,d] or [N2,d] -> returns [N,N2]
    return a @ b.t()

def info_nce_ce(sim_logits: torch.Tensor, tau: float) -> torch.Tensor:
    # sim_logits: raw cosine sims; divide by tau then CE to diag labels
    N = sim_logits.size(0)
    logits = sim_logits / tau
    labels = torch.arange(N, device=logits.device)
    return F.cross_entropy(logits, labels)

@torch.no_grad()
def embed_global(mods: Dict[str, torch.nn.Module], mri: torch.Tensor, pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns L2-normalized global embeddings (no grad)
    zM = l2_normalize(mods["proj_M"](mods["enc_M"](mri)))
    zP = l2_normalize(mods["proj_P"](mods["enc_P"](pet)))
    return zM, zP

def embed_global_hat(mods: Dict[str, torch.nn.Module], pet_hat: torch.Tensor) -> torch.Tensor:
    # Grad flows into pet_hat path when computing loss in GAN stage
    zPh = l2_normalize(mods["proj_P"](mods["enc_P"](pet_hat)))
    return zPh

def contrastive_aux_loss(
    mods: Dict[str, torch.nn.Module],
    mri: torch.Tensor, pet: torch.Tensor, pet_hat: torch.Tensor,
    brain_mask: Optional[torch.Tensor],
    tau: float,
    use_patches: bool,
    patch_size: Tuple[int,int,int],
    patches_per_subj: int
):
    """
    Returns: L_contrast (scalar), diagnostics dict
    """
    # Global terms (MRI<->PET_hat, PET_hat<->PET)
    with torch.no_grad():
        zM, zP = embed_global(mods, mri, pet)
    zPh = embed_global_hat(mods, pet_hat)

    L_MPh = info_nce_ce(cosine_sim(zM,  zPh), tau)
    L_PhP = info_nce_ce(cosine_sim(zPh, zP),  tau)
    L_total = L_MPh + L_PhP
    diag = {"L_MPh": L_MPh.item(), "L_PhP": L_PhP.item()}

    # Optional patch-level (GAN stage only)
    if use_patches:
        uM, uP, uPh = sample_aligned_patches(
            mods, mri, pet, pet_hat, brain_mask, patch_size, patches_per_subj
        )
        L_patch_MPh = info_nce_ce(cosine_sim(uM,  uPh), tau)
        L_patch_PhP = info_nce_ce(cosine_sim(uPh, uP),  tau)
        L_total = L_total + (L_patch_MPh + L_patch_PhP)
        diag.update({"L_patch_MPh": L_patch_MPh.item(), "L_patch_PhP": L_patch_PhP.item()})

    return L_total, diag
