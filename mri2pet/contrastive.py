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

# --- BEGIN PATCH: mri2pet/contrastive.py::contrastive_aux_loss ---
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
    Returns:
      L_m2ph_total, L_ph2p_total, diag
    where each total = global + (patch if enabled).
    """
    # Global terms (MRI<->PET_hat, PET_hat<->PET)
    with torch.no_grad():
        zM, zP = embed_global(mods, mri, pet)      # [B,d], no grad
    zPh = embed_global_hat(mods, pet_hat)          # [B,d], grad wrt pet_hat path

    L_MPh_glob = info_nce_ce(cosine_sim(zM,  zPh), tau)   # MRI → PET̂
    L_PhP_glob = info_nce_ce(cosine_sim(zPh, zP),  tau)   # PET̂ → PET

    diag = {"L_MPh_global": L_MPh_glob.item(), "L_PhP_global": L_PhP_glob.item()}

    L_MPh_patch = torch.tensor(0.0, device=pet_hat.device)
    L_PhP_patch = torch.tensor(0.0, device=pet_hat.device)

    if use_patches:
        uM, uP, uPh = sample_aligned_patches(
            mods, mri, pet, pet_hat, brain_mask, patch_size, patches_per_subj
        )
        L_MPh_patch = info_nce_ce(cosine_sim(uM,  uPh), tau)
        L_PhP_patch = info_nce_ce(cosine_sim(uPh, uP),  tau)
        diag.update({
            "L_MPh_patch": L_MPh_patch.item(),
            "L_PhP_patch": L_PhP_patch.item()
        })

    L_m2ph_total = L_MPh_glob + L_MPh_patch
    L_ph2p_total = L_PhP_glob + L_PhP_patch
    return L_m2ph_total, L_ph2p_total, diag
# --- END PATCH ---
