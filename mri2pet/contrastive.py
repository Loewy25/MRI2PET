from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from .encoders import l2_normalize
from .patches import sample_aligned_patches
from typing import Dict
from .patches import sample_aligned_patches_per_roi

def contrastive_aux_loss_roi(
    mods: Dict[str, torch.nn.Module],
    mri: torch.Tensor, pet: torch.Tensor, pet_hat: torch.Tensor,
    roi_masks: Dict[str, torch.Tensor],
    tau: float,
    patches_per_roi: int,
    patch_size: Tuple[int,int,int],
    roi_weights: Dict[str, float],
    roi_memory: Optional[Dict[str, Dict[str, torch.Tensor]]] = None  # optional
):
    """
    Builds ROI-only contrast:
      - Per ROI r: L_M->Ph^(r) and L_Ph->P^(r) using ONLY in-ROI negatives (K-1 other samples)
      - Aggregate across ROIs with weights alpha_r -> two totals (same shape as old code)
    Returns: (L_m2ph_total, L_ph2p_total, diag)
    """
    # 1) sample & embed per ROI
    per_roi = sample_aligned_patches_per_roi(
        mods, mri, pet, pet_hat, roi_masks,
        patch_size=patch_size, patches_per_roi=patches_per_roi
    )

    # 2) compute per-ROI InfoNCE (two directions) with in-ROI negatives only
    L_m_list = []; L_p_list = []; diag = {}
    for name, (uM, uP, uPh) in per_roi.items():
        # N x N cosine for in-ROI candidates
        sim_M_Ph = cosine_sim(uM,  uPh)  # anchors uM; positives are diagonal
        sim_Ph_P = cosine_sim(uPh, uP)   # anchors uPh; positives are diagonal

        L_MPh_r = info_nce_ce(sim_M_Ph, tau)  # MRI -> P̂
        L_PhP_r = info_nce_ce(sim_Ph_P, tau)  # P̂  -> P

        w = float(roi_weights.get(name, 0.0))
        L_m_list.append(w * L_MPh_r)
        L_p_list.append(w * L_PhP_r)

        diag[f"ROI_{name}_M2Ph"] = float(L_MPh_r.item())
        diag[f"ROI_{name}_Ph2P"] = float(L_PhP_r.item())

    # 3) aggregate to two totals (same outputs as your non-ROI contrast)
    L_m2ph_total = torch.stack(L_m_list).sum() if len(L_m_list) else torch.tensor(0.0, device=mri.device)
    L_ph2p_total = torch.stack(L_p_list).sum() if len(L_p_list) else torch.tensor(0.0, device=mri.device)

    return L_m2ph_total, L_ph2p_total, diag

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [N,d], b: [M,d] -> returns [N,M] cosine-sim matrix (since inputs are L2-normalized).
    Here we pass dot products because we normalize beforehand.
    """
    return a @ b.t()

def info_nce_ce(sim_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Cross-entropy InfoNCE with diagonal positives.
    sim_logits: raw cosine similarities (not divided by tau yet), shape [N,N]
    """
    N = sim_logits.size(0)
    logits = sim_logits / tau
    labels = torch.arange(N, device=logits.device)
    return F.cross_entropy(logits, labels)

@torch.no_grad()
def embed_global(mods: Dict[str, torch.nn.Module], mri: torch.Tensor, pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns L2-normalized global embeddings for MRI and PET (no grad)
    zM = l2_normalize(mods["proj_M"](mods["enc_M"](mri)))  # [B,d]
    zP = l2_normalize(mods["proj_P"](mods["enc_P"](pet)))  # [B,d]
    return zM, zP

def embed_global_hat(mods: Dict[str, torch.nn.Module], pet_hat: torch.Tensor) -> torch.Tensor:
    # Keep grad into pet_hat path (teacher params are frozen)
    zPh = l2_normalize(mods["proj_P"](mods["enc_P"](pet_hat)))  # [B,d]
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
    Returns: (L_m2ph_total, L_ph2p_total, diag)
      - L_m2ph_total = global(M→P̂) + patch(M→P̂)
      - L_ph2p_total = global(P̂→P) + patch(P̂→P)
    Both totals are scalars. 'diag' contains individual terms for logging.
    """
    # ----- Global terms (B >= 1) -----
    with torch.no_grad():
        zM, zP = embed_global(mods, mri, pet)  # [B,d]
    zPh = embed_global_hat(mods, pet_hat)      # [B,d]

    # cosine_sim outputs [B,B]; CE with diag positives
    L_MPh = info_nce_ce(cosine_sim(zM,  zPh), tau)  # MRI anchors, PET_hat candidates
    L_PhP = info_nce_ce(cosine_sim(zPh, zP),  tau)  # PET_hat anchors, PET candidates

    L_m2ph_total = L_MPh
    L_ph2p_total = L_PhP
    diag = {"L_MPh_global": float(L_MPh.item()), "L_PhP_global": float(L_PhP.item())}

    # ----- Patch-level terms (B*K >= 1) -----
    if use_patches and patches_per_subj > 0:
        uM, uP, uPh = sample_aligned_patches(
            mods, mri, pet, pet_hat, brain_mask, patch_size, patches_per_subj
        )  # each [B*K, d]

        L_patch_MPh = info_nce_ce(cosine_sim(uM,  uPh), tau)  # M anchors vs P̂
        L_patch_PhP = info_nce_ce(cosine_sim(uPh, uP),  tau)  # P̂ anchors vs P

        L_m2ph_total = L_m2ph_total + L_patch_MPh
        L_ph2p_total = L_ph2p_total + L_patch_PhP

        diag.update({
            "L_MPh_patch": float(L_patch_MPh.item()),
            "L_PhP_patch": float(L_patch_PhP.item()),
        })

    return L_m2ph_total, L_ph2p_total, diag

