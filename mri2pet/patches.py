from typing import Tuple, List
import torch
import torch.nn.functional as F
from .encoders import l2_normalize

# ------------ helpers ------------

def _rand_center_within(mask3d: torch.Tensor, ps: Tuple[int,int,int]) -> Tuple[int,int,int]:
    """
    mask3d: [D,H,W] boolean; ps: (pd,ph,pw)
    Pick a random center inside mask (fallback to volume if mask sparse).
    """
    assert mask3d.dim() == 3, f"mask3d must be 3D [D,H,W], got {tuple(mask3d.shape)}"
    D,H,W = mask3d.shape
    pd,ph,pw = ps

    # If mask very sparse, sample from valid interior region
    if mask3d.sum() < 8:
        cd = torch.randint(pd//2, max(pd//2+1, D-pd//2), (1,)).item()
        ch = torch.randint(ph//2, max(ph//2+1, H-ph//2), (1,)).item()
        cw = torch.randint(pw//2, max(pw//2+1, W-pw//2), (1,)).item()
        return cd, ch, cw

    idx = mask3d.nonzero(as_tuple=False)
    pick = idx[torch.randint(0, idx.shape[0], (1,)).item()]
    cd, ch, cw = int(pick[0]), int(pick[1]), int(pick[2])

    # Clamp to keep full patch inside bounds
    cd = min(max(cd, pd//2), max(pd//2, D - pd//2 - 1))
    ch = min(max(ch, ph//2), max(ph//2, H - ph//2 - 1))
    cw = min(max(cw, pw//2), max(pw//2, W - pw//2 - 1))
    return cd, ch, cw


def _crop_3d(x: torch.Tensor, center: Tuple[int,int,int], ps: Tuple[int,int,int]) -> torch.Tensor:
    """
    x: [B,1,D,H,W] ; returns [B,1,pd,ph,pw] centered crop (pads if near borders).
    """
    _,_,D,H,W = x.shape
    pd,ph,pw = ps
    cd,ch,cw = center

    d0 = cd - pd//2; d1 = d0 + pd
    h0 = ch - ph//2; h1 = h0 + ph
    w0 = cw - pw//2; w1 = w0 + pw

    d0 = max(0, d0); h0 = max(0, h0); w0 = max(0, w0)
    d1 = min(D, d1); h1 = min(H, h1); w1 = min(W, w1)

    patch = x[:, :, d0:d1, h0:h1, w0:w1]
    # pad if hit borders
    pdz = pd - patch.size(2); phz = ph - patch.size(3); pwz = pw - patch.size(4)
    if pdz>0 or phz>0 or pwz>0:
        patch = F.pad(patch, (0,pwz, 0,phz, 0,pdz), value=0.)
    return patch


def _embed(mods, x: torch.Tensor, which: str) -> torch.Tensor:
    """
    x: [N,1,pd,ph,pw]; returns L2-normalized projection [N, d]
    """
    if which == "M":
        f = mods["enc_M"](x); p = mods["proj_M"](f)
    else:
        f = mods["enc_P"](x); p = mods["proj_P"](f)
    return l2_normalize(p)


def _standardize_masks(brain_mask, mri: torch.Tensor) -> torch.Tensor:
    """
    Returns a batched boolean mask of shape [B,D,H,W].
    Accepts: None, [D,H,W], [B,D,H,W], or [H,W] (replicated across D).
    """
    B, _, D, H, W = mri.shape
    dev = mri.device

    if brain_mask is None:
        return (mri[:, 0] != 0)

    if not torch.is_tensor(brain_mask):
        brain_mask = torch.from_numpy(brain_mask)

    bm = brain_mask.to(dev)

    if bm.dim() == 4:
        # [B,D,H,W] or [1,D,H,W]
        if bm.shape[0] == B:
            return bm.bool()
        if bm.shape[0] == 1:
            return bm.bool().expand(B, -1, -1, -1).contiguous()
        # Mismatched batch -> fallback
        return (mri[:, 0] != 0)

    if bm.dim() == 3:
        # [D,H,W] -> tile across batch
        return bm.bool().unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    if bm.dim() == 2:
        # [H,W] -> repeat across depth D, then across batch
        bm3 = bm.bool().unsqueeze(0).repeat(D, 1, 1)
        return bm3.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    # Unknown shape -> fallback
    return (mri[:, 0] != 0)


# ------------ main API ------------

def sample_aligned_patches(
    mods,
    mri: torch.Tensor, pet: torch.Tensor, pet_hat: torch.Tensor,
    brain_mask,
    patch_size: Tuple[int,int,int], K: int
):
    """
    Returns L2-normalized patch embeddings (uM, uP, uPh) each of shape [B*K, d].
    - MRI, PET are encoded with teachers (no grad)
    - PET_hat path keeps grad to update G
    """
    assert mri.shape == pet.shape == pet_hat.shape, "MRI/PET/PET_hat shapes must match"
    B, _, D, H, W = mri.shape
    dev = mri.device

    masks = _standardize_masks(brain_mask, mri)  # [B,D,H,W] bool

    # Collect patches
    pd,ph,pw = patch_size
    m_list: List[torch.Tensor]  = []
    p_list: List[torch.Tensor]  = []
    ph_list: List[torch.Tensor] = []

    for b in range(B):
        mask3d = masks[b]  # [D,H,W]
        for _ in range(K):
            center = _rand_center_within(mask3d, patch_size)
            m_list.append( _crop_3d(mri[b:b+1],     center, patch_size) )
            p_list.append( _crop_3d(pet[b:b+1],     center, patch_size) )
            ph_list.append(_crop_3d(pet_hat[b:b+1], center, patch_size) )

    M_batch  = torch.cat(m_list,  dim=0)  # [B*K,1,pd,ph,pw]
    P_batch  = torch.cat(p_list,  dim=0)
    Ph_batch = torch.cat(ph_list, dim=0)

    # Teachers (MRI, PET) -> no grad
    with torch.no_grad():
        uM = _embed(mods, M_batch,  "M")  # [B*K,d]
        uP = _embed(mods, P_batch,  "P")  # [B*K,d]

    # PET_hat: keep grad so loss updates G via Ph
    uPh = _embed(mods, Ph_batch, "P")     # [B*K,d] (params frozen but grad flows to inputs)

    return uM, uP, uPh

