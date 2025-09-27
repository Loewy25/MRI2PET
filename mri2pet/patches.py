# mri2pet/patches.py
from typing import Tuple
import torch
import torch.nn.functional as F
from .encoders import l2_normalize

def _rand_center_within(mask3d: torch.Tensor, ps: Tuple[int,int,int]) -> Tuple[int,int,int]:
    # mask3d: [D,H,W] boolean; ps: (pd,ph,pw)
    D,H,W = mask3d.shape
    pd,ph,pw = ps
    # fall back to whole volume if mask is empty
    if mask3d.sum() < 8:
        # center in valid range
        cd = torch.randint(pd//2, max(pd//2+1, D-pd//2), (1,)).item()
        ch = torch.randint(ph//2, max(ph//2+1, H-ph//2), (1,)).item()
        cw = torch.randint(pw//2, max(pw//2+1, W-pw//2), (1,)).item()
        return cd, ch, cw
    idx = mask3d.nonzero()
    pick = idx[torch.randint(0, idx.shape[0], (1,)).item()]
    cd, ch, cw = int(pick[0]), int(pick[1]), int(pick[2])
    cd = min(max(cd, pd//2), max(pd//2, D - pd//2 - 1))
    ch = min(max(ch, ph//2), max(ph//2, H - ph//2 - 1))
    cw = min(max(cw, pw//2), max(pw//2, W - pw//2 - 1))
    return cd, ch, cw

def _crop_3d(x: torch.Tensor, center: Tuple[int,int,int], ps: Tuple[int,int,int]) -> torch.Tensor:
    # x: [B,1,D,H,W] ; returns [B,1,pd,ph,pw]
    _,_,D,H,W = x.shape
    pd,ph,pw = ps
    cd,ch,cw = center
    d0 = cd - pd//2; d1 = d0 + pd
    h0 = ch - ph//2; h1 = h0 + ph
    w0 = cw - pw//2; w1 = w0 + pw
    d0 = max(0, d0); h0 = max(0, h0); w0 = max(0, w0)
    d1 = min(D, d1); h1 = min(H, h1); w1 = min(W, w1)
    patch = x[:, :, d0:d1, h0:h1, w0:w1]
    # pad if we hit borders
    pdz = pd - patch.size(2); phz = ph - patch.size(3); pwz = pw - patch.size(4)
    if pdz>0 or phz>0 or pwz>0:
        patch = F.pad(patch, (0,pwz, 0,phz, 0,pdz), value=0.)
    return patch


def _embed(mods, x: torch.Tensor, which: str) -> torch.Tensor:
    if which == "M":
        f = mods["enc_M"](x); p = mods["proj_M"](f)
    else:
        f = mods["enc_P"](x); p = mods["proj_P"](f)
    return l2_normalize(p)

def sample_aligned_patches(
    mods,
    mri: torch.Tensor, pet: torch.Tensor, pet_hat: torch.Tensor,
    brain_mask, patch_size: Tuple[int,int,int], K: int
):
    """
    Returns L2-normalized patch embeddings (uM, uP, uPh) each [K, d].
    """
    B, _, D, H, W = mri.shape
    assert B == 1, "B=1 expected in your config; extend here if needed."
    mask3d = brain_mask if isinstance(brain_mask, torch.Tensor) else None
    if mask3d is None:
        mask3d = (mri[0,0] != 0).to(mri.device)

    uM_list, uP_list, uPh_list = [], [], []
    for _ in range(K):
        center = _rand_center_within(mask3d, patch_size)
        m = _crop_3d(mri,     center, patch_size)
        p = _crop_3d(pet,     center, patch_size)
        ph= _crop_3d(pet_hat, center, patch_size)
        # teachers (MRI, PET): no grad
        with torch.no_grad():
            uM_list.append(_embed(mods, m, "M"))
            uP_list.append(_embed(mods, p, "P"))
       # PET-hat: KEEP GRAD so loss updates G via ph
        uPh_list.append(_embed(mods, ph, "P"))

    uM  = torch.cat(uM_list,  dim=0)   # [K,d]
    uP  = torch.cat(uP_list,  dim=0)   # [K,d]
    uPh = torch.cat(uPh_list, dim=0)   # [K,d]
    return uM, uP, uPh
