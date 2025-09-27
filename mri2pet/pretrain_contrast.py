# mri2pet/pretrain_contrast.py
from typing import Dict
import torch
import torch.nn.functional as F
from .contrastive import embed_global, cosine_sim, info_nce_ce
from .encoders import build_encoders_and_heads, set_finetune_pct, save_teachers

def pretrain_encoders(
    train_loader, val_loader, device,
    proj_dim: int, tau: float, finetune_pct: float,
    lr: float, epochs: int, ckpt_path: str
) -> Dict[str, torch.nn.Module]:

    mods = build_encoders_and_heads(in_ch_mri=1, in_ch_pet=1, proj_dim=proj_dim)
    for m in mods.values(): m.to(device)

    # Fine-tune only the top pct initially (helps when starting from random too)
    set_finetune_pct(mods["enc_M"], finetune_pct)
    set_finetune_pct(mods["enc_P"], finetune_pct)
    # Always train projection heads
    for p in mods["proj_M"].parameters(): p.requires_grad = True
    for p in mods["proj_P"].parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(
        [p for m in mods.values() for p in m.parameters() if p.requires_grad],
        lr=lr, betas=(0.9,0.999), weight_decay=1e-4
    )

    mods["enc_M"].train(); mods["enc_P"].train()
    mods["proj_M"].train(); mods["proj_P"].train()

    for ep in range(1, epochs+1):
        tot = 0.0; n = 0
        for mri, pet, _ in train_loader:
            mri = mri.to(device); pet = pet.to(device)
            zM, zP = embed_global(mods, mri, pet)   # no-grad inside embed_global, but we want grads here
            # Recompute without @torch.no_grad(): inline to allow grad
            zM = F.normalize(mods["proj_M"](mods["enc_M"](mri)), dim=1)
            zP = F.normalize(mods["proj_P"](mods["enc_P"](pet)), dim=1)

            L_MP = info_nce_ce(cosine_sim(zM, zP), tau)
            L_PM = info_nce_ce(cosine_sim(zP, zM), tau)
            loss = 0.5*(L_MP + L_PM)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item(); n += 1

        # (Optional) quick val retrieval@1 could be added here
        print(f"[Pretrain] epoch {ep}/{epochs}  loss={tot/max(1,n):.4f}")

    save_teachers(mods, ckpt_path)
    return mods
