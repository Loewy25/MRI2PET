import os, time, itertools
from typing import Any, Dict, Iterable, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scipy.ndimage import zoom as nd_zoom

# Local imports
from .contrastive import contrastive_aux_loss
from .config import (
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1,
    MVIEWS_ENABLE, MVIEWS_WEIGHTS,
)
import mri2pet.config as cfg  # to read Plan-4 floors & logging knobs dynamically

from .losses import l1_loss, ssim3d, psnr, mmd_gaussian
from .utils import _safe_name, _save_nifti, _meta_unbatch


# ---------- MGDA helpers ----------
def _flatten5d(x: torch.Tensor) -> torch.Tensor:
    # [B, C, D, H, W] -> [B, C*D*H*W]
    return x.flatten(start_dim=1)

def _l2_normalize_per_sample(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Normalize each sample vector to unit L2 in output-space (B-wise)
    B = v.size(0)
    v2d = _flatten5d(v)
    n = v2d.norm(dim=1, keepdim=True) + eps
    v2d = v2d / n
    return v2d.view_as(v)

def _cos_batch(v1n: torch.Tensor, v2n: torch.Tensor) -> float:
    # Cosine of normalized per-sample vectors; returns batch-mean
    a = _flatten5d(v1n)
    b = _flatten5d(v2n)
    cos = (a * b).sum(dim=1)  # per-sample cos
    return float(cos.mean().detach().cpu())

@torch.no_grad()
def _mgda2_normed(v1n: torch.Tensor, v2n: torch.Tensor, eps: float = 1e-12):
    """
    2-task MGDA in output-space, closed-form on the line segment.
    v1n, v2n are already per-sample L2-normalized; shape [B,1,D,H,W]
    Returns (alpha_scalar, v_comb) where v_comb has shape like v1n.
    alpha is the weight for v1n; (1-alpha) for v2n.
    """
    V1 = _flatten5d(v1n)  # [B,N]
    V2 = _flatten5d(v2n)  # [B,N]
    diff = (V2 - V1)
    num  = (diff * V2).sum(dim=1)                 # (v2-v1)·v2
    den  = (diff * diff).sum(dim=1) + eps         # ||v1-v2||^2
    alpha_b = torch.clamp(num / den, 0.0, 1.0)    # per-sample
    # robust aggregate (median)
    alpha = alpha_b.median()
    v_comb = alpha * v1n + (1.0 - alpha) * v2n
    return alpha, v_comb

@torch.no_grad()
def _mgda3_normed(vs_normed: list, eps: float = 1e-12):
    """
    3-task MGDA (tiny QP on simplex). We do:
      - try interior solution a ∝ G^{-1} 1 (if all >=0)
      - else try each edge (2-task closed forms) and pick min-norm.
    vs_normed: [v1n, v2n, v3n], each [B,1,D,H,W], per-sample normalized.
    Returns (w_best [3], v_comb). Assumes B>=1, we operate on batch-mean proxy.
    """
    assert len(vs_normed) == 3, "Need 3 normalized vectors"
    U = [_flatten5d(vn).mean(dim=0) for vn in vs_normed]  # reduce B by mean for QP proxy
    G = torch.stack([torch.stack([torch.dot(Ui, Uj) for Uj in U]) for Ui in U])  # [3,3]
    ones = torch.ones(3, device=G.device, dtype=G.dtype)

    candidates = []

    # interior candidate
    try:
        a_tilde = torch.linalg.solve(G + 1e-12 * torch.eye(3, device=G.device, dtype=G.dtype), ones)
    except RuntimeError:
        a_tilde = torch.linalg.lstsq(G, ones).solution
    a_int = a_tilde / (a_tilde.sum() + 1e-12)
    if (a_int >= -1e-8).all():
        a_int = torch.clamp(a_int, 0.0, 1.0)
        a_int = a_int / (a_int.sum() + 1e-12)
        candidates.append(a_int)

    # edges (i,j)
    def edge_2task(u_i, u_j, i, j):
        diff = u_j - u_i
        num = torch.dot(diff, u_j)
        den = torch.dot(diff, diff) + 1e-12
        a = torch.clamp(num / den, 0.0, 1.0)
        w = torch.zeros(3, device=G.device, dtype=G.dtype)
        w[i] = a
        w[j] = 1.0 - a
        return w
    candidates.append(edge_2task(U[0], U[1], 0, 1))
    candidates.append(edge_2task(U[0], U[2], 0, 2))
    candidates.append(edge_2task(U[1], U[2], 1, 2))

    # pick min-norm candidate
    def combo_vec(w):
        return w[0] * U[0] + w[1] * U[1] + w[2] * U[2]
    w_best, n_best = None, None
    for w in candidates:
        c = combo_vec(w)
        n = torch.dot(c, c)
        if (n_best is None) or (n < n_best):
            n_best, w_best = n, w
    v_comb = w_best[0] * vs_normed[0] + w_best[1] * vs_normed[1] + w_best[2] * vs_normed[2]
    return w_best, v_comb


def _apply_weight_floors(w: torch.Tensor, floors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Apply elementwise lower bounds and renormalize to simplex.
    w, floors: shape [n].
    """
    w = torch.maximum(w, floors)
    s = w.sum()
    if s.item() > 0:
        w = w / (s + eps)
    return w


def train_paggan(
    G: nn.Module,
    D: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    device: torch.device,
    epochs: int = EPOCHS,
    gamma: float = GAMMA,
    lambda_gan: float = LAMBDA_GAN,   # kept for logging compatibility
    data_range: float = DATA_RANGE,
    verbose: bool = True,
    # frozen teacher encoders + config dict
    contrastive_mods: Optional[Dict[str, nn.Module]] = None,
    contrast_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    G.to(device); D.to(device)
    G.train(); D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

    sch_G = ReduceLROnPlateau(
        opt_G, mode="min", factor=0.5, patience=15,
        threshold=1e-4, cooldown=5, min_lr=1e-6, verbose=True
    )
    sch_D = ReduceLROnPlateau(
        opt_D, mode="min", factor=0.5, patience=15,
        threshold=1e-4, cooldown=5, min_lr=5e-6, verbose=True
    )

    bce = nn.BCEWithLogitsLoss()

    best_val = float('inf')
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

    # Epoch accumulators for angles and weights
    from collections import defaultdict
    angle_accum = defaultdict(float)
    angle_count = 0

    # Weight accumulators (average per-epoch)
    w_rec_sums  = {"b": torch.zeros(2), "u3": torch.zeros(2), "out": torch.zeros(2)}   # [w_L1, w_SSIM]
    w_con_sums  = {"b": torch.zeros(2), "u3": torch.zeros(2), "out": torch.zeros(2)}   # [w_M2Ph, w_Ph2P]
    w_l2_sums   = {"b": torch.zeros(3), "u3": torch.zeros(3), "out": torch.zeros(3)}   # [w_Recon, w_GAN, w_Contr]
    w_counts    = 0

    # Floors from config (Plan-4)
    flo_L1    = float(getattr(cfg, "HMGDA_FLOOR_L1", 0.0))
    flo_SSIM  = float(getattr(cfg, "HMGDA_FLOOR_SSIM", 0.0))
    flo_M2PH  = float(getattr(cfg, "HMGDA_FLOOR_M2PH", 0.0))
    flo_PH2P  = float(getattr(cfg, "HMGDA_FLOOR_PH2P", 0.0))
    flo_RECON = float(getattr(cfg, "HMGDA_FLOOR_RECON", 0.0))
    flo_GAN   = float(getattr(cfg, "HMGDA_FLOOR_GAN", 0.0))
    flo_CONTR = float(getattr(cfg, "HMGDA_FLOOR_CONTR", 0.0))

    use_contrast = (contrastive_mods is not None) and (contrast_cfg is not None) and contrast_cfg.get("use", False)

    # --- NEW: optional per-ROI memory (persists across batches/epochs)
    roi_memory = None
    if use_contrast and getattr(cfg, "ROI_CONTRAST_ENABLE", False) and getattr(cfg, "ROI_MEMORY_ENABLE", False):
        from .memory import ROIMemory
        roi_memory = ROIMemory(max_len=getattr(cfg, "ROI_MEMORY_LEN", 512))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0

        # reset per-epoch accumulators
        for k in w_rec_sums:  w_rec_sums[k].zero_()
        for k in w_con_sums:  w_con_sums[k].zero_()
        for k in w_l2_sums:   w_l2_sums[k].zero_()
        w_counts = 0
        for k in list(angle_accum.keys()): angle_accum[k] = 0.0
        angle_count = 0

        for batch in train_loader:
            # --- Unpack batch & build B-aware brain mask ---
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, meta = batch
            else:
                raise ValueError("There is something wrong happened when passing data")

            mri = mri.to(device, non_blocking=True)  # [B,1,D,H,W] or [1,1,D,H,W] after collate tweak
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0)

            # meta may be a list of dicts (B>1) or a dict (B=1)
            if isinstance(meta, list) and len(meta) == B and isinstance(meta[0], dict) and ("brain_mask" in meta[0]):
                mask_list = []
                for i in range(B):
                    bm_np = meta[i].get("brain_mask", None)
                    if bm_np is not None:
                        bm_t = bm_np if torch.is_tensor(bm_np) else torch.from_numpy(bm_np)
                        mask_list.append(bm_t.to(device).bool())
                    else:
                        mask_list.append((pet[i, 0] > 0))
                brain_mask = torch.stack(mask_list, dim=0)  # [B,D,H,W]
            else:
                brain_mask = (pet[:, 0] > 0)

            # ---- Update D ----
            mri5 = mri if mri.dim() == 5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim() == 5 else pet.unsqueeze(0)

            with torch.no_grad():
                fake = G(mri5)

            D.zero_grad(set_to_none=True)
            pair_real = torch.cat([mri5, pet5], dim=1)           # [B,2,...]
            pair_fake = torch.cat([mri5, fake.detach()], dim=1)  # [B,2,...]

            out_real = D(pair_real)   # [B,1,d,h,w]
            out_fake = D(pair_fake)   # [B,1,d,h,w]

            loss_D = bce(out_real, torch.ones_like(out_real)) + \
                     bce(out_fake, torch.zeros_like(out_fake))
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_D.step()

            # ---- Update G (Hierarchical MGDA) ----
            G.zero_grad(set_to_none=True)

            # Forward ONCE to get all views
            fake, u3, b = G.forward_with_intermediates(mri5)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            # Atomic losses
            loss_gan  = bce(out_fake_for_G, torch.ones_like(out_fake_for_G))
            loss_l1   = l1_loss(fake, pet5)
            ssim_val  = ssim3d(fake, pet5, data_range=data_range)      # ∈ [0,1]
            loss_ssim = (1.0 - ssim_val)

            # Gradients wrt EACH VIEW (b, u3, fake). Normalize per-sample.
            # --- bottleneck view ---
            v_l1_b   = torch.autograd.grad(loss_l1,   b, retain_graph=True)[0]
            v_ssim_b = torch.autograd.grad(loss_ssim, b, retain_graph=True)[0]
            v_gan_b  = torch.autograd.grad(loss_gan,  b, retain_graph=True)[0]
            v_l1_b_n, v_ssim_b_n, v_gan_b_n = (
                _l2_normalize_per_sample(v_l1_b),
                _l2_normalize_per_sample(v_ssim_b),
                _l2_normalize_per_sample(v_gan_b),
            )

            # --- u3 view ---
            v_l1_u3   = torch.autograd.grad(loss_l1,   u3, retain_graph=True)[0]
            v_ssim_u3 = torch.autograd.grad(loss_ssim, u3, retain_graph=True)[0]
            v_gan_u3  = torch.autograd.grad(loss_gan,  u3, retain_graph=True)[0]
            v_l1_u3_n, v_ssim_u3_n, v_gan_u3_n = (
                _l2_normalize_per_sample(v_l1_u3),
                _l2_normalize_per_sample(v_ssim_u3),
                _l2_normalize_per_sample(v_gan_u3),
            )

            # --- output view (fake) ---
            v_l1_out   = torch.autograd.grad(loss_l1,   fake, retain_graph=True)[0]
            v_ssim_out = torch.autograd.grad(loss_ssim, fake, retain_graph=True)[0]
            v_gan_out  = torch.autograd.grad(loss_gan,  fake, retain_graph=True)[0]
            v_l1_out_n, v_ssim_out_n, v_gan_out_n = (
                _l2_normalize_per_sample(v_l1_out),
                _l2_normalize_per_sample(v_ssim_out),
                _l2_normalize_per_sample(v_gan_out),
            )

            # ===== Contrast losses (two directions) =====
            L_m2ph_total = torch.tensor(0.0, device=device)
            L_ph2p_total = torch.tensor(0.0, device=device)
            have_contrast = False
            if use_contrast:
                # Gather ROI masks (B=1 typical)
                roi_masks = None
                if isinstance(meta, dict) and ("roi_masks" in meta) and (len(meta["roi_masks"]) > 0):
                    roi_masks = {k: torch.as_tensor(v > 0, device=device) for k, v in meta["roi_masks"].items()}

                if getattr(cfg, "ROI_CONTRAST_ENABLE", False) and roi_masks:
                    # ROI-only contrast (in-ROI positives/negatives + optional memory)
                    from .contrastive import contrastive_aux_loss_roi
                    L_m2ph_total, L_ph2p_total, _ = contrastive_aux_loss_roi(
                        mods=contrastive_mods,
                        mri=mri5, pet=pet5, pet_hat=fake,
                        roi_masks=roi_masks,
                        tau=cfg.CONTRAST_TAU,
                        patches_per_roi=cfg.ROI_PATCHES_PER_ROI,
                        patch_size=cfg.ROI_PATCH_SIZE,
                        roi_weights=cfg.ROI_AGG_WEIGHTS,
                        roi_memory=roi_memory if getattr(cfg, "ROI_MEMORY_ENABLE", False) else None
                    )
                else:
                    # fallback to your original brain-mask patch contrast
                    from .contrastive import contrastive_aux_loss
                    L_m2ph_total, L_ph2p_total, _ = contrastive_aux_loss(
                        mods=contrastive_mods,
                        mri=mri5, pet=pet5, pet_hat=fake,
                        brain_mask=brain_mask,
                        tau=contrast_cfg["tau"],
                        use_patches=contrast_cfg["use_patches"],
                        patch_size=contrast_cfg["patch_size"],
                        patches_per_subj=contrast_cfg["patches_per_subj"]
                    )

                # Per-view contrast grads (two directions)
                v_m2ph_b   = torch.autograd.grad(L_m2ph_total, b,    retain_graph=True)[0]
                v_m2ph_u3  = torch.autograd.grad(L_m2ph_total, u3,   retain_graph=True)[0]
                v_m2ph_out = torch.autograd.grad(L_m2ph_total, fake, retain_graph=True)[0]

                v_ph2p_b   = torch.autograd.grad(L_ph2p_total, b,    retain_graph=True)[0]
                v_ph2p_u3  = torch.autograd.grad(L_ph2p_total, u3,   retain_graph=True)[0]
                v_ph2p_out = torch.autograd.grad(L_ph2p_total, fake, retain_graph=True)[0]

                # Normalize
                v_m2ph_b_n,  v_m2ph_u3_n,  v_m2ph_out_n  = (
                    _l2_normalize_per_sample(v_m2ph_b), _l2_normalize_per_sample(v_m2ph_u3), _l2_normalize_per_sample(v_m2ph_out)
                )
                v_ph2p_b_n,  v_ph2p_u3_n,  v_ph2p_out_n  = (
                    _l2_normalize_per_sample(v_ph2p_b), _l2_normalize_per_sample(v_ph2p_u3), _l2_normalize_per_sample(v_ph2p_out)
                )
                have_contrast = True

            # ===== Level-1: inside-group MGDA =====
            def _group2(v_a_n, v_b_n, flo_a, flo_b):
                alpha, v_ab = _mgda2_normed(v_a_n, v_b_n)  # alpha for v_a, 1-alpha for v_b
                w = torch.stack([alpha, 1.0 - alpha]).to(v_a_n.device)
                # floors (post-solve)
                floors = torch.tensor([flo_a, flo_b], device=v_a_n.device, dtype=w.dtype)
                if floors.max().item() > 0:
                    w = _apply_weight_floors(w, floors)
                    v_ab = w[0] * v_a_n + w[1] * v_b_n
                return w, v_ab

            w_rec_b,   v_rec_b   = _group2(v_l1_b_n,   v_ssim_b_n,  flo_L1,   flo_SSIM)
            w_rec_u3,  v_rec_u3  = _group2(v_l1_u3_n,  v_ssim_u3_n, flo_L1,   flo_SSIM)
            w_rec_out, v_rec_out = _group2(v_l1_out_n, v_ssim_out_n, flo_L1,   flo_SSIM)

            # Contrast = {M→P̂, P̂→P}
            if have_contrast:
                w_con_b,   v_con_b   = _group2(v_m2ph_b_n,   v_ph2p_b_n,   flo_M2PH, flo_PH2P)
                w_con_u3,  v_con_u3  = _group2(v_m2ph_u3_n,  v_ph2p_u3_n,  flo_M2PH, flo_PH2P)
                w_con_out, v_con_out = _group2(v_m2ph_out_n, v_ph2p_out_n, flo_M2PH, flo_PH2P)
            else:
                w_con_b = w_con_u3 = w_con_out = torch.tensor([0.0, 0.0], device=device)
                v_con_b = v_con_u3 = v_con_out = None

            # ===== Level-2: MGDA across {Recon, GAN, Contrast} per view =====
            def _level2(v_rec, v_gan_n, v_contr):
                if v_contr is not None:
                    w_raw, v_comb = _mgda3_normed([v_rec, v_gan_n, v_contr])
                    floors = torch.tensor([flo_RECON, flo_GAN, flo_CONTR], device=v_rec.device, dtype=w_raw.dtype)
                    if floors.max().item() > 0:
                        w_adj = _apply_weight_floors(w_raw, floors)
                        v_comb = w_adj[0] * v_rec + w_adj[1] * v_gan_n + w_adj[2] * v_contr
                        return w_adj, v_comb
                    else:
                        return w_raw, v_comb
                else:
                    alpha_rg, v_rg = _mgda2_normed(v_rec, v_gan_n)
                    w2 = torch.stack([alpha_rg, 1.0 - alpha_rg, torch.tensor(0.0, device=v_rec.device, dtype=alpha_rg.dtype)])
                    return w2, v_rg

            w_l2_b,   v_b_hier   = _level2(v_rec_b,   v_gan_b_n,   v_con_b)
            w_l2_u3,  v_u3_hier  = _level2(v_rec_u3,  v_gan_u3_n,  v_con_u3)
            w_l2_out, v_out_hier = _level2(v_rec_out, v_gan_out_n, v_con_out)

            # ===== Angle logging (task & group) at OUTPUT view =====
            v_rec_out_n  = _l2_normalize_per_sample(v_rec_out)
            v_out_hier_n = _l2_normalize_per_sample(v_out_hier)
            angle_accum["cos(L1,SSIM)"]   += _cos_batch(v_l1_out_n,  v_ssim_out_n)
            angle_accum["cos(L1,GAN)"]    += _cos_batch(v_l1_out_n,  v_gan_out_n)
            angle_accum["cos(SSIM,GAN)"]  += _cos_batch(v_ssim_out_n, v_gan_out_n)
            angle_accum["cos(Recon,GAN)"] += _cos_batch(v_rec_out_n, v_gan_out_n)
            if have_contrast:
                v_con_out_n = _l2_normalize_per_sample(v_con_out)
                angle_accum["cos(Recon,Contrast)"]  += _cos_batch(v_rec_out_n,   v_con_out_n)
                angle_accum["cos(Contrast,GAN)"]    += _cos_batch(v_con_out_n,   v_gan_out_n)
                angle_accum["cos(HierOut,Contrast)"]+= _cos_batch(v_out_hier_n,  v_con_out_n)
            angle_accum["cos(HierOut,Recon)"] += _cos_batch(v_out_hier_n, v_rec_out_n)
            angle_accum["cos(HierOut,GAN)"]   += _cos_batch(v_out_hier_n, v_gan_out_n)
            angle_count += 1

            # ===== Accumulate weights (for per-epoch averages) =====
            def _acc(dst, key, w):
                dst[key] += w.detach().cpu()

            _acc(w_rec_sums, 'b',   w_rec_b)
            _acc(w_rec_sums, 'u3',  w_rec_u3)
            _acc(w_rec_sums, 'out', w_rec_out)

            _acc(w_con_sums, 'b',   w_con_b)
            _acc(w_con_sums, 'u3',  w_con_u3)
            _acc(w_con_sums, 'out', w_con_out)

            _acc(w_l2_sums, 'b',    w_l2_b)
            _acc(w_l2_sums, 'u3',   w_l2_u3)
            _acc(w_l2_sums, 'out',  w_l2_out)
            w_counts += 1

            # ===== View fusion and backward =====
            w_b, w_u3, w_out = (MVIEWS_WEIGHTS if MVIEWS_ENABLE else (0.0, 0.0, 1.0))
            s = (w_b + w_u3 + w_out) or 1.0
            w_b, w_u3, w_out = w_b/s, w_u3/s, w_out/s

            grad_b, grad_u3, grad_out = w_b * v_b_hier, w_u3 * v_u3_hier, w_out * v_out_hier

            opt_G.zero_grad(set_to_none=True)
            torch.autograd.backward(
                tensors=[b, u3, fake],
                grad_tensors=[grad_b, grad_u3, grad_out],
                retain_graph=False
            )

            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # Logging proxy (only for monitoring)
            if have_contrast:
                loss_G_log = (gamma * (loss_l1 + loss_ssim) + loss_gan +
                              L_m2ph_total.item() + L_ph2p_total.item())
            else:
                loss_G_log = (gamma * (loss_l1 + loss_ssim) + loss_gan).detach().item()

            g_running += float(loss_G_log)
            d_running += loss_D.item()
            n_batches += 1

        # ---- End of epoch aggregates ----
        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)
        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)

        # ---- Validation ----
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon = 0.0
                v_batches = 0
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri_v, pet_v, _ = batch
                    else:
                        mri_v, pet_v = batch

                    mri_v = mri_v.to(device, non_blocking=True)
                    pet_v = pet_v.to(device, non_blocking=True)
                    if mri_v.dim() == 4:  # [1,D,H,W] -> [B=1,1,D,H,W]
                        mri_v = mri_v.unsqueeze(0)
                    if pet_v.dim() == 4:
                        pet_v = pet_v.unsqueeze(0)

                    fake_v = G(mri_v)
                    L1_v   = l1_loss(fake_v, pet_v)
                    SSIM_v = ssim3d(fake_v, pet_v, data_range=data_range)
                    val_recon += (L1_v + (1.0 - SSIM_v)).item()
                    v_batches += 1

                val_recon = val_recon / max(1, v_batches)
                hist["val_recon"].append(val_recon)

                if val_recon < best_val:
                    best_val = val_recon
                    best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                    best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                    torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                    torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))

                # Update LR schedulers
                sch_G.step(val_recon)
                sch_D.step(val_recon)

                if verbose:
                    lrG = opt_G.param_groups[0]["lr"]
                    lrD = opt_D.param_groups[0]["lr"]
                    dt  = time.time() - t0
                    print(f"    LR update: G={lrG:.2e}, D={lrD:.2e}")
                    print(f"Epoch [{epoch:03d}/{epochs}]  "
                          f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                          f"ValRecon(L1 + 1-SSIM): {val_recon:.4f}  "
                          f"| best {best_val:.4f}  | λ_g={float(lambda_gan):.4f}  | {dt:.1f}s")
            G.train()
        else:
            if verbose:
                dt = time.time() - t0
                print(f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  | {dt:.1f}s")

        # ---- Epoch-end: print cosines and MGDA weights ----
        if getattr(cfg, "PRINT_GRAD_COSINES", True) and angle_count > 0:
            print("  Pairwise cosine averages @OUTPUT view this epoch:")
            keys = [
                "cos(L1,SSIM)", "cos(L1,GAN)", "cos(SSIM,GAN)",
                "cos(Recon,GAN)", "cos(Recon,Contrast)", "cos(Contrast,GAN)",
                "cos(HierOut,Recon)", "cos(HierOut,GAN)", "cos(HierOut,Contrast)"
            ]
            for k in keys:
                if k in angle_accum:
                    print(f"    {k:>20}: {angle_accum[k] / angle_count:.3f}")

        if w_counts > 0:
            def _avg(v): return (v / w_counts).tolist()
            print("  Level-1 Recon weights [L1, SSIM] avg:")
            print(f"    bottleneck: {_avg(w_rec_sums['b'])}")
            print(f"    u3       : {_avg(w_rec_sums['u3'])}")
            print(f"    output   : {_avg(w_rec_sums['out'])}")

            print("  Level-1 Contrast weights [M→P̂, P̂→P] avg:")
            print(f"    bottleneck: {_avg(w_con_sums['b'])}")
            print(f"    u3       : {_avg(w_con_sums['u3'])}")
            print(f"    output   : {_avg(w_con_sums['out'])}")

            print("  Level-2 GROUP weights [Recon, GAN, Contrast] avg:")
            print(f"    bottleneck: {_avg(w_l2_sums['b'])}")
            print(f"    u3       : {_avg(w_l2_sums['u3'])}")
            print(f"    output   : {_avg(w_l2_sums['out'])}")

    # restore best
    if best_G is not None:
        G.load_state_dict(best_G)
    if best_D is not None:
        D.load_state_dict(best_D)

    return {"history": hist, "best_G": best_G, "best_D": best_D}

@torch.no_grad()
def evaluate_paggan(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    data_range: float = DATA_RANGE,
    mmd_voxels: int = 2048,
) -> Dict[str, float]:
    G.to(device)
    G.eval()

    ssim_sum = 0.0
    psnr_sum = 0.0
    mse_sum = 0.0
    mmd_sum = 0.0
    n = 0

    for batch in test_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {}
        mri = mri.to(device, non_blocking=True)
        pet = pet.to(device, non_blocking=True)
        fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))

        pet_for_metric  = pet if pet.dim()==5 else pet.unsqueeze(0)

        # masked metrics
        brain_mask_np = meta.get("brain_mask", None) if isinstance(meta, dict) else None
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            raise TypeError("No Mask")

        fake_m = fake * brain
        pet_m  = pet_for_metric * brain

        ssim_sum += ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_sum += psnr(fake_m, pet_m, data_range=data_range)
        mse_sum  += F.mse_loss(fake_m, pet_m).item()
        mmd_sum  += mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)
        n += 1

    return {
        "SSIM": ssim_sum / max(1, n),
        "PSNR": psnr_sum / max(1, n),
        "MSE":  mse_sum  / max(1, n),
        "MMD":  mmd_sum  / max(1, n),
    }


@torch.no_grad()
def save_test_volumes(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    out_dir: str,
    resample_back_to_t1: bool = RESAMPLE_BACK_TO_T1,
):
    print(f"Saving test volumes to: {out_dir}  (resample_back_to_t1={resample_back_to_t1})")
    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    for i, batch in enumerate(test_loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {"sid": f"sample_{i:04d}", "t1_affine": np.eye(4), "orig_shape": tuple(mri.shape[2:]), "cur_shape": tuple(mri.shape[2:]), "resized_to": None}
        meta = _meta_unbatch(meta)

        sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
        subdir = os.path.join(out_dir, sid)
        os.makedirs(subdir, exist_ok=True)

        mri_t  = mri.to(device, non_blocking=True)
        fake_t = G(mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0))

        mri_np  = (mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        pet_np  = (pet   if pet.dim()==5   else pet.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        fake_np = fake_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        err_np  = np.abs(fake_np - pet_np)

        cur_shape  = tuple(mri_np.shape)
        orig_shape = tuple(meta.get("orig_shape", cur_shape))

        if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
            zf = (float(orig_shape[0]) / float(cur_shape[0]),
                  float(orig_shape[1]) / float(cur_shape[1]),
                  float(orig_shape[2]) / float(cur_shape[2]))
            mri_np  = nd_zoom(mri_np,  zf, order=1)
            pet_np  = nd_zoom(pet_np,  zf, order=1)
            fake_np = nd_zoom(fake_np, zf, order=1)
            err_np  = nd_zoom(err_np,  zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            resized_to = meta.get("resized_to", None)
            if resized_to is None:
                affine_to_use = meta.get("t1_affine", np.eye(4))
            else:
                affine_to_use = np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

        print(f"  saved {sid}: MRI/PET_gt/PET_fake/PET_abs_error")


@torch.no_grad()
@torch.no_grad()
def evaluate_and_save(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    out_dir: str,
    data_range: float = DATA_RANGE,
    mmd_voxels: int = 2048,
    resample_back_to_t1: bool = RESAMPLE_BACK_TO_T1,
):
    """
    Evaluates on test_loader, saves volumes, and returns a dict of aggregate metrics.
    Also writes per-subject metrics CSV (sid, SSIM, PSNR, MSE, MMD) for later aggregation,
    and includes 95% confidence intervals in the returned dict.
    """
    import csv, json
    import numpy as np
    try:
        from scipy.stats import t as _t_dist
        def _tcrit(df): return float(_t_dist.ppf(0.975, df)) if df > 0 else float('nan')
    except Exception:
        def _tcrit(df): return 1.96 if df > 0 else float('nan')  # normal approx fallback

    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    # Per-subject collections
    sids, ssim_list, psnr_list, mse_list, mmd_list = [], [], [], [], []

    run_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) else out_dir

    for i, batch in enumerate(test_loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {"sid": f"sample_{i:04d}", "t1_affine": np.eye(4),
                    "orig_shape": tuple(mri.shape[2:]), "cur_shape": tuple(mri.shape[2:]), "resized_to": None}
        meta = _meta_unbatch(meta)

        sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
        sids.append(sid)
        subdir = os.path.join(out_dir, sid)
        os.makedirs(subdir, exist_ok=True)

        mri_t = mri.to(device, non_blocking=True)
        pet_t = pet.to(device, non_blocking=True)
        fake_t = G(mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0))

        pet_for_metric  = pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)

        # mask (prefer brain mask; else pet>0)
        brain_mask_np = meta.get("brain_mask", None)
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake_t * brain
        pet_m  = pet_for_metric * brain

        # per-subject metrics
        ssim_val = ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_val = psnr(fake_m,  pet_m, data_range=data_range)
        mse_val  = F.mse_loss(fake_m, pet_m).item()
        mmd_val  = mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)

        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        mse_list.append(mse_val)
        mmd_list.append(mmd_val)

        # Save volumes (unchanged)
        mri_np  = (mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        pet_np  = (pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        fake_np =  fake_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        err_np  = np.abs(fake_np - pet_np)

        cur_shape  = tuple(mri_np.shape)
        orig_shape = tuple(meta.get("orig_shape", cur_shape))

        if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
            zf = (float(orig_shape[0]) / float(cur_shape[0]),
                  float(orig_shape[1]) / float(cur_shape[1]),
                  float(orig_shape[2]) / float(cur_shape[2]))
            mri_np  = nd_zoom(mri_np,  zf, order=1)
            pet_np  = nd_zoom(pet_np,  zf, order=1)
            fake_np = nd_zoom(fake_np, zf, order=1)
            err_np  = nd_zoom(err_np,  zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            resized_to = meta.get("resized_to", None)
            affine_to_use = meta.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

    # ---- Aggregate + CI (keep old keys as means) ----
    def _mean_std_ci(vals):
        a = np.asarray(vals, dtype=np.float64)
        n = a.size
        mean = float(a.mean()) if n > 0 else float("nan")
        std  = float(a.std(ddof=1)) if n > 1 else float("nan")
        se   = (std / np.sqrt(n)) if n > 1 else float("nan")
        tcrit = _tcrit(n - 1)
        lo = mean - tcrit * se if n > 1 else float("nan")
        hi = mean + tcrit * se if n > 1 else float("nan")
        return mean, std, n, lo, hi

    m_ssim, sd_ssim, n_ssim, lo_ssim, hi_ssim = _mean_std_ci(ssim_list)
    m_psnr, sd_psnr, n_psnr, lo_psnr, hi_psnr = _mean_std_ci(psnr_list)
    m_mse,  sd_mse,  n_mse,  lo_mse,  hi_mse  = _mean_std_ci(mse_list)
    m_mmd,  sd_mmd,  n_mmd,  lo_mmd,  hi_mmd  = _mean_std_ci(mmd_list)

    # Per-subject CSV (next to volumes/)
    per_subj_csv = os.path.join(run_dir, "per_subject_metrics.csv")
    with open(per_subj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "SSIM", "PSNR", "MSE", "MMD"])
        for sid, ssim_v, psnr_v, mse_v, mmd_v in zip(sids, ssim_list, psnr_list, mse_list, mmd_list):
            w.writerow([sid, ssim_v, psnr_v, mse_v, mmd_v])

    # Summary JSON with CI
    summary_json = os.path.join(run_dir, "test_metrics_summary.json")
    summary = {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE":  m_mse,  "MSE_std":  sd_mse,  "MSE_lo95":  lo_mse,  "MSE_hi95":  hi_mse,
        "MMD":  m_mmd,  "MMD_std":  sd_mmd,  "MMD_lo95":  lo_mmd,  "MMD_hi95":  hi_mmd,
        "per_subject_csv": per_subj_csv,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Backward-compatible return
    return {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE":  m_mse,  "MSE_std":  sd_mse,  "MSE_lo95":  lo_mse,  "MSE_hi95":  hi_mse,
        "MMD":  m_mmd,  "MMD_std":  sd_mmd,  "MMD_lo95":  lo_mmd,  "MMD_hi95":  hi_mmd,
        "per_subject_csv": per_subj_csv,
        "summary_json": summary_json,
    }
