import os, time
from typing import Any, Dict, Iterable, Optional
import numpy as np
from scipy.ndimage import zoom as nd_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1
)
from .losses import l1_loss, ssim3d, psnr, mmd_gaussian
from .utils import _safe_name, _save_nifti, _meta_unbatch
from .config import (
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1,
    # cosine-gated multi-view fusion
    MVIEWS_ENABLE, MVIEWS_COS_FUSION, MVIEWS_KAPPA, MVIEWS_FLOOR, MVIEWS_EMA_BETA,
)


# ---------- MGDA-UB helpers for dynamic grouping ----------
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
    # Cosine of normalized per-sample vectors; returns batch-mean (B=1 -> scalar)
    a = _flatten5d(v1n)
    b = _flatten5d(v2n)
    # v1n, v2n should already be per-sample normalized
    cos = (a * b).sum(dim=1)  # per-sample cos
    return float(cos.mean().detach().cpu())

@torch.no_grad()
def _mgda2_normed(v1n: torch.Tensor, v2n: torch.Tensor, eps: float = 1e-12):
    """
    2-task MGDA in output-space, closed-form on the line segment.
    v1n, v2n are already per-sample L2-normalized; shape [B,1,D,H,W]
    Returns (alpha_scalar, v_comb) where v_comb has shape like v1n.
    """
    V1 = _flatten5d(v1n)  # [B,N]
    V2 = _flatten5d(v2n)  # [B,N]
    diff = (V2 - V1)
    num  = (diff * V2).sum(dim=1)                 # (v2-v1)·v2
    den  = (diff * diff).sum(dim=1) + eps         # ||v1-v2||^2
    alpha_b = torch.clamp(num / den, 0.0, 1.0)    # per-sample solution
    # robust scalar for the batch (B=1 -> just that value)
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
    Returns (w_best [3], v_comb)
    NOTE: assumes B=1 (your setup). If B>1, run per-sample & reduce by median.
    """
    assert len(vs_normed) == 3, "Need 3 normalized vectors"
    # flatten (assume B=1)
    U = [_flatten5d(vn).squeeze(0) for vn in vs_normed]  # each [N]
    # Gram matrix
    G = torch.stack([torch.stack([torch.dot(Ui, Uj) for Uj in U]) for Ui in U])  # [3,3]
    ones = torch.ones(3, device=G.device, dtype=G.dtype)

    candidates = []

    # interior candidate: a_tilde = G^{-1} 1, then normalize to sum=1
    try:
        a_tilde = torch.linalg.solve(G + eps * torch.eye(3, device=G.device, dtype=G.dtype), ones)
    except RuntimeError:
        a_tilde = torch.linalg.lstsq(G, ones).solution
    a_int = a_tilde / (a_tilde.sum() + eps)
    if (a_int >= -1e-8).all():  # feasible
        a_int = torch.clamp(a_int, 0.0, 1.0)
        a_int = a_int / (a_int.sum() + eps)
        candidates.append(a_int)

    # edges: (i,j) pairs -> 2-task closed forms
    def edge_2task(u_i, u_j, i, j):
        diff = u_j - u_i
        num = torch.dot(diff, u_j)
        den = torch.dot(diff, diff) + eps
        a = torch.clamp(num / den, 0.0, 1.0)  # weight on i
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
# ------------------------------------------------------------


def train_paggan(
    G: nn.Module,
    D: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    device: torch.device,
    epochs: int = EPOCHS,
    gamma: float = GAMMA,
    lambda_gan: float = LAMBDA_GAN,
    data_range: float = DATA_RANGE,
    verbose: bool = True,
) -> Dict[str, Any]:
    G.to(device); D.to(device)
    G.train(); D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    bce = nn.BCEWithLogitsLoss()

    # === Global Gradient‑Ratio Controller (dynamic lambda_g) ===
    lambda_g = float(lambda_gan)  # initialize from config LAMBDA_GAN
    ema_beta = 0.9                # EMA smoothing for B=1 noise
    LAM_MIN, LAM_MAX = 0.05, 5.0  # clamp range for stability
    EPS = 1e-8
    TRUST_TAU = 0.5               # trust-region: max 50% change per step

    best_val = float('inf')
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}
        # === Dynamic grouping controller state (Idea 1) ===
    from .config import DYN_GROUP, COS_EMA_BETA, COS_HIGH, COS_LOW, MIN_HOLD_EPOCHS, ADAM_AWARE_NORM
    group_state = "merged"   # start merged: L1+SSIM act as one Recon task
    epochs_since_switch = 0
    cos_ema = None           # EMA of cos(L1,SSIM) in output-space

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, _ = batch
            else:
                raise ValueError("There is something wrong happened when passing data")
            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0) if mri.dim() == 5 else 1
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # ---- Update D ----
            mri5 = mri if mri.dim()==5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim()==5 else pet.unsqueeze(0)

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
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)  # <— clip D
            opt_D.step()



                        # ---- Update G (Dynamic merge/split of L1 & SSIM + MGDA-UB) ----
            G.zero_grad(set_to_none=True)

            # Forward through G and D (for GAN loss)
            fake = G(mri5)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            # Atomic losses (keep separate; we'll merge/split on gradients)
            loss_gan = bce(out_fake_for_G, torch.ones_like(out_fake_for_G))
            loss_l1  = l1_loss(fake, pet5)
            ssim_val = ssim3d(fake, pet5, data_range=data_range)      # ∈ [0,1]
            loss_ssim = (1.0 - ssim_val)                               # make it a loss
            loss_recon = gamma * (loss_l1 + loss_ssim)                 # for logging

            # Output-space grads wrt 'fake' for each task
            # We will not backprop them; just read. We'll do ONE backward with the combined direction.
            v_l1   = torch.autograd.grad(loss_l1,   fake, retain_graph=True)[0]
            v_ssim = torch.autograd.grad(loss_ssim, fake, retain_graph=True)[0]
            v_gan  = torch.autograd.grad(loss_gan,  fake, retain_graph=True)[0]

            # Per-sample L2 normalization (stable across objectives)
            v_l1n   = _l2_normalize_per_sample(v_l1)
            v_ssimn = _l2_normalize_per_sample(v_ssim)
            v_gann  = _l2_normalize_per_sample(v_gan)

            # Track cosine(L1, SSIM) this epoch (for the controller)
            cos_l1_ssim = _cos_batch(v_l1n, v_ssimn)  # scalar (B=1)
            # Accumulate epoch stats
            if 'cos_sum' not in locals():
                cos_sum, cos_cnt = 0.0, 0
            cos_sum += cos_l1_ssim
            cos_cnt += 1

            # ---- Build final output-space direction v_comb depending on state ----
            if DYN_GROUP and group_state == "merged":
                # (1) Inside the group: MGDA-UB on {L1, SSIM} -> v_recon^†
                _, v_recon_dag = _mgda2_normed(v_l1n, v_ssimn)
                # (2) Top level: MGDA-UB on {v_recon^†, GAN}
                _, v_comb = _mgda2_normed(v_recon_dag, v_gann)
                mode_str = "MERGED"
            else:
                # Split: MGDA-UB on {L1, SSIM, GAN}
                _, v_comb = _mgda3_normed([v_l1n, v_ssimn, v_gann])
                mode_str = "SPLIT"

            # Single backward through G with the combined direction
            opt_G.zero_grad(set_to_none=True)
            fake.backward(v_comb)   # ONE backward
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # For logging only (proxy scalar; not used for backward)
            loss_G_log = (loss_recon + loss_gan).detach().item()

            g_running += float(loss_G_log)
            d_running += loss_D.item()
            n_batches += 1


        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)
        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)
        # ---- End-of-epoch: update dynamic grouping state ----
        if DYN_GROUP and cos_cnt > 0:
            cos_epoch = cos_sum / max(1, cos_cnt)
            cos_ema = (cos_epoch if (cos_ema is None)
                       else COS_EMA_BETA * cos_ema + (1.0 - COS_EMA_BETA) * cos_epoch)

            # Switch with hysteresis & min-hold
            switched = False
            if group_state == "merged" and epochs_since_switch >= MIN_HOLD_EPOCHS and cos_ema is not None and cos_ema <= COS_LOW:
                group_state = "split"
                epochs_since_switch = 0
                switched = True
            elif group_state == "split" and epochs_since_switch >= MIN_HOLD_EPOCHS and cos_ema is not None and cos_ema >= COS_HIGH:
                group_state = "merged"
                epochs_since_switch = 0
                switched = True
            else:
                epochs_since_switch += 1

            # Log controller status
            print(f"[Epoch {epoch:03d}] Grouping={group_state} | cos(L1,SSIM)_ema={cos_ema:.3f}"
                  + ("  (state changed)" if switched else ""))

            # reset accumulators for next epoch
            del cos_sum, cos_cnt


        # ---- Validation ----
        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon, v_batches = 0.0, 0
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri, pet, _ = batch
                    else:
                        mri, pet = batch
                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    fake = G(mri if mri.dim()==5 else mri.unsqueeze(0))
                    loss_l1_v = l1_loss(fake, pet if pet.dim()==5 else pet.unsqueeze(0))
                    ssim_v  = ssim3d(fake, pet if pet.dim()==5 else pet.unsqueeze(0),
                                     data_range=data_range)
                    val_recon += (loss_l1_v + (1.0 - ssim_v)).item()
                    v_batches += 1
            val_recon /= max(1, v_batches)
            hist["val_recon"].append(val_recon)

            if val_recon < best_val:
                best_val = val_recon
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))

            if verbose:
                dt = time.time() - t0
                print(f"Epoch [{epoch:03d}/{epochs}]  "
                      f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                      f"ValRecon(L1 + 1-SSIM): {val_recon:.4f}  "
                      f"| best {best_val:.4f}  | λ_g={lambda_g:.4f}  | {dt:.1f}s")

            G.train()
        elif verbose:
            dt = time.time() - t0
            print(f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  | {dt:.1f}s")

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
            brain = (pet_for_metric > 0).float()

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
    import numpy as np
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
def evaluate_and_save(
    G: nn.Module,
    test_loader: Iterable,
    device: torch.device,
    out_dir: str,
    data_range: float = DATA_RANGE,
    mmd_voxels: int = 2048,
    resample_back_to_t1: bool = RESAMPLE_BACK_TO_T1,
):
    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    ssim_sum = 0.0
    psnr_sum = 0.0
    mse_sum  = 0.0
    mmd_sum  = 0.0
    n = 0

    for i, batch in enumerate(test_loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            mri, pet, meta = batch
        else:
            mri, pet = batch
            meta = {"sid": f"sample_{i:04d}", "t1_affine": np.eye(4), "orig_shape": tuple(mri.shape[2:]), "cur_shape": tuple(mri.shape[2:]), "resized_to": None}
        meta = _meta_unbatch(meta)

        mri_t = mri.to(device, non_blocking=True)
        pet_t = pet.to(device, non_blocking=True)
        fake_t = G(mri_t if mri_t.dim()==5 else mri_t.unsqueeze(0))

        pet_for_metric  = pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)

        brain_mask_np = meta.get("brain_mask", None)
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake_t * brain
        pet_m  = pet_for_metric * brain

        ssim_sum += ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_sum += psnr(fake_m,  pet_m, data_range=data_range)
        mse_sum  += F.mse_loss(fake_m, pet_m).item()
        mmd_sum  += mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)
        n += 1

        sid = _safe_name(meta.get("sid", f"sample_{i:04d}"))
        subdir = os.path.join(out_dir, sid)
        os.makedirs(subdir, exist_ok=True)

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
            if resized_to is None:
                affine_to_use = meta.get("t1_affine", np.eye(4))
            else:
                affine_to_use = np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

    return {
        "SSIM": ssim_sum / max(1, n),
        "PSNR": psnr_sum / max(1, n),
        "MSE":  mse_sum  / max(1, n),
        "MMD":  mmd_sum  / max(1, n),
    }
