# train_eval.py
import os
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.ndimage import zoom as nd_zoom

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1,
)

from .losses import l1_loss, ssim3d, psnr, mmd_gaussian
from .utils import _safe_name, _save_nifti, _meta_unbatch
import wandb


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


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

    # ---- NEW: FAMO hyperparams (env-overridable) ----
    famo_beta: Optional[float] = None,   # β for ξ update
    famo_decay: Optional[float] = None,  # γ decay for ξ
    famo_eps: Optional[float] = None,    # ε for positivity/log stability

    # ---- epoch-scheduled cortex contrast (env-overridable) ----
    e_normal1: Optional[int] = None,
    e_contrast: Optional[int] = None,
    contrast_eps: Optional[float] = None,
    contrast_eps0: Optional[float] = None,
    lambda_contrast_out: Optional[float] = None,
    lambda_contrast_ctx: Optional[float] = None,

    # ---- Backward-compat (old Jacobian/FGSM args are accepted but ignored) ----
    lambda_sens: Optional[float] = None,
    lambda_local: Optional[float] = None,
    fgsm_eps: Optional[float] = None,
    sens_tau: Optional[float] = None,

    verbose: bool = True,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Trains PatchGAN + Generator using **FAMO** task weighting over 3 losses:
      1) global recon: gamma * (L1 + (1-SSIM))
      2) cortex ROI recon: masked L1
      3) GAN loss (LSGAN objective for G)

    Additionally (only during scheduled contrast epochs):
      - cortex contrast loss (unchanged from your code)

    FAMO follows Algorithm 1 from the paper:
      - compute z = softmax(ξ)
      - compute w_i ∝ z_i / ℓ_i, normalized to sum to 1 (via c_t)
      - update ξ using change in log-losses log ℓ_t - log ℓ_{t+1}
    """

    # ---- Resolve env-overridable hyperparams (only if not explicitly passed) ----
    if e_normal1 is None:
        e_normal1 = _env_int("E_NORMAL1", 0)
    if e_contrast is None:
        e_contrast = _env_int("E_CONTRAST", 0)

    if contrast_eps is None:
        contrast_eps = _env_float("CONTRAST_EPS", 0.005)
    if contrast_eps0 is None:
        contrast_eps0 = _env_float("CONTRAST_EPS0", 1e-6)

    if lambda_contrast_out is None:
        lambda_contrast_out = _env_float("LAMBDA_CONTRAST_OUT", 0.05)
    if lambda_contrast_ctx is None:
        lambda_contrast_ctx = _env_float("LAMBDA_CONTRAST_CTX", 0.005)

    # ---- FAMO params ----
    if famo_beta is None:
        famo_beta = _env_float("FAMO_BETA", 0.01)
    if famo_decay is None:
        famo_decay = _env_float("FAMO_DECAY", 0.001)
    if famo_eps is None:
        famo_eps = _env_float("FAMO_EPS", 1e-8)

    # One-time note if old (deprecated) args were passed
    if verbose and any(x is not None for x in [lambda_sens, lambda_local, fgsm_eps, sens_tau]):
        print(
            "[INFO] Deprecated FGSM/Jacobian args (lambda_sens/lambda_local/fgsm_eps/sens_tau) "
            "were provided but are ignored in the new training."
        )

    G.to(device)
    D.to(device)
    G.train()
    D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    adv_criterion = nn.MSELoss()

    best_val = float("inf")
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

    # ---- FAMO state: task logits ξ (task order fixed!) ----
    # [0]=global recon, [1]=ROI recon, [2]=GAN
    xi = torch.zeros(3, device=device, dtype=torch.float32)

    def _meta_to_cortex_mask(meta_any, B_expected: int) -> Optional[torch.Tensor]:
        if isinstance(meta_any, dict):
            cm = meta_any.get("cortex_mask", None)
            if cm is None:
                return None
            t = torch.from_numpy(cm.astype(np.float32))
            if t.dim() == 3:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.dim() == 4:
                t = t.unsqueeze(0)
            return t.to(device, non_blocking=True)

        if isinstance(meta_any, list) and len(meta_any) == B_expected:
            masks = []
            for m in meta_any:
                if not isinstance(m, dict):
                    return None
                cm = m.get("cortex_mask", None)
                if cm is None:
                    return None
                t = torch.from_numpy(cm.astype(np.float32))
                if t.dim() != 3:
                    return None
                masks.append(t.unsqueeze(0).unsqueeze(0))
            return torch.cat(masks, dim=0).to(device, non_blocking=True)

        return None

    def _meta_to_brain_mask(meta_any, B_expected: int) -> Optional[torch.Tensor]:
        if isinstance(meta_any, dict):
            bm = meta_any.get("brain_mask", None)
            if bm is None:
                return None
            t = torch.from_numpy(bm.astype(np.float32))
            if t.dim() == 3:
                t = t.unsqueeze(0).unsqueeze(0)
            return t.to(device, non_blocking=True)

        if isinstance(meta_any, list) and len(meta_any) == B_expected:
            masks = []
            for m in meta_any:
                if not isinstance(m, dict):
                    return None
                bm = m.get("brain_mask", None)
                if bm is None:
                    return None
                t = torch.from_numpy(bm.astype(np.float32))
                if t.dim() != 3:
                    return None
                masks.append(t.unsqueeze(0).unsqueeze(0))
            return torch.cat(masks, dim=0).to(device, non_blocking=True)

        return None

    def _masked_l1(fake5: torch.Tensor, target5: torch.Tensor, mask5: torch.Tensor) -> torch.Tensor:
        diff = (fake5 - target5).abs() * mask5
        num = diff.sum(dim=(1, 2, 3, 4))
        den = mask5.sum(dim=(1, 2, 3, 4)) + 1e-6
        return (num / den).mean()

    for epoch in range(1, epochs + 1):
        contrast_on_epoch = (epoch > int(e_normal1)) and (epoch <= (int(e_normal1) + int(e_contrast)))

        # ---- Per-epoch accumulators ----
        recon_running = 0.0
        roi_running = 0.0
        gan_running = 0.0

        contrast_running = 0.0
        delta_ctx_running = 0.0
        delta_out_running = 0.0
        n_contrast = 0

        # ---- FAMO monitoring (averaged per epoch) ----
        z_running = np.zeros(3, dtype=np.float64)
        w_running = np.zeros(3, dtype=np.float64)
        dlog_running = np.zeros(3, dtype=np.float64)
        ct_running = 0.0
        weighted_running = 0.0
        n_famo = 0

        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, meta = batch
            else:
                raise ValueError("There is something wrong happened when passing data")

            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)

            mri5 = mri if mri.dim() == 5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim() == 5 else pet.unsqueeze(0)

            # ===================== Update D (unchanged) =====================
            for p in D.parameters():
                p.requires_grad_(True)

            with torch.no_grad():
                fake_D = G(mri5)

            D.zero_grad(set_to_none=True)
            pair_real = torch.cat([mri5, pet5], dim=1)
            pair_fake = torch.cat([mri5, fake_D.detach()], dim=1)

            out_real = D(pair_real)
            out_fake = D(pair_fake)

            loss_D_real = adv_criterion(out_real, torch.ones_like(out_real))
            loss_D_fake = adv_criterion(out_fake, torch.zeros_like(out_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_D.step()

            # ===================== Update G (FAMO replaces MGDA) =====================
            for p in D.parameters():
                p.requires_grad_(False)

            mri5 = mri5.detach()
            fake = G(mri5)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            # GAN objective (as-is)
            loss_gan = 0.5 * adv_criterion(out_fake_for_G, torch.ones_like(out_fake_for_G))

            # Global recon objective (as-is)
            loss_l1 = l1_loss(fake, pet5)
            ssim_val = ssim3d(fake, pet5, data_range=data_range)
            loss_recon_global = gamma * (loss_l1 + (1.0 - ssim_val))

            # ROI/cortex recon objective (masked L1)
            cortex5 = _meta_to_cortex_mask(meta, B_expected=(mri5.size(0)))
            use_roi = (cortex5 is not None) and (float(cortex5.sum().item()) > 0.0)
            if use_roi:
                loss_recon_roi = _masked_l1(fake, pet5, cortex5)
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=fake.dtype)

            # ---- FAMO weights over the 3 losses ----
            losses_t = torch.stack([loss_recon_global, loss_recon_roi, loss_gan])
            losses_t_det = losses_t.detach().clamp(min=float(famo_eps))

            z = torch.softmax(xi, dim=0)  # simplex over tasks
            inv = z / losses_t_det        # z_i / ℓ_i
            c_t = 1.0 / (inv.sum() + 1e-12)
            w = (c_t * inv).detach()      # convex weights on task gradients (sum=1)

            loss_weighted = w[0] * loss_recon_global + w[1] * loss_recon_roi + w[2] * loss_gan

            opt_G.zero_grad(set_to_none=True)
            loss_weighted.backward()

            # ---- Contrast regularizer block (UNCHANGED) ----
            loss_contrast = torch.zeros((), device=device, dtype=fake.dtype)
            delta_ctx = torch.zeros((), device=device, dtype=fake.dtype)
            delta_out = torch.zeros((), device=device, dtype=fake.dtype)

            if (
                contrast_on_epoch
                and use_roi
                and float(contrast_eps) > 0.0
                and ((float(lambda_contrast_out) > 0.0) or (float(lambda_contrast_ctx) > 0.0))
            ):
                brain5 = _meta_to_brain_mask(meta, B_expected=mri5.size(0))
                if brain5 is None:
                    brain5 = (pet5 > 0).float()

                noise = torch.sign(torch.randn_like(mri5))
                mri_pert = (mri5 + float(contrast_eps) * noise * cortex5).detach()

                fake_pert = G(mri_pert)
                fake_anchor = fake.detach()

                out_mask = brain5 * (1.0 - cortex5)

                delta_ctx = _masked_l1(fake_pert, fake_anchor, cortex5)
                delta_out = _masked_l1(fake_pert, fake_anchor, out_mask)

                loss_contrast = (
                    float(lambda_contrast_out) * delta_out
                    + float(lambda_contrast_ctx) * (-torch.log(delta_ctx + float(contrast_eps0)))
                )

                loss_contrast.backward()

                contrast_running += float(loss_contrast.detach().item())
                delta_ctx_running += float(delta_ctx.detach().item())
                delta_out_running += float(delta_out.detach().item())
                n_contrast += 1

            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # Re-enable D grads for next iteration
            for p in D.parameters():
                p.requires_grad_(True)

            # ---- FAMO ξ update needs losses at t+1 (recompute on same batch) ----
            with torch.no_grad():
                fake_after = G(mri5)
                out_after = D(torch.cat([mri5, fake_after], dim=1))
                loss_gan_after = 0.5 * adv_criterion(out_after, torch.ones_like(out_after))

                loss_l1_after = l1_loss(fake_after, pet5)
                ssim_after = ssim3d(fake_after, pet5, data_range=data_range)
                loss_global_after = gamma * (loss_l1_after + (1.0 - ssim_after))

                if use_roi:
                    loss_roi_after = _masked_l1(fake_after, pet5, cortex5)
                else:
                    loss_roi_after = torch.zeros((), device=device, dtype=fake_after.dtype)

                losses_after = torch.stack([loss_global_after, loss_roi_after, loss_gan_after]).clamp(min=float(famo_eps))
                delta_log = (torch.log(losses_t_det) - torch.log(losses_after)).detach()

            # δ = J_softmax(ξ)^T * Δlogℓ  (compute via autograd)
            xi_var = xi.detach().clone().requires_grad_(True)
            z_var = torch.softmax(xi_var, dim=0)
            s = torch.sum(z_var * delta_log.to(device=xi_var.device, dtype=xi_var.dtype))
            grad_xi = torch.autograd.grad(s, xi_var, retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                xi = xi - float(famo_beta) * (grad_xi + float(famo_decay) * xi)

            # ---- Accumulate stats ----
            loss_G_log = (loss_recon_global + (loss_recon_roi if use_roi else 0.0) + loss_gan).detach().item()

            recon_running += float(loss_recon_global.detach().item())
            roi_running += float(loss_recon_roi.detach().item()) if use_roi else 0.0
            gan_running += float(loss_gan.detach().item())

            g_running += float(loss_G_log)
            d_running += float(loss_D.detach().item())
            n_batches += 1

            z_running += z.detach().cpu().numpy()
            w_running += w.detach().cpu().numpy()
            dlog_running += delta_log.detach().cpu().numpy()
            ct_running += float(c_t.detach().item())
            weighted_running += float(loss_weighted.detach().item())
            n_famo += 1

        # ---- Epoch aggregates ----
        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)

        avg_recon = recon_running / max(1, n_batches)
        avg_roi = roi_running / max(1, n_batches)
        avg_gan = gan_running / max(1, n_batches)

        avg_contrast = contrast_running / max(1, n_contrast) if n_contrast > 0 else 0.0
        avg_delta_ctx = delta_ctx_running / max(1, n_contrast) if n_contrast > 0 else 0.0
        avg_delta_out = delta_out_running / max(1, n_contrast) if n_contrast > 0 else 0.0

        avg_z = z_running / max(1, n_famo)
        avg_w = w_running / max(1, n_famo)
        avg_dlog = dlog_running / max(1, n_famo)
        avg_ct = ct_running / max(1, n_famo)
        avg_weighted = weighted_running / max(1, n_famo)

        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)

        # ---- Validation (UNCHANGED) ----
        val_recon_epoch: Optional[float] = None

        if val_loader is not None:
            G.eval()
            w_roi_val = 1.0

            with torch.no_grad():
                val_combo_sum = 0.0
                val_global_sum = 0.0
                val_roi_sum = 0.0
                v_batches = 0

                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri, pet, meta = batch
                    else:
                        mri, pet = batch
                        meta = None

                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)

                    fake = G(mri if mri.dim() == 5 else mri.unsqueeze(0))
                    pet_for_metric = pet if pet.dim() == 5 else pet.unsqueeze(0)

                    loss_l1_v = l1_loss(fake, pet_for_metric)
                    ssim_v = ssim3d(fake, pet_for_metric, data_range=data_range)
                    loss_global_v = (loss_l1_v + (1.0 - ssim_v))

                    loss_roi_v = torch.zeros((), device=device, dtype=fake.dtype)
                    if meta is not None:
                        cortex5_v = _meta_to_cortex_mask(meta, B_expected=fake.size(0))
                        if (cortex5_v is not None) and (float(cortex5_v.sum().item()) > 0.0):
                            loss_roi_v = _masked_l1(fake, pet_for_metric, cortex5_v)

                    loss_combo_v = loss_global_v + w_roi_val * loss_roi_v

                    val_global_sum += float(loss_global_v.item())
                    val_roi_sum += float(loss_roi_v.item())
                    val_combo_sum += float(loss_combo_v.item())
                    v_batches += 1

            val_combo = val_combo_sum / max(1, v_batches)
            val_global = val_global_sum / max(1, v_batches)
            val_roi = val_roi_sum / max(1, v_batches)

            val_recon = val_combo
            val_recon_epoch = val_recon
            hist["val_recon"].append(val_recon)

            if val_recon < best_val:
                best_val = val_recon
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))

            if verbose:
                dt = time.time() - t0
                print(
                    f"Epoch [{epoch:03d}/{epochs}]  "
                    f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                    f"ValCombo(Global + ROI, w=1): {val_recon:.4f}  "
                    f"(Global={val_global:.4f}, ROI={val_roi:.4f})  "
                    f"| best {best_val:.4f}  | {dt:.1f}s"
                )
                print(
                    f"      [Loss] recon={avg_recon:.4f}  roi={avg_roi:.4f}  gan={avg_gan:.4f}  "
                    f"contrast={'ON' if contrast_on_epoch else 'OFF'}  "
                    f"Lc={avg_contrast:.4f}  Δctx={avg_delta_ctx:.4f}  Δout={avg_delta_out:.4f}"
                )
                print(
                    f"      [FAMO] z=[{avg_z[0]:.3f},{avg_z[1]:.3f},{avg_z[2]:.3f}]  "
                    f"w=[{avg_w[0]:.3f},{avg_w[1]:.3f},{avg_w[2]:.3f}]  "
                    f"ct={avg_ct:.3e}  "
                    f"Δlog=[{avg_dlog[0]:.3e},{avg_dlog[1]:.3e},{avg_dlog[2]:.3e}]  "
                    f"xi=[{float(xi[0].item()):.3f},{float(xi[1].item()):.3f},{float(xi[2].item()):.3f}]"
                )

            G.train()

        elif verbose:
            dt = time.time() - t0
            print(f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  | {dt:.1f}s")
            print(
                f"      [Loss] recon={avg_recon:.4f}  roi={avg_roi:.4f}  gan={avg_gan:.4f}  "
                f"contrast={'ON' if contrast_on_epoch else 'OFF'}  "
                f"Lc={avg_contrast:.4f}  Δctx={avg_delta_ctx:.4f}  Δout={avg_delta_out:.4f}"
            )
            print(
                f"      [FAMO] z=[{avg_z[0]:.3f},{avg_z[1]:.3f},{avg_z[2]:.3f}]  "
                f"w=[{avg_w[0]:.3f},{avg_w[1]:.3f},{avg_w[2]:.3f}]  "
                f"ct={avg_ct:.3e}  "
                f"Δlog=[{avg_dlog[0]:.3e},{avg_dlog[1]:.3e},{avg_dlog[2]:.3e}]  "
                f"xi=[{float(xi[0].item()):.3f},{float(xi[1].item()):.3f},{float(xi[2].item()):.3f}]"
            )

        if log_to_wandb and wandb.run is not None:
            log_dict = {
                "epoch": epoch,

                # your existing loss logging
                "train/G_loss": avg_g,
                "train/D_loss": avg_d,
                "train/recon_global": avg_recon,
                "train/recon_roi": avg_roi,
                "train/gan": avg_gan,

                # contrast logs (unchanged)
                "contrast/active": int(contrast_on_epoch),
                "contrast/loss": avg_contrast,
                "contrast/delta_ctx": avg_delta_ctx,
                "contrast/delta_out": avg_delta_out,
                "contrast/eps": float(contrast_eps),
                "contrast/lambda_out": float(lambda_contrast_out),
                "contrast/lambda_ctx": float(lambda_contrast_ctx),
                "contrast/e_normal1": int(e_normal1),
                "contrast/e_contrast": int(e_contrast),

                # ---- FAMO logs ----
                "famo/beta": float(famo_beta),
                "famo/decay": float(famo_decay),
                "famo/eps": float(famo_eps),

                "famo/z_global": float(avg_z[0]),
                "famo/z_roi": float(avg_z[1]),
                "famo/z_gan": float(avg_z[2]),

                "famo/w_global": float(avg_w[0]),
                "famo/w_roi": float(avg_w[1]),
                "famo/w_gan": float(avg_w[2]),

                "famo/ct": float(avg_ct),
                "famo/weighted_loss": float(avg_weighted),

                "famo/dlog_global": float(avg_dlog[0]),
                "famo/dlog_roi": float(avg_dlog[1]),
                "famo/dlog_gan": float(avg_dlog[2]),

                "famo/xi_global": float(xi[0].item()),
                "famo/xi_roi": float(xi[1].item()),
                "famo/xi_gan": float(xi[2].item()),
            }
            if val_recon_epoch is not None:
                log_dict["val/recon_loss"] = val_recon_epoch
                log_dict["val/best_recon_loss"] = best_val
            wandb.log(log_dict, step=epoch)

    # ---- Load best weights ----
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
        fake = G(mri if mri.dim() == 5 else mri.unsqueeze(0))

        pet_for_metric = pet if pet.dim() == 5 else pet.unsqueeze(0)

        # masked metrics
        brain_mask_np = meta.get("brain_mask", None) if isinstance(meta, dict) else None
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            raise TypeError("No Mask")

        fake_m = fake * brain
        pet_m = pet_for_metric * brain

        ssim_sum += ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_sum += psnr(fake_m, pet_m, data_range=data_range)
        mse_sum += F.mse_loss(fake_m, pet_m).item()
        mmd_sum += mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)
        n += 1

    return {
        "SSIM": ssim_sum / max(1, n),
        "PSNR": psnr_sum / max(1, n),
        "MSE": mse_sum / max(1, n),
        "MMD": mmd_sum / max(1, n),
    }


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
        fake_t = G(mri_t if mri_t.dim() == 5 else mri_t.unsqueeze(0))

        pet_for_metric = pet_t if pet_t.dim() == 5 else pet_t.unsqueeze(0)

        # mask (prefer brain mask from meta; else pet>0)
        brain_mask_np = meta.get("brain_mask", None)
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake_t * brain
        pet_m = pet_for_metric * brain

        # per-subject metrics
        ssim_val = ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_val = psnr(fake_m, pet_m, data_range=data_range)
        mse_val = F.mse_loss(fake_m, pet_m).item()
        mmd_val = mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)

        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        mse_list.append(mse_val)
        mmd_list.append(mmd_val)

        # Save volumes (same as before)
        mri_np = (mri_t if mri_t.dim() == 5 else mri_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        pet_np = (pet_t if pet_t.dim() == 5 else pet_t.unsqueeze(0)).squeeze(0).squeeze(0).detach().cpu().numpy()
        fake_np = fake_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        err_np = np.abs(fake_np - pet_np)

        cur_shape = tuple(mri_np.shape)
        orig_shape = tuple(meta.get("orig_shape", cur_shape))

        if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
            zf = (float(orig_shape[0]) / float(cur_shape[0]),
                  float(orig_shape[1]) / float(cur_shape[1]),
                  float(orig_shape[2]) / float(cur_shape[2]))
            mri_np = nd_zoom(mri_np, zf, order=1)
            pet_np = nd_zoom(pet_np, zf, order=1)
            fake_np = nd_zoom(fake_np, zf, order=1)
            err_np = nd_zoom(err_np, zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            resized_to = meta.get("resized_to", None)
            affine_to_use = meta.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

        _save_nifti(mri_np, affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np, affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np, affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

    # ---- Aggregate + CI (keep old keys as means) ----
    def _mean_std_ci(vals):
        a = np.asarray(vals, dtype=np.float64)
        n = a.size
        mean = float(a.mean()) if n > 0 else float("nan")
        std = float(a.std(ddof=1)) if n > 1 else float("nan")
        se = (std / np.sqrt(n)) if n > 1 else float("nan")
        tcrit = _tcrit(n - 1)
        lo = mean - tcrit * se if n > 1 else float("nan")
        hi = mean + tcrit * se if n > 1 else float("nan")
        return mean, std, n, lo, hi

    m_ssim, sd_ssim, n_ssim, lo_ssim, hi_ssim = _mean_std_ci(ssim_list)
    m_psnr, sd_psnr, n_psnr, lo_psnr, hi_psnr = _mean_std_ci(psnr_list)
    m_mse, sd_mse, n_mse, lo_mse, hi_mse = _mean_std_ci(mse_list)
    m_mmd, sd_mmd, n_mmd, lo_mmd, hi_mmd = _mean_std_ci(mmd_list)

    # Per-subject CSV in run directory (sits next to 'volumes/')
    per_subj_csv = os.path.join(run_dir, "per_subject_metrics.csv")
    with open(per_subj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "SSIM", "PSNR", "MSE", "MMD"])
        for sid, ssim_v, psnr_v, mse_v, mmd_v in zip(sids, ssim_list, psnr_list, mse_list, mmd_list):
            w.writerow([sid, ssim_v, psnr_v, mse_v, mmd_v])

    # Also write a machine-readable summary JSON (optional, handy later)
    summary_json = os.path.join(run_dir, "test_metrics_summary.json")
    summary = {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE": m_mse, "MSE_std": sd_mse, "MSE_lo95": lo_mse, "MSE_hi95": hi_mse,
        "MMD": m_mmd, "MMD_std": sd_mmd, "MMD_lo95": lo_mmd, "MMD_hi95": hi_mmd,
        "per_subject_csv": per_subj_csv,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Keep backward-compatible keys as means for your main() writer
    return {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE": m_mse, "MSE_std": sd_mse, "MSE_lo95": lo_mse, "MSE_hi95": hi_mse,
        "MMD": m_mmd, "MMD_std": sd_mmd, "MMD_lo95": lo_mmd, "MMD_hi95": hi_mmd,
        "per_subject_csv": per_subj_csv,
        "summary_json": summary_json,
    }



