import os, time
from typing import Any, Dict, Iterable, Optional
import numpy as np
from scipy.ndimage import zoom as nd_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    EPOCHS, GAMMA, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1,
    AUG_ENABLE, AUG_PROB, AUG_FLIP_PROB,
    AUG_INTENSITY_PROB, AUG_NOISE_STD,
    AUG_SCALE_MIN, AUG_SCALE_MAX,
    AUG_SHIFT_MIN, AUG_SHIFT_MAX,
    ROI_HI_Q, ROI_HI_LAMBDA, ROI_HI_MIN_VOXELS,
)

from .losses import l1_loss, ssim3d, psnr, mmd_gaussian
from .utils import _safe_name, _save_nifti, _meta_unbatch
import wandb


def train_paggan(
    G: nn.Module,
    D: nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    device: torch.device,
    epochs: int = EPOCHS,
    gamma: float = GAMMA,
    data_range: float = DATA_RANGE,
    verbose: bool = True,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
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

    # =========================================================================
    # [CAGrad STANDARDIZATION INIT]
    # Objectives:
    #   - global recon gradient (v_global)
    #   - ROI/cortex recon gradient (v_roi)
    #   - GAN gradient (v_gan)
    # We standardize each objective gradient with EMA norm, then apply CAGrad.
    # =========================================================================
    avg_norm_recon_global = 0.0
    avg_norm_recon_roi = 0.0
    avg_norm_gan = 0.0
    norm_decay = 0.9
    cagrad_alpha = float(os.environ.get("CAGRAD_ALPHA", "0.5"))
    cagrad_rescale = int(os.environ.get("CAGRAD_RESCALE", "1"))

    def _meta_to_mask(meta_any, key: str, B_expected: int) -> Optional[torch.Tensor]:
        """
        Returns mask tensor [B,1,D,H,W] on device, or None if unavailable.
        Supports meta as dict (B=1) or list of dicts (B>1).
        """
        if isinstance(meta_any, dict):
            arr = meta_any.get(key, None)
            if arr is None:
                return None
            t = torch.from_numpy(arr.astype(np.float32))
            if t.dim() == 3:
                t = t.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
            return t.to(device, non_blocking=True)

        if isinstance(meta_any, list) and len(meta_any) == B_expected:
            masks = []
            for m in meta_any:
                if not isinstance(m, dict):
                    return None
                arr = m.get(key, None)
                if arr is None:
                    return None
                t = torch.from_numpy(arr.astype(np.float32))
                if t.dim() != 3:
                    return None
                masks.append(t.unsqueeze(0).unsqueeze(0))  # [1,1,D,H,W]
            return torch.cat(masks, dim=0).to(device, non_blocking=True)

        return None

    def _masked_l1_high_uptake(
        fake5: torch.Tensor,
        pet5: torch.Tensor,
        mask5: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cortex ROI loss with high-uptake emphasis:
          L = L1(cortex) + ROI_HI_LAMBDA * L1(top-quantile uptake in cortex)
        Quantile is computed from ground-truth PET within cortex mask, per sample.
        fake5, pet5, mask5: [B,1,D,H,W]
        """
        diff = (fake5 - pet5).abs()
        losses = []
        q = float(min(max(ROI_HI_Q, 0.0), 1.0))
        lambda_hi = float(ROI_HI_LAMBDA)
        min_vox = max(1, int(ROI_HI_MIN_VOXELS))

        for b in range(diff.size(0)):
            cortex = (mask5[b, 0] > 0.5)
            n_cortex = int(cortex.sum().item())
            if n_cortex == 0:
                losses.append(torch.zeros((), device=diff.device, dtype=diff.dtype))
                continue

            d_b = diff[b, 0]
            p_b = pet5[b, 0]

            l1_cortex = d_b[cortex].mean()

            if n_cortex < min_vox:
                hi_mask = cortex
            else:
                p_roi = p_b[cortex]
                thr = torch.quantile(p_roi, q)
                hi_mask = cortex & (p_b >= thr)
                if int(hi_mask.sum().item()) == 0:
                    hi_mask = cortex

            l1_high = d_b[hi_mask].mean()
            losses.append(l1_cortex + lambda_hi * l1_high)

        return torch.stack(losses).mean()

    def _maybe_augment_pair(
        mri5: torch.Tensor,
        pet5: torch.Tensor,
        brain5: Optional[torch.Tensor],
        cortex5: Optional[torch.Tensor],
    ):
        """
        Train-only augmentation:
          - paired random flips on MRI/PET/brain/cortex masks
          - MRI-only intensity jitter inside brain mask
        """
        if not AUG_ENABLE:
            return mri5, pet5, brain5, cortex5

        # apply augmentation to this batch with probability AUG_PROB
        if torch.rand((), device=mri5.device) > float(AUG_PROB):
            return mri5, pet5, brain5, cortex5

        # --- paired random flips (D/H/W axes) ---
        for dim in (-1, -2, -3):  # W, H, D in [B,1,D,H,W]
            if torch.rand((), device=mri5.device) < float(AUG_FLIP_PROB):
                mri5 = torch.flip(mri5, dims=(dim,))
                pet5 = torch.flip(pet5, dims=(dim,))
                if brain5 is not None:
                    brain5 = torch.flip(brain5, dims=(dim,))
                if cortex5 is not None:
                    cortex5 = torch.flip(cortex5, dims=(dim,))

        # --- MRI-only intensity augmentation (inside brain mask) ---
        if torch.rand((), device=mri5.device) < float(AUG_INTENSITY_PROB):
            B = mri5.size(0)
            dtype = mri5.dtype
            dev = mri5.device

            # per-sample scale/shift
            s = float(AUG_SCALE_MIN) + (float(AUG_SCALE_MAX) - float(AUG_SCALE_MIN)) * torch.rand((B, 1, 1, 1, 1), device=dev, dtype=dtype)
            b = float(AUG_SHIFT_MIN) + (float(AUG_SHIFT_MAX) - float(AUG_SHIFT_MIN)) * torch.rand((B, 1, 1, 1, 1), device=dev, dtype=dtype)
            noise = torch.randn_like(mri5) * float(AUG_NOISE_STD)

            if brain5 is not None:
                m = (brain5 > 0.5).to(dtype)
                mri5 = (mri5 * (1.0 - m)) + ((mri5 * s + b + noise) * m)
            else:
                mri5 = (mri5 * s + b + noise)

        return mri5, pet5, brain5, cortex5

    # -----------------------------
    # CAGrad utilities
    # -----------------------------
    def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the probability simplex."""
        n = int(v.numel())
        if n == 1:
            return torch.ones_like(v)
        u = torch.sort(v, descending=True).values
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        cond = (u - cssv / ind) > 0
        if not bool(cond.any()):
            return torch.full_like(v, 1.0 / float(n))
        rho = int(cond.nonzero(as_tuple=False)[-1].item()) + 1
        theta = cssv[rho - 1] / float(rho)
        w = torch.clamp(v - theta, min=0.0)
        s = w.sum()
        if float(s.item()) <= 0.0:
            return torch.full_like(v, 1.0 / float(n))
        return w / s

    def _cagrad_coeffs_from_gram(
        Gm: torch.Tensor,
        calpha: float,
        rescale: int = 1,
        steps: int = 50,
        eps: float = 1e-12,
    ):
        """
        Solve CAGrad on simplex:
          min_x x^T G b + c * sqrt(x^T G x),  s.t. x in simplex,
        where b is uniform and c = calpha * ||g0||.

        Returns:
          x_simplex: simplex solution x
          coeff: effective coefficients for final direction
          lam: lambda term in CAGrad
        """
        dev, dtype = Gm.device, Gm.dtype
        T = int(Gm.size(0))
        b = torch.full((T,), 1.0 / float(T), device=dev, dtype=dtype)

        g0_norm = torch.sqrt(torch.clamp(Gm.mean(), min=0.0) + eps)
        c = float(calpha) * g0_norm

        # If c is effectively zero, fall back to average-gradient direction.
        if float(c.item()) <= 1e-12:
            x = b.clone()
            coeff = b.clone()
            lam = torch.tensor(0.0, device=dev, dtype=dtype)
            return x, coeff, lam

        Gb = Gm @ b

        def _obj(x: torch.Tensor) -> torch.Tensor:
            xGx = torch.dot(x, Gm @ x)
            return torch.dot(x, Gb) + c * torch.sqrt(torch.clamp(xGx, min=0.0) + eps)

        x = b.clone()
        lr = 0.2
        best_x = x
        best_obj = _obj(x)

        for _ in range(steps):
            Gx = Gm @ x
            xGx = torch.dot(x, Gx)
            grad = Gb + c * Gx / torch.sqrt(torch.clamp(xGx, min=0.0) + eps)

            x_try = _project_simplex(x - lr * grad)
            obj_try = _obj(x_try)

            if float(obj_try.item()) <= float(best_obj.item()):
                x = x_try
                best_x = x_try
                best_obj = obj_try
                lr = min(lr * 1.05, 1.0)
            else:
                lr = max(lr * 0.5, 1e-4)

        x = _project_simplex(best_x)
        xGx = torch.dot(x, Gm @ x)
        gw_norm = torch.sqrt(torch.clamp(xGx, min=0.0) + eps)
        lam = c / (gw_norm + eps)

        coeff = b + lam * x
        if rescale == 1:
            coeff = coeff / (1.0 + calpha ** 2)
        elif rescale == 2:
            coeff = coeff / (1.0 + calpha)
        return x, coeff, lam

    def _cagrad_weights(
        grad_views,
        calpha: float,
        rescale: int = 1,
        eps: float = 1e-12,
    ):
        """
        grad_views: list of flattened standardized gradients, each [B, N].
        Returns:
          simplex_batch: [B, T]
          coeff_batch:   [B, T] effective coefficients for final direction
          lambda_batch:  [B]
        """
        T = len(grad_views)
        B = grad_views[0].size(0)
        simplex_list = []
        coeff_list = []
        lambda_list = []

        for b in range(B):
            Gm = torch.empty((T, T), device=grad_views[0].device, dtype=grad_views[0].dtype)
            for i in range(T):
                gi = grad_views[i][b]
                for j in range(i, T):
                    gj = grad_views[j][b]
                    dot_ij = torch.dot(gi, gj)
                    Gm[i, j] = dot_ij
                    Gm[j, i] = dot_ij

            x, coeff, lam = _cagrad_coeffs_from_gram(
                Gm, calpha=calpha, rescale=rescale, eps=eps
            )
            simplex_list.append(x)
            coeff_list.append(coeff)
            lambda_list.append(lam)

        return (
            torch.stack(simplex_list, dim=0),
            torch.stack(coeff_list, dim=0),
            torch.stack(lambda_list, dim=0),
        )

    for epoch in range(1, epochs + 1):
        # --- CAGrad monitoring accumulators (per epoch) ---
        coeff_global_running = 0.0
        coeff_roi_running = 0.0
        coeff_gan_running = 0.0
        simplex_global_running = 0.0
        simplex_roi_running = 0.0
        simplex_gan_running = 0.0
        lambda_running = 0.0

        grad_recon_running = 0.0          # raw ||grad_global||
        grad_roi_running = 0.0            # raw ||grad_roi||
        grad_gan_running = 0.0            # raw ||grad_gan||
        loss_recon_global_running = 0.0
        loss_recon_roi_running = 0.0
        loss_gan_running = 0.0

        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, meta = batch
            else:
                raise ValueError("There is something wrong happened when passing data")

            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0) if mri.dim() == 5 else 1

            # ---- Update D ----
            mri5 = mri if mri.dim() == 5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim() == 5 else pet.unsqueeze(0)

            # get masks from meta (for augmentation + ROI loss)
            brain5 = _meta_to_mask(meta, "brain_mask",  B_expected=mri5.size(0))
            cortex5 = _meta_to_mask(meta, "cortex_mask", B_expected=mri5.size(0))

            # train-only augmentation (paired)
            mri5, pet5, brain5, cortex5 = _maybe_augment_pair(mri5, pet5, brain5, cortex5)

            with torch.no_grad():
                fake = G(mri5)

            D.zero_grad(set_to_none=True)
            pair_real = torch.cat([mri5, pet5], dim=1)
            pair_fake = torch.cat([mri5, fake.detach()], dim=1)

            out_real = D(pair_real)
            out_fake = D(pair_fake)

            loss_D_real = adv_criterion(out_real, torch.ones_like(out_real))
            loss_D_fake = adv_criterion(out_fake, torch.zeros_like(out_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_D.step()

            # ---- Update G ----
            G.zero_grad(set_to_none=True)

            fake = G(mri5)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            # GAN objective (as-is)
            loss_gan = 0.5 * adv_criterion(out_fake_for_G, torch.ones_like(out_fake_for_G))

            # Global recon objective (as-is)
            loss_l1 = l1_loss(fake, pet5)
            ssim_val = ssim3d(fake, pet5, data_range=data_range)
            loss_recon_global = gamma * (loss_l1 + (1.0 - ssim_val))

            # ROI/cortex recon objective (masked L1 + high-uptake weighted L1)
            use_roi = (cortex5 is not None) and (float(cortex5.sum().item()) > 0.0)
            if use_roi:
                loss_recon_roi = _masked_l1_high_uptake(fake, pet5, cortex5)
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=fake.dtype)

            # ========================= CAGrad (single stage) =========================

            # ---- global recon grad ----
            v_global = torch.autograd.grad(loss_recon_global, fake, retain_graph=True)[0]
            current_nglobal = v_global.norm().item()
            if avg_norm_recon_global == 0:
                avg_norm_recon_global = current_nglobal
            else:
                avg_norm_recon_global = norm_decay * avg_norm_recon_global + (1 - norm_decay) * current_nglobal
            v_global_s = v_global / (avg_norm_recon_global + 1e-8)

            # ---- ROI recon grad ----
            if use_roi:
                v_roi = torch.autograd.grad(loss_recon_roi, fake, retain_graph=True)[0]
                current_nroi = v_roi.norm().item()
                if avg_norm_recon_roi == 0:
                    avg_norm_recon_roi = current_nroi
                else:
                    avg_norm_recon_roi = norm_decay * avg_norm_recon_roi + (1 - norm_decay) * current_nroi
                v_roi_s = v_roi / (avg_norm_recon_roi + 1e-8)
            else:
                v_roi_s = torch.zeros_like(v_global_s)
                current_nroi = 0.0

            # ---- GAN grad ----
            v_gan = torch.autograd.grad(loss_gan, fake, retain_graph=True)[0]
            current_ngan = v_gan.norm().item()
            if avg_norm_gan == 0:
                avg_norm_gan = current_ngan
            else:
                avg_norm_gan = norm_decay * avg_norm_gan + (1 - norm_decay) * current_ngan
            v_gan_s = v_gan / (avg_norm_gan + 1e-8)

            # Flatten standardized grads
            Vg = v_global_s.reshape(v_global_s.size(0), -1)
            Vr = v_roi_s.reshape(v_roi_s.size(0), -1)
            Vgan = v_gan_s.reshape(v_gan_s.size(0), -1)

            if use_roi:
                simplex_batch, coeff_batch, lambda_batch = _cagrad_weights(
                    [Vg, Vr, Vgan], calpha=cagrad_alpha, rescale=cagrad_rescale
                )
                simplex_med = simplex_batch.median(dim=0).values
                coeff_med = coeff_batch.median(dim=0).values
                lambda_med = lambda_batch.median()

                w_global, w_roi, w_gan_w = coeff_med[0], coeff_med[1], coeff_med[2]
                simplex_global, simplex_roi, simplex_gan = (
                    simplex_med[0], simplex_med[1], simplex_med[2]
                )
            else:
                simplex_batch, coeff_batch, lambda_batch = _cagrad_weights(
                    [Vg, Vgan], calpha=cagrad_alpha, rescale=cagrad_rescale
                )
                simplex_med = simplex_batch.median(dim=0).values
                coeff_med = coeff_batch.median(dim=0).values
                lambda_med = lambda_batch.median()

                w_global = coeff_med[0]
                w_roi = torch.tensor(0.0, device=device, dtype=fake.dtype)
                w_gan_w = coeff_med[1]
                simplex_global = simplex_med[0]
                simplex_roi = torch.tensor(0.0, device=device, dtype=fake.dtype)
                simplex_gan = simplex_med[1]

            # Monitoring
            coeff_global_running += float(w_global.item())
            coeff_roi_running += float(w_roi.item())
            coeff_gan_running += float(w_gan_w.item())
            simplex_global_running += float(simplex_global.item())
            simplex_roi_running += float(simplex_roi.item())
            simplex_gan_running += float(simplex_gan.item())
            lambda_running += float(lambda_med.item())

            grad_recon_running += current_nglobal
            grad_roi_running += current_nroi
            grad_gan_running += current_ngan
            loss_recon_global_running += float(loss_recon_global.detach().item())
            loss_recon_roi_running += float(loss_recon_roi.detach().item())
            loss_gan_running += float(loss_gan.detach().item())

            # Final direction
            v_final = (w_global * v_global_s) + (w_roi * v_roi_s) + (w_gan_w * v_gan_s)

            opt_G.zero_grad(set_to_none=True)
            fake.backward(v_final)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # For logging only (proxy scalar; not used for backward)
            loss_G_log = (loss_recon_global + (loss_recon_roi if use_roi else 0.0) + loss_gan).detach().item()

            g_running += float(loss_G_log)
            d_running += loss_D.item()
            n_batches += 1

        # ---- Epoch aggregates ----
        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)

        avg_coeff_global = coeff_global_running / max(1, n_batches)
        avg_coeff_roi = coeff_roi_running / max(1, n_batches)
        avg_coeff_gan = coeff_gan_running / max(1, n_batches)

        avg_simplex_global = simplex_global_running / max(1, n_batches)
        avg_simplex_roi = simplex_roi_running / max(1, n_batches)
        avg_simplex_gan = simplex_gan_running / max(1, n_batches)
        avg_lambda = lambda_running / max(1, n_batches)

        avg_grad_recon = grad_recon_running / max(1, n_batches)
        avg_grad_roi = grad_roi_running / max(1, n_batches)
        avg_grad_gan = grad_gan_running / max(1, n_batches)
        avg_loss_recon_global = loss_recon_global_running / max(1, n_batches)
        avg_loss_recon_roi = loss_recon_roi_running / max(1, n_batches)
        avg_loss_gan = loss_gan_running / max(1, n_batches)

        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)

        # ---- Validation ----
        val_recon_epoch: Optional[float] = None

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
                    fake = G(mri if mri.dim() == 5 else mri.unsqueeze(0))
                    pet_for_metric = pet if pet.dim() == 5 else pet.unsqueeze(0)
                    loss_l1_v = l1_loss(fake, pet_for_metric)
                    ssim_v = ssim3d(fake, pet_for_metric, data_range=data_range)
                    val_recon += (loss_l1_v + (1.0 - ssim_v)).item()
                    v_batches += 1
            val_recon /= max(1, v_batches)
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
                    f"ValRecon(L1 + 1-SSIM): {val_recon:.4f}  "
                    f"| best {best_val:.4f}  | {dt:.1f}s"
                )
                print(
                    f"      [CAGrad] coeff_global={avg_coeff_global:.3f}  "
                    f"coeff_roi={avg_coeff_roi:.3f}  coeff_gan={avg_coeff_gan:.3f}  "
                    f"| simplex=({avg_simplex_global:.3f},{avg_simplex_roi:.3f},{avg_simplex_gan:.3f})  "
                    f"| lambda={avg_lambda:.3f}  "
                    f"||grad_global||={avg_grad_recon:.3e}  "
                    f"||grad_roi||={avg_grad_roi:.3e}  "
                    f"||grad_gan||={avg_grad_gan:.3e}"
                )

            G.train()
        elif verbose:
            dt = time.time() - t0
            print(
                f"Epoch [{epoch:03d}/{epochs}]  "
                f"G: {avg_g:.4f}  D: {avg_d:.4f}  | {dt:.1f}s"
            )
            print(
                f"      [CAGrad] coeff_global={avg_coeff_global:.3f}  "
                f"coeff_roi={avg_coeff_roi:.3f}  coeff_gan={avg_coeff_gan:.3f}  "
                f"| simplex=({avg_simplex_global:.3f},{avg_simplex_roi:.3f},{avg_simplex_gan:.3f})  "
                f"| lambda={avg_lambda:.3f}  "
                f"||grad_global||={avg_grad_recon:.3e}  "
                f"||grad_roi||={avg_grad_roi:.3e}  "
                f"||grad_gan||={avg_grad_gan:.3e}"
            )

        # ---- wandb logging per epoch ----
        if log_to_wandb and wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "train/G_loss": avg_g,
                "train/D_loss": avg_d,
                "train/loss_recon_global": avg_loss_recon_global,
                "train/loss_recon_roi": avg_loss_recon_roi,
                "train/loss_gan": avg_loss_gan,

                # --- CAGrad effective coefficients used in final direction ---
                "cagrad/coeff_recon_global": avg_coeff_global,
                "cagrad/coeff_recon_roi": avg_coeff_roi,
                "cagrad/coeff_gan": avg_coeff_gan,

                # --- CAGrad simplex solution x ---
                "cagrad/simplex_recon_global": avg_simplex_global,
                "cagrad/simplex_recon_roi": avg_simplex_roi,
                "cagrad/simplex_gan": avg_simplex_gan,
                "cagrad/lambda": avg_lambda,
                "cagrad/alpha": cagrad_alpha,
                "cagrad/rescale_mode": float(cagrad_rescale),

                # keep gradient norm tracking
                "cagrad/grad_recon_global_norm": avg_grad_recon,
                "cagrad/grad_recon_roi_norm": avg_grad_roi,
                "cagrad/grad_gan_norm": avg_grad_gan,
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

        pet_for_metric = pet_t if pet_t.dim()==5 else pet_t.unsqueeze(0)

        # mask (prefer brain mask from meta; else pet>0)
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

        # Save volumes (same as before)
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
        "MSE":  m_mse,  "MSE_std":  sd_mse,  "MSE_lo95":  lo_mse,  "MSE_hi95":  hi_mse,
        "MMD":  m_mmd,  "MMD_std":  sd_mmd,  "MMD_lo95":  lo_mmd,  "MMD_hi95":  hi_mmd,
        "per_subject_csv": per_subj_csv,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Keep backward-compatible keys as means for your main() writer
    return {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE":  m_mse,  "MSE_std":  sd_mse,  "MSE_lo95":  lo_mse,  "MSE_hi95":  hi_mse,
        "MMD":  m_mmd,  "MMD_std":  sd_mmd,  "MMD_lo95":  lo_mmd,  "MMD_hi95":  hi_mmd,
        "per_subject_csv": per_subj_csv,
        "summary_json": summary_json,
    }
