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

        # === Global Gradient‑Ratio Controller (you already added) ===
    # lambda_g, ema_beta, LAM_MIN, LAM_MAX, EPS, TRUST_TAU are already defined above

    # === Feature-Matching (FM) dynamic weight controller ===
    nu_fm = 0.10                 # initial FM weight (kept modest)
    ema_beta_fm = 0.9            # EMA smoothing for ν_fm
    FM_MIN, FM_MAX = 0.01, 0.30  # guard rails: FM stays weaker than GAN
    TRUST_TAU_FM = 0.25          # trust-region: slower ν_fm changes than λ_g
    FM_TARGET = 0.3             # target strength ~0.5× recon (fine-tuner)

    # === Curriculum gate for FM (warmup then ramp) ===
    def fm_gate(epoch: int, total_epochs: int) -> float:
        # No FM for first 10 epochs, then ramp to 1.0 over next 30 epochs
        if epoch < 10:
            return 0.0
        elif epoch < 40:
            return (epoch - 10) / 30.0
        else:
            return 1.0


    best_val = float('inf')
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

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



            # ---- Update G (GAN with dynamic λ_g + Feature-Matching with dynamic ν_fm) ----
            G.zero_grad(set_to_none=True)

            # 1) Forward generator
            fake = G(mri5)

            # 2) Discriminator on FAKE (logits + features) for G's gradients
            out_fake_for_G, feats_fake = D.forward_with_feats(torch.cat([mri5, fake], dim=1))
            loss_gan = bce(out_fake_for_G, torch.ones_like(out_fake_for_G))

            # 3) Discriminator on REAL (features only; no grad into D)
            with torch.no_grad():
                out_real_for_FM, feats_real = D.forward_with_feats(torch.cat([mri5, pet5], dim=1))

            # 4) Feature-Matching loss with D-separability gating (no extra params)
            Lf = min(len(feats_fake), len(feats_real))
            EPS = 1e-8

            loss_fm_layers = []
            sep_scores = []

            for i in range(Lf):
                f_fake = feats_fake[i]               # requires grad
                f_real = feats_real[i].detach()      # from no_grad() call above; keep detached

                # Per-layer FM term: mean L1 across all elements (size-invariant)
                Li = (f_fake - f_real).abs().mean()
                loss_fm_layers.append(Li)

                # D-separability score s_i (stop-grad):
                # Compare per-(B,C) spatial means between real and fake, normalized by real energy.
                mu_r = f_real.mean(dim=(2,3,4))                # [B, C]
                mu_f = f_fake.detach().mean(dim=(2,3,4))       # [B, C]
                gap  = (mu_r - mu_f).abs().mean()              # scalar
                denom = f_real.abs().mean() + EPS              # scalar
                sep_scores.append((gap / denom).detach())
        
            if Lf > 0:
                # Convert scores to simplex weights w (sum=1, each>0)
                s = torch.stack(sep_scores) + 1e-6             # [L]
                w = s / s.sum()

                # Smooth and stabilize (EMA + trust region) to reduce B=1 jitter
                beta = 0.75   # EMA factor
                tau  = 0.25   # trust-region: at most ±25% change vs previous step
                if not hasattr(train_paggan, "_w_gate") or train_paggan._w_gate.numel() != Lf:
                    train_paggan._w_gate = w.clone()
                w = beta * train_paggan._w_gate + (1.0 - beta) * w
                w = (w / w.sum()).clamp_min(1e-6)
                w_up = (1.0 + tau) * train_paggan._w_gate
                w_dn = (1.0 - tau) * train_paggan._w_gate
                w = torch.max(torch.min(w, w_up), w_dn)
                w = (w / w.sum()).detach()           # keep weights as constants for backprop
                train_paggan._w_gate = w

                # Final gated FM loss: weighted sum over layers
                loss_fm = torch.sum(w * torch.stack(loss_fm_layers))
            else:
                loss_fm = torch.tensor(0.0, device=mri.device if mri.dim() == 5 else mri5.device)


            # 5) Reconstruction loss (same as before)
            loss_l1   = l1_loss(fake, pet5)
            ssim_val  = ssim3d(fake, pet5, data_range=data_range)
            loss_recon = gamma * (loss_l1 + (1.0 - ssim_val))

            # 6) Measure grad-norms at last generator conv (proxy subspace)
            params_L = [G.out_conv.weight]
            if getattr(G.out_conv, "bias", None) is not None:
                params_L.append(G.out_conv.bias)

            # recon grad‑norm
            grads_r = torch.autograd.grad(loss_recon, params_L, retain_graph=True, allow_unused=True)
            norm_recon_sq = 0.0
            for g in grads_r:
                if g is not None:
                    norm_recon_sq = norm_recon_sq + g.pow(2).sum()
            norm_recon = torch.sqrt(norm_recon_sq + 0.0)

            # gan grad‑norm
            grads_g = torch.autograd.grad(loss_gan, params_L, retain_graph=True, allow_unused=True)
            norm_gan_sq = 0.0
            for g in grads_g:
                if g is not None:
                    norm_gan_sq = norm_gan_sq + g.pow(2).sum()
            norm_gan = torch.sqrt(norm_gan_sq + 0.0)

            # 7) Update dynamic λ_g (GAN weight) via gradient-ratio vs recon
            if torch.isfinite(norm_recon) and torch.isfinite(norm_gan):
                raw = (norm_recon / (norm_gan + EPS)).item()
                raw = max(LAM_MIN, min(LAM_MAX, raw))  # clip
                lam_new = ema_beta * lambda_g + (1.0 - ema_beta) * raw
                if lambda_g > 0.0:  # trust-region (±50%)
                    max_up = (1.0 + TRUST_TAU) * lambda_g
                    max_dn = (1.0 - TRUST_TAU) * lambda_g
                    lam_new = min(max(lam_new, max_dn), max_up)
                lambda_g = lam_new

            # 8) FM grad‑norm (for dynamic ν_fm)
            grads_fm = torch.autograd.grad(loss_fm, params_L, retain_graph=True, allow_unused=True)
            norm_fm_sq = 0.0
            for g in grads_fm:
                if g is not None:
                    norm_fm_sq = norm_fm_sq + g.pow(2).sum()
            norm_fm = torch.sqrt(norm_fm_sq + 0.0)

            # 9) Update dynamic ν_fm (FM weight) vs recon (weaker target + stricter bounds)
            if torch.isfinite(norm_recon) and torch.isfinite(norm_fm) and norm_fm.item() > 0:
                raw_fm = (FM_TARGET * norm_recon / (norm_fm + EPS)).item()
                raw_fm = max(FM_MIN, min(FM_MAX, raw_fm))  # clip tighter than GAN
                nu_new = ema_beta_fm * nu_fm + (1.0 - ema_beta_fm) * raw_fm
                if nu_fm > 0.0:  # trust-region (±25%)
                    max_up = (1.0 + TRUST_TAU_FM) * nu_fm
                    max_dn = (1.0 - TRUST_TAU_FM) * nu_fm
                    nu_new = min(max(nu_new, max_dn), max_up)
                nu_fm = nu_new
            # else: keep previous nu_fm

            # 10) Final generator loss: recon + λ_g*GAN + gated ν_fm*FM
            nu_fm_effective = fm_gate(epoch, epochs) * nu_fm
            loss_G = loss_recon + lambda_g * loss_gan + nu_fm_effective * loss_fm

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()




            g_running += loss_G.item()
            d_running += loss_D.item()
            n_batches += 1

        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)
        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)

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
                      f"| best {best_val:.4f}  | λ_g={lambda_g:.4f} ν_fm={nu_fm:.4f} | {dt:.1f}s")
                    # Optional: show D-gate weights every 5 epochs
                if (epoch % 5 == 0) and hasattr(train_paggan, "_w_gate"):
                    w_str = ", ".join(f"{float(x):.3f}" for x in train_paggan._w_gate.detach().cpu())
                    print(f"FM w_gate (per D-layer): [{w_str}]")


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
