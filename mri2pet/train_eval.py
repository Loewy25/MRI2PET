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



            # ---- Update G ----
            G.zero_grad(set_to_none=True)

            # forward through G and D (for GAN loss)
            fake = G(mri5)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            # losses (two objectives): Recon = gamma*(L1 + (1-SSIM)), GAN = BCE
            loss_gan   = bce(out_fake_for_G, torch.ones_like(out_fake_for_G))
            loss_l1    = l1_loss(fake, pet5)
            ssim_val   = ssim3d(fake, pet5, data_range=data_range)
            loss_recon = gamma * (loss_l1 + (1.0 - ssim_val))

            # ========================= MGDA-UB (Two tasks) =========================
            # Compute output-space gradients wrt 'fake' for each objective
            v1 = torch.autograd.grad(loss_recon, fake, retain_graph=True)[0]  # ∇_fake L_recon
            v2 = torch.autograd.grad(loss_gan,   fake, retain_graph=True)[0]  # ∇_fake L_gan

            # L2-normalize to avoid scale bias between objectives
            eps = 1e-12
            v1n = v1 / (v1.norm() + eps)
            v2n = v2 / (v2.norm() + eps)

            # Closed-form α* for 2 tasks (per-sample), then take robust median
            V1 = v1n.reshape(v1n.size(0), -1)
            V2 = v2n.reshape(v2n.size(0), -1)
            diff = (V2 - V1)
            num  = (diff * V2).sum(dim=1)                 # (v2 - v1) · v2
            den  = (diff * diff).sum(dim=1) + eps         # ||v1 - v2||^2
            alpha_batch = torch.clamp(num / den, 0.0, 1.0)
            alpha = alpha_batch.median()                  # scalar α* for this batch

            # Combined output-space direction and single backward through G
            v_comb = alpha * v1n + (1.0 - alpha) * v2n
            opt_G.zero_grad(set_to_none=True)
            fake.backward(v_comb)                         # ONE backward through G
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # For logging only (proxy scalar; not used for backward)
            loss_G_log = (loss_recon + loss_gan).detach().item()
            # =========================================================================

            # ------------------ Legacy gradient-ratio controller ------------------
            # (Commented out per your request; leave here for quick rollback)
            """
            # measure grad norms at last decoder conv params (cheap & stable)
            params_L = [G.out_conv.weight]
            if getattr(G.out_conv, "bias", None) is not None:
                params_L.append(G.out_conv.bias)

            grads_r = torch.autograd.grad(loss_recon, params_L, retain_graph=True, allow_unused=True)
            norm_recon_sq = 0.0
            for g in grads_r:
                if g is not None:
                    norm_recon_sq = norm_recon_sq + g.pow(2).sum()
            norm_recon = torch.sqrt(norm_recon_sq + 0.0)

            grads_g = torch.autograd.grad(loss_gan, params_L, retain_graph=True, allow_unused=True)
            norm_gan_sq = 0.0
            for g in grads_g:
                if g is not None:
                    norm_gan_sq = norm_gan_sq + g.pow(2).sum()
            norm_gan = torch.sqrt(norm_gan_sq + 0.0)

            # update dynamic lambda_g with EMA + clipping + trust-region
            if torch.isfinite(norm_recon) and torch.isfinite(norm_gan):
                raw = (norm_recon / (norm_gan + 1e-8)).item()
                raw = max(LAM_MIN, min(LAM_MAX, raw))  # clip
                lam_new = ema_beta * lambda_g + (1.0 - ema_beta) * raw
                if lambda_g > 0.0:
                    max_up = (1.0 + TRUST_TAU) * lambda_g
                    max_dn = (1.0 - TRUST_TAU) * lambda_g
                    lam_new = min(max(lam_new, max_dn), max_up)
                lambda_g = lam_new

            loss_G = loss_recon + lambda_g * loss_gan
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            loss_G_log = loss_G.detach().item()
            """
            # ----------------------------------------------------------------------

            g_running += float(loss_G_log)
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

