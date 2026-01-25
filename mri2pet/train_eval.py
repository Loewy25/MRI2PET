import os, time
from typing import Any, Dict, Iterable, Optional
import numpy as np
from scipy.ndimage import zoom as nd_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    EPOCHS, GAMMA, LAMBDA_GAN, DATA_RANGE,
    LR_G, LR_D, CKPT_DIR, RESAMPLE_BACK_TO_T1,
    AUG_ENABLE, AUG_PROB, AUG_FLIP_PROB,
    AUG_INTENSITY_PROB, AUG_NOISE_STD,
    AUG_SCALE_MIN, AUG_SCALE_MAX,
    AUG_SHIFT_MIN, AUG_SHIFT_MAX,
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
    lambda_gan: float = LAMBDA_GAN,
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

    # === Global Gradient‑Ratio Controller (dynamic lambda_g) ===
    lambda_g = float(lambda_gan)  # (kept as-is; not used in MGDA path, but left unchanged)

    best_val = float("inf")
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {"train_G": [], "train_D": [], "val_recon": []}

    # =========================================================================
    # [MGDA-UB STANDARDIZATION INIT]
    # We now have:
    #   - global recon gradient (v_global)
    #   - ROI/cortex recon gradient (v_roi)
    #   - GAN gradient (v_gan)
    # and we do a single-stage 3-way MGDA on these three.
    # =========================================================================
    avg_norm_recon_global = 0.0
    avg_norm_recon_roi = 0.0
    avg_norm_gan = 0.0
    norm_decay = 0.9

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

    def _masked_l1(fake5: torch.Tensor, pet5: torch.Tensor, mask5: torch.Tensor) -> torch.Tensor:
        """
        Masked mean absolute error over mask voxels.
        fake5, pet5, mask5: [B,1,D,H,W]
        """
        diff = (fake5 - pet5).abs() * mask5
        num = diff.sum(dim=(1, 2, 3, 4))
        den = mask5.sum(dim=(1, 2, 3, 4)) + 1e-6
        return (num / den).mean()

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
    # MGDA-UB (3-way) utilities
    # -----------------------------
    def _mgda_weights_from_gram_3(Gm: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Solve min || sum_i w_i g_i ||^2 s.t. w_i >= 0, sum w_i = 1 for 3 objectives,
        given Gram matrix Gm[i,j] = <g_i, g_j>.
        Active-set enumeration: interior + edges + vertices.
        Returns: w tensor shape [3] (global, roi, gan).
        """
        dev, dtype = Gm.device, Gm.dtype
        one = torch.ones(3, device=dev, dtype=dtype)

        def _obj(w: torch.Tensor) -> torch.Tensor:
            return torch.dot(w, Gm @ w)

        candidates = []

        # Interior candidate (if feasible)
        try:
            try:
                pinv = torch.linalg.pinv(Gm)
            except Exception:
                pinv = torch.pinverse(Gm)
            w_int = pinv @ one
            denom = torch.dot(one, w_int)
            if torch.isfinite(denom) and float(denom.abs().item()) > eps:
                w_int = w_int / denom
                if torch.all(w_int >= -1e-6):
                    w_int = torch.clamp(w_int, min=0.0)
                    w_int = w_int / (w_int.sum() + eps)
                    candidates.append(w_int)
        except Exception:
            pass

        # Edge candidates (two objectives active)
        def _edge(i: int, j: int) -> torch.Tensor:
            Gii = Gm[i, i]
            Gjj = Gm[j, j]
            Gij = Gm[i, j]
            num = (Gjj - Gij)
            den = (Gii + Gjj - 2.0 * Gij) + eps
            a = torch.clamp(num / den, 0.0, 1.0)
            w = torch.zeros(3, device=dev, dtype=dtype)
            w[i] = a
            w[j] = 1.0 - a
            return w

        candidates += [_edge(0, 1), _edge(0, 2), _edge(1, 2)]

        # Vertices
        candidates += [
            torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dtype),
            torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype),
        ]

        best_w = candidates[0]
        best_val = _obj(best_w)
        for w in candidates[1:]:
            v = _obj(w)
            if float(v.item()) < float(best_val.item()):
                best_val = v
                best_w = w

        best_w = torch.clamp(best_w, min=0.0)
        best_w = best_w / (best_w.sum() + eps)
        return best_w

    def _mgda_weights_3(Vg: torch.Tensor, Vr: torch.Tensor, Vgan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Vg, Vr, Vgan: [B, N] flattened standardized gradients.
        Returns: weights per-sample [B, 3] = (w_global, w_roi, w_gan)
        """
        B = Vg.size(0)
        ws = []
        for b in range(B):
            g0 = Vg[b]
            g1 = Vr[b]
            g2 = Vgan[b]

            Gm = torch.empty((3, 3), device=Vg.device, dtype=Vg.dtype)
            Gm[0, 0] = torch.dot(g0, g0)
            Gm[0, 1] = torch.dot(g0, g1)
            Gm[0, 2] = torch.dot(g0, g2)
            Gm[1, 0] = Gm[0, 1]
            Gm[1, 1] = torch.dot(g1, g1)
            Gm[1, 2] = torch.dot(g1, g2)
            Gm[2, 0] = Gm[0, 2]
            Gm[2, 1] = Gm[1, 2]
            Gm[2, 2] = torch.dot(g2, g2)

            ws.append(_mgda_weights_from_gram_3(Gm, eps=eps))

        return torch.stack(ws, dim=0)

    for epoch in range(1, epochs + 1):
        # --- MGDA monitoring accumulators (per epoch) ---
        w_global_running = 0.0
        w_roi_running = 0.0
        w_gan_running = 0.0

        grad_recon_running = 0.0          # raw ||grad_global||
        grad_roi_running = 0.0            # raw ||grad_roi||
        grad_gan_running = 0.0            # raw ||grad_gan||

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

            # ROI/cortex recon objective (masked L1)
            use_roi = (cortex5 is not None) and (float(cortex5.sum().item()) > 0.0)
            if use_roi:
                loss_recon_roi = _masked_l1(fake, pet5, cortex5)
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=fake.dtype)

            # ========================= MGDA-UB (3-way, single stage) =========================

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
                # 3-way MGDA weights (per-sample), then robust median across batch
                w_batch = _mgda_weights_3(Vg, Vr, Vgan)         # [B,3]
                w_med = w_batch.median(dim=0).values            # [3]
                w_sum = (w_med.sum() + 1e-12)
                w_med = w_med / w_sum
                w_global, w_roi, w_gan_w = w_med[0], w_med[1], w_med[2]
            else:
                # fallback: 2-way MGDA between global recon and GAN
                diff = Vgan - Vg
                num = (diff * Vgan).sum(dim=1)
                den = (diff * diff).sum(dim=1) + 1e-12
                a_batch = torch.clamp(num / den, 0.0, 1.0)
                a = a_batch.median()
                w_global = a
                w_roi = torch.tensor(0.0, device=device, dtype=fake.dtype)
                w_gan_w = 1.0 - a

            # Monitoring
            w_global_running += float(w_global.item())
            w_roi_running += float(w_roi.item())
            w_gan_running += float(w_gan_w.item())

            grad_recon_running += current_nglobal
            grad_roi_running += current_nroi
            grad_gan_running += current_ngan

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

        avg_w_global = w_global_running / max(1, n_batches)
        avg_w_roi = w_roi_running / max(1, n_batches)
        avg_w_gan = w_gan_running / max(1, n_batches)

        avg_grad_recon = grad_recon_running / max(1, n_batches)
        avg_grad_roi = grad_roi_running / max(1, n_batches)
        avg_grad_gan = grad_gan_running / max(1, n_batches)

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
                    f"| best {best_val:.4f}  | λ_g={lambda_g:.4f}  | {dt:.1f}s"
                )
                print(
                    f"      [MGDA-UB-3] w_global={avg_w_global:.3f}  "
                    f"w_roi={avg_w_roi:.3f}  w_gan={avg_w_gan:.3f}  "
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
                f"      [MGDA-UB-3] w_global={avg_w_global:.3f}  "
                f"w_roi={avg_w_roi:.3f}  w_gan={avg_w_gan:.3f}  "
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

                # --- NEW: 3-way MGDA weights ---
                "mgda/w_recon_global": avg_w_global,
                "mgda/w_recon_roi": avg_w_roi,
                "mgda/w_gan": avg_w_gan,

                # keep grad norm tracking
                "mgda/grad_recon_global_norm": avg_grad_recon,
                "mgda/grad_recon_roi_norm": avg_grad_roi,
                "mgda/grad_gan_norm": avg_grad_gan,
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


