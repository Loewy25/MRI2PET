import os, time, csv
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
    FREEZE_BASE_EPOCHS, BASE_LR_MULT,
    LAMBDA_STAGE_ORD, LAMBDA_BRAAK, LAMBDA_DELTA_OUT,
    USE_GT_STAGE_HINT_TRAIN, MASK_GLOBAL_RECON,
    LR_PLATEAU_PATIENCE, EARLY_STOP_PATIENCE,
    AMP_ENABLE, USE_CHECKPOINT, VAL_ROI_WEIGHT,
    ABLATION_STEP,
)

from .losses import l1_loss, ssim3d, psnr, mmd_gaussian
from .utils import _safe_name, _save_nifti, _meta_unbatch
from .models import build_stage_onehot, ordinal_logits_to_stage_probs
import wandb


# =========================================================================
# Meta helpers (shared)
# =========================================================================
def _meta_as_list(meta_any, B: int) -> List[Dict[str, Any]]:
    """Normalize meta (dict for B=1 or list[dict]) into a list of length B."""
    if isinstance(meta_any, dict):
        return [meta_any]
    if isinstance(meta_any, list):
        return meta_any
    return [{}] * B


def _meta_to_tensor(metas: List[Dict[str, Any]], key: str, device: torch.device,
                    dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    """Stack a per-sample key from meta list into a batched tensor."""
    vals = []
    for m in metas:
        v = m.get(key, None)
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            vals.append(torch.from_numpy(v))
        elif isinstance(v, (int, float)):
            vals.append(torch.tensor(v))
        elif isinstance(v, torch.Tensor):
            vals.append(v)
        else:
            return None
    return torch.stack(vals, dim=0).to(device=device, dtype=dtype, non_blocking=True)


def _extract_masks(metas: List[Dict[str, Any]], device: torch.device
                   ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract brain and cortex masks as [B,1,D,H,W] tensors."""
    brain_list, cortex_list = [], []
    for m in metas:
        ba = m.get("brain_mask", None)
        ca = m.get("cortex_mask", None)
        if ba is None:
            return None, None
        brain_list.append(torch.from_numpy(ba.astype(np.float32)).unsqueeze(0).unsqueeze(0))
        if ca is not None:
            cortex_list.append(torch.from_numpy(ca.astype(np.float32)).unsqueeze(0).unsqueeze(0))

    brain5 = torch.cat(brain_list, dim=0).to(device, non_blocking=True)
    cortex5 = torch.cat(cortex_list, dim=0).to(device, non_blocking=True) if len(cortex_list) == len(metas) else None
    return brain5, cortex5


def _extract_new_variant_inputs(metas: List[Dict[str, Any]], device: torch.device):
    """Extract flair, clinical_vector, stage_ord, braak_values from meta list."""
    flair_list, clin_list, stage_list, braak_list = [], [], [], []
    for m in metas:
        fl = m.get("flair", None)
        cl = m.get("clinical_vector", None)
        so = m.get("stage_ord", None)
        bv = m.get("braak_values", None)
        if fl is None or cl is None or so is None or bv is None:
            return None, None, None, None
        flair_list.append(torch.from_numpy(fl) if isinstance(fl, np.ndarray) else fl)
        clin_list.append(torch.from_numpy(cl) if isinstance(cl, np.ndarray) else cl)
        stage_list.append(torch.tensor(so, dtype=torch.long))
        braak_list.append(torch.from_numpy(bv) if isinstance(bv, np.ndarray) else bv)

    flair5 = torch.stack(flair_list, dim=0).to(device, non_blocking=True)
    clinical = torch.stack(clin_list, dim=0).to(device, non_blocking=True)
    stage_ord = torch.stack(stage_list, dim=0).to(device, non_blocking=True)
    braak_vals = torch.stack(braak_list, dim=0).to(device, non_blocking=True)
    return flair5, clinical, stage_ord, braak_vals


# =========================================================================
# Shared augmentation + loss helpers
# =========================================================================
def _masked_l1_high_uptake(
    fake5: torch.Tensor,
    pet5: torch.Tensor,
    mask5: torch.Tensor,
) -> torch.Tensor:
    """
    Cortex ROI loss with high-uptake emphasis:
      L = L1(cortex) + ROI_HI_LAMBDA * L1(top-quantile uptake in cortex)
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
    flair5: Optional[torch.Tensor] = None,
):
    """
    Train-only augmentation:
      - paired random flips on MRI/PET/FLAIR/brain/cortex masks
      - MRI-only intensity jitter inside brain mask
    Returns (mri5, pet5, brain5, cortex5, flair5)
    """
    if not AUG_ENABLE:
        return mri5, pet5, brain5, cortex5, flair5

    if torch.rand((), device=mri5.device) > float(AUG_PROB):
        return mri5, pet5, brain5, cortex5, flair5

    # --- paired random flips (D/H/W axes) ---
    for dim in (-1, -2, -3):
        if torch.rand((), device=mri5.device) < float(AUG_FLIP_PROB):
            mri5 = torch.flip(mri5, dims=(dim,))
            pet5 = torch.flip(pet5, dims=(dim,))
            if brain5 is not None:
                brain5 = torch.flip(brain5, dims=(dim,))
            if cortex5 is not None:
                cortex5 = torch.flip(cortex5, dims=(dim,))
            if flair5 is not None:
                flair5 = torch.flip(flair5, dims=(dim,))

    # --- MRI-only intensity augmentation (inside brain mask) ---
    if torch.rand((), device=mri5.device) < float(AUG_INTENSITY_PROB):
        B = mri5.size(0)
        dtype = mri5.dtype
        dev = mri5.device

        s = float(AUG_SCALE_MIN) + (float(AUG_SCALE_MAX) - float(AUG_SCALE_MIN)) * torch.rand((B, 1, 1, 1, 1), device=dev, dtype=dtype)
        b = float(AUG_SHIFT_MIN) + (float(AUG_SHIFT_MAX) - float(AUG_SHIFT_MIN)) * torch.rand((B, 1, 1, 1, 1), device=dev, dtype=dtype)
        noise = torch.randn_like(mri5) * float(AUG_NOISE_STD)

        if brain5 is not None:
            m = (brain5 > 0.5).to(dtype)
            mri5 = (mri5 * (1.0 - m)) + ((mri5 * s + b + noise) * m)
        else:
            mri5 = (mri5 * s + b + noise)

    return mri5, pet5, brain5, cortex5, flair5


# =========================================================================
# MGDA-UB (3-way) utilities
# =========================================================================
def _mgda_weights_from_gram_3(Gm: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Solve min || sum_i w_i g_i ||^2 s.t. w_i >= 0, sum w_i = 1 for 3 objectives.
    Active-set enumeration: interior + edges + vertices.
    """
    dev, dtype = Gm.device, Gm.dtype
    one = torch.ones(3, device=dev, dtype=dtype)

    def _obj(w: torch.Tensor) -> torch.Tensor:
        return torch.dot(w, Gm @ w)

    candidates = []

    # Interior candidate
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

    # Edge candidates
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
    best_obj = _obj(best_w)
    for w in candidates[1:]:
        v = _obj(w)
        if float(v.item()) < float(best_obj.item()):
            best_obj = v
            best_w = w

    best_w = torch.clamp(best_w, min=0.0)
    best_w = best_w / (best_w.sum() + eps)
    return best_w


def _mgda_weights_3(Vg: torch.Tensor, Vr: torch.Tensor, Vgan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample 3-way MGDA. Vg, Vr, Vgan: [B, N]. Returns [B, 3]."""
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


# =========================================================================
# Baseline training: train_paggan
# =========================================================================
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

    # AMP: prefer BF16 on supported hardware (A100+), fall back to FP16
    use_amp = AMP_ENABLE and device.type == "cuda"
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # LR scheduler + early stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_G, mode="min", factor=0.5, patience=LR_PLATEAU_PATIENCE
    )

    best_val = float("inf")
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    hist: Dict[str, list] = {"train_G": [], "train_D": [], "val_recon": [], "val_roi": [], "val_score": []}

    avg_norm_recon_global = 0.0
    avg_norm_recon_roi = 0.0
    avg_norm_gan = 0.0
    norm_decay = 0.9

    for epoch in range(1, epochs + 1):
        w_global_running = 0.0
        w_roi_running = 0.0
        w_gan_running = 0.0
        grad_recon_running = 0.0
        grad_roi_running = 0.0
        grad_gan_running = 0.0

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

            mri5 = mri if mri.dim() == 5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim() == 5 else pet.unsqueeze(0)

            metas = _meta_as_list(meta, B)
            brain5, cortex5 = _extract_masks(metas, device)

            mri5, pet5, brain5, cortex5, _ = _maybe_augment_pair(mri5, pet5, brain5, cortex5)

            # ---- Update D ----
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    fake = G(mri5)

            D.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                pair_real = torch.cat([mri5, pet5], dim=1)
                pair_fake = torch.cat([mri5, fake.detach()], dim=1)
                out_real = D(pair_real)
                out_fake = D(pair_fake)
                loss_D_real = adv_criterion(out_real, torch.ones_like(out_real))
                loss_D_fake = adv_criterion(out_fake, torch.zeros_like(out_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            scaler.step(opt_D)
            scaler.update()

            # ---- Update G ----
            G.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                fake = G(mri5)
                out_fake_for_G = D(torch.cat([mri5, fake], dim=1))
                loss_gan = 0.5 * adv_criterion(out_fake_for_G, torch.ones_like(out_fake_for_G))

            # Recon losses in float32 for MGDA stability
            # NOTE: baseline does NOT use MASK_GLOBAL_RECON — PET target is
            # already brain-masked, and mean-reducing over zero-masked voxels
            # would dilute the global gradient, letting GAN dominate in MGDA.
            fake_f32 = fake.float()
            pet5_f32 = pet5.float()

            loss_l1 = l1_loss(fake_f32, pet5_f32)
            ssim_val = ssim3d(fake_f32, pet5_f32, data_range=data_range)
            loss_recon_global = gamma * (loss_l1 + (1.0 - ssim_val))

            use_roi = (cortex5 is not None) and (float(cortex5.sum().item()) > 0.0)
            if use_roi:
                loss_recon_roi = _masked_l1_high_uptake(fake_f32, pet5_f32, cortex5.float())
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=torch.float32)

            # ---- MGDA-UB 3-way ----
            v_global = torch.autograd.grad(loss_recon_global, fake, retain_graph=True)[0].float()
            current_nglobal = v_global.norm().item()
            if avg_norm_recon_global == 0:
                avg_norm_recon_global = current_nglobal
            else:
                avg_norm_recon_global = norm_decay * avg_norm_recon_global + (1 - norm_decay) * current_nglobal
            v_global_s = v_global / (avg_norm_recon_global + 1e-8)

            if use_roi:
                v_roi = torch.autograd.grad(loss_recon_roi, fake, retain_graph=True)[0].float()
                current_nroi = v_roi.norm().item()
                if avg_norm_recon_roi == 0:
                    avg_norm_recon_roi = current_nroi
                else:
                    avg_norm_recon_roi = norm_decay * avg_norm_recon_roi + (1 - norm_decay) * current_nroi
                v_roi_s = v_roi / (avg_norm_recon_roi + 1e-8)
            else:
                v_roi_s = torch.zeros_like(v_global_s)
                current_nroi = 0.0

            v_gan = torch.autograd.grad(loss_gan, fake, retain_graph=True)[0].float()
            current_ngan = v_gan.norm().item()
            if avg_norm_gan == 0:
                avg_norm_gan = current_ngan
            else:
                avg_norm_gan = norm_decay * avg_norm_gan + (1 - norm_decay) * current_ngan
            v_gan_s = v_gan / (avg_norm_gan + 1e-8)

            Vg = v_global_s.reshape(v_global_s.size(0), -1)
            Vr = v_roi_s.reshape(v_roi_s.size(0), -1)
            Vgan = v_gan_s.reshape(v_gan_s.size(0), -1)

            if use_roi:
                w_batch = _mgda_weights_3(Vg, Vr, Vgan)
                w_med = w_batch.median(dim=0).values
                w_sum = (w_med.sum() + 1e-12)
                w_med = w_med / w_sum
                w_global, w_roi, w_gan_w = w_med[0], w_med[1], w_med[2]
            else:
                diff = Vgan - Vg
                num = (diff * Vgan).sum(dim=1)
                den = (diff * diff).sum(dim=1) + 1e-12
                a_batch = torch.clamp(num / den, 0.0, 1.0)
                a = a_batch.median()
                w_global = a
                w_roi = torch.tensor(0.0, device=device, dtype=torch.float32)
                w_gan_w = 1.0 - a

            w_global_running += float(w_global.item())
            w_roi_running += float(w_roi.item())
            w_gan_running += float(w_gan_w.item())
            grad_recon_running += current_nglobal
            grad_roi_running += current_nroi
            grad_gan_running += current_ngan

            v_final = (w_global * v_global_s) + (w_roi * v_roi_s) + (w_gan_w * v_gan_s)

            opt_G.zero_grad(set_to_none=True)
            fake.backward(v_final)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

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
        val_roi_epoch: Optional[float] = None
        val_score_epoch: Optional[float] = None

        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon, val_roi_sum, v_batches = 0.0, 0.0, 0
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri, pet, meta_v = batch
                    else:
                        mri, pet = batch
                        meta_v = {}
                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    Bv = mri.size(0) if mri.dim() == 5 else 1
                    mri5v = mri if mri.dim() == 5 else mri.unsqueeze(0)
                    pet5v = pet if pet.dim() == 5 else pet.unsqueeze(0)

                    metas_v = _meta_as_list(meta_v, Bv)
                    brain5_v, cortex5_v = _extract_masks(metas_v, device)

                    with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                        fake_v = G(mri5v)

                    fake_eval = fake_v.float()
                    pet_eval = pet5v.float()

                    loss_l1_v = l1_loss(fake_eval, pet_eval)
                    ssim_v = ssim3d(fake_eval, pet_eval, data_range=data_range)
                    val_recon += (loss_l1_v + (1.0 - ssim_v)).item()

                    if cortex5_v is not None and float(cortex5_v.sum().item()) > 0.0:
                        val_roi_sum += _masked_l1_high_uptake(fake_eval, pet_eval, cortex5_v.float()).item()

                    v_batches += 1

            val_recon /= max(1, v_batches)
            val_roi_sum /= max(1, v_batches)
            val_score = val_recon + VAL_ROI_WEIGHT * val_roi_sum
            val_recon_epoch = val_recon
            val_roi_epoch = val_roi_sum
            val_score_epoch = val_score
            hist["val_recon"].append(val_recon)
            hist["val_roi"].append(val_roi_sum)
            hist["val_score"].append(val_score)

            # LR scheduler step on combined val_score
            scheduler.step(val_score)

            if val_score < best_val:
                best_val = val_score
                patience_counter = 0
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))
            else:
                patience_counter += 1

            if verbose:
                dt = time.time() - t0
                cur_lr = opt_G.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:03d}/{epochs}]  "
                    f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                    f"ValRecon: {val_recon:.4f}  ValROI: {val_roi_sum:.4f}  "
                    f"ValScore: {val_score:.4f}  "
                    f"| best {best_val:.4f}  "
                    f"patience={patience_counter}/{EARLY_STOP_PATIENCE}  "
                    f"lr={cur_lr:.1e}  | {dt:.1f}s"
                )
                print(
                    f"      [MGDA-UB-3] w_global={avg_w_global:.3f}  "
                    f"w_roi={avg_w_roi:.3f}  w_gan={avg_w_gan:.3f}  "
                    f"||grad_global||={avg_grad_recon:.3e}  "
                    f"||grad_roi||={avg_grad_roi:.3e}  "
                    f"||grad_gan||={avg_grad_gan:.3e}"
                )

            G.train()

            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

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

        if log_to_wandb and wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "train/G_loss": avg_g,
                "train/D_loss": avg_d,
                "mgda/w_recon_global": avg_w_global,
                "mgda/w_recon_roi": avg_w_roi,
                "mgda/w_gan": avg_w_gan,
                "mgda/grad_recon_global_norm": avg_grad_recon,
                "mgda/grad_recon_roi_norm": avg_grad_roi,
                "mgda/grad_gan_norm": avg_grad_gan,
            }
            if val_recon_epoch is not None:
                log_dict["val/recon_loss"] = val_recon_epoch
            if val_roi_epoch is not None:
                log_dict["val/roi_loss"] = val_roi_epoch
            if val_score_epoch is not None:
                log_dict["val/score"] = val_score_epoch
                log_dict["val/best_score"] = best_val
            wandb.log(log_dict, step=epoch)

    if best_G is not None:
        G.load_state_dict(best_G)
    if best_D is not None:
        D.load_state_dict(best_D)

    return {"history": hist, "best_G": best_G, "best_D": best_D}


# =========================================================================
# Prompt-Residual-Braak training
# =========================================================================
def train_prompt_residual_braak(
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
    """
    Train PromptResidualBraakGenerator with:
      - Base freeze schedule (frozen for FREEZE_BASE_EPOCHS, then lower LR)
      - GT stage one-hot at train, predicted probs at val
      - MGDA-UB-3 on [global_recon, roi_recon, gan]
      - Aux losses outside MGDA: CORAL stage_ord, braak SmoothL1, delta outside-cortex reg
      - AMP with float32 for MGDA-sensitive losses
      - LR scheduler + early stopping
    """
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    # Two param groups: base (frozen initially) and new branch
    base_params = list(G.base.parameters())
    base_param_ids = set(id(p) for p in base_params)
    new_params = [p for p in G.parameters() if id(p) not in base_param_ids]

    opt_G = torch.optim.Adam([
        {"params": base_params, "lr": 0.0},   # frozen initially
        {"params": new_params, "lr": LR_G},
    ])
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    adv_criterion = nn.MSELoss()

    # AMP: prefer BF16 on supported hardware (A100+), fall back to FP16
    use_amp = AMP_ENABLE and device.type == "cuda"
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    # GradScaler is only needed for FP16; BF16 does not need loss scaling
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # LR scheduler on validation loss (new params group only)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_G, mode="min", factor=0.5, patience=LR_PLATEAU_PATIENCE
    )

    best_val = float("inf")
    best_G_state: Optional[Dict[str, torch.Tensor]] = None
    best_D_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    hist: Dict[str, list] = {
        "train_G": [], "train_D": [], "val_recon": [], "val_roi": [], "val_score": [],
        "train_stage_ord": [], "train_braak": [], "train_delta_out": [],
        "train_alpha": [], "val_braak": [],
    }

    avg_norm_recon_global = 0.0
    avg_norm_recon_roi = 0.0
    avg_norm_gan = 0.0
    norm_decay = 0.9

    for epoch in range(1, epochs + 1):
        # ---- Base freeze schedule ----
        if epoch <= FREEZE_BASE_EPOCHS:
            for p in base_params:
                p.requires_grad = False
            opt_G.param_groups[0]["lr"] = 0.0
        else:
            for p in base_params:
                p.requires_grad = True
            # Proportional: base LR tracks new-branch LR * BASE_LR_MULT
            opt_G.param_groups[0]["lr"] = opt_G.param_groups[1]["lr"] * BASE_LR_MULT

        t0 = time.time()
        g_running, d_running, n_batches = 0.0, 0.0, 0
        stage_ord_running, braak_running, delta_out_running, alpha_running = 0.0, 0.0, 0.0, 0.0
        w_global_running, w_roi_running, w_gan_running = 0.0, 0.0, 0.0
        grad_recon_running, grad_roi_running, grad_gan_running = 0.0, 0.0, 0.0
        # D(real)/D(fake) tracking
        d_real_running, d_fake_running = 0.0, 0.0
        # Residual branch tracking
        delta_in_cortex_running, delta_out_cortex_running = 0.0, 0.0
        pet_diff_running = 0.0
        # Gradient conflict tracking (recon vs aux on shared params)
        grad_cos_running, grad_norm_recon_shared_running, grad_norm_aux_shared_running = 0.0, 0.0, 0.0
        has_aux_grads = (LAMBDA_STAGE_ORD > 0 or LAMBDA_BRAAK > 0 or LAMBDA_DELTA_OUT > 0)

        G.train()
        D.train()

        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                mri, pet, meta = batch
            else:
                raise ValueError("Expected (mri, pet, meta) batch")

            mri = mri.to(device, non_blocking=True)
            pet = pet.to(device, non_blocking=True)
            B = mri.size(0) if mri.dim() == 5 else 1

            mri5 = mri if mri.dim() == 5 else mri.unsqueeze(0)
            pet5 = pet if pet.dim() == 5 else pet.unsqueeze(0)

            metas = _meta_as_list(meta, B)
            brain5, cortex5 = _extract_masks(metas, device)
            flair5, clinical, stage_ord, braak_gt = _extract_new_variant_inputs(metas, device)

            if flair5 is None:
                raise RuntimeError("FLAIR/clinical/stage_ord/braak missing from meta")

            mri5, pet5, brain5, cortex5, flair5 = _maybe_augment_pair(mri5, pet5, brain5, cortex5, flair5)

            # Build GT stage one-hot for training
            if USE_GT_STAGE_HINT_TRAIN:
                stage_weights = build_stage_onehot(stage_ord, num_classes=4).to(device)
            else:
                stage_weights = None

            # ---- Update D ----
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    fake = G(mri5, flair5, clinical, stage_prompt_weights=stage_weights)

            D.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                pair_real = torch.cat([mri5, pet5], dim=1)
                pair_fake = torch.cat([mri5, fake.detach()], dim=1)
                out_real = D(pair_real)
                out_fake = D(pair_fake)
                loss_D_real = adv_criterion(out_real, torch.ones_like(out_real))
                loss_D_fake = adv_criterion(out_fake, torch.zeros_like(out_fake))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

            d_real_running += out_real.detach().mean().item()
            d_fake_running += out_fake.detach().mean().item()

            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            scaler.step(opt_D)
            scaler.update()

            # ---- Update G ----
            G.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                pet_hat, aux = G(mri5, flair5, clinical,
                                 stage_prompt_weights=stage_weights,
                                 return_aux=True)
                out_fake_for_G = D(torch.cat([mri5, pet_hat], dim=1))
                loss_gan = 0.5 * adv_criterion(out_fake_for_G, torch.ones_like(out_fake_for_G))

            # Recon losses in float32 for MGDA stability
            pet_hat_f32 = pet_hat.float()
            pet5_f32 = pet5.float()

            if MASK_GLOBAL_RECON and brain5 is not None:
                loss_l1 = l1_loss(pet_hat_f32 * brain5.float(), pet5_f32 * brain5.float())
                ssim_val = ssim3d(pet_hat_f32 * brain5.float(), pet5_f32 * brain5.float(), data_range=data_range)
            else:
                loss_l1 = l1_loss(pet_hat_f32, pet5_f32)
                ssim_val = ssim3d(pet_hat_f32, pet5_f32, data_range=data_range)
            loss_recon_global = gamma * (loss_l1 + (1.0 - ssim_val))

            use_roi = (cortex5 is not None) and (float(cortex5.sum().item()) > 0.0)
            if use_roi:
                loss_recon_roi = _masked_l1_high_uptake(pet_hat_f32, pet5_f32, cortex5.float())
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=torch.float32)

            # ---- MGDA-UB 3-way on recon losses ----
            v_global = torch.autograd.grad(loss_recon_global, pet_hat, retain_graph=True)[0].float()
            current_nglobal = v_global.norm().item()
            if avg_norm_recon_global == 0:
                avg_norm_recon_global = current_nglobal
            else:
                avg_norm_recon_global = norm_decay * avg_norm_recon_global + (1 - norm_decay) * current_nglobal
            v_global_s = v_global / (avg_norm_recon_global + 1e-8)

            if use_roi:
                v_roi = torch.autograd.grad(loss_recon_roi, pet_hat, retain_graph=True)[0].float()
                current_nroi = v_roi.norm().item()
                if avg_norm_recon_roi == 0:
                    avg_norm_recon_roi = current_nroi
                else:
                    avg_norm_recon_roi = norm_decay * avg_norm_recon_roi + (1 - norm_decay) * current_nroi
                v_roi_s = v_roi / (avg_norm_recon_roi + 1e-8)
            else:
                v_roi_s = torch.zeros_like(v_global_s)
                current_nroi = 0.0

            v_gan_grad = torch.autograd.grad(loss_gan, pet_hat, retain_graph=True)[0].float()
            current_ngan = v_gan_grad.norm().item()
            if avg_norm_gan == 0:
                avg_norm_gan = current_ngan
            else:
                avg_norm_gan = norm_decay * avg_norm_gan + (1 - norm_decay) * current_ngan
            v_gan_s = v_gan_grad / (avg_norm_gan + 1e-8)

            Vg = v_global_s.reshape(v_global_s.size(0), -1)
            Vr = v_roi_s.reshape(v_roi_s.size(0), -1)
            Vgan_flat = v_gan_s.reshape(v_gan_s.size(0), -1)

            if use_roi:
                w_batch = _mgda_weights_3(Vg, Vr, Vgan_flat)
                w_med = w_batch.median(dim=0).values
                w_sum = (w_med.sum() + 1e-12)
                w_med = w_med / w_sum
                w_g, w_r, w_a = w_med[0], w_med[1], w_med[2]
            else:
                diff_v = Vgan_flat - Vg
                num = (diff_v * Vgan_flat).sum(dim=1)
                den = (diff_v * diff_v).sum(dim=1) + 1e-12
                a_batch = torch.clamp(num / den, 0.0, 1.0)
                a = a_batch.median()
                w_g = a
                w_r = torch.tensor(0.0, device=device, dtype=torch.float32)
                w_a = 1.0 - a

            w_global_running += float(w_g.item())
            w_roi_running += float(w_r.item())
            w_gan_running += float(w_a.item())
            grad_recon_running += current_nglobal
            grad_roi_running += current_nroi
            grad_gan_running += current_ngan

            v_final = (w_g * v_global_s) + (w_r * v_roi_s) + (w_a * v_gan_s)

            # ---- Aux losses (outside MGDA) ----
            # CORAL ordinal stage loss: BCE on cumulative logits
            stage_logits = aux["stage_logits"]  # [B, 3]
            target_cum = (stage_ord.unsqueeze(1) >= torch.arange(1, 4, device=device).unsqueeze(0)).float()
            loss_stage_ord = F.binary_cross_entropy_with_logits(stage_logits.float(), target_cum)

            # Braak prediction loss: SmoothL1
            braak_pred = aux["braak_pred"]  # [B, 3]
            loss_braak = F.smooth_l1_loss(braak_pred.float(), braak_gt.float())

            # Delta outside-cortex regularization
            delta_pet = aux["delta_pet"]
            if cortex5 is not None:
                outside_cortex = (1.0 - cortex5).float()
                loss_delta_out = (delta_pet.float().abs() * outside_cortex).mean()
            else:
                loss_delta_out = delta_pet.float().abs().mean() * 0.01

            loss_aux = (LAMBDA_STAGE_ORD * loss_stage_ord
                        + LAMBDA_BRAAK * loss_braak
                        + LAMBDA_DELTA_OUT * loss_delta_out)

            # ---- Residual branch behavior ----
            with torch.no_grad():
                delta_abs = delta_pet.float().abs()
                if cortex5 is not None:
                    cortex_f = cortex5.float()
                    outside_f = (1.0 - cortex_f)
                    n_in = cortex_f.sum().clamp(min=1)
                    n_out = outside_f.sum().clamp(min=1)
                    delta_in_cortex_running += (delta_abs * cortex_f).sum().item() / n_in.item()
                    delta_out_cortex_running += (delta_abs * outside_f).sum().item() / n_out.item()
                else:
                    delta_in_cortex_running += delta_abs.mean().item()
                    delta_out_cortex_running += delta_abs.mean().item()
                pet_diff_running += (pet_hat_f32 - aux["pet_base"].float()).abs().mean().item()

            # Combined backward: MGDA direction + aux
            # NOTE: both backward passes are in float32 (MGDA grads are float32,
            # aux losses are cast to float32), so we skip the scaler for G to
            # avoid mixing scaled/unscaled gradients.
            opt_G.zero_grad(set_to_none=True)

            # ---- Gradient conflict monitoring (recon vs aux on shared params) ----
            if has_aux_grads:
                # Use torch.autograd.grad to probe without consuming the graph
                shared_params = [p for p in G.parameters() if p.requires_grad]
                if shared_params:
                    recon_grads_tuple = torch.autograd.grad(
                        outputs=pet_hat, inputs=shared_params,
                        grad_outputs=v_final,
                        retain_graph=True, allow_unused=True,
                    )
                    aux_grads_tuple = torch.autograd.grad(
                        outputs=loss_aux, inputs=shared_params,
                        retain_graph=True, allow_unused=True,
                    )
                    # Flatten and concatenate (replace None with zeros)
                    recon_flat = torch.cat([
                        g.detach().flatten() if g is not None else torch.zeros(p.numel(), device=device)
                        for g, p in zip(recon_grads_tuple, shared_params)
                    ])
                    aux_flat = torch.cat([
                        g.detach().flatten() if g is not None else torch.zeros(p.numel(), device=device)
                        for g, p in zip(aux_grads_tuple, shared_params)
                    ])
                    recon_norm = recon_flat.norm().item()
                    aux_norm = aux_flat.norm().item()
                    cos_sim = (torch.dot(recon_flat, aux_flat) /
                               (recon_norm * aux_norm + 1e-12)).item()
                    grad_cos_running += cos_sim
                    grad_norm_recon_shared_running += recon_norm
                    grad_norm_aux_shared_running += aux_norm

            # Actual backward: MGDA direction + aux (single pass)
            pet_hat.backward(v_final, retain_graph=True)
            loss_aux.backward()

            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            # Logging accumulators
            alpha_val = float(aux["alpha"].detach().item())
            loss_G_log = (loss_recon_global + (loss_recon_roi if use_roi else 0.0) + loss_gan).detach().item()
            g_running += float(loss_G_log)
            d_running += loss_D.item()
            stage_ord_running += loss_stage_ord.detach().item()
            braak_running += loss_braak.detach().item()
            delta_out_running += loss_delta_out.detach().item()
            alpha_running += alpha_val
            n_batches += 1

        # ---- Epoch aggregates ----
        nb = max(1, n_batches)
        avg_g = g_running / nb
        avg_d = d_running / nb
        avg_stage = stage_ord_running / nb
        avg_braak = braak_running / nb
        avg_dout = delta_out_running / nb
        avg_alpha = alpha_running / nb
        avg_w_global = w_global_running / nb
        avg_w_roi = w_roi_running / nb
        avg_w_gan = w_gan_running / nb
        avg_d_real = d_real_running / nb
        avg_d_fake = d_fake_running / nb
        avg_delta_in = delta_in_cortex_running / nb
        avg_delta_out = delta_out_cortex_running / nb
        avg_pet_diff = pet_diff_running / nb
        avg_grad_cos = grad_cos_running / nb if has_aux_grads else None
        avg_grad_recon_shared = grad_norm_recon_shared_running / nb if has_aux_grads else None
        avg_grad_aux_shared = grad_norm_aux_shared_running / nb if has_aux_grads else None

        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)
        hist["train_stage_ord"].append(avg_stage)
        hist["train_braak"].append(avg_braak)
        hist["train_delta_out"].append(avg_dout)
        hist["train_alpha"].append(avg_alpha)

        # ---- Validation ----
        val_recon_epoch: Optional[float] = None
        val_roi_epoch: Optional[float] = None
        val_score_epoch: Optional[float] = None
        val_braak_epoch: Optional[float] = None
        val_stage_acc_epoch: Optional[float] = None
        val_braak_mae_epoch: Optional[List[float]] = None
        val_braak_corr_epoch: Optional[List[float]] = None

        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_recon, val_roi_sum, val_braak_sum, v_batches = 0.0, 0.0, 0.0, 0
                # Stage accuracy and Braak MAE/correlation collectors
                all_stage_pred, all_stage_gt = [], []
                all_braak_pred, all_braak_gt = [], []
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        mri, pet, meta = batch
                    else:
                        raise ValueError("Expected (mri, pet, meta)")

                    mri = mri.to(device, non_blocking=True)
                    pet = pet.to(device, non_blocking=True)
                    Bv = mri.size(0) if mri.dim() == 5 else 1
                    mri5v = mri if mri.dim() == 5 else mri.unsqueeze(0)
                    pet5v = pet if pet.dim() == 5 else pet.unsqueeze(0)

                    metas_v = _meta_as_list(meta, Bv)
                    flair_v, clin_v, stage_v, braak_v = _extract_new_variant_inputs(metas_v, device)
                    brain5_v, cortex5_v = _extract_masks(metas_v, device)

                    if flair_v is not None:
                        # At validation: use predicted probs (no GT stage hint)
                        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                            pet_hat_v, aux_v = G(mri5v, flair_v, clin_v,
                                                 stage_prompt_weights=None,
                                                 return_aux=True)
                        fake_eval = pet_hat_v.float()
                        pet_eval = pet5v.float()
                        if MASK_GLOBAL_RECON and brain5_v is not None:
                            fake_eval = fake_eval * brain5_v.float()
                            pet_eval = pet_eval * brain5_v.float()
                        loss_l1_v = l1_loss(fake_eval, pet_eval)
                        ssim_v = ssim3d(fake_eval, pet_eval, data_range=data_range)
                        val_recon += (loss_l1_v + (1.0 - ssim_v)).item()
                        # Val ROI loss
                        if cortex5_v is not None and float(cortex5_v.sum().item()) > 0.0:
                            val_roi_sum += _masked_l1_high_uptake(fake_eval, pet_eval, cortex5_v.float()).item()
                        val_braak_sum += F.smooth_l1_loss(aux_v["braak_pred"].float(), braak_v.float()).item()

                        # Collect stage predictions (ablation >= 4)
                        if ABLATION_STEP >= 4:
                            stage_probs = aux_v["stage_probs"].float()  # [B, 4]
                            stage_pred_cls = stage_probs.argmax(dim=1)  # [B]
                            all_stage_pred.append(stage_pred_cls.cpu())
                            all_stage_gt.append(stage_v.cpu())

                        # Collect Braak predictions (ablation >= 5)
                        if ABLATION_STEP >= 5:
                            all_braak_pred.append(aux_v["braak_pred"].float().cpu())
                            all_braak_gt.append(braak_v.float().cpu())
                    else:
                        # Fallback: T1-only
                        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                            fake_v = G.base(mri5v)
                        loss_l1_v = l1_loss(fake_v.float(), pet5v.float())
                        ssim_v = ssim3d(fake_v.float(), pet5v.float(), data_range=data_range)
                        val_recon += (loss_l1_v + (1.0 - ssim_v)).item()

                    v_batches += 1

                val_recon /= max(1, v_batches)
                val_roi_sum /= max(1, v_batches)
                val_braak_sum /= max(1, v_batches)
                val_score = val_recon + VAL_ROI_WEIGHT * val_roi_sum
                val_recon_epoch = val_recon
                val_roi_epoch = val_roi_sum
                val_score_epoch = val_score
                val_braak_epoch = val_braak_sum
                hist["val_recon"].append(val_recon)
                hist["val_roi"].append(val_roi_sum)
                hist["val_score"].append(val_score)
                hist["val_braak"].append(val_braak_sum)

                # Stage accuracy
                if all_stage_pred:
                    sp = torch.cat(all_stage_pred)
                    sg = torch.cat(all_stage_gt)
                    val_stage_acc_epoch = float((sp == sg).float().mean().item())

                # Braak MAE + Pearson correlation per component
                if all_braak_pred:
                    bp = torch.cat(all_braak_pred)  # [N, 3]
                    bg = torch.cat(all_braak_gt)    # [N, 3]
                    val_braak_mae_epoch = [(bp[:, i] - bg[:, i]).abs().mean().item() for i in range(3)]
                    val_braak_corr_epoch = []
                    for i in range(3):
                        p, g = bp[:, i], bg[:, i]
                        if p.std() > 1e-6 and g.std() > 1e-6:
                            r = float(torch.corrcoef(torch.stack([p, g]))[0, 1].item())
                        else:
                            r = 0.0
                        val_braak_corr_epoch.append(r)

                # LR scheduler step on combined val_score
                scheduler.step(val_score)

                # Sync base LR proportionally after scheduler step
                if epoch > FREEZE_BASE_EPOCHS:
                    opt_G.param_groups[0]["lr"] = opt_G.param_groups[1]["lr"] * BASE_LR_MULT

                if val_score < best_val:
                    best_val = val_score
                    patience_counter = 0
                    best_G_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
                    best_D_state = {k: v.detach().clone() for k, v in D.state_dict().items()}
                    torch.save(best_G_state, os.path.join(CKPT_DIR, "best_G.pth"))
                    torch.save(best_D_state, os.path.join(CKPT_DIR, "best_D.pth"))
                else:
                    patience_counter += 1

            if verbose:
                dt = time.time() - t0
                cur_lr_base = opt_G.param_groups[0]["lr"]
                cur_lr_new = opt_G.param_groups[1]["lr"]
                frozen_str = "FROZEN" if epoch <= FREEZE_BASE_EPOCHS else f"lr={cur_lr_base:.1e}"
                print(
                    f"Epoch [{epoch:03d}/{epochs}]  "
                    f"G: {avg_g:.4f}  D: {avg_d:.4f}  "
                    f"ValRecon: {val_recon:.4f}  ValROI: {val_roi_sum:.4f}  "
                    f"ValScore: {val_score:.4f}  "
                    f"ValBraak: {val_braak_sum:.4f}  "
                    f"| best {best_val:.4f}  "
                    f"patience={patience_counter}/{EARLY_STOP_PATIENCE}  "
                    f"| {dt:.1f}s"
                )
                print(
                    f"      alpha={avg_alpha:.4f}  base={frozen_str}  "
                    f"lr_new={cur_lr_new:.1e}  "
                    f"[MGDA] w_g={avg_w_global:.3f} w_r={avg_w_roi:.3f} w_a={avg_w_gan:.3f}  "
                    f"[AUX] stage={avg_stage:.4f} braak={avg_braak:.4f} dout={avg_dout:.4f}"
                )
                print(
                    f"      [GAN] D(real)={avg_d_real:.4f}  D(fake)={avg_d_fake:.4f}  "
                    f"[RESID] |delta|_in={avg_delta_in:.4f} |delta|_out={avg_delta_out:.4f} "
                    f"|hat-base|={avg_pet_diff:.4f}"
                )
                if has_aux_grads:
                    print(
                        f"      [GRAD-CONFLICT] cos(recon,aux)={avg_grad_cos:.4f}  "
                        f"||recon||={avg_grad_recon_shared:.3e}  ||aux||={avg_grad_aux_shared:.3e}  "
                        f"ratio=||aux||/||recon||={avg_grad_aux_shared / (avg_grad_recon_shared + 1e-12):.3f}"
                    )
                if val_stage_acc_epoch is not None:
                    print(f"      [VAL-STAGE] acc={val_stage_acc_epoch:.4f}")
                if val_braak_mae_epoch is not None:
                    print(
                        f"      [VAL-BRAAK] MAE(norm): B12={val_braak_mae_epoch[0]:.4f} "
                        f"B34={val_braak_mae_epoch[1]:.4f} B56={val_braak_mae_epoch[2]:.4f}"
                    )
                if val_braak_corr_epoch is not None:
                    print(
                        f"      [VAL-BRAAK] Pearson r: B12={val_braak_corr_epoch[0]:.4f} "
                        f"B34={val_braak_corr_epoch[1]:.4f} B56={val_braak_corr_epoch[2]:.4f}"
                    )

            G.train()

            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

        elif verbose:
            dt = time.time() - t0
            print(
                f"Epoch [{epoch:03d}/{epochs}]  "
                f"G: {avg_g:.4f}  D: {avg_d:.4f}  alpha={avg_alpha:.4f}  | {dt:.1f}s"
            )
            print(
                f"      [GAN] D(real)={avg_d_real:.4f}  D(fake)={avg_d_fake:.4f}  "
                f"[RESID] |delta|_in={avg_delta_in:.4f} |delta|_out={avg_delta_out:.4f} "
                f"|hat-base|={avg_pet_diff:.4f}"
            )

        if log_to_wandb and wandb.run is not None:
            cur_lr_base = opt_G.param_groups[0]["lr"]
            cur_lr_new = opt_G.param_groups[1]["lr"]
            log_dict = {
                "epoch": epoch,
                "train/G_loss": avg_g,
                "train/D_loss": avg_d,
                "train/stage_ord_loss": avg_stage,
                "train/braak_loss": avg_braak,
                "train/delta_out_loss": avg_dout,
                "train/alpha": avg_alpha,
                # LR / freeze status
                "optim/lr_base": cur_lr_base,
                "optim/lr_new": cur_lr_new,
                "optim/base_frozen": 1 if epoch <= FREEZE_BASE_EPOCHS else 0,
                # GAN health
                "gan/D_real": avg_d_real,
                "gan/D_fake": avg_d_fake,
                # MGDA
                "mgda/w_recon_global": avg_w_global,
                "mgda/w_recon_roi": avg_w_roi,
                "mgda/w_gan": avg_w_gan,
                "mgda/grad_recon_global_norm": grad_recon_running / nb,
                "mgda/grad_recon_roi_norm": grad_roi_running / nb,
                "mgda/grad_gan_norm": grad_gan_running / nb,
                # Residual branch
                "residual/delta_in_cortex": avg_delta_in,
                "residual/delta_out_cortex": avg_delta_out,
                "residual/in_out_ratio": avg_delta_in / (avg_delta_out + 1e-12),
                "residual/pet_hat_minus_base": avg_pet_diff,
            }
            # Gradient conflict (only when aux losses active)
            if has_aux_grads:
                log_dict["grad_conflict/cos_recon_vs_aux"] = avg_grad_cos
                log_dict["grad_conflict/norm_recon_shared"] = avg_grad_recon_shared
                log_dict["grad_conflict/norm_aux_shared"] = avg_grad_aux_shared
                log_dict["grad_conflict/ratio_aux_over_recon"] = avg_grad_aux_shared / (avg_grad_recon_shared + 1e-12)
            if val_recon_epoch is not None:
                log_dict["val/recon_loss"] = val_recon_epoch
            if val_roi_epoch is not None:
                log_dict["val/roi_loss"] = val_roi_epoch
            if val_score_epoch is not None:
                log_dict["val/score"] = val_score_epoch
                log_dict["val/best_score"] = best_val
            if val_braak_epoch is not None:
                log_dict["val/braak_loss"] = val_braak_epoch
            if val_stage_acc_epoch is not None:
                log_dict["val/stage_acc"] = val_stage_acc_epoch
            if val_braak_mae_epoch is not None:
                log_dict["val/braak_mae_norm_B12"] = val_braak_mae_epoch[0]
                log_dict["val/braak_mae_norm_B34"] = val_braak_mae_epoch[1]
                log_dict["val/braak_mae_norm_B56"] = val_braak_mae_epoch[2]
            if val_braak_corr_epoch is not None:
                log_dict["val/braak_corr_B12"] = val_braak_corr_epoch[0]
                log_dict["val/braak_corr_B34"] = val_braak_corr_epoch[1]
                log_dict["val/braak_corr_B56"] = val_braak_corr_epoch[2]
            wandb.log(log_dict, step=epoch)

    # Load best weights
    if best_G_state is not None:
        G.load_state_dict(best_G_state)
    if best_D_state is not None:
        D.load_state_dict(best_D_state)

    return {"history": hist, "best_G": best_G_state, "best_D": best_D_state}


# =========================================================================
# Evaluation (shared, but variant-aware)
# =========================================================================
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
    is_prompt_residual: bool = False,
):
    """
    Evaluates on test_loader, saves volumes, returns aggregate metrics with 95% CI.
    If is_prompt_residual=True, also saves PET_base, PET_delta, and per-subject aux CSV.
    Requires batch_size=1 (per-subject metrics/saves assume single-sample batches).
    """
    assert getattr(test_loader, "batch_size", 1) == 1, (
        f"evaluate_and_save requires batch_size=1, got {getattr(test_loader, 'batch_size', '?')}"
    )
    import json
    try:
        from scipy.stats import t as _t_dist
        def _tcrit(df): return float(_t_dist.ppf(0.975, df)) if df > 0 else float('nan')
    except Exception:
        def _tcrit(df): return 1.96 if df > 0 else float('nan')

    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    sids, ssim_list, psnr_list, mse_list, mmd_list = [], [], [], [], []
    aux_rows: List[Dict[str, Any]] = []

    run_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) else out_dir

    use_amp = AMP_ENABLE and device.type == "cuda"
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

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
        mri5 = mri_t if mri_t.dim() == 5 else mri_t.unsqueeze(0)
        pet5 = pet_t if pet_t.dim() == 5 else pet_t.unsqueeze(0)

        if is_prompt_residual:
            metas_list = _meta_as_list(meta, 1)
            flair5, clinical, stage_ord_t, braak_gt = _extract_new_variant_inputs(metas_list, device)

            if flair5 is not None:
                with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                    pet_hat, aux = G(mri5, flair5, clinical,
                                     stage_prompt_weights=None,
                                     return_aux=True)
                fake_t = pet_hat
                # Save base and delta
                pet_base_np = aux["pet_base"].squeeze(0).squeeze(0).float().cpu().numpy()
                delta_np = aux["delta_pet"].squeeze(0).squeeze(0).float().cpu().numpy()
                alpha_v = float(aux["alpha"].item())
                stage_probs_np = aux["stage_probs"].squeeze(0).float().cpu().numpy()
                braak_pred_np = aux["braak_pred"].squeeze(0).float().cpu().numpy()

                braak_raw_gt = meta.get("braak_values_raw", None)

                aux_rows.append({
                    "sid": sid,
                    "alpha": alpha_v,
                    "stage_pred_0": float(stage_probs_np[0]),
                    "stage_pred_12": float(stage_probs_np[1]),
                    "stage_pred_34": float(stage_probs_np[2]),
                    "stage_pred_56": float(stage_probs_np[3]),
                    "braak_pred_12": float(braak_pred_np[0]),
                    "braak_pred_34": float(braak_pred_np[1]),
                    "braak_pred_56": float(braak_pred_np[2]),
                    "braak_raw_gt_12": float(braak_raw_gt[0]) if braak_raw_gt is not None else "",
                    "braak_raw_gt_34": float(braak_raw_gt[1]) if braak_raw_gt is not None else "",
                    "braak_raw_gt_56": float(braak_raw_gt[2]) if braak_raw_gt is not None else "",
                    "stage_ord_gt": int(stage_ord_t.item()) if stage_ord_t is not None else "",
                })
            else:
                fake_t = G.base(mri5)
                pet_base_np = None
                delta_np = None
        else:
            fake_t = G(mri5)
            pet_base_np = None
            delta_np = None

        pet_for_metric = pet5

        # Mask
        brain_mask_np = meta.get("brain_mask", None)
        if brain_mask_np is not None:
            brain = torch.from_numpy(brain_mask_np.astype(np.float32))[None, None].to(device)
        else:
            brain = (pet_for_metric > 0).float()

        fake_m = fake_t.float() * brain
        pet_m  = pet_for_metric.float() * brain

        ssim_val = ssim3d(fake_m, pet_m, data_range=data_range).item()
        psnr_val = psnr(fake_m,  pet_m, data_range=data_range)
        mse_val  = F.mse_loss(fake_m, pet_m).item()
        mmd_val  = mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)

        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)
        mse_list.append(mse_val)
        mmd_list.append(mmd_val)

        # Save volumes
        mri_np  = mri5.squeeze(0).squeeze(0).cpu().numpy()
        pet_np  = pet5.squeeze(0).squeeze(0).cpu().numpy()
        fake_np = fake_t.squeeze(0).squeeze(0).float().cpu().numpy()
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
            if pet_base_np is not None:
                pet_base_np = nd_zoom(pet_base_np, zf, order=1)
            if delta_np is not None:
                delta_np = nd_zoom(delta_np, zf, order=1)
            affine_to_use = meta.get("t1_affine", np.eye(4))
        else:
            resized_to = meta.get("resized_to", None)
            affine_to_use = meta.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

        _save_nifti(mri_np,  affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
        _save_nifti(pet_np,  affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
        _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
        _save_nifti(err_np,  affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

        if pet_base_np is not None:
            _save_nifti(pet_base_np, affine_to_use, os.path.join(subdir, "PET_base.nii.gz"))
        if delta_np is not None:
            _save_nifti(delta_np, affine_to_use, os.path.join(subdir, "PET_delta.nii.gz"))

    # ---- Aggregate + CI ----
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

    per_subj_csv = os.path.join(run_dir, "per_subject_metrics.csv")
    with open(per_subj_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "SSIM", "PSNR", "MSE", "MMD"])
        for sid, ssim_v, psnr_v, mse_v, mmd_v in zip(sids, ssim_list, psnr_list, mse_list, mmd_list):
            w.writerow([sid, ssim_v, psnr_v, mse_v, mmd_v])

    # Per-subject aux CSV (prompt-residual only)
    if is_prompt_residual and aux_rows:
        aux_csv = os.path.join(run_dir, "per_subject_aux.csv")
        aux_cols = [
            "sid", "alpha",
            "stage_pred_0", "stage_pred_12", "stage_pred_34", "stage_pred_56",
            "braak_pred_12", "braak_pred_34", "braak_pred_56",
            "braak_raw_gt_12", "braak_raw_gt_34", "braak_raw_gt_56",
            "stage_ord_gt",
        ]
        with open(aux_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(aux_cols)
            for row in aux_rows:
                w.writerow([row.get(c, "") for c in aux_cols])

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

    return {
        "N": n_ssim,
        "SSIM": m_ssim, "SSIM_std": sd_ssim, "SSIM_lo95": lo_ssim, "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr, "PSNR_std": sd_psnr, "PSNR_lo95": lo_psnr, "PSNR_hi95": hi_psnr,
        "MSE":  m_mse,  "MSE_std":  sd_mse,  "MSE_lo95":  lo_mse,  "MSE_hi95":  hi_mse,
        "MMD":  m_mmd,  "MMD_std":  sd_mmd,  "MMD_lo95":  lo_mmd,  "MMD_hi95":  hi_mmd,
        "per_subject_csv": per_subj_csv,
        "summary_json": summary_json,
    }
