import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy.ndimage import zoom as nd_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .config import (
    AUG_ENABLE,
    AUG_FLIP_PROB,
    AUG_INTENSITY_PROB,
    AUG_NOISE_STD,
    AUG_PROB,
    AUG_SCALE_MAX,
    AUG_SCALE_MIN,
    AUG_SHIFT_MAX,
    AUG_SHIFT_MIN,
    CKPT_DIR,
    CONTRAST_TEMP,
    DATA_RANGE,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    GAMMA,
    LAMBDA_56,
    LAMBDA_CON,
    LAMBDA_HIGH,
    LR_PLATEAU_PATIENCE,
    LR_D,
    LR_G,
    RESAMPLE_BACK_TO_T1,
    ROI_HI_LAMBDA,
    ROI_HI_MIN_VOXELS,
    ROI_HI_Q,
)
from .losses import l1_loss, mmd_gaussian, psnr, ssim3d
from .utils import _safe_name, _save_nifti


def _meta_as_list(meta_any: Any) -> List[Dict[str, Any]]:
    if isinstance(meta_any, dict):
        return [meta_any]
    if isinstance(meta_any, list) and all(isinstance(m, dict) for m in meta_any):
        return meta_any
    raise TypeError("meta must be a dict or a list of dicts")


def _sample_meta_tensor(value: Any) -> torch.Tensor:
    t = torch.as_tensor(value)
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 0:
        return t.view(1)
    return t


def _meta_to_tensor(
    meta_any: Any,
    key: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    metas = _meta_as_list(meta_any)
    vals = []
    for meta in metas:
        if key not in meta:
            raise KeyError(f"Missing meta key '{key}'")
        vals.append(_sample_meta_tensor(meta[key]))
    out = torch.stack(vals, dim=0)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out.to(device, non_blocking=True)


def _extract_generator_inputs(meta_any: Any, device: torch.device, batch_size: int):
    flair = _meta_to_tensor(meta_any, "flair", device, dtype=torch.float32)
    clinical = _meta_to_tensor(meta_any, "clinical_vector", device, dtype=torch.float32)
    if flair.size(0) != batch_size or clinical.size(0) != batch_size:
        raise RuntimeError("Meta batch size does not match MRI/PET batch size")
    return flair, clinical


def _extract_training_targets(meta_any: Any, device: torch.device, batch_size: int):
    brain = _meta_to_tensor(meta_any, "brain_mask", device, dtype=torch.float32)
    cortex = _meta_to_tensor(meta_any, "cortex_mask", device, dtype=torch.float32)
    y_high = _meta_to_tensor(meta_any, "y_high", device, dtype=torch.float32)
    y_56 = _meta_to_tensor(meta_any, "y_56", device, dtype=torch.float32)
    contrast_group = _meta_to_tensor(meta_any, "contrast_group", device, dtype=torch.long).view(-1)
    tensors = [brain, cortex, y_high, y_56]
    if any(t.size(0) != batch_size for t in tensors) or contrast_group.numel() != batch_size:
        raise RuntimeError("Training target batch size does not match MRI/PET batch size")
    return brain, cortex, y_high, y_56, contrast_group


def _masked_l1_high_uptake(
    fake5: torch.Tensor,
    pet5: torch.Tensor,
    mask5: torch.Tensor,
) -> torch.Tensor:
    diff = (fake5 - pet5).abs()
    losses = []
    q = float(min(max(ROI_HI_Q, 0.0), 1.0))
    lambda_hi = float(ROI_HI_LAMBDA)
    min_vox = max(1, int(ROI_HI_MIN_VOXELS))

    for b in range(diff.size(0)):
        cortex = mask5[b, 0] > 0.5
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


def _maybe_augment_batch(
    mri5: torch.Tensor,
    pet5: torch.Tensor,
    flair5: torch.Tensor,
    brain5: torch.Tensor,
    cortex5: torch.Tensor,
):
    if not AUG_ENABLE:
        return mri5, pet5, flair5, brain5, cortex5

    if torch.rand((), device=mri5.device) > float(AUG_PROB):
        return mri5, pet5, flair5, brain5, cortex5

    for dim in (-1, -2, -3):
        if torch.rand((), device=mri5.device) < float(AUG_FLIP_PROB):
            mri5 = torch.flip(mri5, dims=(dim,))
            pet5 = torch.flip(pet5, dims=(dim,))
            flair5 = torch.flip(flair5, dims=(dim,))
            brain5 = torch.flip(brain5, dims=(dim,))
            cortex5 = torch.flip(cortex5, dims=(dim,))

    if torch.rand((), device=mri5.device) < float(AUG_INTENSITY_PROB):
        batch = mri5.size(0)
        dtype = mri5.dtype
        dev = mri5.device
        scale = float(AUG_SCALE_MIN) + (float(AUG_SCALE_MAX) - float(AUG_SCALE_MIN)) * torch.rand(
            (batch, 1, 1, 1, 1), device=dev, dtype=dtype
        )
        shift = float(AUG_SHIFT_MIN) + (float(AUG_SHIFT_MAX) - float(AUG_SHIFT_MIN)) * torch.rand(
            (batch, 1, 1, 1, 1), device=dev, dtype=dtype
        )
        noise = torch.randn_like(mri5) * float(AUG_NOISE_STD)
        mask = (brain5 > 0.5).to(dtype)
        mri5 = (mri5 * (1.0 - mask)) + ((mri5 * scale + shift + noise) * mask)

    return mri5, pet5, flair5, brain5, cortex5


def _alignment_supcon_loss(
    anchor: torch.Tensor,
    fused: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=1)
    fused = F.normalize(fused, dim=1)
    logits = torch.matmul(anchor, fused.t()) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1)
    pos_mask = labels[:, None].eq(labels[None, :])
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    return -(pos_log_prob.sum(dim=1) / pos_count).mean()


def _contrastive_loss(aux: Dict[str, torch.Tensor], contrast_group: torch.Tensor) -> torch.Tensor:
    return (
        _alignment_supcon_loss(aux["z_mri_con"], aux["z_fuse"], contrast_group, CONTRAST_TEMP)
        + _alignment_supcon_loss(aux["z_flair"], aux["z_fuse"], contrast_group, CONTRAST_TEMP)
        + _alignment_supcon_loss(aux["z_clin"], aux["z_fuse"], contrast_group, CONTRAST_TEMP)
    )


def _mgda_weights_from_gram_3(Gm: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dev, dtype = Gm.device, Gm.dtype
    one = torch.ones(3, device=dev, dtype=dtype)

    def _obj(w: torch.Tensor) -> torch.Tensor:
        return torch.dot(w, Gm @ w)

    candidates = []
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

    def _edge(i: int, j: int) -> torch.Tensor:
        Gii = Gm[i, i]
        Gjj = Gm[j, j]
        Gij = Gm[i, j]
        num = Gjj - Gij
        den = (Gii + Gjj - 2.0 * Gij) + eps
        a = torch.clamp(num / den, 0.0, 1.0)
        w = torch.zeros(3, device=dev, dtype=dtype)
        w[i] = a
        w[j] = 1.0 - a
        return w

    candidates += [_edge(0, 1), _edge(0, 2), _edge(1, 2)]
    candidates += [
        torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dtype),
        torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dtype),
        torch.tensor([0.0, 0.0, 1.0], device=dev, dtype=dtype),
    ]

    best_w = candidates[0]
    best_val = _obj(best_w)
    for w in candidates[1:]:
        val = _obj(w)
        if float(val.item()) < float(best_val.item()):
            best_val = val
            best_w = w

    best_w = torch.clamp(best_w, min=0.0)
    return best_w / (best_w.sum() + eps)


def _mgda_weights_3(Vg: torch.Tensor, Vr: torch.Tensor, Vgan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    weights = []
    for b in range(Vg.size(0)):
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
        weights.append(_mgda_weights_from_gram_3(Gm, eps=eps))
    return torch.stack(weights, dim=0)


def _loader_items(loader: Iterable) -> List[Dict[str, Any]]:
    ds = loader.dataset
    if isinstance(ds, torch.utils.data.Subset):
        base = ds.dataset
        return [base.items[i] for i in ds.indices]
    if hasattr(ds, "items"):
        return list(ds.items)
    raise TypeError("Could not read dataset items for label statistics")


def _pos_weight_from_items(items: List[Dict[str, Any]], key: str, device: torch.device) -> torch.Tensor:
    positives = sum(int(float(item[key]) > 0.5) for item in items)
    negatives = len(items) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError(f"Train split does not contain both classes for {key}")
    return torch.tensor([negatives / positives], device=device, dtype=torch.float32)


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
    sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_G, mode="min", patience=LR_PLATEAU_PATIENCE)
    adv_criterion = nn.MSELoss()

    train_items = _loader_items(train_loader)
    pos_weight_high = _pos_weight_from_items(train_items, "y_high", device)
    pos_weight_56 = _pos_weight_from_items(train_items, "y_56", device)

    best_val_score = float("inf")
    best_val_global = float("inf")
    best_val_roi = float("inf")
    best_G: Optional[Dict[str, torch.Tensor]] = None
    best_D: Optional[Dict[str, torch.Tensor]] = None

    hist = {
        "train_G": [],
        "train_D": [],
        "train_global": [],
        "train_roi": [],
        "train_gan": [],
        "train_con": [],
        "train_high": [],
        "train_56": [],
        "train_aux": [],
        "val_global": [],
        "val_roi": [],
        "val_score": [],
        "val_recon": [],
    }

    avg_norm_recon_global = 0.0
    avg_norm_recon_roi = 0.0
    avg_norm_gan = 0.0
    norm_decay = 0.9
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        w_global_running = 0.0
        w_roi_running = 0.0
        w_gan_running = 0.0
        grad_recon_running = 0.0
        grad_roi_running = 0.0
        grad_gan_running = 0.0

        t0 = time.time()
        g_running = 0.0
        d_running = 0.0
        global_running = 0.0
        roi_running = 0.0
        gan_running = 0.0
        con_running = 0.0
        high_running = 0.0
        severe_running = 0.0
        aux_running = 0.0
        p_high_running = 0.0
        p_56_running = 0.0
        n_batches = 0

        for batch in train_loader:
            if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
                raise ValueError("Expected train batch as (MRI, PET, meta)")

            mri, pet, meta = batch
            mri5 = mri.to(device, non_blocking=True)
            pet5 = pet.to(device, non_blocking=True)
            if mri5.dim() == 4:
                mri5 = mri5.unsqueeze(0)
            if pet5.dim() == 4:
                pet5 = pet5.unsqueeze(0)

            batch_size = mri5.size(0)
            flair5, clinical = _extract_generator_inputs(meta, device, batch_size)
            brain5, cortex5, y_high, y_56, contrast_group = _extract_training_targets(meta, device, batch_size)
            mri5, pet5, flair5, brain5, cortex5 = _maybe_augment_batch(mri5, pet5, flair5, brain5, cortex5)

            with torch.no_grad():
                fake = G(mri5, flair5, clinical)

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

            fake, aux = G(mri5, flair5, clinical, return_aux=True)
            out_fake_for_G = D(torch.cat([mri5, fake], dim=1))

            loss_gan = 0.5 * adv_criterion(out_fake_for_G, torch.ones_like(out_fake_for_G))
            loss_l1 = l1_loss(fake, pet5)
            ssim_val = ssim3d(fake, pet5, data_range=data_range)
            loss_recon_global = gamma * (loss_l1 + (1.0 - ssim_val))

            use_roi = float(cortex5.sum().item()) > 0.0
            if use_roi:
                loss_recon_roi = _masked_l1_high_uptake(fake, pet5, cortex5)
            else:
                loss_recon_roi = torch.zeros((), device=device, dtype=fake.dtype)

            loss_high = F.binary_cross_entropy_with_logits(aux["high_logit"], y_high, pos_weight=pos_weight_high)
            loss_56 = F.binary_cross_entropy_with_logits(aux["severe_logit"], y_56, pos_weight=pos_weight_56)
            loss_con = _contrastive_loss(aux, contrast_group)
            aux_loss = (LAMBDA_CON * loss_con) + (LAMBDA_HIGH * loss_high) + (LAMBDA_56 * loss_56)

            if not torch.isfinite(aux_loss):
                raise RuntimeError("Auxiliary multimodal loss became non-finite")

            v_global = torch.autograd.grad(loss_recon_global, fake, retain_graph=True)[0]
            current_nglobal = v_global.norm().item()
            if avg_norm_recon_global == 0:
                avg_norm_recon_global = current_nglobal
            else:
                avg_norm_recon_global = norm_decay * avg_norm_recon_global + (1 - norm_decay) * current_nglobal
            v_global_s = v_global / (avg_norm_recon_global + 1e-8)

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

            v_gan = torch.autograd.grad(loss_gan, fake, retain_graph=True)[0]
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
                w_med = w_med / (w_med.sum() + 1e-12)
                w_global, w_roi, w_gan_w = w_med[0], w_med[1], w_med[2]
            else:
                diff = Vgan - Vg
                num = (diff * Vgan).sum(dim=1)
                den = (diff * diff).sum(dim=1) + 1e-12
                a_batch = torch.clamp(num / den, 0.0, 1.0)
                a = a_batch.median()
                w_global = a
                w_roi = torch.tensor(0.0, device=device, dtype=fake.dtype)
                w_gan_w = 1.0 - a

            w_global_running += float(w_global.item())
            w_roi_running += float(w_roi.item())
            w_gan_running += float(w_gan_w.item())
            grad_recon_running += current_nglobal
            grad_roi_running += current_nroi
            grad_gan_running += current_ngan

            v_final = (w_global * v_global_s) + (w_roi * v_roi_s) + (w_gan_w * v_gan_s)

            opt_G.zero_grad(set_to_none=True)
            fake.backward(v_final, retain_graph=True)
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_G.step()

            loss_G_log = (
                loss_recon_global
                + (loss_recon_roi if use_roi else 0.0)
                + loss_gan
                + aux_loss
            ).detach().item()

            g_running += float(loss_G_log)
            d_running += loss_D.item()
            global_running += float(loss_recon_global.detach().item())
            roi_running += float(loss_recon_roi.detach().item()) if use_roi else 0.0
            gan_running += float(loss_gan.detach().item())
            con_running += float(loss_con.detach().item())
            high_running += float(loss_high.detach().item())
            severe_running += float(loss_56.detach().item())
            aux_running += float(aux_loss.detach().item())
            p_high_running += float(aux["p_high"].mean().detach().item())
            p_56_running += float(aux["p_56"].mean().detach().item())
            n_batches += 1

        avg_g = g_running / max(1, n_batches)
        avg_d = d_running / max(1, n_batches)
        avg_global = global_running / max(1, n_batches)
        avg_roi = roi_running / max(1, n_batches)
        avg_gan = gan_running / max(1, n_batches)
        avg_con = con_running / max(1, n_batches)
        avg_high = high_running / max(1, n_batches)
        avg_56 = severe_running / max(1, n_batches)
        avg_aux = aux_running / max(1, n_batches)
        avg_p_high = p_high_running / max(1, n_batches)
        avg_p_56 = p_56_running / max(1, n_batches)

        avg_w_global = w_global_running / max(1, n_batches)
        avg_w_roi = w_roi_running / max(1, n_batches)
        avg_w_gan = w_gan_running / max(1, n_batches)
        avg_grad_recon = grad_recon_running / max(1, n_batches)
        avg_grad_roi = grad_roi_running / max(1, n_batches)
        avg_grad_gan = grad_gan_running / max(1, n_batches)

        hist["train_G"].append(avg_g)
        hist["train_D"].append(avg_d)
        hist["train_global"].append(avg_global)
        hist["train_roi"].append(avg_roi)
        hist["train_gan"].append(avg_gan)
        hist["train_con"].append(avg_con)
        hist["train_high"].append(avg_high)
        hist["train_56"].append(avg_56)
        hist["train_aux"].append(avg_aux)

        val_global_epoch: Optional[float] = None
        val_roi_epoch: Optional[float] = None
        val_score_epoch: Optional[float] = None

        if val_loader is not None:
            G.eval()
            with torch.no_grad():
                val_global_sum = 0.0
                val_roi_sum = 0.0
                v_batches = 0
                for batch in val_loader:
                    if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
                        raise ValueError("Expected validation batch as (MRI, PET, meta)")
                    mri, pet, meta = batch
                    mri5 = mri.to(device, non_blocking=True)
                    pet5 = pet.to(device, non_blocking=True)
                    if mri5.dim() == 4:
                        mri5 = mri5.unsqueeze(0)
                    if pet5.dim() == 4:
                        pet5 = pet5.unsqueeze(0)
                    batch_size = mri5.size(0)
                    flair5, clinical = _extract_generator_inputs(meta, device, batch_size)
                    cortex5 = _meta_to_tensor(meta, "cortex_mask", device, dtype=torch.float32)
                    fake = G(mri5, flair5, clinical)

                    loss_l1_v = l1_loss(fake, pet5)
                    ssim_v = ssim3d(fake, pet5, data_range=data_range)
                    val_global_sum += float((loss_l1_v + (1.0 - ssim_v)).item())

                    use_roi = float(cortex5.sum().item()) > 0.0
                    if use_roi:
                        val_roi_sum += float(_masked_l1_high_uptake(fake, pet5, cortex5).item())
                    v_batches += 1

            val_global_epoch = val_global_sum / max(1, v_batches)
            val_roi_epoch = val_roi_sum / max(1, v_batches)
            val_score_epoch = val_global_epoch + val_roi_epoch

            hist["val_global"].append(val_global_epoch)
            hist["val_roi"].append(val_roi_epoch)
            hist["val_score"].append(val_score_epoch)
            hist["val_recon"].append(val_global_epoch)

            if val_score_epoch < best_val_score:
                best_val_score = val_score_epoch
                best_val_global = val_global_epoch
                best_val_roi = val_roi_epoch
                epochs_without_improve = 0
                best_G = {k: v.detach().clone() for k, v in G.state_dict().items()}
                best_D = {k: v.detach().clone() for k, v in D.state_dict().items()}
                torch.save(best_G, os.path.join(CKPT_DIR, "best_G.pth"))
                torch.save(best_D, os.path.join(CKPT_DIR, "best_D.pth"))
            else:
                epochs_without_improve += 1

            sched_G.step(val_score_epoch)
            current_lr_g = float(opt_G.param_groups[0]["lr"])

            if verbose:
                dt = time.time() - t0
                print(
                    f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  "
                    f"ValGlobal: {val_global_epoch:.4f}  ValROI: {val_roi_epoch:.4f}  "
                    f"ValScore: {val_score_epoch:.4f}  | best {best_val_score:.4f}  "
                    f"| lr_G {current_lr_g:.2e}  | {dt:.1f}s"
                )
                print(
                    f"      [MGDA-UB-3] w_global={avg_w_global:.3f}  w_roi={avg_w_roi:.3f}  "
                    f"w_gan={avg_w_gan:.3f}  ||grad_global||={avg_grad_recon:.3e}  "
                    f"||grad_roi||={avg_grad_roi:.3e}  ||grad_gan||={avg_grad_gan:.3e}"
                )
                print(
                    f"      [AUX] global={avg_global:.4f} roi={avg_roi:.4f} gan={avg_gan:.4f} "
                    f"con={avg_con:.4f} high={avg_high:.4f} severe={avg_56:.4f} "
                    f"p_high={avg_p_high:.3f} p_56={avg_p_56:.3f}"
                )
            G.train()
        elif verbose:
            current_lr_g = float(opt_G.param_groups[0]["lr"])
            dt = time.time() - t0
            print(
                f"Epoch [{epoch:03d}/{epochs}]  G: {avg_g:.4f}  D: {avg_d:.4f}  "
                f"| lr_G {current_lr_g:.2e}  | {dt:.1f}s"
            )
            print(
                f"      [MGDA-UB-3] w_global={avg_w_global:.3f}  w_roi={avg_w_roi:.3f}  "
                f"w_gan={avg_w_gan:.3f}  ||grad_global||={avg_grad_recon:.3e}  "
                f"||grad_roi||={avg_grad_roi:.3e}  ||grad_gan||={avg_grad_gan:.3e}"
            )
            print(
                f"      [AUX] global={avg_global:.4f} roi={avg_roi:.4f} gan={avg_gan:.4f} "
                f"con={avg_con:.4f} high={avg_high:.4f} severe={avg_56:.4f} "
                f"p_high={avg_p_high:.3f} p_56={avg_p_56:.3f}"
            )

        if log_to_wandb and wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "train/G_loss": avg_g,
                "train/D_loss": avg_d,
                "train/global_loss": avg_global,
                "train/roi_loss": avg_roi,
                "train/gan_loss": avg_gan,
                "train/aux_loss": avg_aux,
                "train/con_loss": avg_con,
                "train/high_loss": avg_high,
                "train/severe_loss": avg_56,
                "train/p_high_mean": avg_p_high,
                "train/p_56_mean": avg_p_56,
                "train/lr_G": float(opt_G.param_groups[0]["lr"]),
                "mgda/w_recon_global": avg_w_global,
                "mgda/w_recon_roi": avg_w_roi,
                "mgda/w_gan": avg_w_gan,
                "mgda/grad_recon_global_norm": avg_grad_recon,
                "mgda/grad_recon_roi_norm": avg_grad_roi,
                "mgda/grad_gan_norm": avg_grad_gan,
            }
            if val_global_epoch is not None:
                log_dict["val/global_loss"] = val_global_epoch
                log_dict["val/roi_loss"] = val_roi_epoch
                log_dict["val/score"] = val_score_epoch
                log_dict["val/best_score"] = best_val_score
                log_dict["val/best_global"] = best_val_global
                log_dict["val/best_roi"] = best_val_roi
                log_dict["train/epochs_without_improve"] = epochs_without_improve
            wandb.log(log_dict, step=epoch)

        if val_loader is not None and epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(
                f"[INFO] Early stopping at epoch {epoch}: no ValScore improvement for "
                f"{EARLY_STOP_PATIENCE} epochs."
            )
            break

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
        if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
            raise ValueError("Expected test batch as (MRI, PET, meta)")
        mri, pet, meta = batch
        mri5 = mri.to(device, non_blocking=True)
        pet5 = pet.to(device, non_blocking=True)
        if mri5.dim() == 4:
            mri5 = mri5.unsqueeze(0)
        if pet5.dim() == 4:
            pet5 = pet5.unsqueeze(0)
        batch_size = mri5.size(0)
        flair5, clinical = _extract_generator_inputs(meta, device, batch_size)
        brain5 = _meta_to_tensor(meta, "brain_mask", device, dtype=torch.float32)
        fake5 = G(mri5, flair5, clinical)

        for i in range(batch_size):
            brain = brain5[i : i + 1]
            fake_m = fake5[i : i + 1] * brain
            pet_m = pet5[i : i + 1] * brain
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
    try:
        from scipy.stats import t as _t_dist

        def _tcrit(df):
            return float(_t_dist.ppf(0.975, df)) if df > 0 else float("nan")

    except Exception:

        def _tcrit(df):
            return 1.96 if df > 0 else float("nan")

    os.makedirs(out_dir, exist_ok=True)
    G.to(device)
    G.eval()

    sids: List[str] = []
    ssim_list: List[float] = []
    psnr_list: List[float] = []
    mse_list: List[float] = []
    mmd_list: List[float] = []

    run_dir = os.path.dirname(out_dir) if os.path.basename(out_dir) else out_dir

    for batch_idx, batch in enumerate(test_loader):
        if not (isinstance(batch, (list, tuple)) and len(batch) == 3):
            raise ValueError("Expected test batch as (MRI, PET, meta)")
        mri, pet, meta = batch
        metas = _meta_as_list(meta)

        mri5 = mri.to(device, non_blocking=True)
        pet5 = pet.to(device, non_blocking=True)
        if mri5.dim() == 4:
            mri5 = mri5.unsqueeze(0)
        if pet5.dim() == 4:
            pet5 = pet5.unsqueeze(0)
        batch_size = mri5.size(0)
        if len(metas) != batch_size:
            raise RuntimeError("Meta batch size does not match evaluation batch size")

        flair5, clinical = _extract_generator_inputs(meta, device, batch_size)
        fake5 = G(mri5, flair5, clinical)

        for i, meta_i in enumerate(metas):
            sid = _safe_name(meta_i.get("sid", f"sample_{batch_idx:04d}_{i:02d}"))
            sids.append(sid)
            subdir = os.path.join(out_dir, sid)
            os.makedirs(subdir, exist_ok=True)

            mri_i = mri5[i : i + 1]
            pet_i = pet5[i : i + 1]
            fake_i = fake5[i : i + 1]

            if "brain_mask" not in meta_i:
                raise RuntimeError(f"{sid}: missing brain_mask in meta")
            brain = _sample_meta_tensor(meta_i["brain_mask"]).unsqueeze(0).to(device=device, dtype=torch.float32)

            fake_m = fake_i * brain
            pet_m = pet_i * brain

            ssim_val = ssim3d(fake_m, pet_m, data_range=data_range).item()
            psnr_val = psnr(fake_m, pet_m, data_range=data_range)
            mse_val = F.mse_loss(fake_m, pet_m).item()
            mmd_val = mmd_gaussian(pet_m, fake_m, num_voxels=mmd_voxels, mask=brain)

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)
            mse_list.append(mse_val)
            mmd_list.append(mmd_val)

            mri_np = mri_i.squeeze(0).squeeze(0).detach().cpu().numpy()
            pet_np = pet_i.squeeze(0).squeeze(0).detach().cpu().numpy()
            fake_np = fake_i.squeeze(0).squeeze(0).detach().cpu().numpy()
            err_np = np.abs(fake_np - pet_np)

            cur_shape = tuple(mri_np.shape)
            orig_shape = tuple(meta_i.get("orig_shape", cur_shape))
            if resample_back_to_t1 and tuple(orig_shape) != tuple(cur_shape):
                zf = (
                    float(orig_shape[0]) / float(cur_shape[0]),
                    float(orig_shape[1]) / float(cur_shape[1]),
                    float(orig_shape[2]) / float(cur_shape[2]),
                )
                mri_np = nd_zoom(mri_np, zf, order=1)
                pet_np = nd_zoom(pet_np, zf, order=1)
                fake_np = nd_zoom(fake_np, zf, order=1)
                err_np = nd_zoom(err_np, zf, order=1)
                affine_to_use = meta_i.get("t1_affine", np.eye(4))
            else:
                resized_to = meta_i.get("resized_to", None)
                affine_to_use = meta_i.get("t1_affine", np.eye(4)) if resized_to is None else np.eye(4)

            _save_nifti(mri_np, affine_to_use, os.path.join(subdir, "MRI.nii.gz"))
            _save_nifti(pet_np, affine_to_use, os.path.join(subdir, "PET_gt.nii.gz"))
            _save_nifti(fake_np, affine_to_use, os.path.join(subdir, "PET_fake.nii.gz"))
            _save_nifti(err_np, affine_to_use, os.path.join(subdir, "PET_abs_error.nii.gz"))

    def _mean_std_ci(vals):
        arr = np.asarray(vals, dtype=np.float64)
        n = arr.size
        mean = float(arr.mean()) if n > 0 else float("nan")
        std = float(arr.std(ddof=1)) if n > 1 else float("nan")
        se = (std / np.sqrt(n)) if n > 1 else float("nan")
        tcrit = _tcrit(n - 1)
        lo = mean - tcrit * se if n > 1 else float("nan")
        hi = mean + tcrit * se if n > 1 else float("nan")
        return mean, std, n, lo, hi

    m_ssim, sd_ssim, n_ssim, lo_ssim, hi_ssim = _mean_std_ci(ssim_list)
    m_psnr, sd_psnr, n_psnr, lo_psnr, hi_psnr = _mean_std_ci(psnr_list)
    m_mse, sd_mse, n_mse, lo_mse, hi_mse = _mean_std_ci(mse_list)
    m_mmd, sd_mmd, n_mmd, lo_mmd, hi_mmd = _mean_std_ci(mmd_list)

    per_subj_csv = os.path.join(run_dir, "per_subject_metrics.csv")
    with open(per_subj_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sid", "SSIM", "PSNR", "MSE", "MMD"])
        for sid, ssim_v, psnr_v, mse_v, mmd_v in zip(sids, ssim_list, psnr_list, mse_list, mmd_list):
            writer.writerow([sid, ssim_v, psnr_v, mse_v, mmd_v])

    summary_json = os.path.join(run_dir, "test_metrics_summary.json")
    summary = {
        "N": n_ssim,
        "SSIM": m_ssim,
        "SSIM_std": sd_ssim,
        "SSIM_lo95": lo_ssim,
        "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr,
        "PSNR_std": sd_psnr,
        "PSNR_lo95": lo_psnr,
        "PSNR_hi95": hi_psnr,
        "MSE": m_mse,
        "MSE_std": sd_mse,
        "MSE_lo95": lo_mse,
        "MSE_hi95": hi_mse,
        "MMD": m_mmd,
        "MMD_std": sd_mmd,
        "MMD_lo95": lo_mmd,
        "MMD_hi95": hi_mmd,
        "per_subject_csv": per_subj_csv,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "N": n_ssim,
        "SSIM": m_ssim,
        "SSIM_std": sd_ssim,
        "SSIM_lo95": lo_ssim,
        "SSIM_hi95": hi_ssim,
        "PSNR": m_psnr,
        "PSNR_std": sd_psnr,
        "PSNR_lo95": lo_psnr,
        "PSNR_hi95": hi_psnr,
        "MSE": m_mse,
        "MSE_std": sd_mse,
        "MSE_lo95": lo_mse,
        "MSE_hi95": hi_mse,
        "MMD": m_mmd,
        "MMD_std": sd_mmd,
        "MMD_lo95": lo_mmd,
        "MMD_hi95": hi_mmd,
        "per_subject_csv": per_subj_csv,
        "summary_json": summary_json,
    }
