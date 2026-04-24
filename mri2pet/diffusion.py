from typing import Any, Dict, Optional, Tuple

import torch

from .config import DIFF_BETA_END, DIFF_BETA_START, DIFF_TIMESTEPS


def make_beta_schedule(
    timesteps: int = DIFF_TIMESTEPS,
    beta_start: float = DIFF_BETA_START,
    beta_end: float = DIFF_BETA_END,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    out = a.gather(0, t.to(device=a.device, dtype=torch.long))
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_ab = _extract(schedule["sqrt_alphas_cumprod"], t, tuple(x0.shape)).to(dtype=x0.dtype)
    sqrt_omab = _extract(schedule["sqrt_one_minus_alphas_cumprod"], t, tuple(x0.shape)).to(dtype=x0.dtype)
    return sqrt_ab * x0 + sqrt_omab * noise


def predict_x0_from_eps(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_ab = _extract(schedule["sqrt_alphas_cumprod"], t, tuple(xt.shape)).to(dtype=xt.dtype)
    sqrt_omab = _extract(schedule["sqrt_one_minus_alphas_cumprod"], t, tuple(xt.shape)).to(dtype=xt.dtype)
    return (xt - sqrt_omab * eps) / sqrt_ab.clamp_min(1e-8)


@torch.no_grad()
def ddim_sample_loop(
    model,
    shape: Tuple[int, int, int, int, int],
    schedule: Dict[str, torch.Tensor],
    steps: int,
    t1: torch.Tensor,
    flair: torch.Tensor,
    pet_base: torch.Tensor,
    brain_mask: torch.Tensor,
    cortex_mask: torch.Tensor,
    clinical: torch.Tensor,
    x_start: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    device = t1.device
    total_t = int(schedule["betas"].numel())
    sample_steps = max(1, min(int(steps), total_t))
    step_ids = torch.linspace(total_t - 1, 0, sample_steps, device=device).round().long()
    step_ids = torch.unique_consecutive(step_ids)

    x = x_start if x_start is not None else torch.randn(shape, device=device, dtype=t1.dtype)
    aux_last: Dict[str, Any] = {}
    for i, step in enumerate(step_ids):
        t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
        eps_pred, aux_last = model(
            x, t,
            t1=t1,
            flair=flair,
            pet_base=pet_base,
            brain_mask=brain_mask,
            cortex_mask=cortex_mask,
            clinical=clinical,
            return_aux=True,
        )
        x0_pred = predict_x0_from_eps(x, t, eps_pred, schedule)
        if i == len(step_ids) - 1:
            x = x0_pred
            break

        next_step = step_ids[i + 1]
        ab_prev = schedule["alphas_cumprod"][next_step].to(device=device, dtype=x.dtype)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_pred

    return x, aux_last
