from typing import Tuple, Optional
from math import log10
import torch
import torch.nn.functional as F

def ssim3d(x: torch.Tensor, y: torch.Tensor, ksize: int = 3,
           k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0) -> torch.Tensor:
    pad = ksize // 2
    mu_x = F.avg_pool3d(x, ksize, stride=1, padding=pad)
    mu_y = F.avg_pool3d(y, ksize, stride=1, padding=pad)
    sigma_x = F.avg_pool3d(x * x, ksize, stride=1, padding=pad) - mu_x ** 2
    sigma_y = F.avg_pool3d(y * y, ksize, stride=1, padding=pad) - mu_y ** 2
    sigma_xy = F.avg_pool3d(x * y, ksize, stride=1, padding=pad) - mu_x * mu_y
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return ssim_map.mean()

def ms_ssim3d(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0,
              ksize: int = 11, levels: int = 4,
              weights: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)) -> torch.Tensor:
    pad = ksize // 2
    def _ssim3d_local(a, b):
        mu_a = F.avg_pool3d(a, ksize, stride=1, padding=pad)
        mu_b = F.avg_pool3d(b, ksize, stride=1, padding=pad)
        sigma_a = F.avg_pool3d(a*a, ksize, stride=1, padding=pad) - mu_a**2
        sigma_b = F.avg_pool3d(b*b, ksize, stride=1, padding=pad) - mu_b**2
        sigma_ab = F.avg_pool3d(a*b, ksize, stride=1, padding=pad) - mu_a*mu_b
        k1_, k2_ = 0.01, 0.03
        C1 = (k1_ * data_range) ** 2
        C2 = (k2_ * data_range) ** 2
        ssim_map = ((2*mu_a*mu_b + C1)*(2*sigma_ab + C2)) / ((mu_a**2 + mu_b**2 + C1)*(sigma_a + sigma_b + C2) + 1e-8)
        ssim_mean = ssim_map.mean()
        ssim_01 = torch.clamp((ssim_mean + 1.0) * 0.5, min=1e-6, max=1.0)
        return ssim_01
    ws = torch.tensor(weights[:levels], device=x.device, dtype=x.dtype)
    ws = ws / ws.sum()
    vals = []
    a, b = x, y
    for l in range(levels):
        vals.append(_ssim3d_local(a, b))
        if l < levels - 1:
            a = F.avg_pool3d(a, kernel_size=2, stride=2, padding=0)
            b = F.avg_pool3d(b, kernel_size=2, stride=2, padding=0)
    vals_t = torch.stack(vals)
    ms = torch.exp(torch.sum(ws * torch.log(vals_t)))
    return ms

def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x, y)

def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)

def ssim3d_masked(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                  ksize: int = 3, k1: float = 0.01, k2: float = 0.03,
                  data_range: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    pad = ksize // 2
    k = torch.ones((1, 1, ksize, ksize, ksize), device=x.device, dtype=x.dtype)
    def wavg(z):
        z_sum = F.conv3d(z * mask, k, padding=pad)
        m_sum = F.conv3d(mask,     k, padding=pad).clamp_min(eps)
        return z_sum / m_sum
    mu_x = wavg(x); mu_y = wavg(y)
    mu_x2 = mu_x * mu_x; mu_y2 = mu_y * mu_y; mu_xy = mu_x * mu_y
    sigma_x  = wavg(x * x) - mu_x2
    sigma_y  = wavg(y * y) - mu_y2
    sigma_xy = wavg(x * y) - mu_xy
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + eps)
    win_hits = (F.conv3d(mask, k, padding=pad) > 0).to(x.dtype)
    return (ssim_map * win_hits).sum() / win_hits.sum().clamp_min(eps)

def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    num = (((x - y) ** 2) * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def masked_psnr(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                data_range: float = 1.0, eps: float = 1e-6) -> float:
    mse = masked_mse(x, y, mask, eps=eps)
    if mse.item() == 0:
        return float('inf')
    return 10.0 * log10((data_range ** 2) / mse.item())

@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return float('inf')
    return 10.0 * log10((data_range ** 2) / mse)

@torch.no_grad()
def mmd_gaussian(real: torch.Tensor,
                 fake: torch.Tensor,
                 num_voxels: int = 2048,
                 sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
                 mask: Optional[torch.Tensor] = None) -> float:
    B = real.size(0)
    dev = real.device
    total = 0.0
    n_valid = 0
    for i in range(B):
        r = real[i, 0].reshape(-1)
        f = fake[i, 0].reshape(-1)

        if mask is not None:
            m = (mask[i, 0].reshape(-1) > 0.5)
            idx_pool = torch.nonzero(m, as_tuple=False).reshape(-1)
            if idx_pool.numel() == 0:
                continue
            S = min(num_voxels, idx_pool.numel())
            sel = idx_pool[torch.randint(0, idx_pool.numel(), (S,), device=dev)]
            r_s = r[sel].view(S, 1)
            f_s = f[sel].view(S, 1)
        else:
            S = min(num_voxels, r.numel(), f.numel())
            ridx = torch.randint(0, r.numel(), (S,), device=dev)
            fidx = torch.randint(0, f.numel(), (S,), device=dev)
            r_s = r[ridx].view(S, 1)
            f_s = f[fidx].view(S, 1)

        d_rr = (r_s - r_s.t()).pow(2)
        d_ff = (f_s - f_s.t()).pow(2)
        d_rf = (r_s - f_s.t()).pow(2)

        mmd = 0.0
        for s in sigmas:
            Krr = torch.exp(-d_rr / (2 * s * s))
            Kff = torch.exp(-d_ff / (2 * s * s))
            Krf = torch.exp(-d_rf / (2 * s * s))
            mmd += Krr.mean() + Kff.mean() - 2 * Krf.mean()
        mmd /= len(sigmas)
        total += mmd.item()
        n_valid += 1
    return total / max(1, n_valid)
