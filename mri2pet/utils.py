from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

def _pad_or_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center pad/crop x spatially to match ref's D,H,W."""
    _, _, D, H, W = x.shape
    _, _, Dr, Hr, Wr = ref.shape

    d_pad = max(0, Dr - D)
    h_pad = max(0, Hr - H)
    w_pad = max(0, Wr - W)

    if d_pad or h_pad or w_pad:
        pad = (w_pad // 2, w_pad - w_pad // 2,
               h_pad // 2, h_pad - h_pad // 2,
               d_pad // 2, d_pad - d_pad // 2)
        x = F.pad(x, pad, mode='constant', value=0.)

    _, _, D2, H2, W2 = x.shape
    d_start = max(0, (D2 - Dr) // 2)
    h_start = max(0, (H2 - Hr) // 2)
    w_start = max(0, (W2 - Wr) // 2)
    x = x[:, :, d_start:d_start+Dr, h_start:h_start+Hr, w_start:w_start+Wr]
    return x

def _safe_name(s: str) -> str:
    s = str(s)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

def _save_nifti(vol: np.ndarray, affine: np.ndarray, path: str):
    img = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(img, path)

def _as_int_tuple3(x: Union[Tuple[int,int,int], List[Any], np.ndarray, torch.Tensor]) -> Tuple[int,int,int]:
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if isinstance(x, (list, tuple)):
        vals = []
        for v in x:
            if isinstance(v, torch.Tensor):
                vals.append(int(v.detach().cpu().reshape(-1)[0].item()))
            else:
                vals.append(int(v))
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])
        raise ValueError(f"orig/cur shape has unexpected length: {vals}")
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            return tuple(int(t) for t in x)
        except Exception:
            raise ValueError(f"Cannot parse shape from type: {type(x)}")
    flat = np.array(arr).astype(np.int64).reshape(-1)
    if flat.size < 3:
        raise ValueError(f"Shape vector too small: {flat}")
    return (int(flat[0]), int(flat[1]), int(flat[2]))

def _meta_unbatch(meta: Any) -> Dict[str, Any]:
    """Convert collated meta back to a plain dict for B=1."""
    if isinstance(meta, list):
        if len(meta) == 0:
            return {}
        if isinstance(meta[0], dict):
            return meta[0]
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in meta.items():
        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        out[k] = v
    if "orig_shape" in out:
        out["orig_shape"] = _as_int_tuple3(out["orig_shape"])
    if "cur_shape" in out:
        out["cur_shape"]  = _as_int_tuple3(out["cur_shape"])
    return out
