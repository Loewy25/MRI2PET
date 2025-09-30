# mri2pet/memory.py
from typing import Dict, List, Optional
import torch

class ROIMemory:
    """
    Two queues per ROI:
      - 'Phat' : negatives for M->P̂ (store P̂ embeddings)
      - 'P'    : negatives for P̂->P (store P   embeddings)
    """
    def __init__(self, max_len: int = 512):
        self.max_len = int(max_len)
        self.device: Optional[torch.device] = None
        self.dim: Optional[int] = None
        self.store: Dict[str, Dict[str, torch.Tensor]] = {'Phat': {}, 'P': {}}
        self.ptr:   Dict[str, Dict[str, int]]          = {'Phat': {}, 'P': {}}
        self.size:  Dict[str, Dict[str, int]]          = {'Phat': {}, 'P': {}}

    def maybe_init(self, roi_names: List[str], dim: int, device: torch.device):
        if self.dim is None:
            self.dim = int(dim)
        if self.device is None:
            self.device = device
        for kind in ('Phat','P'):
            for name in roi_names:
                if name not in self.store[kind]:
                    buf = torch.empty(self.max_len, self.dim, device=self.device)
                    self.store[kind][name] = buf
                    self.ptr[kind][name]   = 0
                    self.size[kind][name]  = 0

    @torch.no_grad()
    def get(self, roi_name: str, kind: str) -> torch.Tensor:
        if (kind not in self.store) or (roi_name not in self.store[kind]):
            # return empty tensor on the right device/dim if known
            d = self.dim if self.dim is not None else 1
            dev = self.device if self.device is not None else 'cpu'
            return torch.empty(0, d, device=dev)
        sz = self.size[kind][roi_name]
        buf = self.store[kind][roi_name]
        if sz == 0:
            return buf[:0]
        p = self.ptr[kind][roi_name]
        if sz < self.max_len:
            return buf[:sz]
        # full: roll so oldest first (optional; not required for CE)
        return torch.cat([buf[p:], buf[:p]], dim=0)

    @torch.no_grad()
    def enqueue(self, roi_name: str, kind: str, feats: torch.Tensor):
        if feats is None or feats.numel() == 0:
            return
        feats = feats.detach()
        if roi_name not in self.store[kind]:
            self.maybe_init([roi_name], feats.shape[-1], feats.device)
        buf = self.store[kind][roi_name]
        p   = self.ptr[kind][roi_name]
        sz  = self.size[kind][roi_name]
        n   = feats.size(0)
        for i in range(n):
            buf[p].copy_(feats[i])
            p = (p + 1) % self.max_len
            if sz < self.max_len:
                sz += 1
        self.ptr[kind][roi_name]  = p
        self.size[kind][roi_name] = sz
