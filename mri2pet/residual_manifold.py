import csv
import os
from typing import Dict, Tuple

import numpy as np


def _basis_to_kdhw(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 5:
        if arr.shape[1] != 1:
            raise ValueError(f"{name} expected channel dimension 1, got shape {arr.shape}")
        arr = arr[:, 0]
    if arr.ndim != 4:
        raise ValueError(f"{name} expected [K,D,H,W] or [K,1,D,H,W], got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_basis_arrays(basis_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    if not basis_dir:
        raise RuntimeError("CDRM_BASIS_DIR/basis_dir is required for residual_manifold")
    cal_path = os.path.join(basis_dir, "B_cal.npy")
    dis_path = os.path.join(basis_dir, "B_dis.npy")
    if not os.path.isfile(cal_path):
        raise FileNotFoundError(f"Missing calibration basis: {cal_path}")
    if not os.path.isfile(dis_path):
        raise FileNotFoundError(f"Missing disease basis: {dis_path}")
    return _basis_to_kdhw(np.load(cal_path), "B_cal"), _basis_to_kdhw(np.load(dis_path), "B_dis")


def load_coeff_targets(csv_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    if not csv_path:
        raise RuntimeError("CDRM_COEFF_CSV is required for residual_manifold training")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Coefficient target CSV not found: {csv_path}")

    targets: Dict[str, Dict[str, np.ndarray]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "sid" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a sid column")
        c_cols = sorted([c for c in reader.fieldnames if c.startswith("c") and c[1:].isdigit()],
                        key=lambda x: int(x[1:]))
        a_cols = sorted([c for c in reader.fieldnames if c.startswith("a") and c[1:].isdigit()],
                        key=lambda x: int(x[1:]))
        if not c_cols or not a_cols:
            raise ValueError(f"{csv_path} must contain c0.. and a0.. coefficient columns")
        for row in reader:
            sid = (row.get("sid") or "").strip()
            if not sid:
                continue
            targets[sid] = {
                "c": np.asarray([float(row[c]) for c in c_cols], dtype=np.float32),
                "a": np.asarray([float(row[c]) for c in a_cols], dtype=np.float32),
            }
    return targets


def solve_signed_coefficients(
    residual: np.ndarray,
    basis: np.ndarray,
    mask: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    basis = _basis_to_kdhw(basis, "basis")
    mask = np.asarray(mask).astype(bool)
    y = np.asarray(residual, dtype=np.float32)[mask].astype(np.float64)
    A = basis[:, mask].T.astype(np.float64)  # [V,K]
    if A.size == 0:
        return np.zeros((basis.shape[0],), dtype=np.float32)
    gram = A.T @ A
    rhs = A.T @ y
    gram = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    coeff = np.linalg.solve(gram, rhs)
    return coeff.astype(np.float32)


def solve_nonnegative_coefficients(
    target: np.ndarray,
    basis: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    basis = _basis_to_kdhw(basis, "basis")
    mask = np.asarray(mask).astype(bool)
    y = np.asarray(target, dtype=np.float32)[mask].astype(np.float64)
    A = basis[:, mask].T.astype(np.float64)  # [V,K]
    if A.size == 0 or float(np.abs(y).sum()) == 0.0:
        return np.zeros((basis.shape[0],), dtype=np.float32)
    try:
        from scipy.optimize import nnls

        coeff, _ = nnls(A, y)
    except Exception:
        coeff = np.linalg.lstsq(A, y, rcond=None)[0]
        coeff = np.maximum(coeff, 0.0)
    return coeff.astype(np.float32)


def reconstruct_from_basis(
    c: np.ndarray,
    a: np.ndarray,
    B_cal: np.ndarray,
    B_dis: np.ndarray,
) -> np.ndarray:
    B_cal = _basis_to_kdhw(B_cal, "B_cal")
    B_dis = _basis_to_kdhw(B_dis, "B_dis")
    return (
        np.einsum("k,kdhw->dhw", np.asarray(c, dtype=np.float32), B_cal)
        + np.einsum("k,kdhw->dhw", np.asarray(a, dtype=np.float32), B_dis)
    ).astype(np.float32)


def fit_joint_coefficients(
    residual: np.ndarray,
    B_cal: np.ndarray,
    B_dis: np.ndarray,
    brain_mask: np.ndarray,
    ridge: float = 1e-4,
    iters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    B_cal = _basis_to_kdhw(B_cal, "B_cal")
    B_dis = _basis_to_kdhw(B_dis, "B_dis")
    brain = np.asarray(brain_mask).astype(bool)
    c = solve_signed_coefficients(residual, B_cal, brain, ridge=ridge)
    a = np.zeros((B_dis.shape[0],), dtype=np.float32)
    dis_mask = np.logical_and(brain, np.any(np.abs(B_dis) > 0.0, axis=0))
    if not np.any(dis_mask):
        dis_mask = brain

    for _ in range(max(1, int(iters))):
        cal_res = np.einsum("k,kdhw->dhw", c, B_cal)
        target_a = np.asarray(residual, dtype=np.float32) - cal_res
        a = solve_nonnegative_coefficients(target_a, B_dis, dis_mask)
        dis_res = np.einsum("k,kdhw->dhw", a, B_dis)
        c = solve_signed_coefficients(np.asarray(residual, dtype=np.float32) - dis_res, B_cal, brain, ridge=ridge)
    return c.astype(np.float32), a.astype(np.float32)
