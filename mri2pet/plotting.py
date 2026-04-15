import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Sequence
import csv

def _has_series(history: Dict[str, Sequence[float]], key: str, require_nonzero: bool = False) -> bool:
    vals = history.get(key, [])
    if len(vals) == 0:
        return False
    if not require_nonzero:
        return True
    return any(abs(float(v)) > 1e-12 for v in vals)

def save_loss_curves(history: Dict[str, Sequence[float]], out_path: str):
    n_plots = 1
    has_aux = (
        _has_series(history, "train_braak")
        or _has_series(history, "train_delta_sup")
    )
    has_prior = (
        _has_series(history, "train_prior_in")
        or _has_series(history, "train_prior_out")
        or _has_series(history, "train_prior_ratio")
        or _has_series(history, "train_mod_in")
        or _has_series(history, "train_mod_out")
        or _has_series(history, "train_mod_ratio")
        or _has_series(history, "train_router_entropy")
        or _has_series(history, "train_router_top1")
    )
    if has_aux:
        n_plots = 2
    if has_prior:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Panel 1: G/D/Val recon
    ax = axes[0]
    if "train_G" in history:
        ax.plot(history["train_G"], label="Train G")
    if "train_D" in history:
        ax.plot(history["train_D"], label="Train D")
    if "val_recon" in history and len(history["val_recon"]) > 0:
        ax.plot(history["val_recon"], label="Val Recon")
    if "val_roi" in history and len(history["val_roi"]) > 0:
        ax.plot(history["val_roi"], label="Val ROI")
    if "val_score" in history and len(history["val_score"]) > 0:
        ax.plot(history["val_score"], label="Val Score", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Losses")
    ax.legend()

    if has_aux:
        # Panel 2: Aux losses
        ax2 = axes[1]
        if _has_series(history, "train_braak"):
            ax2.plot(history["train_braak"], label="Braak (SmoothL1)")
        if _has_series(history, "train_delta_sup"):
            ax2.plot(history["train_delta_sup"], label="Delta Sup")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Auxiliary Losses")
        ax2.legend()

    if has_prior:
        ax_idx = 2 if has_aux else 1
        ax3 = axes[ax_idx]
        if _has_series(history, "train_prior_in"):
            ax3.plot(history["train_prior_in"], label="Prior In Cortex")
        if _has_series(history, "train_mod_in"):
            ax3.plot(history["train_mod_in"], label="Mod In Cortex")
        if _has_series(history, "train_prior_out"):
            ax3.plot(history["train_prior_out"], label="Prior Out Cortex")
        if _has_series(history, "train_mod_out"):
            ax3.plot(history["train_mod_out"], label="Mod Out Cortex")
        if _has_series(history, "train_prior_ratio"):
            ax3.plot(history["train_prior_ratio"], label="Prior In/Out Ratio", linestyle="--")
        if _has_series(history, "train_mod_ratio"):
            ax3.plot(history["train_mod_ratio"], label="Mod In/Out Ratio", linestyle="--")
        if _has_series(history, "train_router_entropy"):
            ax3.plot(history["train_router_entropy"], label="Router Entropy")
        if _has_series(history, "train_router_top1"):
            ax3.plot(history["train_router_top1"], label="Router Top1 Mean")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Value")
        ax3.set_title("Regional Modulation Activity")
        ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_history_csv(history: Dict[str, Sequence[float]], out_csv: str):
    all_keys = [k for k in history if len(history[k]) > 0]
    L = max(len(history[k]) for k in all_keys) if all_keys else 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + all_keys)
        for i in range(L):
            row = [i + 1]
            for k in all_keys:
                vals = history[k]
                row.append(vals[i] if i < len(vals) else "")
            w.writerow(row)
