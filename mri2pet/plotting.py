import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Sequence
import csv

def save_loss_curves(history: Dict[str, Sequence[float]], out_path: str):
    n_plots = 1
    has_aux = any(k in history for k in ("train_stage_ord", "train_braak", "train_delta_out", "train_alpha"))
    if has_aux:
        n_plots = 3

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
        if "train_stage_ord" in history:
            ax2.plot(history["train_stage_ord"], label="Stage Ord (BCE)")
        if "train_braak" in history:
            ax2.plot(history["train_braak"], label="Braak (SmoothL1)")
        if "train_delta_out" in history:
            ax2.plot(history["train_delta_out"], label="Delta Out Reg")
        if "val_braak" in history and len(history["val_braak"]) > 0:
            ax2.plot(history["val_braak"], label="Val Braak")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Auxiliary Losses")
        ax2.legend()

        # Panel 3: Alpha
        ax3 = axes[2]
        if "train_alpha" in history:
            ax3.plot(history["train_alpha"], label="Alpha (sigmoid)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Value")
        ax3.set_title("Residual Alpha")
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
