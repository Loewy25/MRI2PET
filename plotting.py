import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Sequence
import csv

def save_loss_curves(history: Dict[str, Sequence[float]], out_path: str):
    plt.figure(figsize=(7,5))
    if "train_G" in history:
        plt.plot(history["train_G"], label="Train G")
    if "train_D" in history:
        plt.plot(history["train_D"], label="Train D")
    if "val_recon" in history and len(history["val_recon"]) > 0:
        plt.plot(history["val_recon"], label="Val (L1 + 1-SSIM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_history_csv(history: Dict[str, Sequence[float]], out_csv: str):
    L = max(len(history.get("train_G", [])),
            len(history.get("train_D", [])),
            len(history.get("val_recon", [])))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_G", "train_D", "val_recon"])
        for i in range(L):
            row = [i+1,
                   history.get("train_G",  [None]*L)[i] if i < len(history.get("train_G",[])) else "",
                   history.get("train_D",  [None]*L)[i] if i < len(history.get("train_D",[])) else "",
                   history.get("val_recon",[None]*L)[i] if i < len(history.get("val_recon",[])) else ""]
            w.writerow(row)
