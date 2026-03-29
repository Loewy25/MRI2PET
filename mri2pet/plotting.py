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
    if "val_global" in history and len(history["val_global"]) > 0:
        plt.plot(history["val_global"], label="Val Global")
    if "val_roi" in history and len(history["val_roi"]) > 0:
        plt.plot(history["val_roi"], label="Val ROI")
    if "val_score" in history and len(history["val_score"]) > 0:
        plt.plot(history["val_score"], label="Val Score")
    if "train_aux" in history and len(history["train_aux"]) > 0:
        plt.plot(history["train_aux"], label="Train Aux")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_history_csv(history: Dict[str, Sequence[float]], out_csv: str):
    keys = [
        "train_G",
        "train_D",
        "train_global",
        "train_roi",
        "train_gan",
        "train_con",
        "train_high",
        "train_56",
        "train_aux",
        "val_global",
        "val_roi",
        "val_score",
    ]
    L = max(len(history.get(k, [])) for k in keys)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(L):
            row = [i + 1]
            for key in keys:
                vals = history.get(key, [])
                row.append(vals[i] if i < len(vals) else "")
            w.writerow(row)
