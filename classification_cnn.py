#!/usr/bin/env python3
import os, argparse, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# --- seeds & speed ---
torch.backends.cudnn.benchmark = True
torch.manual_seed(0); np.random.seed(0)

# --- label rule ---
def braak_stage(row, thr=1.2):
    """Return 0,1,2,3 Braak stage based on CSV SUVR numbers."""
    if row["Braak5_6"] > thr: return 3
    elif row["Braak3_4"] > thr: return 2
    elif row["Braak1_2"] > thr: return 1
    else: return 0

def braak_label(row, thr=1.2):
    """
    Binary label:
      0 -> stages 0 or 1
      1 -> stages 2 or 3
    """
    s = braak_stage(row, thr)
    return 1 if s >= 2 else 0

# --- binary AUC (safe) ---
def binary_auc_safe(y_true, score):
    # y_true in {0,1}, score is any continuous "more positive -> more likely class 1"
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, score))

# --- dataset (no normalization, no augmentation) ---
class VolDataset(Dataset):
    """
    Returns:
      vol:  [1, D, H, W] float32 torch tensor (resized to size^3)
      yreg: [3] float32 torch tensor = (Braak1_2, Braak3_4, Braak5_6)
      ybin: int (0/1) derived from CSV rule
    """
    def __init__(self, df, path_col, size=128, thr=1.2):
        self.paths = df[path_col].tolist()
        self.yreg = df[["Braak1_2", "Braak3_4", "Braak5_6"]].astype(np.float32).values
        self.ybin = df.apply(lambda r: braak_label(r, thr), axis=1).astype(int).values
        self.size = size
        self.thr = thr

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = nib.load(self.paths[i]).get_fdata().astype(np.float32)

        # Make sure a is 3D: (D, H, W)
        if a.ndim == 4:
            if a.shape[-1] == 1:
                a = a[..., 0]
            else:
                a = a.mean(axis=-1)
        elif a.ndim != 3:
            raise ValueError(f"Unexpected volume shape {a.shape} for {self.paths[i]}")

        t = torch.from_numpy(a)[None, None, ...]  # [1,1,D,H,W]
        t = F.interpolate(t, size=(self.size,)*3, mode="trilinear", align_corners=False)
        t = t[0]  # -> [1,D,H,W]

        yreg = torch.from_numpy(self.yreg[i])      # [3]
        ybin = int(self.ybin[i])
        return t, yreg, ybin

# --- 3D CNN (removed normalization layers; reduced pooling depth) ---
class CNNReg(nn.Module):
    """
    Predicts 3 continuous values: (Braak1_2, Braak3_4, Braak5_6)
    Pooling is only 5x (ends at 4x4x4 when input is 128^3).
    """
    def __init__(self, out_dim=3):
        super().__init__()

        def blk(cin, cout, pool=True):
            layers = [
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool3d(2))
            return nn.Sequential(*layers)

        # 128 -> 64 -> 32 -> 16 -> 8 -> 4 (5 pools), then one more conv block w/o pooling
        self.features = nn.Sequential(
            blk(1,   16, pool=True),
            blk(16,  32, pool=True),
            blk(32,  48, pool=True),
            blk(48,  64, pool=True),
            blk(64,  96, pool=True),
            blk(96, 128, pool=False),  # keep 4x4x4 (no 6th pool)
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        z = self.features(x)        # [B,128,4,4,4] for size=128
        z = z.mean(dim=(2,3,4))     # GAP -> [B,128]
        return self.head(z)         # [B,3]

# --- one fold run ---
def run_fold(df, tr_idx, te_idx, path_col, size, device, epochs=70, bs=2, thr=1.2):
    # stratify splits using the same binary labels as before
    y_all_bin = df.apply(lambda r: braak_label(r, thr), axis=1).astype(int).values
    tr_sub, val_sub = train_test_split(
        tr_idx, test_size=0.2, stratify=y_all_bin[tr_idx], random_state=0
    )

    ds_tr  = VolDataset(df.iloc[tr_sub].reset_index(drop=True), path_col, size=size, thr=thr)
    ds_val = VolDataset(df.iloc[val_sub].reset_index(drop=True), path_col, size=size, thr=thr)
    ds_te  = VolDataset(df.iloc[te_idx].reset_index(drop=True),  path_col, size=size, thr=thr)

    dl_tr  = DataLoader(ds_tr,  batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    dl_te  = DataLoader(ds_te,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNReg(out_dim=3).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    # Regression loss (Huber / Smooth L1 is usually more robust than MSE)
    crit = nn.SmoothL1Loss()

    best, best_state, bad, patience = np.inf, None, 0, 50
    for epoch in range(1, epochs+1):
        # --- train ---
        model.train()
        tloss, ntr = 0.0, 0
        for vol, yreg, _ybin in dl_tr:
            vol = vol.to(device, non_blocking=True)
            yreg = yreg.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(vol)                 # [B,3]
            loss = crit(pred, yreg)
            loss.backward()
            opt.step()

            tloss += loss.item() * vol.size(0)
            ntr += vol.size(0)

        train_loss = tloss / max(1, ntr)

        # --- val ---
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for vol, yreg, _ybin in dl_val:
                vol = vol.to(device, non_blocking=True)
                yreg = yreg.to(device, non_blocking=True)
                pred = model(vol)
                vloss += crit(pred, yreg).item() * vol.size(0)
                n += vol.size(0)
        vloss /= max(1, n)

        prev_lr = opt.param_groups[0]['lr']
        sched.step(vloss)
        new_lr = opt.param_groups[0]['lr']
        lr_msg = f" | lr↓->{new_lr:.6f}" if new_lr < prev_lr - 1e-12 else ""

        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        print(f"Epoch {epoch:03d}: train_reg_loss={train_loss:.4f} val_reg_loss={vloss:.4f} best_val={best:.4f} bad={bad}{lr_msg}")
        if bad >= patience:
            print(f"[EarlyStopping] Stop at epoch {epoch} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- test: regression outputs -> threshold rule -> binary classification metrics ---
    model.eval()
    pred_reg_list, yreg_list, ybin_list = [], [], []
    with torch.no_grad():
        for vol, yreg, ybin in dl_te:
            vol = vol.to(device, non_blocking=True)
            pred = model(vol).cpu().numpy()
            pred_reg_list.append(pred)
            yreg_list.append(yreg.numpy())
            ybin_list.append(np.asarray(ybin, dtype=np.int64))

    pred_reg = np.concatenate(pred_reg_list, axis=0)   # [N,3]
    yreg_true = np.concatenate(yreg_list, axis=0)      # [N,3]
    ybin_true = np.concatenate(ybin_list, axis=0)      # [N]

    # Binary rule for your label: stage>=2 <=> (Braak3_4>thr OR Braak5_6>thr)
    score = np.maximum(pred_reg[:, 1], pred_reg[:, 2])      # continuous score for AUC
    pred_bin = (score > thr).astype(np.int64)

    acc = accuracy_score(ybin_true, pred_bin)
    auc = binary_auc_safe(ybin_true, score)
    cm  = confusion_matrix(ybin_true, pred_bin, labels=[0, 1])

    # Useful regression diagnostics (optional but helpful)
    mae = np.mean(np.abs(pred_reg - yreg_true), axis=0)     # per-target MAE
    mae_mean = float(np.mean(mae))

    return acc, auc, cm, mae, mae_mean

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--modality", choices=["mri","pet_gt","pet_fake","pet_raw"], default="pet_raw")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bs", type=int, default=5)
    args = ap.parse_args()

    thr = 1.2

    df = pd.read_csv(args.csv)
    # Checks for metadata. We do NOT check for all image columns (pet_raw/gt/fake)
    # because the user might only have one of them in the CSV.
    needed = ["subject", "Braak1_2", "Braak3_4", "Braak5_6"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    path_col = args.modality
    if path_col not in df.columns:
        raise ValueError(f"The selected modality column '{path_col}' is missing from the CSV.")
    df = df[df[path_col].apply(lambda p: isinstance(p, str) and os.path.exists(p))].reset_index(drop=True)
    if len(df) == 0:
        print("[fatal] No valid volumes found for selected modality.")
        return
    print(f"[info] using {len(df)} rows with existing files from column '{path_col}'.")

    # Stratify folds on the same binary label as before
    y_all_bin = df.apply(lambda r: braak_label(r, thr), axis=1).astype(int).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    accs, aucs, cms, maes = [], [], [], []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y_all_bin), 1):
        acc, auc, cm, mae, mae_mean = run_fold(
            df, tr, te, path_col, args.size, device,
            epochs=args.epochs, bs=args.bs, thr=thr
        )
        accs.append(acc); aucs.append(auc); cms.append(cm); maes.append(mae)

        print(f"Fold{k}: AUC={auc:.3f} ACC={acc:.3f} | MAE(B12,B34,B56)={mae} (mean={mae_mean:.4f})")
        print(f"Confusion:\n{cm}\n")

    acc_mean = float(np.mean(accs))
    acc_var  = float(np.var(accs, ddof=1) if len(accs) > 1 else 0.0)
    auc_mean = float(np.nanmean(aucs))
    auc_var  = float(np.nanvar(aucs, ddof=1) if np.sum(~np.isnan(aucs)) > 1 else 0.0)
    cm_sum   = sum(cms)

    maes = np.stack(maes, axis=0)  # [folds, 3]
    mae_mean = np.mean(maes, axis=0)
    mae_var  = np.var(maes, axis=0, ddof=1) if maes.shape[0] > 1 else np.zeros(3, dtype=np.float32)

    print(f"== {args.modality} (train regression; eval binary: {0,1}->{0}, {2,3}->{1}) ==")
    print(f"ACC mean={acc_mean:.3f} var={acc_var:.4f} | AUC mean={auc_mean:.3f} var={auc_var:.4f}")
    print(f"MAE mean(B12,B34,B56)={mae_mean} | MAE var={mae_var}")
    print("Confusion (sum over folds):\n", cm_sum)

if __name__ == "__main__":
    main()
