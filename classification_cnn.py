#!/usr/bin/env python3
import os, argparse, glob, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# --- seeds & speed ---
torch.backends.cudnn.benchmark = True
torch.manual_seed(0); np.random.seed(0)

# --- labels ---
def braak_label(row, thr=1.2):
    if row["Braak5_6"] > thr: return 3
    elif row["Braak3_4"] > thr: return 2
    elif row["Braak1_2"] > thr: return 1
    else: return 0

# --- dataset (no augmentation) ---
class VolDataset(Dataset):
    def __init__(self, df, path_col, size=128):
        self.paths = df[path_col].tolist()
        self.y = df.apply(braak_label, axis=1).astype(int).values
        self.size = size

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, i):
        a = nib.load(self.paths[i]).get_fdata().astype(np.float32)
        m = a != 0
        if m.any():
            x = a[m]
            a[m] = (x - x.mean()) / (x.std() + 1e-6)
            a[~m] = 0.0
        t = torch.from_numpy(a)[None, None, ...]  # [1,1,D,H,W]
        t = F.interpolate(t, size=(self.size,)*3, mode="trilinear", align_corners=False)
        t = t[0]  # -> [1,D,H,W]
        return t, int(self.y[i])

# --- model: 6 conv blocks + GAP head (InstanceNorm3d) ---
class CNN6(nn.Module):
    def __init__(self, ncls=4):
        super().__init__()
        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv3d(cin, cout, 3, padding=1),
                nn.InstanceNorm3d(cout, affine=True, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, 3, padding=1),
                nn.InstanceNorm3d(cout, affine=True, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2)  # /2
            )
        self.features = nn.Sequential(
            blk(1, 16),   # 128 -> 64
            blk(16, 32),  # 64  -> 32
            blk(32, 48),  # 32  -> 16
            blk(48, 64),  # 16  -> 8
            blk(64, 96),  # 8   -> 4
            blk(96, 128), # 4   -> 2
        )
        self.head = nn.Linear(128, ncls)

    def forward(self, x):                 # x: [B,1,D,H,W]
        z = self.features(x)              # [B,128,2,2,2]
        z = z.mean(dim=(2,3,4))           # GAP -> [B,128]
        return self.head(z)               # logits

# --- robust macro AUC (works even if a class is missing in test) ---
def macro_auc_safe(y_true, proba, n_classes=4):
    aucs = []
    for c in range(n_classes):
        yb = (y_true == c).astype(int)
        if yb.min() == 0 and yb.max() == 1:  # need pos & neg
            aucs.append(roc_auc_score(yb, proba[:, c]))
    return float(np.mean(aucs)) if aucs else float('nan')

# --- helper: resolve pet_gt to PUP SUVR path ---
def resolve_pet_gt_to_pup_suvr(pet_gt_path):
    """
    Example input path in CSV:
      /scratch/l.peiwang/kari_brainv33_top300/20520060F_T807_v1/PET_in_T1_masked.nii.gz
    We extract '20520060F_T807_v1' and search under:
      /ceph/chpc/mapped/benz04_kari/pup/20520060F_T807_v1/**/*msum_SUVR.nii.gz
    If not found -> print 'we don't have it' and return None.
    """
    if not isinstance(pet_gt_path, str) or not pet_gt_path:
        print("we don't have it")
        return None
    subj_id = os.path.basename(os.path.dirname(pet_gt_path))
    base = os.path.join("/ceph/chpc/mapped/benz04_kari/pup", subj_id)
    pattern = os.path.join(base, "**", "*msum_SUVR.nii.gz")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        print(f"we don't have it: {subj_id}")
        return None
    # take the first match
    return hits[0]

# --- one fold ---
def run_fold(df, tr_idx, te_idx, path_col, size, device, epochs=70, bs=2):
    y_all = df.apply(braak_label, axis=1).astype(int).values
    tr_sub, val_sub = train_test_split(tr_idx, test_size=0.2, stratify=y_all[tr_idx], random_state=0)

    # Datasets (no augmentation)
    ds_tr  = VolDataset(df.iloc[tr_sub].reset_index(drop=True), path_col, size=size)
    ds_val = VolDataset(df.iloc[val_sub].reset_index(drop=True), path_col, size=size)
    ds_te  = VolDataset(df.iloc[te_idx].reset_index(drop=True),  path_col, size=size)

    # Loaders (no weighted sampler; simple shuffle on train)
    dl_tr  = DataLoader(ds_tr,  batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    dl_te  = DataLoader(ds_te,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # class weights for CE (kept)
    y_tr = y_all[tr_sub]
    counts = np.bincount(y_tr, minlength=4)
    w = counts.sum() / (counts + 1e-6); w = (w / w.mean()).astype(np.float32)
    w = torch.tensor(w, device=device)

    model = CNN6().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    crit  = nn.CrossEntropyLoss(weight=w)

    best, best_state, bad, patience = np.inf, None, 0, 50
    for epoch in range(1, epochs+1):
        # --------- train ----------
        model.train()
        tloss, ntr = 0.0, 0
        for vol, y in dl_tr:
            vol, y = vol.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(vol), y)
            loss.backward(); opt.step()
            tloss += loss.item() * y.size(0); ntr += y.size(0)
        train_loss = tloss / max(1, ntr)

        # --------- val ------------
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            for vol, y in dl_val:
                vol, y = vol.to(device), y.to(device)
                vloss += crit(model(vol), y).item() * y.size(0); n += y.size(0)
        vloss /= max(1, n)

        # --------- scheduler + logs ----------
        prev_lr = opt.param_groups[0]['lr']
        sched.step(vloss)
        new_lr = opt.param_groups[0]['lr']
        lr_msg = f" | lr↓->{new_lr:.6f}" if new_lr < prev_lr - 1e-12 else ""

        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={vloss:.4f} "
              f"best_val={best:.4f} bad={bad}{lr_msg}")

        if bad >= patience:
            print(f"[EarlyStopping] Stop at epoch {epoch} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------- test ----------
    model.eval(); logits_, y_ = [], []
    with torch.no_grad():
        for vol, y in dl_te:
            vol = vol.to(device)
            logits_.append(model(vol).cpu().numpy())
            y_.append(y.numpy())
    logits = np.concatenate(logits_); y = np.concatenate(y_)
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = prob.argmax(1)

    acc = accuracy_score(y, pred)
    auc = macro_auc_safe(y, prob, n_classes=4)
    cm  = confusion_matrix(y, pred, labels=[0,1,2,3])
    return acc, auc, cm

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--modality", choices=["mri","pet_gt","pet_fake"], default="pet_gt")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bs", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # columns check (same as before)
    for c in ["subject","mri","pet_gt","pet_fake","Braak1_2","Braak3_4","Braak5_6"]:
        if c not in df.columns: 
            raise ValueError(f"missing column: {c}")

    # resolve the path column we will actually use for this modality
    if args.modality == "pet_gt":
        # build new path from subject id extracted from pet_gt column
        df = df.copy()
        df["resolved_path"] = df["pet_gt"].apply(resolve_pet_gt_to_pup_suvr)
    else:
        df = df.copy()
        df["resolved_path"] = df[args.modality]

    # drop rows with missing/unreadable files (print already done in resolver for pet_gt)
    before = len(df)
    df = df[df["resolved_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].reset_index(drop=True)
    after = len(df)
    if after == 0:
        raise RuntimeError("No valid volumes found for the selected modality after path resolution.")
    print(f"[info] using {after}/{before} rows after path resolution & existence check.")

    y_all = df.apply(braak_label, axis=1).astype(int).values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=0)

    accs, aucs, cms = [], [], []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y_all), 1):
        acc, auc, cm = run_fold(df, tr, te, "resolved_path", args.size, device, epochs=args.epochs, bs=args.bs)
        accs.append(acc); aucs.append(auc); cms.append(cm)
        print(f"Fold{k}: AUC={auc:.3f} ACC={acc:.3f}\nConfusion:\n{cm}\n")

    acc_mean, acc_var = float(np.mean(accs)), float(np.var(accs, ddof=1) if len(accs)>1 else 0.0)
    auc_mean = float(np.nanmean(aucs))
    auc_var  = float(np.nanvar(aucs, ddof=1) if np.sum(~np.isnan(aucs))>1 else 0.0)
    cm_sum = sum(cms)

    print(f"== {args.modality} ==")
    print(f"ACC mean={acc_mean:.3f} var={acc_var:.4f} | AUC mean={auc_mean:.3f} var={auc_var:.4f}")
    print("Confusion (sum over folds):\n", cm_sum)

if __name__ == "__main__":
    main()

