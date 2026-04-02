import csv
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split

from .config import (
    BATCH_SIZE,
    BRAAK_THRESHOLD,
    DEMOGRAPHICS_CSV,
    MR_AMY_TAU_CDR_CSV,
    MR_COG_PET_CSV,
    NUM_WORKERS,
    OVERSAMPLE_ENABLE,
    OVERSAMPLE_LABEL3_TARGET,
    OVERSAMPLE_MAX_WEIGHT,
    PIN_MEMORY,
    RESIZE_TO,
    ROOT_DIR,
    TRAIN_FRACTION,
    VAL_FRACTION,
)


# =========================================================================
# Clinical feature definitions
# =========================================================================
CLINICAL_FEATURE_NAMES = [
    "Age_MR",
    "cdr",
    "MMSE",
    "MRFreePET_Centiloid",
    "MR_PET_span",
    "MR_COG_span",
    "sex",
    "education",
    "APOE_e2_count",
    "APOE_e4_count",
]
CLINICAL_CONTINUOUS = [
    "Age_MR",
    "cdr",
    "MMSE",
    "MRFreePET_Centiloid",
    "MR_PET_span",
    "MR_COG_span",
    "education",
]
CSV1_COLS = ["TAU_PET_Session", "MR_Session", "Braak1_2", "Braak3_4", "Braak5_6"]
CSV2_COLS = [
    "MR_Session",
    "ID",
    "Age_MR",
    "cdr",
    "MMSE",
    "apoe",
    "MRFreePET_Centiloid",
    "MR_PET_span",
    "MR_COG_span",
]
CSV3_COLS = ["ID", "EDUC", "sex"]


# =========================================================================
# Volume helpers
# =========================================================================
def _maybe_resize(vol: np.ndarray, target: Optional[Tuple[int, int, int]], order: int = 1) -> np.ndarray:
    if target is None:
        return vol.astype(np.float32)
    dz, hy, wx = vol.shape
    td, th, tw = target
    if (dz, hy, wx) == (td, th, tw):
        return vol.astype(np.float32)
    zoom_factors = (td / dz, th / hy, tw / wx)
    return nd_zoom(vol, zoom_factors, order=order).astype(np.float32)


def norm_mri_to_01(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    x = vol.astype(np.float32)
    if mask is None:
        raise TypeError("no mask")
    vals = x[mask]
    if vals.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    mean = float(vals.mean())
    std = float(vals.std() + 1e-6)
    z = (x - mean) / std
    z[~mask] = 0.0
    return z.astype(np.float32)


def norm_pet_to_01(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    x = vol.astype(np.float32)
    if mask is None:
        raise TypeError("No Mask")
    x_out = x.copy()
    x_out[~mask] = 0.0
    return x_out.astype(np.float32)


# =========================================================================
# CSV parsing helpers
# =========================================================================
def norm_key(x: Any) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def has_value(x: Any) -> bool:
    return pd.notna(x) and str(x).strip() not in {"", "nan", "none", "na", "n/a"}


def _first_value(series: pd.Series):
    for x in series:
        if has_value(x):
            return x
    return pd.NA


def _collapse(df: pd.DataFrame, key: str, cols: List[str]) -> pd.DataFrame:
    keep = [key] + cols
    out = df[keep].copy()
    out["_key"] = out[key].map(norm_key)
    out = out[out["_key"] != ""]
    agg = {c: _first_value for c in keep}
    return out.groupby("_key", as_index=False).agg(agg).set_index("_key")


def _clean_cols(cols: List[Any]) -> List[str]:
    return [str(c).strip().strip('"').lstrip("\ufeff") for c in cols]


def _xml_children(elem: ET.Element, suffix: str) -> List[ET.Element]:
    return [child for child in list(elem) if child.tag.endswith(suffix)]


def _xml_first(elem: ET.Element, suffix: str) -> Optional[ET.Element]:
    for child in elem.iter():
        if child.tag.endswith(suffix):
            return child
    return None


def _xlsx_cell_value(cell: ET.Element, shared: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join((node.text or "") for node in cell.iter() if node.tag.endswith("t"))
    v = _xml_first(cell, "v")
    if v is None or v.text is None:
        return ""
    if cell_type == "s":
        idx = int(v.text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    return v.text


def _xlsx_col_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(idx - 1, 0)


def _read_xlsx_any(path: str, required_cols: List[str]) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(path) as zf:
            sheet_names = sorted(
                name for name in zf.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            )
            if not sheet_names:
                return None
            shared = []
            if "xl/sharedStrings.xml" in zf.namelist():
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                shared = ["".join(node.itertext()) for node in root.iter() if node.tag.endswith("si")]
            for sheet_name in sheet_names:
                root = ET.fromstring(zf.read(sheet_name))
                sheet_data = next((child for child in root if child.tag.endswith("sheetData")), None)
                if sheet_data is None:
                    continue
                rows = []
                for row in _xml_children(sheet_data, "row"):
                    vals = []
                    for cell in _xml_children(row, "c"):
                        idx = _xlsx_col_index(cell.attrib.get("r", "A1"))
                        while len(vals) <= idx:
                            vals.append("")
                        vals[idx] = _xlsx_cell_value(cell, shared)
                    rows.append(vals)
                for i, row in enumerate(rows[:200]):
                    cols = _clean_cols(row)
                    if all(col in cols for col in required_cols):
                        width = len(cols)
                        data = []
                        for vals in rows[i + 1:]:
                            vals = list(vals) + [""] * max(0, width - len(vals))
                            data.append(dict(zip(cols, vals[:width])))
                        return pd.DataFrame(data, columns=cols)
    except (OSError, zipfile.BadZipFile, ET.ParseError):
        return None
    return None


def _read_text_table(path: str, required_cols: List[str]) -> Optional[pd.DataFrame]:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", "\t", ";", "|", None]
    for enc in encodings:
        for sep in seps:
            try:
                kwargs = {"encoding": enc, "sep": sep, "dtype": str, "on_bad_lines": "skip"}
                if sep is None:
                    kwargs["engine"] = "python"
                df = pd.read_csv(path, **kwargs)
                if all(col in df.columns for col in required_cols):
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError, csv.Error):
                continue
    return None


def _read_table_any(path: str, required_cols: List[str]) -> pd.DataFrame:
    df = _read_xlsx_any(path, required_cols)
    if df is not None:
        return df
    df = _read_text_table(path, required_cols)
    if df is not None:
        return df
    raise RuntimeError(f"Could not parse {path} with expected columns {required_cols}")


def _to_float(x: Any, field: str, sid: str) -> float:
    if not has_value(x):
        raise RuntimeError(f"{sid}: missing clinical value for {field}")
    try:
        val = float(x)
    except Exception as exc:
        raise RuntimeError(f"{sid}: could not parse {field}={x!r} as float") from exc
    if not np.isfinite(val):
        raise RuntimeError(f"{sid}: non-finite clinical value for {field}")
    return float(val)


def _parse_sex(x: Any, sid: str) -> float:
    if not has_value(x):
        raise RuntimeError(f"{sid}: missing sex")
    text = str(x).strip().lower()
    if text in {"f", "female", "0", "false"}:
        return 0.0
    if text in {"m", "male", "1", "true"}:
        return 1.0
    raise RuntimeError(f"{sid}: unsupported sex value {x!r}")


def _parse_apoe_counts(x: Any, sid: str) -> Tuple[float, float]:
    if not has_value(x):
        raise RuntimeError(f"{sid}: missing apoe")
    digits = "".join(re.findall(r"\d", str(x)))
    if len(digits) >= 2:
        digits = digits[:2]
    mapping = {
        "22": (2.0, 0.0), "23": (1.0, 0.0), "24": (1.0, 1.0),
        "33": (0.0, 0.0), "34": (0.0, 1.0), "44": (0.0, 2.0),
    }
    if digits not in mapping:
        raise RuntimeError(f"{sid}: unsupported apoe value {x!r}")
    return mapping[digits]


def _braak_stage(b12: Any, b34: Any, b56: Any) -> int:
    v12 = float(pd.to_numeric(b12, errors="coerce"))
    v34 = float(pd.to_numeric(b34, errors="coerce"))
    v56 = float(pd.to_numeric(b56, errors="coerce"))
    if np.isfinite(v56) and v56 >= BRAAK_THRESHOLD:
        return 3
    if np.isfinite(v34) and v34 >= BRAAK_THRESHOLD:
        return 2
    if np.isfinite(v12) and v12 >= BRAAK_THRESHOLD:
        return 1
    return 0


def _subject_records(subject_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    df1 = _read_table_any(MR_AMY_TAU_CDR_CSV, CSV1_COLS)
    df2 = _read_table_any(MR_COG_PET_CSV, CSV2_COLS)
    df3 = _read_table_any(DEMOGRAPHICS_CSV, CSV3_COLS)

    idx1 = _collapse(df1, "TAU_PET_Session", CSV1_COLS[1:])
    idx2 = _collapse(df2, "MR_Session", CSV2_COLS[1:])
    idx3 = _collapse(df3, "ID", CSV3_COLS[1:])

    out: Dict[str, Dict[str, Any]] = {}
    for sid in subject_ids:
        key1 = norm_key(sid)
        if key1 not in idx1.index:
            raise RuntimeError(f"{sid}: not found in TAU_PET_Session column of {MR_AMY_TAU_CDR_CSV}")
        row1 = idx1.loc[key1]
        mr_session = row1["MR_Session"]
        key2 = norm_key(mr_session)
        if key2 not in idx2.index:
            raise RuntimeError(f"{sid}: MR_Session={mr_session!r} not found in {MR_COG_PET_CSV}")
        row2 = idx2.loc[key2]
        subj_id = row2["ID"]
        key3 = norm_key(subj_id)
        if key3 not in idx3.index:
            raise RuntimeError(f"{sid}: ID={subj_id!r} not found in {DEMOGRAPHICS_CSV}")
        row3 = idx3.loc[key3]
        apoe_e2, apoe_e4 = _parse_apoe_counts(row2["apoe"], sid)

        b12_raw = _to_float(row1["Braak1_2"], "Braak1_2", sid)
        b34_raw = _to_float(row1["Braak3_4"], "Braak3_4", sid)
        b56_raw = _to_float(row1["Braak5_6"], "Braak5_6", sid)
        label = _braak_stage(row1["Braak1_2"], row1["Braak3_4"], row1["Braak5_6"])

        out[sid] = {
            "mr_session": str(mr_session).strip(),
            "subject_id": str(subj_id).strip(),
            "label": int(label),
            "stage_ord": int(label),
            "braak_values_raw": np.array([b12_raw, b34_raw, b56_raw], dtype=np.float32),
            "clinical_raw": {
                "Age_MR": _to_float(row2["Age_MR"], "Age_MR", sid),
                "cdr": _to_float(row2["cdr"], "cdr", sid),
                "MMSE": _to_float(row2["MMSE"], "MMSE", sid),
                "MRFreePET_Centiloid": _to_float(row2["MRFreePET_Centiloid"], "MRFreePET_Centiloid", sid),
                "MR_PET_span": _to_float(row2["MR_PET_span"], "MR_PET_span", sid),
                "MR_COG_span": _to_float(row2["MR_COG_span"], "MR_COG_span", sid),
                "sex": _parse_sex(row3["sex"], sid),
                "education": _to_float(row3["EDUC"], "EDUC", sid),
                "APOE_e2_count": apoe_e2,
                "APOE_e4_count": apoe_e4,
            },
        }
    return out


def _compute_clinical_stats(ds: "KariAV1451Dataset", indices: List[int]) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for name in CLINICAL_CONTINUOUS:
        vals = np.asarray([ds.items[idx]["clinical_raw"][name] for idx in indices], dtype=np.float32)
        mean = float(vals.mean())
        std = float(vals.std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[name] = (mean, std)
    return stats


def _compute_braak_stats(ds: "KariAV1451Dataset", indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.stack([ds.items[idx]["braak_values_raw"] for idx in indices], axis=0)
    mean = vals.mean(axis=0).astype(np.float32)
    std = vals.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


# =========================================================================
# Dataset
# =========================================================================
class KariAV1451Dataset(Dataset):
    def __init__(
        self,
        root_dir: str = ROOT_DIR,
        resize_to: Optional[Tuple[int, int, int]] = RESIZE_TO,
        sid_to_label: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = root_dir
        self.resize_to = resize_to
        self.sid_to_label = sid_to_label or {}
        self.clinical_stats: Optional[Dict[str, Tuple[float, float]]] = None
        self.braak_mean: Optional[np.ndarray] = None
        self.braak_std: Optional[np.ndarray] = None

        def _is_subject_dir(path: str) -> bool:
            required = [
                "T1_masked.nii.gz",
                "FLAIR_in_T1_masked.nii.gz",
                "PET_in_T1_masked.nii.gz",
                "aseg_brainmask.nii.gz",
                "mask_cortex.nii.gz",
            ]
            return all(os.path.exists(os.path.join(path, name)) for name in required)

        subject_dirs = sorted(
            entry.path for entry in os.scandir(root_dir)
            if entry.is_dir() and _is_subject_dir(entry.path)
        )
        if not subject_dirs:
            raise RuntimeError(f"No subject folders found under {root_dir}")

        subject_ids = [os.path.basename(path) for path in subject_dirs]
        records = _subject_records(subject_ids)

        self.items: List[Dict[str, Any]] = []
        for d in subject_dirs:
            sid = os.path.basename(d)
            paths = {
                "t1_path": os.path.join(d, "T1_masked.nii.gz"),
                "flair_path": os.path.join(d, "FLAIR_in_T1_masked.nii.gz"),
                "pet_path": os.path.join(d, "PET_in_T1_masked.nii.gz"),
                "mask_path": os.path.join(d, "aseg_brainmask.nii.gz"),
                "cortex_path": os.path.join(d, "mask_cortex.nii.gz"),
            }
            item = {"sid": sid, **paths, **records[sid]}
            item["label_train_csv"] = int(self.sid_to_label.get(sid, item["label"]))
            self.items.append(item)

    def set_clinical_stats(self, stats: Dict[str, Tuple[float, float]]) -> None:
        self.clinical_stats = stats

    def set_braak_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.braak_mean = mean
        self.braak_std = std

    def __len__(self) -> int:
        return len(self.items)

    def _clinical_vector(self, raw: Dict[str, float]) -> np.ndarray:
        if self.clinical_stats is None:
            raise RuntimeError("clinical normalization stats are not set")
        vals = []
        for name in CLINICAL_FEATURE_NAMES:
            value = float(raw[name])
            if name in CLINICAL_CONTINUOUS:
                mean, std = self.clinical_stats[name]
                value = (value - mean) / std
            vals.append(value)
        return np.asarray(vals, dtype=np.float32)

    def _normalized_braak(self, raw: np.ndarray) -> np.ndarray:
        if self.braak_mean is None or self.braak_std is None:
            return raw.copy()
        return ((raw - self.braak_mean) / self.braak_std).astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        sid = item["sid"]

        t1_img = nib.load(item["t1_path"])
        flair_img = nib.load(item["flair_path"])
        pet_img = nib.load(item["pet_path"])
        mask_img = nib.load(item["mask_path"])
        cortex_img = nib.load(item["cortex_path"])

        t1 = np.asarray(t1_img.get_fdata(), dtype=np.float32)
        flair = np.asarray(flair_img.get_fdata(), dtype=np.float32)
        pet = np.asarray(pet_img.get_fdata(), dtype=np.float32)
        mask = np.asarray(mask_img.get_fdata()) > 0
        cortex = np.asarray(cortex_img.get_fdata()) > 0

        if t1.shape != pet.shape or t1.shape != flair.shape:
            raise RuntimeError(f"{sid}: T1/FLAIR/PET are not in the same grid")
        if mask.shape != t1.shape or cortex.shape != t1.shape:
            raise RuntimeError(f"{sid}: mask grids do not match T1")

        cortex = np.logical_and(cortex, mask)

        orig_shape = tuple(t1.shape)
        t1_affine = t1_img.affine
        pet_affine = pet_img.affine

        t1 = _maybe_resize(t1, self.resize_to, order=1)
        flair = _maybe_resize(flair, self.resize_to, order=1)
        pet = _maybe_resize(pet, self.resize_to, order=1)
        cur_shape = tuple(t1.shape)

        if self.resize_to is not None:
            dz, hy, wx = mask.shape
            td, th, tw = self.resize_to
            if (dz, hy, wx) != (td, th, tw):
                mask = nd_zoom(mask.astype(np.float32), (td / dz, th / hy, tw / wx), order=0) > 0.5
                cortex = nd_zoom(cortex.astype(np.float32), (td / dz, th / hy, tw / wx), order=0) > 0.5
                cortex = np.logical_and(cortex, mask)

        t1n = norm_mri_to_01(t1, mask)
        flairn = norm_mri_to_01(flair, mask)
        petn = norm_pet_to_01(pet, mask=mask)
        clin = self._clinical_vector(item["clinical_raw"])
        braak_norm = self._normalized_braak(item["braak_values_raw"])

        t1n_t = torch.from_numpy(np.expand_dims(t1n, axis=0))
        petn_t = torch.from_numpy(np.expand_dims(petn, axis=0))

        meta = {
            "sid": sid,
            "t1_path": item["t1_path"],
            "flair_path": item["flair_path"],
            "pet_path": item["pet_path"],
            "t1_affine": t1_affine,
            "pet_affine": pet_affine,
            "orig_shape": orig_shape,
            "cur_shape": cur_shape,
            "label": int(item["label"]),
            "label_train_csv": int(item["label_train_csv"]),
            "resized_to": self.resize_to,
            "brain_mask": mask.astype(np.uint8),
            "cortex_mask": cortex.astype(np.uint8),
            "flair": np.expand_dims(flairn, axis=0).astype(np.float32),
            "clinical_vector": clin.astype(np.float32),
            "stage_ord": int(item["stage_ord"]),
            "braak_values": braak_norm,
        }
        return t1n_t, petn_t, meta


# =========================================================================
# Collate / loaders
# =========================================================================
def _collate_keep_meta(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    if len(batch) == 1:
        return batch[0]
    mri = torch.stack([b[0] for b in batch], dim=0)
    pet = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return mri, pet, metas


def build_loaders(
    root: str = ROOT_DIR,
    resize_to: Optional[Tuple[int, int, int]] = RESIZE_TO,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    seed: int = 1999,
):
    ds = KariAV1451Dataset(root_dir=root, resize_to=resize_to)
    n_total = len(ds)
    n_train = int(round(train_fraction * n_total))
    n_val = int(round(val_fraction * n_total))
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test], generator=gen)
    ds.set_clinical_stats(_compute_clinical_stats(ds, list(train_set.indices)))
    braak_mean, braak_std = _compute_braak_stats(ds, list(train_set.indices))
    ds.set_braak_stats(braak_mean, braak_std)

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=False, collate_fn=_collate_keep_meta)
    dl_val = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=False, collate_fn=_collate_keep_meta)
    dl_test = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=False, collate_fn=_collate_keep_meta)
    return dl_train, dl_val, dl_test, n_total, n_train, n_val, n_test


# =========================================================================
# CSV-driven fold loaders
# =========================================================================
def _read_fold_csv_lists(path: str):
    train, val, test = [], [], []
    train_sid_to_label: Dict[str, int] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        tcol = cols.get("train")
        vcol = cols.get("validation")
        ccol = cols.get("test")
        lcol = cols.get("label")
        if not (tcol and vcol and ccol):
            raise ValueError(f"fold CSV missing required columns train/validation/test: {path}")
        if OVERSAMPLE_ENABLE and lcol is None:
            raise ValueError(f"OVERSAMPLE_ENABLE=1 but fold CSV has no 'label' column: {path}")

        for row in reader:
            t = (row.get(tcol) or "").strip()
            v = (row.get(vcol) or "").strip()
            c = (row.get(ccol) or "").strip()
            if t:
                train.append(t)
                if lcol is not None:
                    lab_str = (row.get(lcol) or "").strip()
                    if lab_str == "":
                        raise RuntimeError(f"Missing train label for {t} in {path}")
                    train_sid_to_label[t] = int(float(lab_str))
            if v:
                val.append(v)
            if c:
                test.append(c)

    return train, val, test, train_sid_to_label


def _sid_for_item(item: Dict[str, Any]) -> str:
    return item["sid"]


def build_loaders_from_fold_csv(
    fold_csv_path: str,
    root: str = ROOT_DIR,
    resize_to: Optional[Tuple[int, int, int]] = RESIZE_TO,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    train_sids, val_sids, test_sids, train_sid_to_label = _read_fold_csv_lists(fold_csv_path)

    ds = KariAV1451Dataset(root_dir=root, resize_to=resize_to, sid_to_label=train_sid_to_label)
    n_total = len(ds)

    sid_list = [_sid_for_item(item) for item in ds.items]
    sid_to_index = {sid: i for i, sid in enumerate(sid_list)}

    def _to_indices(sids: List[str]) -> List[int]:
        idxs = []
        missing = []
        for sid in sids:
            if sid in sid_to_index:
                idxs.append(sid_to_index[sid])
            else:
                missing.append(sid)
        if missing:
            raise RuntimeError(f"{len(missing)} subjects from {fold_csv_path} not found on disk. Examples: {missing[:8]}")
        return sorted(idxs)

    idx_train = _to_indices(train_sids)
    idx_val = _to_indices(val_sids)
    idx_test = _to_indices(test_sids)

    ds.set_clinical_stats(_compute_clinical_stats(ds, idx_train))
    braak_mean, braak_std = _compute_braak_stats(ds, idx_train)
    ds.set_braak_stats(braak_mean, braak_std)

    train_set = Subset(ds, idx_train)
    val_set = Subset(ds, idx_val)
    test_set = Subset(ds, idx_test)

    sampler = None
    if OVERSAMPLE_ENABLE:
        train_labels = []
        for idx in idx_train:
            sid = sid_list[idx]
            lab = train_sid_to_label.get(sid)
            if lab is None:
                raise RuntimeError(f"Train sid {sid} has no label in {fold_csv_path}")
            if lab not in (0, 1, 2, 3):
                raise RuntimeError(f"Train sid {sid} has out-of-range label={lab}")
            train_labels.append(int(lab))

        ntr = len(train_labels)
        counts = {k: 0 for k in (0, 1, 2, 3)}
        for lab in train_labels:
            counts[lab] += 1

        if counts[3] == 0:
            print("[INFO] No label=3 samples in train split; oversampling disabled for this fold.")
        else:
            p = {k: counts[k] / max(1, ntr) for k in (0, 1, 2, 3)}
            target_p3 = float(np.clip(OVERSAMPLE_LABEL3_TARGET, 0.0, 0.95))
            rest = 1.0 - target_p3
            p_rest = p[0] + p[1] + p[2]
            q = {0: 0.0, 1: 0.0, 2: 0.0, 3: target_p3}
            if p_rest > 0:
                for k in (0, 1, 2):
                    q[k] = rest * (p[k] / p_rest)
            class_w = {}
            for k in (0, 1, 2, 3):
                class_w[k] = float(np.clip((q[k] / p[k]) if p[k] > 0 else 0.0, 0.0, OVERSAMPLE_MAX_WEIGHT))
            weights = torch.DoubleTensor([class_w[lab] for lab in train_labels])
            if float(weights.sum().item()) > 0.0:
                sampler = WeightedRandomSampler(weights, num_samples=ntr, replacement=True)
                print(
                    f"[INFO] Oversampling train enabled: target_p3={target_p3:.2f}, "
                    f"counts={counts}, class_w={class_w}"
                )

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=(sampler is None),
                          sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=False, collate_fn=_collate_keep_meta)
    dl_val = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=False, collate_fn=_collate_keep_meta)
    dl_test = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory,
                         drop_last=False, collate_fn=_collate_keep_meta)
    return dl_train, dl_val, dl_test, n_total, len(idx_train), len(idx_val), len(idx_test)
