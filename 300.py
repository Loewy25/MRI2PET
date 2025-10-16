#!/usr/bin/env python3
import os
import sys
import shutil
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

# ========= EDIT THESE =========
CSV_PATH          = "/scratch/l.peiwang/MR_AMY_TAU_CDR_merge_DF26.csv"
SOURCE_DIR        = "/scratch/l.peiwang/kari_brainv33"            # ~400 T807 subjects (folders)
DEST_DIR          = "/scratch/l.peiwang/kari_brainv33_top300"     # output with 300 selected
TARGET_N          = 300
CN_CENTILOID_THR  = 20.0
BRAAK_THR         = 1.2       # band is "positive" if value >= 1.2
COPY_MODE         = "copy"    # "symlink" (fast) or "copy" (full copy)
RANDOM_SEED       = 42
# ==============================

def die(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)

def norm_colname(s: str) -> str:
    # Lower, strip, remove non-alphanumerics to neutralize weird whitespace/Unicode
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Try several parsing strategies to handle comma/tab/mixed delimiters.
    """
    # 1) Sniff delimiter
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", engine="python", sep=None)
        if len(df.columns) >= 10:  # sanity: got more than a handful of columns
            return df
    except Exception:
        pass
    # 2) Explicit tab
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", sep="\t", engine="python")
        if len(df.columns) >= 10:
            return df
    except Exception:
        pass
    # 3) Regex: tab OR comma
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", sep=r"[\t,]", engine="python")
        if len(df.columns) >= 10:
            return df
    except Exception:
        pass
    # 4) Last resort: comma
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df

def find_col_any(df: pd.DataFrame, aliases: List[str]) -> str:
    cmap = {norm_colname(c): c for c in df.columns}
    # exact (normalized) match first
    for a in aliases:
        k = norm_colname(a)
        if k in cmap:
            return cmap[k]
    # soft contains-match fallback
    for a in aliases:
        k = norm_colname(a)
        for nk, orig in cmap.items():
            if k in nk:
                return orig
    raise KeyError(f"Missing expected column. Tried {aliases}. Available: {list(df.columns)}")

def parse_numeric_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def list_subdirs(path: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except FileNotFoundError:
        die(f"Folder not found: {path}")

def make_dir_map(dirnames: List[str]) -> Dict[str, str]:
    """normalized -> actual name"""
    return {d.strip().lower(): d for d in dirnames}

# ---- Braak 12/34/56 band detection & stage derivation ----

def find_braak_band_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Prefer exactly the band columns you showed: Braak1_2, Braak3_4, Braak5_6.
    Fall back to normalized contains if needed.
    """
    # Try exact names first
    exact = {"Braak1_2": None, "Braak3_4": None, "Braak5_6": None}
    for c in df.columns:
        if c.strip() in exact:
            exact[c.strip()] = c
    if all(exact.values()):
        return exact["Braak1_2"], exact["Braak3_4"], exact["Braak5_6"]

    # Fallback by normalized contains
    want = {"12": None, "34": None, "56": None}
    for c in df.columns:
        nc = norm_colname(c)
        if "braak" in nc and "12" in nc and want["12"] is None:
            want["12"] = c
        if "braak" in nc and "34" in nc and want["34"] is None:
            want["34"] = c
        if "braak" in nc and "56" in nc and want["56"] is None:
            want["56"] = c
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise KeyError(f"Could not find Braak band columns {missing}. Columns: {list(df.columns)}")
    return want["12"], want["34"], want["56"]

def derive_braak_stage_from_bands(df: pd.DataFrame, col12: str, col34: str, col56: str, thr: float) -> pd.DataFrame:
    """
    Create:
      - __braak_stage__ in {0,2,4,6}
      - __braak_band_value__ = value from the winning band (np.nan if stage==0)
    Rule: choose the highest band among {56, 34, 12} whose value >= thr.
    """
    v12 = parse_numeric_col(df[col12])
    v34 = parse_numeric_col(df[col34])
    v56 = parse_numeric_col(df[col56])

    pos12 = (v12 >= thr)
    pos34 = (v34 >= thr)
    pos56 = (v56 >= thr)

    stage = np.where(pos56, 6,
             np.where(pos34, 4,
             np.where(pos12, 2, 0)))

    band_val = np.where(stage == 6, v56,
                np.where(stage == 4, v34,
                np.where(stage == 2, v12, np.nan)))

    df["__braak_stage__"] = stage.astype(float)
    df["__braak_band_value__"] = band_val.astype(float)
    return df

# ---- materialization ----

def materialize_selection(rows: pd.DataFrame, session_col: str, dir_map: Dict[str, str],
                          src_root: str, dst_root: str, mode: str = "symlink"):
    os.makedirs(dst_root, exist_ok=True)
    for sess in rows[session_col]:
        key = str(sess).strip().lower()
        if key not in dir_map:
            print(f"WARNING: folder not found for session: {sess}")
            continue
        src = os.path.join(src_root, dir_map[key])
        dst = os.path.join(dst_root, dir_map[key])
        if os.path.lexists(dst):
            continue
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            shutil.copytree(src, dst)
        else:
            die(f"Unknown COPY_MODE: {mode}")

def main():
    # 1) Read CSV robustly
    if not os.path.isfile(CSV_PATH):
        die(f"CSV not found: {CSV_PATH}")
    df = read_csv_robust(CSV_PATH)

    # 2) Resolve required columns (use the exact names you showed)
    session_col = find_col_any(df, ["TAU_PET_Session", "tau pet session"])
    cdr_col     = find_col_any(df, ["cdr", "CDR", "CDR_Global"])
    cent_col    = find_col_any(df, ["Centiloid", "centiloid", "CL"])

    # 3) Resolve Braak band columns (12/34/56)
    braak12_col, braak34_col, braak56_col = find_braak_band_cols(df)

    # 4) CSV hygiene
    df = df.dropna(subset=[session_col]).copy()
    df[session_col] = df[session_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[session_col], keep="first")

    # 5) Discover folders in SOURCE_DIR and intersect with CSV
    if not os.path.isdir(SOURCE_DIR):
        die(f"Source directory not found: {SOURCE_DIR}")
    subdirs = list_subdirs(SOURCE_DIR)
    dir_map = make_dir_map(subdirs)  # normalized -> actual

    df["__sess_norm__"] = df[session_col].str.lower()
    df_in_src = df[df["__sess_norm__"].isin(dir_map.keys())].copy()
    if df_in_src.empty:
        die("No overlapping sessions between CSV and SOURCE_DIR.")

    # 6) Parse severity features
    df_in_src["__cdr__"]  = parse_numeric_col(df_in_src[cdr_col])
    df_in_src["__cent__"] = parse_numeric_col(df_in_src[cent_col])

    # 7) Derive Braak stage from 12/34/56 with threshold BRAAK_THR
    df_in_src = derive_braak_stage_from_bands(df_in_src, braak12_col, braak34_col, braak56_col, thr=BRAAK_THR)

    # 8) Define impaired vs clinically normal
    impaired_mask = (df_in_src["__cdr__"] > 0) | (df_in_src["__cent__"] >= CN_CENTILOID_THR) | (df_in_src["__braak_stage__"] >= 2)
    cn_mask = ~impaired_mask

    impaired = df_in_src.loc[impaired_mask].copy()
    normal   = df_in_src.loc[cn_mask].copy()

    # 9) Rank impaired by severity: CDR ↓, Centiloid ↓, Braak stage ↓, session name ↑
    impaired = impaired.sort_values(
        by=["__cdr__", "__cent__", "__braak_stage__", "__sess_norm__"],
        ascending=[False,    False,           False,            True]
    )

    # 10) Select up to TARGET_N from impaired; top-up from normal if needed
    selected = impaired.head(TARGET_N).copy()
    need = TARGET_N - len(selected)
    if need > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        if len(normal) >= need:
            chosen_idx = rng.choice(normal.index.to_numpy(), size=need, replace=False)
            selected = pd.concat([selected, normal.loc[chosen_idx]], ignore_index=True)
        else:
            selected = pd.concat([selected, normal], ignore_index=True)
            still = TARGET_N - len(selected)
            if still > 0:
                tail = impaired.iloc[len(impaired.head(TARGET_N)) : len(impaired.head(TARGET_N)) + still]
                selected = pd.concat([selected, tail], ignore_index=True)

    # Safety
    selected = selected.drop_duplicates(subset=[session_col], keep="first")
    assert len(selected) == TARGET_N, f"Expected {TARGET_N}, got {len(selected)}"

    # 11) Materialize into DEST_DIR (COPY_MODE = 'copy')
    materialize_selection(
        rows=selected[[session_col]],
        session_col=session_col,
        dir_map=dir_map,
        src_root=SOURCE_DIR,
        dst_root=DEST_DIR,
        mode=COPY_MODE
    )

    # 12) Write manifest & small QC summary
    manifest_path = "/scratch/l.peiwang/kari_brainv33_top300_manifest.csv"
    selected_out = selected[[session_col,
                             cdr_col, cent_col,
                             braak12_col, braak34_col, braak56_col,
                             "__braak_stage__", "__braak_band_value__"]].copy()
    selected_out = selected_out.rename(columns={
        "__braak_stage__": "BraakStageDerived",
        "__braak_band_value__": "BraakBandValue"
    })
    selected_out.to_csv(manifest_path, index=False, encoding="utf-8")

    print(f"Source folders: {len(subdirs)} | Overlap with CSV: {len(df_in_src)}")
    print(f"Impaired pool: {len(impaired)} | Normal pool: {len(normal)}")
    print(f"Selected {TARGET_N} -> {DEST_DIR} (mode={COPY_MODE})")
    print(f"Manifest -> {manifest_path}")
    print(f"Braak bands: 12='{braak12_col}', 34='{braak34_col}', 56='{braak56_col}'; threshold = {BRAAK_THR}")

if __name__ == "__main__":
    main()

