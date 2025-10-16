#!/usr/bin/env python3
import os
import sys
import shutil
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple

# ========= EDIT THESE =========
CSV_PATH          = "/scratch/l.peiwang/MR_AMY_TAU_merge_DF26.csv"
SOURCE_DIR        = "/scratch/l.peiwang/kari_brainv33"            # ~400 T807 subjects (folders)
DEST_DIR          = "/scratch/l.peiwang/kari_brainv33_top300"     # output with 300 selected
TARGET_N          = 300
CN_CENTILOID_THR  = 20.0
BRAAK_THR         = 1.2       # band is "positive" if value >= 1.2
COPY_MODE         = "copy" # "symlink" (fast) or "copy" (full copy)
RANDOM_SEED       = 42
# ==============================

def die(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)

def norm_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

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

def parse_numeric_col(s: pd.Series, default: float = np.nan) -> pd.Series:
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

def _score_braak_band_name(nc: str, tag: str) -> int:
    """
    Heuristic scoring: prefer names that clearly look like braak bands.
    Higher score = better.
    """
    score = 0
    if "braak" in nc: score += 3
    if tag in nc:     score += 2
    # roman variants
    if tag == "12" and ("i_ii" in nc or "iiii" in nc or "iii" in nc): score += 1
    if tag == "34" and ("iii_iv" in nc or "iiiiv" in nc):             score += 1
    if tag == "56" and ("v_vi" in nc or "vvi" in nc):                 score += 1
    return score

def find_braak_band_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Try to locate the three Braak band columns (12, 34, 56).
    We search for columns that include 'braak' and '12'/'34'/'56' (normalized),
    but fall back to plain '12'/'34'/'56' if needed.
    """
    norm_map = {norm_colname(c): c for c in df.columns}
    best = {"12": (None, -1), "34": (None, -1), "56": (None, -1)}  # (colname, score)

    for c in df.columns:
        nc = norm_colname(c)
        for tag in ["12", "34", "56"]:
            if tag in nc or ("i_ii" in nc and tag=="12") or ("iii_iv" in nc and tag=="34") or ("v_vi" in nc and tag=="56"):
                s = _score_braak_band_name(nc, tag)
                if s > best[tag][1]:
                    best[tag] = (c, s)

    col12, _ = best["12"]
    col34, _ = best["34"]
    col56, _ = best["56"]

    # last-resort fallback: exact simple names '12','34','56'
    if col12 is None and "12" in norm_map: col12 = norm_map["12"]
    if col34 is None and "34" in norm_map: col34 = norm_map["34"]
    if col56 is None and "56" in norm_map: col56 = norm_map["56"]

    missing = [k for k, v in {"12": col12, "34": col34, "56": col56}.items() if v is None]
    if missing:
        raise KeyError(f"Could not find Braak band columns {missing}. "
                       f"Looked for names containing 'braak' and one of 12/34/56. Columns: {list(df.columns)}")
    return col12, col34, col56

def derive_braak_stage_from_bands(df: pd.DataFrame, col12: str, col34: str, col56: str, thr: float) -> pd.DataFrame:
    """
    Create two columns:
      - __braak_stage__ in {0,2,4,6}
      - __braak_band_value__ = the value from the winning band (np.nan if stage==0)
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

    # band value from the winning level (for auditing)
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
    # 1) Read CSV
    if not os.path.isfile(CSV_PATH):
        die(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", engine="python", sep=None)

    # 2) Resolve required columns
    session_col = find_col_any(df, ["TAU_PET_Session", "tau pet session", "session", "folder"])
    cdr_col     = find_col_any(df, ["CDR", "CDR_Global", "cdr", "cdr global"])
    cent_col    = find_col_any(df, ["Centiloid", "Amyloid Centiloid", "CL", "centiloid"])

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

    # 9) Rank impaired by severity: CDR ↓, Centiloid ↓, Braak stage ↓, session name ↑ (deterministic tiebreak)
    impaired = impaired.sort_values(
        by=["__cdr__", "__cent__", "__braak_stage__", "__sess_norm__"],
        ascending=[False,    False,           False,            True]
    )

    # 10) Select top TARGET_N; top up from clinical normals if needed (random, seeded)
    selected = impaired.head(TARGET_N).copy()
    need = TARGET_N - len(selected)
    if need > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        if len(normal) >= need:
            chosen_idx = rng.choice(normal.index.to_numpy(), size=need, replace=False)
            selected = pd.concat([selected, normal.loc[chosen_idx]], ignore_index=True)
        else:
            # take all normals; if still short, fill from remaining impaired tail (unlikely)
            selected = pd.concat([selected, normal], ignore_index=True)
            still = TARGET_N - len(selected)
            if still > 0:
                tail = impaired.iloc[len(impaired.head(TARGET_N)) : len(impaired.head(TARGET_N)) + still]
                selected = pd.concat([selected, tail], ignore_index=True)

    # Safety
    selected = selected.drop_duplicates(subset=[session_col], keep="first")
    assert len(selected) == TARGET_N, f"Expected {TARGET_N}, got {len(selected)}"

    # 11) Materialize into DEST_DIR
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
    selected_out = selected[[session_col, cdr_col, cent_col, braak12_col, braak34_col, braak56_col, "__braak_stage__", "__braak_band_value__"]].copy()
    selected_out.rename(columns={"__braak_stage__": "BraakStageDerived", "__braak_band_value__": "BraakBandValue"}, inplace=True)
    selected_out.to_csv(manifest_path, index=False, encoding="utf-8")

    print(f"Source folders: {len(subdirs)} | Overlap with CSV: {len(df_in_src)}")
    print(f"Impaired pool: {len(impaired)} | Normal pool: {len(normal)}")
    print(f"Selected {TARGET_N} -> {DEST_DIR}")
    print(f"Manifest -> {manifest_path}")
    print(f"Braak bands used: 12='{braak12_col}', 34='{braak34_col}', 56='{braak56_col}'; threshold = {BRAAK_THR}")

if __name__ == "__main__":
    main()
