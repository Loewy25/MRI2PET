#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

ROOT = Path("/scratch/l.peiwang")
DATASET = ROOT / "kari_all_falir"
CSV1 = ROOT / "MR_AMY_TAU_CDR_merge_DF26.csv"
CSV2 = ROOT / "MR_COG_PET_rsfMRI.csv"
CSV3 = ROOT / "demographics.csv"

FIELDS2 = ["Age_MR", "cdr", "MMSE", "apoe", "MRFreePET_Centiloid", "MR_PET_span", "MR_COG_span"]
FIELDS3 = ["EDUC", "sex"]


def norm(x):
    return "" if pd.isna(x) else str(x).strip().lower()


def has_value(x):
    return pd.notna(x) and str(x).strip() not in {"", "nan", "none"}


def first_value(series):
    for x in series:
        if has_value(x):
            return x
    return pd.NA


def collapse(df, key, cols):
    keep = [key] + cols
    out = df[keep].copy()
    out["_key"] = out[key].map(norm)
    out = out[out["_key"] != ""]
    return out.groupby("_key", as_index=False).agg({c: first_value for c in keep}).set_index("_key")


def main():
    for path in [DATASET, CSV1, CSV2, CSV3]:
        if not path.exists():
            raise FileNotFoundError(path)

    df1 = pd.read_csv(CSV1, low_memory=False)
    df2 = pd.read_csv(CSV2, low_memory=False)
    df3 = pd.read_csv(CSV3, low_memory=False)

    need1 = ["TAU_PET_Session", "MR_Session"]
    need2 = ["MR_Session", "ID"] + FIELDS2
    need3 = ["ID"] + FIELDS3
    for name, df, cols in [("MR_AMY_TAU_CDR_merge_DF26.csv", df1, need1), ("MR_COG_PET_rsfMRI.csv", df2, need2), ("demographics.csv", df3, need3)]:
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f"{name} missing columns: {miss}")

    idx1 = collapse(df1, "TAU_PET_Session", ["MR_Session"])
    idx2 = collapse(df2, "MR_Session", ["ID"] + FIELDS2)
    idx3 = collapse(df3, "ID", FIELDS3)

    rows = []
    for subj in sorted(p.name for p in DATASET.iterdir() if p.is_dir()):
        key1 = norm(subj)
        row1 = idx1.loc[key1] if key1 in idx1.index else None
        mr_session = row1["MR_Session"] if row1 is not None else pd.NA

        key2 = norm(mr_session)
        row2 = idx2.loc[key2] if key2 in idx2.index else None
        subj_id = row2["ID"] if row2 is not None else pd.NA

        key3 = norm(subj_id)
        row3 = idx3.loc[key3] if key3 in idx3.index else None

        rec = {
            "subject": subj,
            "has_merge_row": row1 is not None,
            "MR_Session": mr_session,
            "has_mr_cog_row": row2 is not None,
            "ID": subj_id,
            "has_demo_row": row3 is not None,
        }
        for col in FIELDS2:
            rec[f"has_{col}"] = has_value(row2[col]) if row2 is not None else False
        for col in FIELDS3:
            rec[f"has_{col}"] = has_value(row3[col]) if row3 is not None else False
        rec["has_all_requested_info"] = all(rec[f"has_{col}"] for col in FIELDS2 + FIELDS3)
        rows.append(rec)

    out = pd.DataFrame(rows)
    by_subject = ROOT / "kari_all_falir_clinical_summary_by_subject.csv"
    out.to_csv(by_subject, index=False)

    bool_cols = [c for c in out.columns if c.startswith("has_")]
    summary = pd.DataFrame({
        "field": bool_cols,
        "n_have": [int(out[c].sum()) for c in bool_cols],
        "n_total": len(out),
        "pct_have": [round(100 * out[c].mean(), 2) if len(out) else 0.0 for c in bool_cols],
    })
    summary_csv = ROOT / "kari_all_falir_clinical_summary_coverage.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Saved: {by_subject}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
