#!/usr/bin/env python3
import csv
from pathlib import Path
import sys
import zipfile
from xml.etree import ElementTree as ET

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


def _clean_cols(cols):
    return [str(c).strip().strip('"').lstrip("\ufeff") for c in cols]


def _xml_children(elem, suffix):
    return [child for child in list(elem) if child.tag.endswith(suffix)]


def _xml_first(elem, suffix):
    for child in elem.iter():
        if child.tag.endswith(suffix):
            return child
    return None


def _xlsx_cell_value(cell, shared):
    t = cell.attrib.get("t")
    if t == "inlineStr":
        return "".join((node.text or "") for node in cell.iter() if node.tag.endswith("t"))

    v = _xml_first(cell, "v")
    if v is None or v.text is None:
        return ""

    if t == "s":
        idx = int(v.text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    return v.text


def _xlsx_col_index(ref):
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(idx - 1, 0)


def _read_xlsx_any(path, required_cols):
    try:
        with zipfile.ZipFile(path) as zf:
            sheet_names = sorted(
                n for n in zf.namelist()
                if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
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
                        for vals in rows[i + 1 :]:
                            vals = list(vals) + [""] * max(0, width - len(vals))
                            data.append(dict(zip(cols, vals[:width])))
                        df = pd.DataFrame(data, columns=cols)
                        print(f"[INFO] Loaded {Path(path).name} as xlsx sheet={sheet_name.rsplit('/', 1)[-1]}")
                        return df
    except (OSError, zipfile.BadZipFile, ET.ParseError):
        return None
    return None


def _manual_read_table(path, required_cols):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", "\t", ";", "|"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore", newline="") as f:
                lines = f.read().splitlines()
        except OSError:
            continue

        for sep in seps:
            header = None
            header_i = None
            for i, line in enumerate(lines[:200]):
                cols = _clean_cols(line.split(sep))
                if all(col in cols for col in required_cols):
                    header = cols
                    header_i = i
                    break
            if header is None:
                continue

            rows = []
            for line in lines[header_i + 1 :]:
                if not line.strip():
                    continue
                try:
                    vals = next(csv.reader([line], delimiter=sep))
                except Exception:
                    vals = line.split(sep)
                vals = [str(v).strip() for v in vals]
                if len(vals) < len(header):
                    vals += [""] * (len(header) - len(vals))
                rows.append(dict(zip(header, vals[: len(header)])))

            df = pd.DataFrame(rows, columns=header)
            if all(col in df.columns for col in required_cols):
                print(f"[INFO] Loaded {path.name} with manual parser encoding={enc} sep={repr(sep)}")
                return df
    return None


def read_csv_any(path, required_cols, allow_empty=False):
    df = _read_xlsx_any(path, required_cols)
    if df is not None:
        return df

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    seps = [",", "\t", ";", "|", None]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                kwargs = {
                    "encoding": enc,
                    "sep": sep,
                    "on_bad_lines": "skip",
                    "dtype": str,
                }
                if sep is None:
                    kwargs["engine"] = "python"
                df = pd.read_csv(path, **kwargs)
                if all(col in df.columns for col in required_cols):
                    sep_name = "auto" if sep is None else repr(sep)
                    print(f"[INFO] Loaded {path.name} with encoding={enc} sep={sep_name}")
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError, csv.Error) as exc:
                last_err = exc

    df = _manual_read_table(path, required_cols)
    if df is not None:
        return df

    if allow_empty:
        print(f"[WARN] Could not parse {path.name}; continuing with empty table")
        return pd.DataFrame(columns=required_cols)

    raise RuntimeError(f"Could not parse {path} with expected columns {required_cols}") from last_err


def main():
    for path in [DATASET, CSV1, CSV2, CSV3]:
        if not path.exists():
            raise FileNotFoundError(path)

    need1 = ["TAU_PET_Session", "MR_Session"]
    need2 = ["MR_Session", "ID"] + FIELDS2
    need3 = ["ID"] + FIELDS3
    df1 = read_csv_any(CSV1, need1)
    df2 = read_csv_any(CSV2, need2)
    df3 = read_csv_any(CSV3, need3, allow_empty=True)
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

    bool_cols = [c for c in out.columns if c.startswith("has_")]
    summary = pd.DataFrame({
        "field": bool_cols,
        "n_have": [int(out[c].sum()) for c in bool_cols],
        "n_total": len(out),
        "pct_have": [round(100 * out[c].mean(), 2) if len(out) else 0.0 for c in bool_cols],
    })
    print(f"Subjects in dataset: {len(out)}")
    print("=== BY_SUBJECT ===")
    out.to_csv(sys.stdout, index=False)
    print("=== COVERAGE ===")
    summary.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
