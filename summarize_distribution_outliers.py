#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple


PRIMARY_FEATURES = [
    "pet_iqr",
    "pet_cortex_iqr",
    "pet_cortex_p95",
    "pet_p95_to_median",
    "pet_cortex_std",
]

TAIL_FEATURES = {
    "pet_cortex_p95",
    "pet_cortex_p99",
    "pet_p95_to_median",
    "pet_p99_to_median",
}

SPREAD_FEATURES = {
    "pet_iqr",
    "pet_std",
    "pet_cortex_iqr",
    "pet_cortex_std",
}

GEOMETRY_FEATURES = {
    "spacing_x_mm",
    "spacing_y_mm",
    "spacing_z_mm",
    "voxel_volume_mm3",
    "brain_voxels",
    "cortex_voxels",
    "cortex_fraction",
}


def _safe_float(value: str) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _ratio(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    if not math.isfinite(value) or not math.isfinite(baseline):
        return None
    if abs(baseline) < 1e-8:
        return None
    return float(value / baseline)


def _parse_reason_text(text: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        feature, raw_score = chunk.split(":", 1)
        z = _safe_float(raw_score)
        if z is None:
            continue
        out.append((feature.strip(), z))
    return out


def _severity(score: Optional[float]) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 12.0:
        return "EXTREME"
    if score >= 8.0:
        return "HIGH"
    if score >= 5.0:
        return "MEDIUM"
    return "LOW"


def _pattern(reason_pairs: Sequence[Tuple[str, float]]) -> str:
    if not reason_pairs:
        return "unclear"
    tail = max((abs(z) for name, z in reason_pairs if name in TAIL_FEATURES), default=0.0)
    spread = max((abs(z) for name, z in reason_pairs if name in SPREAD_FEATURES), default=0.0)
    geom = max((abs(z) for name, z in reason_pairs if name in GEOMETRY_FEATURES), default=0.0)
    if spread >= tail and spread >= geom and spread > 0:
        return "wide PET distribution"
    if tail >= spread and tail >= geom and tail > 0:
        return "heavy high-uptake tail"
    if geom > 0:
        return "geometry or mask shift"
    return "mixed PET shift"


def _format_ratio(name: str, ratio: Optional[float], comparison_label: str) -> Optional[str]:
    if ratio is None or not math.isfinite(ratio):
        return None
    pretty = {
        "pet_iqr": "brain PET spread",
        "pet_cortex_iqr": "cortex PET spread",
        "pet_cortex_p95": "cortex hot tail",
        "pet_p95_to_median": "tail-to-median",
        "pet_cortex_std": "cortex variability",
    }.get(name, name)
    return "{0} {1:.2f}x {2}".format(pretty, ratio, comparison_label)


def _simple_summary(row: Dict[str, str], baseline_medians: Dict[str, float], comparison_label: str) -> str:
    bits: List[str] = []
    for name in PRIMARY_FEATURES:
        value = _safe_float(row.get(name, ""))
        ratio = _ratio(value, baseline_medians.get(name))
        bit = _format_ratio(name, ratio, comparison_label)
        if bit is not None:
            bits.append(bit)
    return "; ".join(bits[:3])


def _load_feature_baselines(path: str, median_col: str) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = (row.get("feature") or "").strip()
            median_value = _safe_float(row.get(median_col) or "")
            if feature and median_value is not None:
                baselines[feature] = median_value
    return baselines


def _score_column(fieldnames: Sequence[str]) -> str:
    for name in fieldnames:
        if name.startswith("outlier_score_vs_"):
            return name
    for name in fieldnames:
        if name.startswith("outlier_score"):
            return name
    raise RuntimeError("Could not find an outlier score column in input CSV")


def _load_outliers(
    path: str,
    baseline_medians: Dict[str, float],
    comparison_label: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Outlier CSV has no header")
        score_col = _score_column(reader.fieldnames)
        reason_col = "top_vs_ref_features" if "top_vs_ref_features" in reader.fieldnames else "top_outlier_features"
        for row in reader:
            score = _safe_float(row.get(score_col) or "")
            reason_pairs = _parse_reason_text(row.get(reason_col, ""))
            simple = {
                "sid": row.get("sid", ""),
                "outlier_score": "" if score is None else "{0:.3f}".format(score),
                "severity": _severity(score),
                "main_pattern": _pattern(reason_pairs),
                "simple_summary": _simple_summary(row, baseline_medians, comparison_label),
                "top_reason_1": "",
                "top_reason_2": "",
                "top_reason_3": "",
                "comparison_label": comparison_label,
            }
            for idx, (name, z) in enumerate(reason_pairs[:3], start=1):
                ratio = _ratio(_safe_float(row.get(name, "")), baseline_medians.get(name))
                if ratio is not None:
                    text = "{0}: {1:.2f}x {2}".format(name, ratio, comparison_label)
                else:
                    text = "{0}: z={1:+.2f}".format(name, z)
                simple["top_reason_{0}".format(idx)] = text
            for name in PRIMARY_FEATURES:
                ratio = _ratio(_safe_float(row.get(name, "")), baseline_medians.get(name))
                simple["{0}_x_baseline".format(name)] = "" if ratio is None else "{0:.3f}".format(ratio)
            rows.append(simple)
    rows.sort(key=lambda row: _safe_float(row["outlier_score"]) or -1.0, reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = str(idx)
    return rows


def _write_csv(path: str, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "rank",
        "sid",
        "severity",
        "outlier_score",
        "comparison_label",
        "main_pattern",
        "simple_summary",
        "top_reason_1",
        "top_reason_2",
        "top_reason_3",
        "pet_iqr_x_baseline",
        "pet_cortex_iqr_x_baseline",
        "pet_cortex_p95_x_baseline",
        "pet_p95_to_median_x_baseline",
        "pet_cortex_std_x_baseline",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_txt(path: str, rows: Sequence[Dict[str, str]], top_k: int, title: str) -> None:
    with open(path, "w") as f:
        f.write("{0}\n".format(title))
        f.write("=" * len(title))
        f.write("\n\n")
        for row in rows[:top_k]:
            f.write(
                "#{rank} {sid}: {severity} (score={outlier_score}) | {main_pattern} | {simple_summary}\n".format(**row)
            )
            for key in ("top_reason_1", "top_reason_2", "top_reason_3"):
                if row.get(key):
                    f.write("  - {0}\n".format(row[key]))
            f.write("\n")


def _print_rows(title: str, rows: Sequence[Dict[str, str]], top_k: int) -> None:
    print(title)
    print("=" * len(title))
    print("")
    for row in rows[:top_k]:
        print(
            "#{rank} {sid}: {severity} (score={outlier_score}) | {main_pattern} | {simple_summary}".format(**row)
        )
        for key in ("top_reason_1", "top_reason_2", "top_reason_3"):
            if row.get(key):
                print("  - {0}".format(row[key]))
        print("")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn hard-to-read outlier CSVs into a compact human-readable summary."
    )
    parser.add_argument("--outliers-csv", help="Single CSV like datasetB_outliers_vs_datasetA.csv")
    parser.add_argument("--feature-comparison-csv", required=True, help="CSV like feature_comparison_A_vs_B.csv")
    parser.add_argument("--vs-reference-csv", help="Outliers CSV ranking subjects against the reference dataset")
    parser.add_argument("--within-dataset-csv", help="Outliers CSV ranking subjects within their own dataset")
    parser.add_argument(
        "--out-prefix",
        default="easy_outlier_summary",
        help="Prefix for output CSV/TXT files",
    )
    parser.add_argument("--top-k", type=int, default=50, help="How many top outliers to include in TXT report")
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Also write CSV/TXT files. By default this script only prints to stdout.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for compact outputs",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.write_files:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.vs_reference_csv and args.within_dataset_csv:
        ref_medians = _load_feature_baselines(args.feature_comparison_csv, "median_a")
        own_medians = _load_feature_baselines(args.feature_comparison_csv, "median_b")
        rows_vs = _load_outliers(args.vs_reference_csv, ref_medians, "reference median")
        rows_within = _load_outliers(args.within_dataset_csv, own_medians, "dataset median")

        if args.write_files:
            vs_csv_path = os.path.join(args.out_dir, "{0}_vs_reference_top{1}.csv".format(args.out_prefix, args.top_k))
            vs_txt_path = os.path.join(args.out_dir, "{0}_vs_reference_top{1}.txt".format(args.out_prefix, args.top_k))
            within_csv_path = os.path.join(args.out_dir, "{0}_within_dataset_top{1}.csv".format(args.out_prefix, args.top_k))
            within_txt_path = os.path.join(args.out_dir, "{0}_within_dataset_top{1}.txt".format(args.out_prefix, args.top_k))

            _write_csv(vs_csv_path, rows_vs[: args.top_k])
            _write_txt(vs_txt_path, rows_vs, args.top_k, "Top {0} subjects vs reference".format(args.top_k))
            _write_csv(within_csv_path, rows_within[: args.top_k])
            _write_txt(within_txt_path, rows_within, args.top_k, "Top {0} subjects within dataset".format(args.top_k))

            print("Wrote:")
            print("  {0}".format(os.path.abspath(vs_csv_path)))
            print("  {0}".format(os.path.abspath(vs_txt_path)))
            print("  {0}".format(os.path.abspath(within_csv_path)))
            print("  {0}".format(os.path.abspath(within_txt_path)))
            print("")

        _print_rows("Top {0} subjects vs reference".format(min(args.top_k, len(rows_vs))), rows_vs, args.top_k)
        _print_rows("Top {0} subjects within dataset".format(min(args.top_k, len(rows_within))), rows_within, args.top_k)
    else:
        if not args.outliers_csv:
            raise RuntimeError("Use --outliers-csv for single mode, or both --vs-reference-csv and --within-dataset-csv")
        baseline_medians = _load_feature_baselines(args.feature_comparison_csv, "median_a")
        rows = _load_outliers(args.outliers_csv, baseline_medians, "reference median")

        if args.write_files:
            csv_path = os.path.join(args.out_dir, "{0}.csv".format(args.out_prefix))
            txt_path = os.path.join(args.out_dir, "{0}.txt".format(args.out_prefix))
            _write_csv(csv_path, rows[: args.top_k])
            _write_txt(txt_path, rows, args.top_k, "Top {0} subjects".format(args.top_k))

            print("Wrote:")
            print("  {0}".format(os.path.abspath(csv_path)))
            print("  {0}".format(os.path.abspath(txt_path)))
            print("")

        _print_rows("Top {0} subjects".format(min(args.top_k, len(rows))), rows, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
