#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
except Exception:
    ks_2samp = None
    mannwhitneyu = None
    wasserstein_distance = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


REQUIRED_FILES = [
    "T1_masked.nii.gz",
    "PET_in_T1_masked.nii.gz",
    "aseg_brainmask.nii.gz",
    "mask_cortex.nii.gz",
]

IGNORE_DIR_NAMES = {
    ".git",
    "__pycache__",
    "results",
    "CV5_ab_strat",
    "CV5_braak_strat",
    "CV5_tau_strat",
}

PLOT_TOP_K = 6
EPS = 1e-6


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned or "dataset"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    out = np.asarray(list(values), dtype=np.float64)
    return out[np.isfinite(out)]


def _percentiles(values: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    if values.size == 0:
        return {"p{0:02d}".format(int(round(q * 100))): math.nan for q in qs}
    pct = np.quantile(values, qs)
    return {
        "p{0:02d}".format(int(round(q * 100))): float(v)
        for q, v in zip(qs, pct)
    }


def _masked_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            "{0}_count".format(prefix): 0.0,
            "{0}_mean".format(prefix): math.nan,
            "{0}_std".format(prefix): math.nan,
            "{0}_min".format(prefix): math.nan,
            "{0}_max".format(prefix): math.nan,
            "{0}_median".format(prefix): math.nan,
            "{0}_iqr".format(prefix): math.nan,
            "{0}_p01".format(prefix): math.nan,
            "{0}_p05".format(prefix): math.nan,
            "{0}_p25".format(prefix): math.nan,
            "{0}_p75".format(prefix): math.nan,
            "{0}_p95".format(prefix): math.nan,
            "{0}_p99".format(prefix): math.nan,
        }
    qs = _percentiles(values, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    return {
        "{0}_count".format(prefix): float(values.size),
        "{0}_mean".format(prefix): float(values.mean()),
        "{0}_std".format(prefix): float(values.std()),
        "{0}_min".format(prefix): float(values.min()),
        "{0}_max".format(prefix): float(values.max()),
        "{0}_median".format(prefix): float(np.median(values)),
        "{0}_iqr".format(prefix): float(qs["p75"] - qs["p25"]),
        "{0}_p01".format(prefix): float(qs["p01"]),
        "{0}_p05".format(prefix): float(qs["p05"]),
        "{0}_p25".format(prefix): float(qs["p25"]),
        "{0}_p75".format(prefix): float(qs["p75"]),
        "{0}_p95".format(prefix): float(qs["p95"]),
        "{0}_p99".format(prefix): float(qs["p99"]),
    }


def _robust_center_scale(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = _finite_array(values)
    if arr.size == 0:
        return None, None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < EPS:
        q75, q25 = np.quantile(arr, [0.75, 0.25])
        mad = float((q75 - q25) / 1.349)
    if mad < EPS:
        std = float(arr.std())
        mad = std if std >= EPS else math.nan
    return med, mad


def _robust_z(value: float, center: Optional[float], scale: Optional[float]) -> Optional[float]:
    if center is None or scale is None or not math.isfinite(value) or not math.isfinite(scale) or scale < EPS:
        return None
    return float(0.67448975 * (value - center) / scale)


def _list_subject_dirs(root: str, max_subjects: Optional[int]) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError("Dataset root does not exist: {0}".format(root))
    subject_dirs: List[str] = []
    for entry in sorted(os.scandir(root), key=lambda x: x.name):
        if not entry.is_dir():
            continue
        if entry.name in IGNORE_DIR_NAMES:
            continue
        if not all(os.path.exists(os.path.join(entry.path, name)) for name in REQUIRED_FILES):
            continue
        subject_dirs.append(entry.path)
        if max_subjects is not None and len(subject_dirs) >= max_subjects:
            break
    return subject_dirs


def _load_volume(path: str, dtype: Any = np.float32) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(path)
    arr = np.asarray(img.get_fdata(dtype=dtype), dtype=dtype)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    return arr, zooms


def _subject_features(subject_dir: str, dataset_name: str) -> Dict[str, Any]:
    sid = os.path.basename(subject_dir)
    paths = {name: os.path.join(subject_dir, name) for name in REQUIRED_FILES}
    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "sid": sid,
        "subject_dir": subject_dir,
        "error": "",
    }
    try:
        t1, t1_zooms = _load_volume(paths["T1_masked.nii.gz"])
        pet, pet_zooms = _load_volume(paths["PET_in_T1_masked.nii.gz"])
        mask, mask_zooms = _load_volume(paths["aseg_brainmask.nii.gz"], dtype=np.float32)
        cortex, cortex_zooms = _load_volume(paths["mask_cortex.nii.gz"], dtype=np.float32)

        if t1.shape != pet.shape or t1.shape != mask.shape or t1.shape != cortex.shape:
            raise RuntimeError(
                "shape mismatch T1={0} PET={1} mask={2} cortex={3}".format(
                    t1.shape, pet.shape, mask.shape, cortex.shape
                )
            )

        mask_bool = mask > 0
        cortex_bool = np.logical_and(cortex > 0, mask_bool)
        if not np.any(mask_bool):
            raise RuntimeError("brain mask is empty")

        t1_vals = t1[mask_bool].astype(np.float64, copy=False)
        pet_vals = pet[mask_bool].astype(np.float64, copy=False)
        pet_cortex_vals = pet[cortex_bool].astype(np.float64, copy=False)
        t1_cortex_vals = t1[cortex_bool].astype(np.float64, copy=False)

        brain_voxels = int(mask_bool.sum())
        cortex_voxels = int(cortex_bool.sum())
        shape_x, shape_y, shape_z = (int(v) for v in t1.shape)
        spacing_x, spacing_y, spacing_z = (float(v) for v in t1_zooms)
        voxel_volume = spacing_x * spacing_y * spacing_z

        row.update(
            {
                "shape_x": shape_x,
                "shape_y": shape_y,
                "shape_z": shape_z,
                "spacing_x_mm": spacing_x,
                "spacing_y_mm": spacing_y,
                "spacing_z_mm": spacing_z,
                "voxel_volume_mm3": voxel_volume,
                "brain_voxels": float(brain_voxels),
                "cortex_voxels": float(cortex_voxels),
                "cortex_fraction": float(cortex_voxels / max(brain_voxels, 1)),
                "mask_fraction_of_volume": float(brain_voxels / float(np.prod(t1.shape))),
                "t1_pet_spacing_absdiff": float(
                    abs(t1_zooms[0] - pet_zooms[0]) + abs(t1_zooms[1] - pet_zooms[1]) + abs(t1_zooms[2] - pet_zooms[2])
                ),
                "t1_mask_spacing_absdiff": float(
                    abs(t1_zooms[0] - mask_zooms[0]) + abs(t1_zooms[1] - mask_zooms[1]) + abs(t1_zooms[2] - mask_zooms[2])
                ),
                "t1_cortex_spacing_absdiff": float(
                    abs(t1_zooms[0] - cortex_zooms[0]) + abs(t1_zooms[1] - cortex_zooms[1]) + abs(t1_zooms[2] - cortex_zooms[2])
                ),
                "pet_negative_fraction": float(np.mean(pet_vals < 0)),
                "pet_zero_fraction": float(np.mean(np.isclose(pet_vals, 0.0))),
                "pet_nan_fraction": float(1.0 - np.mean(np.isfinite(pet_vals))),
                "t1_nan_fraction": float(1.0 - np.mean(np.isfinite(t1_vals))),
            }
        )

        row.update(_masked_stats(t1_vals[np.isfinite(t1_vals)], "t1"))
        row.update(_masked_stats(t1_cortex_vals[np.isfinite(t1_cortex_vals)], "t1_cortex"))
        row.update(_masked_stats(pet_vals[np.isfinite(pet_vals)], "pet"))
        row.update(_masked_stats(pet_cortex_vals[np.isfinite(pet_cortex_vals)], "pet_cortex"))

        row["pet_p95_to_median"] = float(row["pet_p95"] / max(abs(row["pet_median"]), EPS))
        row["pet_p99_to_median"] = float(row["pet_p99"] / max(abs(row["pet_median"]), EPS))
        row["pet_max_to_p99"] = float(row["pet_max"] / max(abs(row["pet_p99"]), EPS))
        row["pet_cortex_to_brain_mean_ratio"] = float(row["pet_cortex_mean"] / max(abs(row["pet_mean"]), EPS))
        row["t1_cortex_to_brain_mean_ratio"] = float(row["t1_cortex_mean"] / max(abs(row["t1_mean"]), EPS))
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _subject_features_from_args(args: Tuple[str, str]) -> Dict[str, Any]:
    return _subject_features(args[0], args[1])


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _collect_features(
    subject_dirs: Sequence[str],
    dataset_name: str,
    workers: int,
) -> List[Dict[str, Any]]:
    if workers <= 1:
        return [_subject_features(path, dataset_name) for path in subject_dirs]
    args = [(path, dataset_name) for path in subject_dirs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_subject_features_from_args, args))


def _numeric_feature_names(rows: Sequence[Dict[str, Any]]) -> List[str]:
    exclude = {"dataset", "sid", "subject_dir", "error", "top_outlier_features", "top_vs_ref_features"}
    names = []
    for key in sorted({k for row in rows for k in row.keys()}):
        if key in exclude:
            continue
        values = [_safe_float(row.get(key)) for row in rows]
        if any(v is not None for v in values):
            names.append(key)
    return names


def _feature_array(rows: Sequence[Dict[str, Any]], feature: str) -> np.ndarray:
    vals = [_safe_float(row.get(feature)) for row in rows]
    return _finite_array(v for v in vals if v is not None)


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return math.nan
    var = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / max(a.size + b.size - 2, 1)
    return float(math.sqrt(max(var, 0.0)))


def _feature_group(name: str) -> str:
    if name.startswith("pet_"):
        return "pet"
    if name.startswith("t1_"):
        return "t1"
    return "geometry"


def _compare_feature(name: str, rows_a: Sequence[Dict[str, Any]], rows_b: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    a = _feature_array(rows_a, name)
    b = _feature_array(rows_b, name)
    if a.size == 0 or b.size == 0:
        return None
    pooled = _pooled_std(a, b)
    out: Dict[str, Any] = {
        "feature": name,
        "group": _feature_group(name),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "std_a": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "std_b": float(b.std(ddof=1)) if b.size > 1 else 0.0,
        "mean_diff": float(b.mean() - a.mean()),
        "median_diff": float(np.median(b) - np.median(a)),
        "standardized_mean_diff": None if not math.isfinite(pooled) or pooled < EPS else float((b.mean() - a.mean()) / pooled),
        "wasserstein": None,
        "ks_p_value": None,
        "mw_p_value": None,
    }
    if wasserstein_distance is not None:
        out["wasserstein"] = float(wasserstein_distance(a, b))
    if ks_2samp is not None:
        out["ks_p_value"] = float(ks_2samp(a, b, alternative="two-sided", mode="auto").pvalue)
    if mannwhitneyu is not None:
        out["mw_p_value"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return out


def _sort_feature_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(row: Dict[str, Any]) -> Tuple[float, float]:
        p = row.get("mw_p_value")
        effect = row.get("standardized_mean_diff")
        p_score = float(p) if p is not None and math.isfinite(float(p)) else 1.0
        effect_score = abs(float(effect)) if effect is not None and math.isfinite(float(effect)) else 0.0
        return (p_score, -effect_score)

    return sorted(rows, key=key_fn)


def _dataset_summary(dataset_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if not row.get("error")]
    err_rows = [row for row in rows if row.get("error")]
    return {
        "dataset": dataset_name,
        "subject_count_ok": int(len(ok_rows)),
        "subject_count_error": int(len(err_rows)),
        "errors": [{"sid": row["sid"], "error": row["error"]} for row in err_rows[:20]],
    }


def _attach_outlier_scores(
    rows: List[Dict[str, Any]],
    reference_rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    score_field: str,
    feature_field: str,
) -> None:
    centers: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        name: _robust_center_scale([_safe_float(row.get(name)) for row in reference_rows])
        for name in feature_names
    }
    for row in rows:
        z_pairs: List[Tuple[str, float]] = []
        for name in feature_names:
            value = _safe_float(row.get(name))
            if value is None:
                continue
            center, scale = centers[name]
            z = _robust_z(value, center, scale)
            if z is None:
                continue
            z_pairs.append((name, z))
        if not z_pairs:
            row[score_field] = math.nan
            row[feature_field] = ""
            continue
        abs_z = np.asarray([abs(z) for _, z in z_pairs], dtype=np.float64)
        row[score_field] = float(math.sqrt(np.mean(abs_z ** 2)))
        top = sorted(z_pairs, key=lambda item: abs(item[1]), reverse=True)[:5]
        row[feature_field] = ";".join("{0}:{1:+.2f}".format(name, z) for name, z in top)


def _top_outliers(rows: Sequence[Dict[str, Any]], score_field: str, limit: int = 20) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        score = _safe_float(row.get(score_field))
        if score is None:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: float(row[score_field]), reverse=True)
    return list(filtered[:limit])


def _plot_top_shifted_features(
    feature_rows: Sequence[Dict[str, Any]],
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    name_a: str,
    name_b: str,
    out_path: str,
) -> Optional[str]:
    if plt is None:
        return None
    chosen = [row["feature"] for row in feature_rows[:PLOT_TOP_K]]
    if not chosen:
        return None
    ncols = 2
    nrows = int(math.ceil(len(chosen) / float(ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.5 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, feature in zip(axes_arr, chosen):
        a = _feature_array(rows_a, feature)
        b = _feature_array(rows_b, feature)
        if a.size == 0 or b.size == 0:
            ax.set_visible(False)
            continue
        bins = 30
        ax.hist(a, bins=bins, alpha=0.5, density=True, label=name_a)
        ax.hist(b, bins=bins, alpha=0.5, density=True, label=name_b)
        ax.set_title(feature)
        ax.legend(fontsize=8)
    for ax in axes_arr[len(chosen):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _render_report(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    feature_rows: Sequence[Dict[str, Any]],
    outliers_b_vs_a: Sequence[Dict[str, Any]],
    out_dir: str,
) -> str:
    lines = [
        "{0}: ok={1}, errors={2}".format(
            summary_a["dataset"], summary_a["subject_count_ok"], summary_a["subject_count_error"]
        ),
        "{0}: ok={1}, errors={2}".format(
            summary_b["dataset"], summary_b["subject_count_ok"], summary_b["subject_count_error"]
        ),
        "",
        "Top shifted features overall:",
    ]
    for row in feature_rows[:10]:
        effect = row.get("standardized_mean_diff")
        effect_text = "NA" if effect is None else "{0:+.2f}".format(effect)
        p = row.get("mw_p_value")
        p_text = "NA" if p is None else "{0:.4g}".format(p)
        lines.append(
            "  {0}: mean_a={1:.4g}, mean_b={2:.4g}, effect={3}, mw_p={4}".format(
                row["feature"], row["mean_a"], row["mean_b"], effect_text, p_text
            )
        )
    pet_geom_rows = [row for row in feature_rows if row.get("group") != "t1"]
    lines.append("")
    lines.append("Top shifted PET/geometry features:")
    for row in pet_geom_rows[:10]:
        effect = row.get("standardized_mean_diff")
        effect_text = "NA" if effect is None else "{0:+.2f}".format(effect)
        p = row.get("mw_p_value")
        p_text = "NA" if p is None else "{0:.4g}".format(p)
        lines.append(
            "  {0}: mean_a={1:.4g}, mean_b={2:.4g}, effect={3}, mw_p={4}".format(
                row["feature"], row["mean_a"], row["mean_b"], effect_text, p_text
            )
        )
    lines.append("")
    lines.append("Top dataset-B outliers vs dataset-A baseline:")
    for row in outliers_b_vs_a[:10]:
        lines.append(
            "  {0}: score={1:.3f}, reasons={2}".format(
                row["sid"],
                float(row["outlier_score_vs_ref_a"]),
                row.get("top_vs_ref_features", ""),
            )
        )
    lines.append("")
    lines.append("Artifacts written to: {0}".format(os.path.abspath(out_dir)))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MRI/PET subject-level distributions across two dataset roots and flag outlier subjects."
        )
    )
    parser.add_argument("--root-a", required=True, help="Reference dataset root")
    parser.add_argument("--root-b", required=True, help="Comparison dataset root")
    parser.add_argument("--name-a", default="dataset_a", help="Label for dataset A")
    parser.add_argument("--name-b", default="dataset_b", help="Label for dataset B")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("results", "distribution_audit"),
        help="Directory for CSV/JSON/plot outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for subject scanning",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional cap per dataset for quick dry runs",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip histogram plot generation",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    slug_a = _safe_slug(args.name_a)
    slug_b = _safe_slug(args.name_b)

    subject_dirs_a = _list_subject_dirs(args.root_a, args.max_subjects)
    subject_dirs_b = _list_subject_dirs(args.root_b, args.max_subjects)
    if not subject_dirs_a:
        raise RuntimeError("No valid subject folders found in dataset A")
    if not subject_dirs_b:
        raise RuntimeError("No valid subject folders found in dataset B")

    rows_a = _collect_features(subject_dirs_a, args.name_a, max(args.workers, 1))
    rows_b = _collect_features(subject_dirs_b, args.name_b, max(args.workers, 1))

    _write_csv(os.path.join(args.out_dir, "{0}_subject_features.csv".format(slug_a)), rows_a)
    _write_csv(os.path.join(args.out_dir, "{0}_subject_features.csv".format(slug_b)), rows_b)

    ok_rows_a = [row for row in rows_a if not row.get("error")]
    ok_rows_b = [row for row in rows_b if not row.get("error")]

    feature_names = _numeric_feature_names(ok_rows_a + ok_rows_b)
    loader_relevant_feature_names = [
        name
        for name in feature_names
        if _feature_group(name) != "t1"
    ]
    comparison_rows = []
    for name in feature_names:
        compared = _compare_feature(name, ok_rows_a, ok_rows_b)
        if compared is not None:
            comparison_rows.append(compared)
    comparison_rows = _sort_feature_rows(comparison_rows)

    _attach_outlier_scores(
        ok_rows_a,
        ok_rows_a,
        loader_relevant_feature_names,
        "outlier_score_within_dataset",
        "top_outlier_features",
    )
    _attach_outlier_scores(
        ok_rows_b,
        ok_rows_b,
        loader_relevant_feature_names,
        "outlier_score_within_dataset",
        "top_outlier_features",
    )
    _attach_outlier_scores(
        ok_rows_b,
        ok_rows_a,
        loader_relevant_feature_names,
        "outlier_score_vs_ref_a",
        "top_vs_ref_features",
    )
    _attach_outlier_scores(
        ok_rows_a,
        ok_rows_b,
        loader_relevant_feature_names,
        "outlier_score_vs_ref_b",
        "top_vs_ref_features",
    )

    _write_csv(
        os.path.join(args.out_dir, "feature_comparison_{0}_vs_{1}.csv".format(slug_a, slug_b)),
        comparison_rows,
    )
    _write_csv(
        os.path.join(args.out_dir, "{0}_outliers_within.csv".format(slug_a)),
        _top_outliers(ok_rows_a, "outlier_score_within_dataset", limit=len(ok_rows_a)),
    )
    _write_csv(
        os.path.join(args.out_dir, "{0}_outliers_within.csv".format(slug_b)),
        _top_outliers(ok_rows_b, "outlier_score_within_dataset", limit=len(ok_rows_b)),
    )
    _write_csv(
        os.path.join(args.out_dir, "{0}_outliers_vs_{1}.csv".format(slug_b, slug_a)),
        _top_outliers(ok_rows_b, "outlier_score_vs_ref_a", limit=len(ok_rows_b)),
    )

    summary_a = _dataset_summary(args.name_a, rows_a)
    summary_b = _dataset_summary(args.name_b, rows_b)
    payload = {
        "summary_a": summary_a,
        "summary_b": summary_b,
        "top_shifted_features": comparison_rows[:20],
        "top_outliers_b_vs_a": [
            {
                "sid": row["sid"],
                "score": float(row["outlier_score_vs_ref_a"]),
                "reasons": row.get("top_vs_ref_features", ""),
            }
            for row in _top_outliers(ok_rows_b, "outlier_score_vs_ref_a", limit=20)
        ],
    }
    _write_json(os.path.join(args.out_dir, "distribution_summary.json"), payload)

    if not args.no_plots:
        _plot_top_shifted_features(
            comparison_rows,
            ok_rows_a,
            ok_rows_b,
            args.name_a,
            args.name_b,
            os.path.join(args.out_dir, "top_feature_histograms.png"),
        )

    report = _render_report(
        summary_a,
        summary_b,
        comparison_rows,
        _top_outliers(ok_rows_b, "outlier_score_vs_ref_a", limit=20),
        args.out_dir,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
