# ---------- main ----------

def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--meta_csv", required=True)
    ap.add_argument("--session_col", default="TAU_PET_Session")
    ap.add_argument("--centiloid_col", default="Centiloid")
    ap.add_argument("--centiloid_thr", type=float, default=18.4)
    ap.add_argument("--modalities", nargs="+", default=["pet_gt", "pet_fake", "mri"])
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--c_grid", nargs="+", type=float, default=[1, 10, 100, 0.1, 0.01, 0.001])
    ap.add_argument("--outdir", default="./svm_results")
    args = ap.parse_args()

    print("[ARGS]", vars(args), flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    folds = strict_fold_dirs(args.root, "ab_MGDA_UB_v33_500_", [1, 2, 3, 4, 5])
    for f in folds:
        print("FOLD:", f, flush=True)
    subj = collect_subjects_from_folds(folds)

    sids = sorted(subj.keys())
    labels = load_amyloid_labels(
        args.meta_csv, sids, args.session_col, args.centiloid_col, args.centiloid_thr
    )
    keep = [s for s in sids if norm_key(s) in labels]
    drop = [s for s in sids if norm_key(s) not in labels]
    if drop:
        print(f"[WARN] unlabeled excluded: {len(drop)} e.g. {drop[:6]}", flush=True)
    if not keep:
        raise RuntimeError("No labeled subjects after join.")
    y = np.array([labels[norm_key(s)] for s in keep], dtype=int)
    print("[LABELS] 0/1:", dict(Counter(y)), flush=True)

    manifest = pd.DataFrame(
        [{"subject": s, "label": int(labels[norm_key(s)]), **subj[s]} for s in keep]
    ).sort_values("subject")
    mpath = os.path.join(args.outdir, "dataset_manifest.csv")
    manifest.to_csv(mpath, index=False)
    print("[WRITE]", mpath, flush=True)

    rows = []
    for mod in args.modalities:
        print("\n=== Running SVM for", mod, "===", flush=True)
        paths = [subj[s][mod] for s in keep]

        metrics, preds = nested_cv_precomputed_kernel(
            img_paths=paths,
            y=y,
            subjects=keep,
            outer_splits=args.outer,
            inner_splits=args.inner,
            C_grid=tuple(args.c_grid),
            seed=10,
            mod_tag=mod,
        )

        ppath = os.path.join(args.outdir, f"predictions_{mod}.csv")
        preds.to_csv(ppath, index=False)
        print("[WRITE]", ppath, flush=True)

        print(
            f"[{mod}] AUC={metrics['AUC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
            f"Acc={metrics['Accuracy']:.4f}  BalAcc={metrics['BalancedAccuracy']:.4f}",
            flush=True,
        )
        rows.append({"modality": mod, **{k: float(v) for k, v in metrics.items()}})

    summary = pd.DataFrame(rows).set_index("modality")
    spath = os.path.join(args.outdir, "metrics_summary.csv")
    summary.to_csv(spath)
    print("\n[WRITE]", spath, flush=True)
    print(summary.to_string(), flush=True)

if __name__ == "__main__":
    main()



