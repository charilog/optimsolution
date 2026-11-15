#!/usr/bin/env python3
"""
bootstrap_lodo_ci.py
---------------------
Puts a 95% confidence interval on the LODO R2 / MAE / Spearman numbers that
train_gap_regressor.py (or its log-transform variant) already reports, by
cluster-bootstrapping over DATASETS -- the natural resampling unit for a
Leave-One-Dataset-Out evaluation, since each dataset (not each run) is one
independent fold.

This does not need the *_features.csv files again: it reads the
merged_dataset.csv that train_gap_regressor.py already wrote on its last
run, which already has every meta column, every landscape column, the
target column, and the one-hot optimizer columns joined together.

How the bootstrap works
------------------------
For B replicates (default 1000):
  1. Draw a bootstrap sample of the *dataset names*, with replacement, the
     same size as the original number of datasets (this is the standard
     cluster/block bootstrap for grouped data: resample the groups, not the
     rows within them).
  2. Re-run the exact same Leave-One-Dataset-Out loop as the original
     script, treating each name drawn as its own held-out fold in turn (a
     dataset drawn twice is simply evaluated twice, once per draw, against
     a training set built from the OTHER datasets in that draw).
  3. Record R2, MAE, and Spearman rho pooled over that replicate's held-out
     predictions, for both the meta-only and the meta+landscape feature
     sets.
The 2.5th and 97.5th percentiles across the B replicates give a 95% CI for
each metric. This directly addresses the "how stable is this R2 with only
25 datasets" question raised in the article's Discussion and Limitations
section, without requiring a new experiment.

Usage:
    python3 bootstrap_lodo_ci.py <merged_dataset.csv> [--n-boot 1000] [--seed 42] [--jobs 1]

--jobs N (default 1, i.e. the original fully serial behavior, unchanged):
    run N bootstrap replicates concurrently as separate worker processes
    (concurrent.futures.ProcessPoolExecutor). Each replicate only reads the
    same immutable `data` DataFrame and draws its own independent bootstrap
    sample -- there is no shared mutable state between replicates, so this
    is safe to raise up to roughly your CPU core count. NOTE: with --jobs
    > 1, each replicate uses its own sub-seed (--seed + replicate index)
    instead of one continuing RNG stream shared across all replicates, so
    that results never depend on which worker happens to run which
    replicate. This means a --jobs > 1 run's exact bootstrap draws (though
    not their statistical validity) differ slightly from a --jobs 1 run
    with the same --seed -- both are equally valid independent draws from
    the same resampling distribution.

The target column and feature-set columns are auto-detected from
merged_dataset.csv the same way train_gap_regressor.py builds them, so this
works unmodified on either the raw or the --log-transform output.

Requires: numpy, pandas, scipy, scikit-learn (same as train_gap_regressor.py).
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


META_CANDIDATES = ["dim", "iterations", "population", "evals", "wbound", "valsplit", "train_loss", "train_acc"]


def detect_columns(data: pd.DataFrame):
    opt_cols = [c for c in data.columns if c.startswith("opt_")]
    meta_cols = [c for c in META_CANDIDATES if c in data.columns]
    non_feature = set(opt_cols) | set(meta_cols) | {
        "run_key", "dataset", "optimizer", "seed", "label",
        "gen_gap_loss", "gen_gap_loss_slog", "val_loss", "val_acc",
    }
    landscape_cols = [c for c in data.columns if c not in non_feature and c.startswith("r")]
    if "gen_gap_loss_slog" in data.columns:
        target_col = "gen_gap_loss_slog"
    else:
        target_col = "gen_gap_loss"
    return meta_cols, opt_cols, landscape_cols, target_col


def run_lodo_once(data: pd.DataFrame, held_out_names, feature_cols, target_col, seed):
    """One full LODO pass over a (possibly bootstrap-resampled, possibly
    duplicated) list of held-out dataset names. A name that appears twice in
    held_out_names is evaluated twice, independently, exactly as a cluster
    bootstrap over groups requires."""
    preds, actuals = [], []
    for held_out in held_out_names:
        train_mask = data["dataset"] != held_out
        test_mask = data["dataset"] == held_out
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        X_train = data.loc[train_mask, feature_cols]
        y_train = data.loc[train_mask, target_col]
        X_test = data.loc[test_mask, feature_cols]
        y_test = data.loc[test_mask, target_col]

        reg = RandomForestRegressor(n_estimators=200, max_depth=4,
                                     min_samples_leaf=1, random_state=seed)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        preds.extend(y_pred)
        actuals.extend(y_test.tolist())

    preds = np.array(preds)
    actuals = np.array(actuals)
    if len(preds) < 2:
        return float("nan"), float("nan"), float("nan")
    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = sstats.spearmanr(actuals, preds)
    return r2, mae, rho


# ---------------------------------------------------------------------
# Parallel path (--jobs > 1 only). Each worker process is initialized ONCE
# (via ProcessPoolExecutor's initializer=, not per-task) with the read-only
# data it needs, so `data` itself is never re-pickled/re-sent per replicate
# -- only the small (b -> result) messages cross the process boundary.
# ---------------------------------------------------------------------
_worker_data = None
_worker_datasets = None
_worker_feature_sets = None
_worker_target_col = None
_worker_seed = None


def _init_bootstrap_worker(data, unique_datasets, feature_sets, target_col, seed):
    global _worker_data, _worker_datasets, _worker_feature_sets, _worker_target_col, _worker_seed
    _worker_data = data
    _worker_datasets = unique_datasets
    _worker_feature_sets = feature_sets
    _worker_target_col = target_col
    _worker_seed = seed


def _run_one_bootstrap_replicate(b):
    """One full bootstrap replicate (draw + LODO pass for every feature set),
    run inside a worker process. Uses its own independent sub-seed (base
    seed + replicate index b) rather than sharing one continuing RNG stream
    across replicates -- see the --jobs note in the module docstring for why."""
    rng_b = np.random.default_rng(_worker_seed + b)
    boot_names = rng_b.choice(_worker_datasets, size=len(_worker_datasets), replace=True).tolist()
    out = {}
    for label, feature_cols in _worker_feature_sets.items():
        out[label] = run_lodo_once(_worker_data, boot_names, feature_cols, _worker_target_col, seed=_worker_seed)
    return out


def summarize(vals, label):
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        print(f"[{label}] no finite bootstrap replicates")
        return
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"[{label}] median={np.median(vals):.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (n_boot={len(vals)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merged_csv", help="merged_dataset.csv written by train_gap_regressor(.py)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1,
                     help="Run this many bootstrap replicates concurrently as separate worker "
                          "processes (default 1 = original serial behavior). See the module "
                          "docstring's --jobs note for the reproducibility caveat.")
    args = ap.parse_args()

    data = pd.read_csv(args.merged_csv)
    meta_cols, opt_cols, landscape_cols, target_col = detect_columns(data)
    print(f"Detected target column: {target_col}")
    print(f"Meta columns ({len(meta_cols)}): {meta_cols}")
    print(f"Optimizer one-hot columns ({len(opt_cols)}): {opt_cols}")
    print(f"Landscape columns ({len(landscape_cols)}): {landscape_cols[:6]}{' ...' if len(landscape_cols) > 6 else ''}")

    unique_datasets = sorted(data["dataset"].unique())
    n_datasets = len(unique_datasets)
    print(f"\n{n_datasets} distinct dataset(s), bootstrapping {args.n_boot} cluster resamples "
          f"of the dataset list (with replacement)...\n")

    rng = np.random.default_rng(args.seed)

    feature_sets = {
        "meta-only baseline (no landscape features)": meta_cols + opt_cols,
        "meta + local-landscape features": meta_cols + opt_cols + landscape_cols,
    }

    results = {label: {"r2": [], "mae": [], "spearman": []} for label in feature_sets}

    if args.jobs > 1:
        print(f"Running {args.n_boot} bootstrap replicates using {args.jobs} parallel worker "
              f"processes (optimsolution.cfg [nnlandscape] bootstrap_jobs > 1)...\n")
        done = 0
        # chunksize keeps IPC overhead down for a large --n-boot without
        # making any one worker wait too long on a straggler.
        chunksize = max(1, args.n_boot // (args.jobs * 4))
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_init_bootstrap_worker,
            initargs=(data, unique_datasets, feature_sets, target_col, args.seed),
        ) as ex:
            for out in ex.map(_run_one_bootstrap_replicate, range(args.n_boot), chunksize=chunksize):
                for label, (r2, mae, rho) in out.items():
                    results[label]["r2"].append(r2)
                    results[label]["mae"].append(mae)
                    results[label]["spearman"].append(rho)
                done += 1
                if done % max(1, args.n_boot // 10) == 0:
                    print(f"  ... {done}/{args.n_boot} bootstrap replicates done")
    else:
        # Original serial path, byte-for-byte unchanged: one continuing RNG
        # stream shared across every replicate.
        for b in range(args.n_boot):
            boot_names = rng.choice(unique_datasets, size=n_datasets, replace=True).tolist()
            for label, feature_cols in feature_sets.items():
                r2, mae, rho = run_lodo_once(data, boot_names, feature_cols, target_col, seed=args.seed)
                results[label]["r2"].append(r2)
                results[label]["mae"].append(mae)
                results[label]["spearman"].append(rho)
            if (b + 1) % max(1, args.n_boot // 10) == 0:
                print(f"  ... {b + 1}/{args.n_boot} bootstrap replicates done")

    print("\n" + "=" * 70)
    print("Bootstrap 95% confidence intervals (cluster bootstrap over datasets)")
    print("=" * 70)
    for label in feature_sets:
        print(f"\n{label}:")
        summarize(results[label]["r2"], "R2")
        summarize(results[label]["mae"], "MAE")
        summarize(results[label]["spearman"], "Spearman")

    out_csv = args.merged_csv.rsplit(".", 1)[0] + "_bootstrap_ci.csv"
    rows = []
    for label in feature_sets:
        for metric in ("r2", "mae", "spearman"):
            vals = np.array([v for v in results[label][metric] if np.isfinite(v)])
            if len(vals) == 0:
                continue
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({"feature_set": label, "metric": metric,
                         "median": np.median(vals), "ci_lo": lo, "ci_hi": hi, "n_boot": len(vals)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote bootstrap summary -> {out_csv}")


if __name__ == "__main__":
    main()
