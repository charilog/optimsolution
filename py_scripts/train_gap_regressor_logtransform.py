#!/usr/bin/env python3
"""
train_gap_regressor_logtransform.py
------------------------------------
Drop-in variant of the project's own py_scripts/train_gap_regressor.py that
adds one optional behaviour: --log-transform.

Why this script exists
-----------------------
The article's regression-pool analysis found that a handful of high-
dimensional datasets (the "iran" family in particular) produce generalization
gaps that are orders of magnitude larger than every other dataset's, which
distorts the raw pooled mean used to compare optimizers (see the "Discussion
and Limitations" section) but does NOT distort a signed-log-compressed
version of the same values, computed straight from the coarse 9-statistic
summary already in nn_results.xlsx. That check could not reach the actual
Leave-One-Dataset-Out (LODO) random-forest result, because the LODO analysis
needs the full local-landscape feature set (mean/std/min/max/range/iqr/skew/
kurtosis/cv per radius, computed from the *_features.csv sampled-point files)
that this script -- not the coarse summary -- was built to consume.

This script performs the same test properly, on the full feature set:
  - With --log-transform OFF (the default), it behaves identically to the
    original train_gap_regressor.py: same inputs, same outputs, same
    columns. Running it with no flags on your existing data should reproduce
    the nn_lodo_results.xlsx / nn_feature_importance.xlsx numbers already in
    the article.
  - With --log-transform ON, every loss-scale-dependent quantity is put
    through a signed-log transform, slog(x) = sign(x) * log1p(|x|), BEFORE
    any statistic (mean, std, skew, ...) is computed from it. This means:
      * the local-sample loss values "f" are slog-transformed before the
        per-radius landscape statistics (r{radius}_f_mean, ..._skew, etc.)
        are computed from them,
      * the "train_loss" meta feature is slog-transformed,
      * the regression target itself, "gen_gap_loss", is replaced by
        slog(gen_gap_loss).
    Everything downstream (the LODO loop, the meta-only vs meta+landscape
    comparison, the feature-importance ranking) is otherwise unchanged.

Why signed-log and not plain log
----------------------------------
gen_gap_loss can be negative (validation loss sometimes ends up below
training loss for a given seed), so a plain log() is undefined for a
fraction of rows. slog(x) = sign(x) * log1p(|x|) is defined everywhere,
strictly monotonic, and reduces to an ordinary log-like compression for
large |x| while staying close to the identity near zero -- exactly the
property needed to tame the "iran"-family outliers without discarding or
reinterpreting any run. This mirrors, on the full feature set, the same
slog check already reported in the article's Table 7 (computed there from
the coarser nn_results.xlsx summary alone).

Usage (identical to the original script, plus the new flag):
    python3 train_gap_regressor_logtransform.py <features_dir> <summary_csv> <out_dir> [--log-transform]

Example:
    # Baseline (should match the numbers already in the article):
    python3 train_gap_regressor_logtransform.py features_raw/ nn_summary.csv out_raw/

    # Scale-hypothesis test:
    python3 train_gap_regressor_logtransform.py features_raw/ nn_summary.csv out_log/ --log-transform

Then compare out_raw/lodo_results.csv against out_log/lodo_results.csv:
if the scale hypothesis is correct, R2 for "meta + local-landscape features"
should improve substantially (and plausibly turn positive) in out_log/
relative to out_raw/, mirroring what Table 7's coarse slog check already
shows for the pooled mean vs Friedman rank agreement.

Requires: numpy, pandas, scipy, scikit-learn (identical to the original
script -- no new dependencies).
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def slog(x):
    """Signed log: sign(x) * log1p(|x|). Defined for all real x, including
    negative and zero. Reduces the article's Table 7 pooled-mean check to a
    per-row transform applied consistently to every loss-scale quantity."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def landscape_stats(f: pd.Series, prefix: str, log_transform: bool) -> dict:
    """Cheap distribution-shape stats over one local sample's loss values.
    If log_transform is set, f is slog-transformed before any statistic is
    computed from it, so every one of these nine numbers is on a compressed
    scale rather than the raw, potentially huge, loss scale."""
    f = f.to_numpy(dtype=float)
    f = f[np.isfinite(f)]
    if log_transform:
        f = slog(f)
    if f.size < 2:
        return {}
    q1, q3 = np.percentile(f, [25, 75])
    out = {
        f"{prefix}_mean": float(np.mean(f)),
        f"{prefix}_std":  float(np.std(f)),
        f"{prefix}_min":  float(np.min(f)),
        f"{prefix}_max":  float(np.max(f)),
        f"{prefix}_range": float(np.max(f) - np.min(f)),
        f"{prefix}_iqr":  float(q3 - q1),
        f"{prefix}_skew": float(sstats.skew(f)) if f.size >= 3 else 0.0,
        f"{prefix}_kurt": float(sstats.kurtosis(f)) if f.size >= 4 else 0.0,
        f"{prefix}_cv":   float(np.std(f) / abs(np.mean(f))) if abs(np.mean(f)) > 1e-12 else np.nan,
    }
    return out


def build_landscape_feature_table(features_dir: str, log_transform: bool) -> pd.DataFrame:
    """One row per run_key, columns = landscape_stats(...) for every radius
    found among that run_key's *_features.csv files."""
    paths = sorted(glob.glob(os.path.join(features_dir, "*_features.csv")))
    if not paths:
        raise SystemExit(f"No *_features.csv files found in {features_dir}")

    rows = {}  # run_key -> dict of feature columns
    for path in paths:
        df = pd.read_csv(path)
        if df.empty or "run_key" not in df.columns or "radius_frac" not in df.columns:
            print(f"  [warn] skipping malformed file: {path}")
            continue
        run_key = df["run_key"].iloc[0]
        radius = df["radius_frac"].iloc[0]
        prefix = f"r{radius}_f"

        stats_dict = landscape_stats(df["f"], prefix, log_transform)
        row = rows.setdefault(run_key, {"run_key": run_key})
        row.update(stats_dict)

    table = pd.DataFrame(list(rows.values()))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features_dir")
    ap.add_argument("summary_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--log-transform", action="store_true",
                     help="Apply a signed-log transform to every loss-scale quantity "
                          "(local-sample f values, train_loss, and the gen_gap_loss target) "
                          "before computing any statistic or fitting any model.")
    args = ap.parse_args()

    features_dir, summary_csv, out_dir = args.features_dir, args.summary_csv, args.out_dir
    log_transform = args.log_transform
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {features_dir} for local-landscape sample files "
          f"({'log-transformed' if log_transform else 'raw'})...")
    landscape_df = build_landscape_feature_table(features_dir, log_transform)
    print(f"  -> {len(landscape_df)} run(s) with landscape features, "
          f"{landscape_df.shape[1] - 1} landscape feature columns")

    summary_df = pd.read_csv(summary_csv)
    print(f"Loaded {summary_csv}: {len(summary_df)} run(s)")

    data = summary_df.merge(landscape_df, on="run_key", how="inner")
    if len(data) < len(summary_df):
        missing = set(summary_df.run_key) - set(data.run_key)
        print(f"[warn] {len(missing)} run(s) in summary.csv had no matching "
              f"landscape features and were dropped: {sorted(missing)[:5]}"
              + (" ..." if len(missing) > 5 else ""))
    print(f"Merged dataset: {len(data)} runs, "
          f"{data['dataset'].nunique()} distinct dataset(s): {sorted(data['dataset'].unique())}")

    # -- Feature sets ---------------------------------------------------------
    landscape_cols = [c for c in landscape_df.columns if c != "run_key"]
    meta_cols = ["dim", "iterations", "population", "evals", "wbound",
                 "valsplit", "train_loss", "train_acc"]
    meta_cols = [c for c in meta_cols if c in data.columns]

    data_enc = pd.get_dummies(data, columns=["optimizer"], prefix="opt")
    opt_cols = [c for c in data_enc.columns if c.startswith("opt_")]

    if log_transform:
        # train_loss is a loss-scale quantity like everything else here.
        if "train_loss" in data_enc.columns:
            data_enc["train_loss"] = slog(data_enc["train_loss"])
        # Build the (compressed) target as its own column so the original
        # "gen_gap_loss" is left untouched in the merged CSV for inspection.
        data_enc["gen_gap_loss_slog"] = slog(data_enc["gen_gap_loss"])
        target_col = "gen_gap_loss_slog"
    else:
        target_col = "gen_gap_loss"

    data_enc = data_enc.replace([np.inf, -np.inf], np.nan)
    data_enc = data_enc.dropna(subset=[target_col])

    for cols in (landscape_cols, meta_cols):
        if cols:
            data_enc[cols] = data_enc[cols].apply(
                lambda s: s.fillna(s.median()) if s.notna().any() else s.fillna(0.0))

    unique_datasets = sorted(data_enc["dataset"].unique())
    print(f"\nLeave-One-Dataset-Out CV over {len(unique_datasets)} dataset(s): {unique_datasets}\n")

    if len(unique_datasets) < 2:
        print("[note] Only one distinct dataset present -- LODO CV needs >=2 "
              "held-out folds to be meaningful. Running it anyway (single fold) "
              "purely to confirm the code path executes; treat any R2/MAE "
              "numbers below as a functional smoke test, not a result.")

    def run_lodo(feature_cols, label):
        preds, actuals, held = [], [], []
        for held_out in unique_datasets:
            train_mask = data_enc["dataset"] != held_out
            test_mask = data_enc["dataset"] == held_out
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            X_train = data_enc.loc[train_mask, feature_cols]
            y_train = data_enc.loc[train_mask, target_col]
            X_test = data_enc.loc[test_mask, feature_cols]
            y_test = data_enc.loc[test_mask, target_col]

            reg = RandomForestRegressor(n_estimators=200, max_depth=4,
                                         min_samples_leaf=1, random_state=42)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            preds.extend(y_pred)
            actuals.extend(y_test.tolist())
            held.extend([held_out] * len(y_test))

        preds = np.array(preds)
        actuals = np.array(actuals)
        if len(preds) < 2:
            print(f"[{label}] not enough held-out points to score (n={len(preds)})")
            return {"feature_set": label, "n": len(preds), "r2": float("nan"),
                    "mae": float("nan"), "spearman": float("nan")}
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = sstats.spearmanr(actuals, preds)
        print(f"[{label}] n={len(preds)}  R2={r2:.3f}  MAE={mae:.4f}  Spearman={rho:.3f}")
        return {"feature_set": label, "n": len(preds), "r2": r2, "mae": mae, "spearman": rho}

    print("=" * 70)
    print(f"LODO results: predicting {'slog(gen_gap_loss)' if log_transform else 'gen_gap_loss (val_loss - train_loss)'}")
    print("=" * 70)
    lodo_rows = [
        run_lodo(meta_cols + opt_cols, "meta-only baseline (no landscape features)"),
        run_lodo(meta_cols + opt_cols + landscape_cols, "meta + local-landscape features"),
    ]

    # -- Feature importance from a model trained on ALL data ------------------
    full_cols = meta_cols + opt_cols + landscape_cols
    reg_full = RandomForestRegressor(n_estimators=300, max_depth=4,
                                      min_samples_leaf=1, random_state=42)
    reg_full.fit(data_enc[full_cols], data_enc[target_col])
    importances = pd.Series(reg_full.feature_importances_, index=full_cols).sort_values(ascending=False)

    print("\nTop feature importances (model trained on all data, reporting only):")
    print(importances.head(15).to_string())

    out_csv = os.path.join(out_dir, "merged_dataset.csv")
    data_enc.to_csv(out_csv, index=False)
    print(f"\nWrote merged feature table -> {out_csv}")

    lodo_csv = os.path.join(out_dir, "lodo_results.csv")
    pd.DataFrame(lodo_rows)[["feature_set", "n", "r2", "mae", "spearman"]].to_csv(lodo_csv, index=False)
    print(f"Wrote LODO summary -> {lodo_csv}")

    importance_csv = os.path.join(out_dir, "feature_importance.csv")
    importances.rename("importance").rename_axis("feature").reset_index().to_csv(importance_csv, index=False)
    print(f"Wrote feature importances -> {importance_csv}")

    print(f"\nDone. log_transform={log_transform}. Compare lodo_results.csv against the "
          f"non-log-transformed run to test the scale hypothesis directly.")


if __name__ == "__main__":
    main()
