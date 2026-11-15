#!/usr/bin/env python3
"""
train_gap_regressor.py
------------------------
Merges the per-run local-landscape sample CSVs produced by ns_ela_cli
(<prefix>_r<alpha>_features.csv, one row per sampled point) with the
run-summary CSV (summary.csv, one row per training run: trainLoss/valLoss/
generalization gap/sharpness) into a single feature table, then runs a
Leave-One-Dataset-Out (LODO) regression to test whether local-landscape
flatness features predict the generalization gap -- the analogue of
train_meta_learner.py's LOPO protocol, adapted from classification
("best_method") to regression (generalization gap).

IMPORTANT / environment note:
  The project's existing extract_ela_features.py computes full pflacco ELA
  feature families (ela_meta, ela_distr, ela_level, nbc, dispersion,
  information_content), which requires pflacco -> rpy2 -> a working R
  installation. That could not be installed in this sandbox (rpy2's build
  step needs R on PATH), so this script computes a SMALL set of
  landscape-summary statistics directly from (x1..xD, f) with plain
  numpy/pandas instead: mean/std/skew/kurtosis/min/max/range/cv/iqr of f
  within each local sample, per radius. This is intentionally NOT a
  replacement for full ELA -- it is a placeholder that keeps the rest of
  the pipeline (join keys, LOPO loop, regressor, reporting) fully working
  and testable now. Swapping in real pflacco output later is a drop-in
  change: run extract_ela_features.py on the same *_features.csv files
  (unchanged, since ns_ela_cli's CSV layout was built to match its column
  contract -- verified directly against its parsing logic) and merge on
  the same (run_key, radius_frac) key used below.

Usage:
    python3 train_gap_regressor.py <features_dir> <summary_csv> <out_dir>
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def landscape_stats(f: pd.Series, prefix: str) -> dict:
    """Cheap distribution-shape stats over one local sample's loss values.
    Placeholder for full pflacco ELA feature families -- see module docstring."""
    f = f.to_numpy(dtype=float)
    f = f[np.isfinite(f)]
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


def build_landscape_feature_table(features_dir: str) -> pd.DataFrame:
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

        stats_dict = landscape_stats(df["f"], prefix)
        row = rows.setdefault(run_key, {"run_key": run_key})
        row.update(stats_dict)

    table = pd.DataFrame(list(rows.values()))
    return table


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    features_dir, summary_csv, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning {features_dir} for local-landscape sample files...")
    landscape_df = build_landscape_feature_table(features_dir)
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

    # ── Feature sets ─────────────────────────────────────────────────────────
    landscape_cols = [c for c in landscape_df.columns if c != "run_key"]
    # "meta-only" baseline: everything a landscape-blind model could already
    # see (problem size, training budget, which optimizer, the training loss
    # itself) -- mirrors train_meta_learner.py's SBS-baseline idea: does the
    # local-landscape flatness signal add anything beyond what's already
    # observable without sampling around the minimum?
    meta_cols = ["dim", "iterations", "population", "evals", "wbound",
                 "valsplit", "train_loss", "train_acc"]
    meta_cols = [c for c in meta_cols if c in data.columns]

    data_enc = pd.get_dummies(data, columns=["optimizer"], prefix="opt")
    opt_cols = [c for c in data_enc.columns if c.startswith("opt_")]

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
    print("LODO results: predicting generalization gap (val_loss - train_loss)")
    print("=" * 70)
    lodo_rows = [
        run_lodo(meta_cols + opt_cols, "meta-only baseline (no landscape features)"),
        run_lodo(meta_cols + opt_cols + landscape_cols, "meta + local-landscape features"),
    ]

    # ── Feature importance from a model trained on ALL data (reporting only,
    #    not used for the LODO numbers above) ─────────────────────────────────
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

    # ── Two extra, GUI-friendly summary tables -- these, not merged_dataset.csv,
    # are what the "LODO Results" / "Feature Importances" tabs in the GUI's
    # Mode 7 output actually read (kept separate from merged_dataset.csv so a
    # human or a spreadsheet doesn't have to dig the summary numbers back out
    # of a large per-run feature table). Column names are deliberately plain
    # (no rounding baked in) so they can be copied straight into an article's
    # results table or re-formatted from the exported XLSX in the GUI.
    lodo_csv = os.path.join(out_dir, "lodo_results.csv")
    pd.DataFrame(lodo_rows)[["feature_set", "n", "r2", "mae", "spearman"]].to_csv(lodo_csv, index=False)
    print(f"Wrote LODO summary -> {lodo_csv}")

    importance_csv = os.path.join(out_dir, "feature_importance.csv")
    importances.rename("importance").rename_axis("feature").reset_index().to_csv(importance_csv, index=False)
    print(f"Wrote feature importances -> {importance_csv}")




if __name__ == "__main__":
    main()
