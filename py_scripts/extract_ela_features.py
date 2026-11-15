#!/usr/bin/env python3
"""
extract_ela_features.py
------------------------
Reads a folder of "*_features.csv" files produced by OptimSolution's
`--features` CLI mode (landscape sampling: x1..xD, f), computes a battery of
Exploratory Landscape Analysis (ELA) feature sets for each (problem, dim)
pair using pflacco, and writes one combined CSV with one row per
(problem, dim) -- ready to be joined against the performance-ranking labels
produced by `--rank` (batch_method_summary.csv / batch_ranked.csv).

Usage:
    python3 extract_ela_features.py <features_dir> <output_csv> [--jobs 1]

--jobs N (default 1, i.e. the original fully serial behavior, unchanged):
    compute ELA features for N files concurrently, in separate worker
    processes (concurrent.futures.ProcessPoolExecutor). Each file is read
    and processed completely independently of every other -- there is no
    shared mutable state between files -- so this is safe to raise up to
    roughly your CPU core count.

Example:
    python3 extract_ela_features.py features_raw/ ela_features.csv

Requires: pandas, numpy, pflacco (pip install pflacco; see README_python_ela.md
for the exact working dependency set, since pflacco's own pinned requirements
are outdated and will fail to build on modern Python otherwise).
"""
import argparse
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pflacco.classical_ela_features import (
    calculate_ela_distribution,
    calculate_ela_meta,
    calculate_ela_level,
    calculate_nbc,
    calculate_dispersion,
    calculate_information_content,
)


def compute_features_for_file(path: str) -> dict:
    """Load one *_features.csv (x1..xD, f + metadata) and compute all ELA
    feature sets for it. Returns a flat dict of {feature_name: value}, plus
    the problem/dim/modality/separability metadata carried through from the
    sampling step."""
    df = pd.read_csv(path)

    meta = {
        "problem": df["problem"].iloc[0],
        "dim": int(df["dim"].iloc[0]),
        "n_samples": int(df["n_samples"].iloc[0]),
        "modality": df["modality"].iloc[0],
        "separability": df["separability"].iloc[0],
    }

    x_cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    # Keep x-columns in x1..xD numeric order (not lexicographic, which would
    # put x10 before x2).
    x_cols = sorted(x_cols, key=lambda c: int(c[1:]))
    X = df[x_cols].reset_index(drop=True)
    y = df["f"].reset_index(drop=True)

    features = dict(meta)

    # Each feature family is computed independently and defensively: a
    # failure in one family (e.g. ela_level's LDA/QDA occasionally failing
    # on a degenerate quantile split) must not discard the other, unrelated
    # feature families for that same problem.
    feature_fns = {
        "ela_distr": lambda: calculate_ela_distribution(X, y),
        "ela_meta": lambda: calculate_ela_meta(X, y),
        "ela_level": lambda: calculate_ela_level(X, y),
        "nbc": lambda: calculate_nbc(X, y),
        "dispersion": lambda: calculate_dispersion(X, y),
        "ic": lambda: calculate_information_content(X, y, seed=100),
    }

    for name, fn in feature_fns.items():
        try:
            result = fn()
            features.update(result)
        except Exception as e:
            print(f"  [warn] {name} failed for {meta['problem']} d={meta['dim']}: {e}")

    return features


def _compute_one_file(path):
    """Wrapper for the parallel path (--jobs > 1): never raises, so one
    bad file can't take down the whole ProcessPoolExecutor.map() -- returns
    (basename, features_dict_or_None, error_str_or_None), exactly mirroring
    the try/except each file already gets in the serial loop below."""
    base = os.path.basename(path)
    try:
        return base, compute_features_for_file(path), None
    except Exception as e:
        return base, None, f"{e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features_dir")
    ap.add_argument("output_csv")
    ap.add_argument("--jobs", type=int, default=1,
                     help="Compute features for this many files concurrently as separate worker "
                          "processes (default 1 = original serial behavior).")
    args = ap.parse_args()

    features_dir = args.features_dir
    output_csv = args.output_csv

    paths = sorted(glob.glob(os.path.join(features_dir, "*_features.csv")))
    if not paths:
        print(f"No *_features.csv files found in {features_dir}")
        raise SystemExit(1)

    print(f"Found {len(paths)} landscape-sample files. Computing ELA features...")

    rows = []
    if args.jobs > 1:
        print(f"Using {args.jobs} parallel worker processes "
              f"(optimsolution.cfg [pipeline] extract_ela_jobs > 1)...")
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for base, feat, err in ex.map(_compute_one_file, paths):
                done += 1
                if err is not None:
                    print(f"[{done}/{len(paths)}] {base}  [ERROR] Skipping: {err}")
                else:
                    print(f"[{done}/{len(paths)}] {base}")
                    rows.append(feat)
    else:
        # Original serial path, unchanged.
        for i, path in enumerate(paths, 1):
            base = os.path.basename(path)
            print(f"[{i}/{len(paths)}] {base}")
            try:
                rows.append(compute_features_for_file(path))
            except Exception as e:
                print(f"  [ERROR] Skipping {base}: {e}")
                traceback.print_exc()

    if not rows:
        print("No features could be computed. Aborting.")
        raise SystemExit(1)

    out_df = pd.DataFrame(rows)
    # Drop any all-NaN / constant-only columns that a completely failed
    # feature family across every file would otherwise leave behind as dead
    # weight in the meta-learning step.
    out_df = out_df.dropna(axis=1, how="all")

    out_df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(out_df)} rows x {len(out_df.columns)} columns -> {output_csv}")
    print(f"Feature columns: {len(out_df.columns) - 5}")  # minus the 5 metadata columns


if __name__ == "__main__":
    main()
