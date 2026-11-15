#!/usr/bin/env python3
"""
Landscape-aware algorithm selection: train a Random Forest classifier on
Exploratory Landscape Analysis (ELA) features to predict, for a previously
unseen problem, which optimizer is likely to perform best.

Evaluation follows a Leave-One-Problem-Out (LOPO) protocol: for each
problem, a classifier is trained on every OTHER problem and tested on the
held-out one, so no instance is ever evaluated by a model that has seen it
during training. We report accuracy against two baselines:
  - Single Best Solver (SBS): always predict whichever method had the best
    average rank on the training folds (a fixed, landscape-blind choice).
  - Virtual Best Solver (VBS): the oracle upper bound -- always the actual
    best method for that instance.
The meaningful question is not "is accuracy high" (with only a handful of
problem families this will not be highly precise) but "does the
landscape-aware model beat the landscape-blind SBS baseline, and how much
of the SBS-to-VBS gap does it close".

Usage:
    python3 train_meta_learner.py <ela_features.csv> <batch_ranked.csv> <output_dir> [--ablation] [--compare-classifiers]

Without any flags this reproduces exactly the previous behaviour (writes
lopo_predictions.csv, feature_importance.csv, summary.txt to <output_dir>).

With --ablation, it ADDITIONALLY re-runs the same LOPO protocol once per
ELA feature family (see run_lopo() and the --ablation section of main()
below) -- addressing Reviewer 2, comment 4's request for feature-group
ablations.

With --compare-classifiers, it ADDITIONALLY re-runs the same fold-safe
LOPO protocol with three alternative classifiers -- an RBF-kernel SVM, a
Gradient Boosting classifier, and multinomial Logistic Regression, each on
standardized features (a StandardScaler fit on training data only, inside
each fold) -- alongside this study's own Random Forest, and writes a
comparison to <output_dir>/classifier_comparison/classifier_comparison.csv.
This addresses Reviewer 2, comment 4's request to compare against
alternative selection models. Every classifier is run through the exact
same run_lopo() routine as the main run, so the comparison differs only in
which classifier (and, for the scale-sensitive ones, whether features are
standardized) is used -- same folds, same tie-break policy, same feature
set.
"""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def compute_tied_sets(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """From the per-row ranked table (one row per method/problem/dim),
    derive one row per (problem, dim) with the set of tied best-performing
    methods (is_best==1). This set is a property of the batch/ranking stage
    alone (Mode 6, "--rank") and does not depend on any tie-break policy,
    so it is safe to compute once, globally, up front -- it is what the
    lenient LOPO accuracy in Table 8 is scored against, for both the model
    and the Single Best Solver baseline.
    """
    rows = []
    skipped_groups = []
    for (problem, dim), grp in ranked_df.groupby(["problem", "dim"]):
        best_rows = grp[grp["is_best"] == 1]
        tied_methods_alpha = sorted(best_rows["method"].tolist())
        if not tied_methods_alpha:
            skipped_groups.append(f"{problem} d{dim}")
            continue
        rows.append({
            "problem": problem,
            "dim": dim,
            "n_tied_best": len(tied_methods_alpha),
            # "|" cannot appear in a method name and is never treated as a
            # delimiter by any stage of the pipeline, so downstream CSV/XLSX
            # conversion never mis-splits a multi-way tie across columns.
            "tied_methods": "|".join(tied_methods_alpha),
        })
    if skipped_groups:
        print(f"[warn] {len(skipped_groups)} (problem,dim) group(s) had no is_best==1 row "
              f"and were skipped (check batch_ranked.csv for these): {skipped_groups}")
    return pd.DataFrame(rows)


def assign_canonical_label(tied_df: pd.DataFrame, avg_rank: pd.Series) -> pd.Series:
    """Given the tied-method sets and a preference order (avg_rank: lower =
    stronger = preferred), pick one canonical label per (problem, dim) row:
    the tied method with the lowest avg_rank. Any tied method absent from
    avg_rank (should not happen with 8 methods and >=34 training problems,
    but guarded defensively) is treated as worst-ranked, i.e. never
    preferred over a method we do have a training-fold rank for.

    LEAKAGE FIX (Reviewer 1, comment 4): 'avg_rank' must be computed from
    training data ONLY (excluding the held-out problem) when this is used
    to build the labels a classifier is fit on inside a Leave-One-Problem-
    Out fold -- see build_fold_labels() below. This function itself is
    fold-agnostic; it just applies whatever preference order it is given.
    """
    worst = (avg_rank.max() if len(avg_rank) else 0.0) + 1.0

    def pick(tied_str: str) -> str:
        tied = tied_str.split("|")
        return min(tied, key=lambda m: avg_rank.get(m, worst))

    return tied_df["tied_methods"].apply(pick)


def build_fold_labels(tied_df: pd.DataFrame, train_ranked: pd.DataFrame) -> pd.Series:
    """Canonical label per (problem, dim) row in `tied_df`, with the
    tie-break preference order computed strictly from `train_ranked`
    (already filtered to the current fold's training problems only). See
    assign_canonical_label()'s docstring for why this must exclude the
    held-out problem.
    """
    fold_avg_rank = train_ranked.groupby("method")["rank"].mean()
    return assign_canonical_label(tied_df, fold_avg_rank)


def feature_family(col: str) -> str:
    """The ELA family a feature column belongs to, e.g.
    'ela_level.mmce_lda_50' -> 'ela_level'. Matches the family names used
    throughout the paper and in feature_importance.csv."""
    return col.split(".")[0]


def default_rf_factory():
    """The Random Forest configuration used throughout this study (Table 8
    onward). Kept as the default classifier_factory for run_lopo() so the
    main run and the --ablation sweep are completely unaffected by the
    --compare-classifiers addition below."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=1,
        random_state=42, class_weight="balanced",
    )


def run_lopo(data: pd.DataFrame, feature_cols: list, ranked_df: pd.DataFrame,
             le: LabelEncoder, problems: np.ndarray,
             classifier_factory=None, use_scaling: bool = False) -> dict:
    """Run one full Leave-One-Problem-Out pass restricted to `feature_cols`,
    using the exact same fold-safe label/SBS construction as the main run
    (see build_fold_labels()). Returns a dict with the lenient accuracy,
    strict accuracy, and the full per-instance results_df, so the main run,
    every --ablation configuration, and every --compare-classifiers model
    in main() all share this single implementation -- only the classifier
    (and, for scale-sensitive classifiers, whether features are
    standardized) ever differs between calls.

    classifier_factory: a zero-argument callable returning a FRESH,
    unfitted scikit-learn classifier instance (a fresh instance per fold,
    so no state leaks between folds). Defaults to default_rf_factory(),
    i.e. this study's Random Forest, so existing callers (the main run,
    --ablation) are completely unaffected.

    use_scaling: if True, a StandardScaler is fit on each fold's TRAINING
    data only and applied to both train and test -- required for
    scale-sensitive classifiers (SVM, logistic regression) and harmless
    (but unnecessary) for tree-based ones. Fitting the scaler inside the
    fold loop, on training data only, avoids the same kind of leakage
    build_fold_labels() was written to avoid for the tie-break policy.
    """
    if classifier_factory is None:
        classifier_factory = default_rf_factory

    X_full = data[feature_cols].copy()
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full = X_full.fillna(X_full.median(numeric_only=True))

    unique_problems = sorted(set(problems))
    results = []
    sbs_correct = 0
    model_correct = 0
    strict_correct = 0
    n_total = 0

    for held_out in unique_problems:
        train_mask = problems != held_out
        test_mask = problems == held_out

        X_train, X_test = X_full[train_mask].values, X_full[test_mask].values
        if use_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        train_problems = set(problems[train_mask])
        train_ranked = ranked_df[ranked_df["problem"].isin(train_problems)]

        fold_canonical = build_fold_labels(data[["problem", "dim", "tied_methods"]], train_ranked)
        y_full_fold = fold_canonical.values
        y_encoded_fold = le.transform(y_full_fold)

        y_train, y_test = y_encoded_fold[train_mask], y_encoded_fold[test_mask]
        y_test_labels = y_full_fold[test_mask]

        sbs_label = train_ranked.groupby("method")["rank"].mean().idxmin()

        clf = classifier_factory()
        clf.fit(X_train, y_train)
        y_pred_encoded = clf.predict(X_test)
        y_pred_labels = le.inverse_transform(y_pred_encoded)

        for i, (_, row) in enumerate(data[test_mask].iterrows()):
            n_total += 1
            actual = y_test_labels[i]
            predicted = y_pred_labels[i]
            tied = set(row["tied_methods"].split("|"))

            model_hit = predicted in tied
            sbs_hit = sbs_label in tied
            model_correct += int(model_hit)
            sbs_correct += int(sbs_hit)
            strict_correct += int(predicted == actual)

            results.append({
                "problem": row["problem"], "dim": row["dim"],
                "actual_best": actual, "tied_best": row["tied_methods"],
                "model_predicted": predicted, "model_correct": model_hit,
                "sbs_predicted": sbs_label, "sbs_correct": sbs_hit,
            })

    return {
        "n_total": n_total,
        "n_features": len(feature_cols),
        "model_acc": model_correct / n_total,
        "strict_acc": strict_correct / n_total,
        "sbs_acc": sbs_correct / n_total,
        "results_df": pd.DataFrame(results),
    }


def main():
    do_ablation = "--ablation" in sys.argv
    do_compare = "--compare-classifiers" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(positional) != 3:
        print(__doc__)
        sys.exit(1)

    ela_path, ranked_path, out_dir = positional
    os.makedirs(out_dir, exist_ok=True)

    ela_df = pd.read_csv(ela_path)
    ranked_df = pd.read_csv(ranked_path)

    # The tied-method SET per (problem, dim) is a property of the batch/
    # ranking stage (Mode 6) alone -- not of any tie-break policy -- so it
    # is computed once, globally; see compute_tied_sets() docstring. The
    # SINGLE canonical label used to fit the classifier is a different
    # matter and is built per LOPO fold, inside run_lopo(), to avoid
    # leakage (Reviewer 1, comment 4).
    tied_df = compute_tied_sets(ranked_df)
    data = ela_df.merge(tied_df, on=["problem", "dim"], how="inner")
    if len(data) < len(ela_df):
        missing = set(zip(ela_df.problem, ela_df.dim)) - set(zip(data.problem, data.dim))
        print(f"[warn] {len(missing)} (problem,dim) row(s) in the ELA table had no "
              f"matching ranked-performance row and were dropped: {sorted(missing)}")

    print(f"Merged dataset: {len(data)} (problem, dim) instances, "
          f"{data['problem'].nunique()} distinct problems.")

    # Feature columns: everything numeric except identifiers/labels.
    non_feature_cols = {
        "problem", "dim", "n_samples", "modality", "separability",
        "n_tied_best", "tied_methods",
    }
    all_feature_cols = [c for c in data.columns if c not in non_feature_cols
                         and pd.api.types.is_numeric_dtype(data[c])]

    n_nan_before = int(data[all_feature_cols].replace([np.inf, -np.inf], np.nan).isna().sum().sum())
    if n_nan_before > 0:
        print(f"[info] {n_nan_before} missing/inf feature values will be imputed with column medians.")

    problems = data["problem"].values

    # Fit the label encoder on every method name that ever appears in the
    # raw ranked table (not just those that end up as a canonical label),
    # so the encoding is stable across folds even though which methods
    # become canonical labels can now legitimately differ fold to fold.
    le = LabelEncoder()
    le.fit(sorted(ranked_df["method"].unique()))

    unique_problems = sorted(set(problems))
    print(f"Leave-One-Problem-Out CV over {len(unique_problems)} problems: {unique_problems}\n")

    # ================= Main run: full feature set (unchanged output) =================
    main_result = run_lopo(data, all_feature_cols, ranked_df, le, problems)
    model_acc, sbs_acc = main_result["model_acc"], main_result["sbs_acc"]
    n_total = main_result["n_total"]
    results_df = main_result["results_df"]

    print("=" * 60)
    print(f"Leave-One-Problem-Out results ({n_total} instances)")
    print("=" * 60)
    print(f"Virtual Best Solver (VBS) accuracy   : 100.0%  (oracle upper bound)")
    print(f"Landscape-aware model accuracy       : {model_acc*100:.1f}%")
    print(f"Single Best Solver (SBS) accuracy    : {sbs_acc*100:.1f}%")
    gap_closed = "N/A" if sbs_acc >= 1.0 else f"{(model_acc - sbs_acc) / (1.0 - sbs_acc) * 100:.1f}%"
    print(f"Share of SBS->VBS gap closed by model : {gap_closed}")
    print()

    # Feature importance from a model trained on ALL data (for reporting
    # which landscape properties drive the selection decision -- NOT used
    # for the LOPO accuracy numbers above, which are strictly out-of-fold).
    global_avg_rank = ranked_df.groupby("method")["rank"].mean()
    global_canonical = assign_canonical_label(
        data[["problem", "dim", "tied_methods"]], global_avg_rank)
    data["best_method"] = global_canonical.values
    y_encoded_full = le.transform(global_canonical.values)

    X_full_all = data[all_feature_cols].replace([np.inf, -np.inf], np.nan)
    X_full_all = X_full_all.fillna(X_full_all.median(numeric_only=True))

    clf_full = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=1,
        random_state=42, class_weight="balanced",
    )
    clf_full.fit(X_full_all, y_encoded_full)
    importances = pd.Series(clf_full.feature_importances_, index=all_feature_cols)
    importances = importances.sort_values(ascending=False)

    print("Top 10 most important ELA features (trained on full data):")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:40s} {imp:.4f}")

    print("\nBest method distribution by hand-curated modality/separability tag:")
    tag_check = data.groupby(["modality", "separability"])["best_method"] \
                     .agg(lambda s: s.value_counts().index[0]).reset_index()
    tag_check.columns = ["modality", "separability", "most_common_best_method"]
    print(tag_check.to_string(index=False))

    # ---- Write main-run outputs (unchanged filenames/format) ----
    results_path = os.path.join(out_dir, "lopo_predictions.csv")
    results_df.to_csv(results_path, index=False)

    importance_path = os.path.join(out_dir, "feature_importance.csv")
    importances.reset_index().rename(
        columns={"index": "feature", 0: "importance"}
    ).to_csv(importance_path, index=False)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Landscape-aware algorithm selection: Leave-One-Problem-Out results\n")
        f.write("=" * 68 + "\n")
        f.write(f"Instances evaluated (problem x dim): {n_total}\n")
        f.write(f"Distinct problems: {len(unique_problems)} -- {unique_problems}\n")
        f.write(f"Distinct methods in candidate pool: {len(le.classes_)} -- {list(le.classes_)}\n\n")
        f.write(f"Virtual Best Solver (oracle) accuracy : 100.0%\n")
        f.write(f"Landscape-aware model accuracy         : {model_acc*100:.1f}%\n")
        f.write(f"Single Best Solver accuracy            : {sbs_acc*100:.1f}%\n")
        f.write(f"Share of SBS-VBS gap closed by model  : {gap_closed}\n\n")
        f.write("Top 10 ELA feature importances (full-data fit, reporting only):\n")
        for feat, imp in importances.head(10).items():
            f.write(f"  {feat:40s} {imp:.4f}\n")

    print(f"\nWrote: {results_path}")
    print(f"Wrote: {importance_path}")
    print(f"Wrote: {summary_path}")

    if do_ablation:
        _run_ablation(data, all_feature_cols, ranked_df, le, problems,
                       model_acc, main_result, out_dir)

    if do_compare:
        _run_classifier_comparison(data, all_feature_cols, ranked_df, le, problems,
                                    model_acc, main_result, out_dir)


def _run_ablation(data, all_feature_cols, ranked_df, le, problems,
                   model_acc, main_result, out_dir):

    # ================= Ablation runs (Reviewer 2, comment 4) =================
    # Every configuration below reuses run_lopo() with the SAME data, same
    # ranked_df, same LabelEncoder, same problems array, same LOPO folds,
    # and the same classifier hyperparameters as the main run above -- the
    # only thing that changes is which columns of feature_cols are passed
    # in. This keeps the comparison apples-to-apples.
    families = sorted(set(feature_family(c) for c in all_feature_cols))
    print("\n" + "=" * 60)
    print(f"Feature-group ablation over {len(families)} ELA families: {families}")
    print("=" * 60)

    ablation_dir = os.path.join(out_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    ablation_rows = [{
        "configuration": "full (all families)",
        "family": "",
        "mode": "full",
        "n_features": len(all_feature_cols),
        "lenient_accuracy": model_acc,
        "strict_accuracy": main_result["strict_acc"],
        "delta_vs_full_pp": 0.0,
    }]

    for fam in families:
        fam_cols = [c for c in all_feature_cols if feature_family(c) == fam]

        # --- exclude: everything EXCEPT this family (necessity check) ---
        exclude_cols = [c for c in all_feature_cols if feature_family(c) != fam]
        res_ex = run_lopo(data, exclude_cols, ranked_df, le, problems)
        ablation_rows.append({
            "configuration": f"minus {fam}",
            "family": fam,
            "mode": "exclude",
            "n_features": len(exclude_cols),
            "lenient_accuracy": res_ex["model_acc"],
            "strict_accuracy": res_ex["strict_acc"],
            "delta_vs_full_pp": (res_ex["model_acc"] - model_acc) * 100.0,
        })
        print(f"  minus {fam:12s} ({len(exclude_cols):2d} features): "
              f"lenient acc = {res_ex['model_acc']*100:5.1f}%  "
              f"(delta {(res_ex['model_acc']-model_acc)*100:+.1f} pp)")

        # --- only: ONLY this family (sufficiency check) ---
        res_only = run_lopo(data, fam_cols, ranked_df, le, problems)
        ablation_rows.append({
            "configuration": f"only {fam}",
            "family": fam,
            "mode": "only",
            "n_features": len(fam_cols),
            "lenient_accuracy": res_only["model_acc"],
            "strict_accuracy": res_only["strict_acc"],
            "delta_vs_full_pp": (res_only["model_acc"] - model_acc) * 100.0,
        })
        print(f"  only  {fam:12s} ({len(fam_cols):2d} features): "
              f"lenient acc = {res_only['model_acc']*100:5.1f}%  "
              f"(delta {(res_only['model_acc']-model_acc)*100:+.1f} pp)")

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(ablation_dir, "ablation_summary.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)

    ablation_txt_path = os.path.join(ablation_dir, "ablation_summary.txt")
    with open(ablation_txt_path, "w") as f:
        f.write("Feature-group ablation (Reviewer 2, comment 4)\n")
        f.write("=" * 68 + "\n")
        f.write(f"Full model (all {len(all_feature_cols)} features): "
                f"lenient accuracy = {model_acc*100:.1f}%, "
                f"strict accuracy = {main_result['strict_acc']*100:.1f}%\n\n")
        f.write(f"{'configuration':24s} {'n_feat':>7s} {'lenient%':>9s} {'strict%':>8s} {'delta_pp':>9s}\n")
        for _, r in ablation_df.iterrows():
            f.write(f"{r['configuration']:24s} {r['n_features']:7d} "
                     f"{r['lenient_accuracy']*100:9.1f} {r['strict_accuracy']*100:8.1f} "
                     f"{r['delta_vs_full_pp']:+9.1f}\n")

    print(f"\nWrote: {ablation_csv_path}")
    print(f"Wrote: {ablation_txt_path}")


def _run_classifier_comparison(data, all_feature_cols, ranked_df, le, problems,
                                model_acc, main_result, out_dir):
    # ============ Classifier comparison (Reviewer 2, comment 4) ============
    # "Comparisons with alternative selection models" -- every classifier
    # below is run through the exact same run_lopo() routine, on the same
    # 53 features, same folds, same tie-break policy as the main Random
    # Forest run, so the comparison isolates the choice of classifier from
    # every other pipeline decision. SVM and Logistic Regression are
    # scale-sensitive, so use_scaling=True fits a StandardScaler on each
    # fold's training data only (see run_lopo()'s docstring).
    print("\n" + "=" * 60)
    print("Classifier comparison over 4 models (Random Forest + 3 alternatives)")
    print("=" * 60)

    compare_dir = os.path.join(out_dir, "classifier_comparison")
    os.makedirs(compare_dir, exist_ok=True)

    classifiers = [
        ("Random Forest (this study)", default_rf_factory, False),
        ("SVM (RBF kernel)", lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42), True),
        ("Gradient Boosting", lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42), False),
        ("Logistic Regression", lambda: LogisticRegression(
            max_iter=2000, random_state=42), True),
    ]

    compare_rows = []
    for name, factory, needs_scaling in classifiers:
        if name == "Random Forest (this study)":
            # Reuse the main run's own result instead of re-fitting it --
            # identical folds, identical classifier config, so re-running
            # it would only waste time and risks looking like a second,
            # slightly different number if anything upstream ever changes.
            res = main_result
        else:
            res = run_lopo(data, all_feature_cols, ranked_df, le, problems,
                            classifier_factory=factory, use_scaling=needs_scaling)
        compare_rows.append({
            "classifier": name,
            "scaled_features": needs_scaling,
            "lenient_accuracy": res["model_acc"],
            "strict_accuracy": res["strict_acc"],
            "delta_vs_rf_pp": (res["model_acc"] - model_acc) * 100.0,
        })
        print(f"  {name:28s}: lenient acc = {res['model_acc']*100:5.1f}%  "
              f"(delta {(res['model_acc']-model_acc)*100:+.1f} pp vs. this study's Random Forest)")

    compare_df = pd.DataFrame(compare_rows)
    compare_csv_path = os.path.join(compare_dir, "classifier_comparison.csv")
    compare_df.to_csv(compare_csv_path, index=False)

    compare_txt_path = os.path.join(compare_dir, "classifier_comparison.txt")
    with open(compare_txt_path, "w") as f:
        f.write("Classifier comparison (Reviewer 2, comment 4)\n")
        f.write("=" * 68 + "\n")
        f.write(f"All 4 classifiers use the same {len(all_feature_cols)} ELA features, "
                f"the same Leave-One-Problem-Out folds, and the same tie-break policy; "
                f"only the classifier (and, where marked, feature standardization) differs.\n\n")
        f.write(f"{'classifier':28s} {'scaled':>7s} {'lenient%':>9s} {'strict%':>8s} {'delta_pp':>9s}\n")
        for _, r in compare_df.iterrows():
            f.write(f"{r['classifier']:28s} {str(bool(r['scaled_features'])):>7s} "
                     f"{r['lenient_accuracy']*100:9.1f} {r['strict_accuracy']*100:8.1f} "
                     f"{r['delta_vs_rf_pp']:+9.1f}\n")

    print(f"\nWrote: {compare_csv_path}")
    print(f"Wrote: {compare_txt_path}")


if __name__ == "__main__":
    main()
