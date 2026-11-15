#!/usr/bin/env python3
"""
run_pilot_batch_mode9.py
------------------------
Batch driver for the "flat minima <-> generalization gap" pilot study, for
the optimsolution Mode 7 (--nnlandscape) CLI -- NOT the standalone NeuralStudio
ns_ela_cli from the earlier iteration of this pipeline. Calls the optimsolution
executable once per (dataset, architecture, optimizer, seed) combination, all
sharing ONE --radii list and ONE summary CSV (mode 7 itself enforces that the
radii stay consistent across every run appended to the same summary file).

Usage:
    python3 run_pilot_batch_mode9.py \
        --cli /path/to/optimsolution \
        --dataset-dir /path/to/optimsolution/datasets \
        --out-dir /path/to/output \
        [--datasets iris,bupa2,...] [--archs 8,16,8-8] [--optimizers de,pso,...]
        [--seeds 42,7,123] [--radii 0.02,0.05,0.1] [--iterations 200]
        [--population 40] [--local-samples 0]
        [--dry-run] [--resume]

Notes:
  - --cli points at the optimsolution EXECUTABLE itself (the one built by
    `cmake --build build --target optimsolution`), not a separate ns_ela_cli
    binary -- mode 7 is invoked as `optimsolution --nnlandscape <flags>`,
    and this script prepends "--nnlandscape" to every call automatically.
  - No --valsplit: mode 7 uses separate <name>.train/<name>.test files
    directly (auto-derived from --dataset by replacing the .train suffix),
    not an internal train/val re-split by ratio.
  - lbfgs/bfgs/gd/nm do NOT read --iterations/--population at all (their
    budget comes from optimsolution.cfg's own [lbfgs]/[bfgs]/[gd]/[nm]
    sections) -- this is a mode-9/optimsolution property, not something this
    driver can override; see INTEGRATION.md.

Notes:
  - Architectures are given comma-separated with '-' joining layer sizes for
    multi-layer nets, e.g. "8,16,8-8" = three archs: [8], [16], [8,8]. This
    is just this driver's own encoding -- ns_ela_cli's --hidden flag itself
    takes comma-separated layer sizes (e.g. "8,8"), so this script converts
    "8-8" -> "8,8" when building the ns_ela_cli command line.
  - --resume skips a (dataset,arch,optimizer,seed) combination if its
    run_key already appears in the summary CSV -- lets a long grid be
    restarted after an interruption without re-doing finished runs.
  - Every subprocess failure is logged and the batch continues (one bad
    combination should not abort the whole grid); a manifest CSV records
    the outcome (ok/error/skipped) of every planned run.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Defaults: the pilot grid discussed in the design phase ──────────────────
DEFAULT_DATASETS = [
    "iris", "circular", "bupa2", "breast", "tic-tac-toe",
    "balance-scale", "glass", "wine_pca", "cancer", "card",
]
DEFAULT_ARCHS = ["8", "16", "8-8"]           # '-' = multi-layer join, see docstring
DEFAULT_OPTIMIZERS = ["lbfgs", "cg", "sa", "de", "pso", "cmaes", "jso"]
DEFAULT_SEEDS = [42, 7, 123]
DEFAULT_RADII = "0.02,0.05,0.1"


def parse_csv_list(s):
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cli", required=True, help="Path to the optimsolution binary (mode 7 = --nnlandscape)")
    p.add_argument("--dataset-dir", required=True,
                   help="Directory containing <name>.train (+ companion .test) files")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for *_features.csv files + summary.csv + manifest")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    p.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--radii", default=DEFAULT_RADII)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--population", type=int, default=40)
    p.add_argument("--wbound", type=float, default=5.0)
    p.add_argument("--local-samples", type=int, default=0, help="0 = ns_ela_cli auto (30*dim)")
    p.add_argument("--activation", default="relu")
    p.add_argument("--label", default="pilot")
    p.add_argument("--dry-run", action="store_true", help="Print planned commands, run nothing")
    p.add_argument("--resume", action="store_true",
                   help="Skip a combination if its run_key is already in summary.csv")
    p.add_argument("--max-runs", type=int, default=0,
                   help="Stop after N successful runs (0 = no limit); useful for smoke tests")
    p.add_argument("--slow-optimizers", default="cmaes",
                   help="Comma-separated optimizer names that get --slow-iterations instead "
                        "of --iterations. NOTE: the CMAES-scales-badly-with-dimension finding "
                        "from the earlier NeuralStudio-based iteration of this pipeline was "
                        "observed against NeuralStudio's OWN fork of CMAES (src/core/optimizers/"
                        "cmaes.cpp), not necessarily the same code as optimsolution's own "
                        "src/methods/cmaes.cpp that mode 7 actually uses -- this flag is kept "
                        "as a cheap safety net, but re-check timing on a high-dim dataset "
                        "(e.g. card with hidden=16 or 8,8) before assuming the same limit applies.")
    p.add_argument("--slow-iterations", type=int, default=20,
                   help="Iteration budget applied to --slow-optimizers instead of --iterations")
    p.add_argument("--run-timeout", type=int, default=90,
                   help="Kill and mark 'error' any single ns_ela_cli invocation that runs "
                        "longer than this many seconds. CMAES was measured to take ~270s "
                        "even with a 20-iteration cap at dim~460 (card, arch=[8]); at the "
                        "larger archs (dim~850) it is expected to be far worse. A hard "
                        "per-run timeout trades a documented gap in the pilot data (a few "
                        "missing high-dim CMAES rows) for never blocking the whole batch on "
                        "one runaway optimizer call.")
    return p


def existing_run_keys(summary_csv):
    keys = set()
    if not os.path.exists(summary_csv):
        return keys
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "run_key" in row:
                keys.add(row["run_key"])
    return keys


def make_run_key(dataset, arch_tag, optimizer, seed, label):
    key = f"{dataset}__{arch_tag}__{optimizer}__seed{seed}"
    if label:
        key += f"__{label}"
    return key


def main():
    args = build_arg_parser().parse_args()

    datasets = parse_csv_list(args.datasets)
    archs = parse_csv_list(args.archs)
    optimizers = parse_csv_list(args.optimizers)
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    slow_optimizers = set(parse_csv_list(args.slow_optimizers))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = str(out_dir / "summary.csv")
    manifest_csv = str(out_dir / "manifest.csv")

    done_keys = existing_run_keys(summary_csv) if args.resume else set()
    if args.resume and done_keys:
        print(f"[resume] {len(done_keys)} run_key(s) already present in {summary_csv}, will be skipped.")

    total_planned = len(datasets) * len(archs) * len(optimizers) * len(seeds)
    print(f"[plan] {len(datasets)} datasets x {len(archs)} archs x "
          f"{len(optimizers)} optimizers x {len(seeds)} seeds = {total_planned} runs")

    manifest_is_new = not os.path.exists(manifest_csv)
    manifest_f = open(manifest_csv, "a", newline="")
    manifest_w = csv.writer(manifest_f)
    if manifest_is_new:
        manifest_w.writerow(["run_key", "dataset", "arch", "optimizer", "seed",
                              "status", "seconds", "message"])

    n_ok = n_err = n_skip = 0
    t_start = time.time()

    for dataset in datasets:
        train_path = Path(args.dataset_dir) / f"{dataset}.train"
        if not train_path.exists():
            print(f"[warn] dataset file not found, skipping: {train_path}")
            continue

        for arch in archs:
            hidden_arg = arch.replace("-", ",")          # "8-8" -> "8,8" for --hidden
            # arch_tag as ns_ela_cli itself computes it (args.hidden + "_" + activation,
            # comma preserved) -- this MUST match exactly, since it's embedded verbatim
            # in the run_key that ends up in summary.csv, and --resume's skip logic
            # compares against that. A filesystem-safe variant (comma -> dash) is used
            # separately, only for this driver's own --out-prefix path.
            arch_tag_key  = f"{hidden_arg}_{args.activation}"
            arch_tag_path = f"{hidden_arg.replace(',', '-')}_{args.activation}"

            for optimizer in optimizers:
                for seed in seeds:
                    run_key = make_run_key(dataset, arch_tag_key, optimizer, seed, args.label)

                    if run_key in done_keys:
                        n_skip += 1
                        manifest_w.writerow([run_key, dataset, arch, optimizer, seed,
                                              "skipped", 0, "already in summary (resume)"])
                        continue

                    if args.max_runs and n_ok >= args.max_runs:
                        print(f"[max-runs] reached {args.max_runs} successful runs, stopping.")
                        manifest_f.close()
                        return

                    prefix = str(out_dir / f"{dataset}_{arch_tag_path}_{optimizer}_seed{seed}")
                    iters = args.slow_iterations if optimizer in slow_optimizers else args.iterations
                    cmd = [
                        args.cli, "--nnlandscape",
                        "--dataset", str(train_path),
                        "--hidden", hidden_arg,
                        "--activation", args.activation,
                        "--optimizer", optimizer,
                        "--iterations", str(iters),
                        "--population", str(args.population),
                        "--wbound", str(args.wbound),
                        "--seed", str(seed),
                        "--radii", args.radii,
                        "--local-samples", str(args.local_samples),
                        "--out-prefix", prefix,
                        "--out-summary", summary_csv,
                        "--label", args.label,
                    ]

                    if args.dry_run:
                        print(" ".join(cmd))
                        continue

                    t0 = time.time()
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.run_timeout)
                        dt = time.time() - t0
                        if result.returncode == 0:
                            n_ok += 1
                            manifest_w.writerow([run_key, dataset, arch, optimizer, seed,
                                                  "ok", f"{dt:.1f}", ""])
                            print(f"[ok]  {run_key}  ({dt:.1f}s)")
                        else:
                            n_err += 1
                            err_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
                            manifest_w.writerow([run_key, dataset, arch, optimizer, seed,
                                                  "error", f"{dt:.1f}", err_line])
                            print(f"[ERR] {run_key}  ({dt:.1f}s)  {err_line}")
                    except subprocess.TimeoutExpired:
                        n_err += 1
                        manifest_w.writerow([run_key, dataset, arch, optimizer, seed,
                                              "error", "timeout", "subprocess timed out"])
                        print(f"[ERR] {run_key}  TIMEOUT")

                    manifest_f.flush()

    manifest_f.close()
    dt_total = time.time() - t_start
    print(f"\n[done] ok={n_ok} error={n_err} skipped={n_skip} "
          f"in {dt_total:.1f}s -> {summary_csv}")


if __name__ == "__main__":
    main()
