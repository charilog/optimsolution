# Python ELA + Meta-Learning Pipeline

**These two scripts are called automatically by the OptimSolution GUI's
"Meta-dataset pipeline (batch + ELA + ML)" mode (buttons 3 and 4) — you do
not need to run them by hand from the command line unless you want to.**
Just make sure this whole `py_scripts/` folder sits at the project root
(next to `CMakeLists.txt`), and that Python + the dependencies below are
installed once (see section 1). The GUI finds this folder automatically and
auto-detects whichever of `py` / `python` / `python3` works on your machine.

They can still be run manually if you prefer (see section 3) — useful for
scripting, re-running just one stage, or debugging.

Two scripts that turn OptimSolution's output into a trained,
landscape-aware algorithm-selection model:

```
optimsolution --features ...   ──┐
                                  ├─► extract_ela_features.py ──► ela_features.csv ──┐
optimsolution --batch + --rank ──┘                                                   ├─► train_meta_learner.py ──► trained model + report
                                          batch_ranked.csv ───────────────────────────┘
```

## 1. Installing pflacco (important — do this exactly)

**`pip install pflacco` will fail out of the box** on any recent Python
(3.10+). pflacco 1.2.2 pins `numpy~=1.24.3`, which has no prebuilt wheel for
modern Python, so pip tries to build numpy 1.24 from source using
`numpy.distutils` — a module Python removed in 3.12. You'll see errors like
`ModuleNotFoundError: No module named 'distutils.msvccompiler'` or
`AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`.

The fix: install pflacco **without** its pinned dependencies, then install
modern, compatible versions of what it actually needs yourself. This works
correctly with pflacco's real code (it does not actually require numpy
1.24 — that pin is just outdated):

```bash
pip install pflacco --no-deps
pip install numpy pandas scikit-learn scipy
pip install pyDOE==0.3.8          # NOT the "pydoe" 1.4.x package — different
                                    # project, different (lowercase) module
                                    # name; pflacco imports "from pyDOE import lhs"
pip install numdifftools SALib
```

The `pyDOE==0.3.8` pin above is deliberate and required: PyPI now also
serves an unrelated, actively-maintained package under the name `pydoe`
(lowercase, version 1.x) that installs a lowercase `pydoe` module — pflacco's
code does `from pyDOE import lhs` (capital D-O-E), which only the old,
abandoned `pyDOE` 0.3.8 package provides.

This combination (numpy 2.x, pandas 3.x, scikit-learn 1.8, scipy 1.17,
pyDOE 0.3.8) was tested end-to-end against all ELA feature families used
here (`ela_distr`, `ela_meta`, `ela_level`, `nbc`, `dispersion`,
`information_content`) and works correctly.

## 2. Generating the inputs

```bash
# One landscape sample per (problem, dim) you care about:
optimsolution --features ackley 10 300
optimsolution --features sphere 10 300
# ... one call per (problem, dim) combination, or drive this from a script.
# Put/collect them all in one folder, e.g. features_raw/.

# Performance data + labels (see earlier README_batch_mode.md):
optimsolution --batch batch_list.txt out_dir
optimsolution --rank out_dir/batch_master.csv out_dir/batch
# -> out_dir/batch_ranked.csv is what train_meta_learner.py needs.
```

**Sample size note:** ELA literature convention is roughly 50×dim to
100×dim samples per instance. 300 samples is enough at low dimensions but
is under that guideline at d=30 (300 = 10×dim) — this is exactly why
`extract_ela_features.py` logs an `ela_level` warning at d=30 in the run
below (an LDA sub-feature needs more samples than dimensions to fit). It
degrades gracefully (that one sub-feature is skipped, everything else is
computed normally) rather than failing the whole extraction.

## 3. Running the pipeline

```bash
python3 extract_ela_features.py features_raw/ ela_features.csv
python3 train_meta_learner.py ela_features.csv out_dir/batch_ranked.csv meta_learning_results/
```

## 4. What we ran, and the real results

We generated all 40 (problem, dim) landscape samples matching your actual
400-row batch run (8 problems × 5 dims, 300 samples each), computed ELA
features for all of them, merged with the real `--rank` output, and ran
Leave-One-Problem-Out cross-validation (the standard protocol — every
dimension of a held-out problem is tested together, so the model never
sees that problem family during training):

```
Instances evaluated (problem x dim): 40
Distinct problems: 8
Distinct methods in candidate pool: 6

Virtual Best Solver (oracle) accuracy  : 100.0%
Landscape-aware model accuracy         : 47.5%
Single Best Solver accuracy            : 42.5%
Share of SBS->VBS gap closed by model  : 8.7%
```

**Honest read of this:** the landscape-aware model beats the naive
"always pick the same method" baseline (47.5% vs 42.5%), which is the
right direction and the whole point of the algorithm-selection framing —
but the margin is small, and with only 8 problem families and 40 total
instances this is not yet a statistically strong result. This is a
correctly-wired, working pipeline and a real first data point, not a
publishable result on its own. To turn this into an article-strength
result you would want, roughly in order of expected impact:

- **More problems** — 8 problem families is very few for training a
  classifier with 6 candidate methods; the CEC2017/CEC2022/GKLS families
  already registered in your framework would multiply this substantially.
- **More samples per landscape** — especially at higher dimensions
  (50-100×dim rather than the fixed 300 used here).
- **More runs per (method, problem, dim)** in `--batch` for less noisy
  performance labels (you already used 30, which is reasonable — this
  matters more as problem count grows).
- Consider collapsing near-identical methods (e.g. many swarm variants)
  into families before classification, since 6+ raw candidate labels with
  only 40 samples is a hard multi-class problem.

## 5. Files in this delivery

- `extract_ela_features.py`, `train_meta_learner.py` — the pipeline, reusable on any future `--features`/`--rank` output.
- `ela_features.csv` — the real 40-row ELA feature table computed from your data.
- `meta_learning_results/summary.txt`, `lopo_predictions.csv`, `feature_importance.csv` — the real results above, in full detail.
