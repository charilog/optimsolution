#pragma once
//
// U-score ("trial-based dominance") ranking, as described in:
//   K.V. Price, A. Kumar, P.N. Suganthan, "Trial-based dominance for
//   comparing both the speed and accuracy of stochastic optimizers with
//   standard non-parametric tests", Swarm and Evolutionary Computation,
//   78, 101287, 2023.
// and illustrated (Figures 1/2) in the IEEE CEC 2025 "numerical
// optimization considering accuracy and speed" competition slides.
//
// This header is intentionally Qt-free and dependency-free (pure
// std C++17) so the core ranking logic can be exercised/unit-tested in
// isolation from the GUI. The GUI wires this up in
// gui/MainWindow_BatchStats.cpp (see exportUScoreReport()).
//
// ---------------------------------------------------------------------
// Algorithm summary (see the slides' Figures 1/2 for the worked example):
//
//   For ONE problem, pool every independent run ("trial") of every
//   algorithm being compared. Rank all trials worst(=1) .. best(=N):
//     * any trial that reached the (shared) target value outranks any
//       trial that did not;
//     * among trials that reached the target, the one that needed fewer
//       function evaluations (faster) outranks the others;
//     * among trials that never reached the target, the one with the
//       lower final error outranks the others.
//   Identical trials (same outcome) share the AVERAGE of the ranks their
//   tied group spans, per Section 3.2 ("Ties") of the paper -- e.g. two
//   trials tied for ranks 5 and 6 both get rank 5.5.
//   For each algorithm, SR = sum of the ranks of its own trials, and
//     U-score = SR - n(n+1)/2   ("n" = number of trials of that algorithm),
//   which is exactly the Mann-Whitney U statistic of that algorithm's
//   trials against the pooled remainder. A higher U-score means the
//   algorithm dominated more of the other algorithms' trials.
//
//   Across several problems, summing each algorithm's U-score ("Sum")
//   and its per-problem rank ("RS") gives the competition-style overall
//   comparison; ascending RS (i.e. average rank) is used here as the
//   overall ranking criterion, since it is the only rule fully
//   determined by the material in the slides.
// ---------------------------------------------------------------------

#include <string>
#include <vector>
#include <limits>

namespace optimsolution {
namespace uscore {

// One sampled point of a run's convergence trace: the best_f value
// observed after `evals` function evaluations. A trace is simply the
// list of such points for one independent run, and must be supplied
// already sorted by ascending `evals` (this is already the case for the
// *_convergence.csv files this project writes).
struct EvalPoint {
    double evals = 0.0;
    double bestF = std::numeric_limits<double>::infinity();
};

// Outcome of a single trial (one independent run), already resolved
// against a shared target/tolerance for the problem it belongs to.
struct TrialOutcome {
    bool   success    = false;                                   // reached the target within tol before the budget ran out
    double hitEvals    = std::numeric_limits<double>::quiet_NaN(); // evals at which the target was first reached (only meaningful if success)
    double finalBestF  = std::numeric_limits<double>::infinity();  // best_f at the end of the run (tie-break for failed trials)
};

// Resolves a raw convergence trace into a TrialOutcome relative to a
// shared `target`/`tol` (see computeSharedTarget()). `trace` must be
// sorted by ascending evals; an empty trace yields a "failed" trial
// with finalBestF = +inf.
TrialOutcome resolveTrial(const std::vector<EvalPoint>& trace, double target, double tol);

// Given the final best_f achieved by every run of every algorithm being
// compared on ONE problem, derives a shared target (the best value found
// by ANY algorithm's ANY run on that problem) and an absolute tolerance
// around it. This mirrors the tolerance convention already used
// elsewhere in this project (see src/main.cpp's `tol` /
// MainWindow::loadBatchCellFromConvergence()'s `out.tol`):
//   tol = max(1e-12, 1e-9 * |target|)
// `allFinalBestF` must be non-empty; NaN/inf entries are ignored.
void computeSharedTarget(const std::vector<double>& allFinalBestF,
                          double& targetOut, double& tolOut);

// All trials (independent runs) of one algorithm on one problem.
struct MethodTrials {
    std::string methodName;
    std::vector<TrialOutcome> trials;
};

// Per-algorithm U-score result for ONE problem.
struct ProblemScore {
    std::string methodName;
    int    nTrials      = 0;
    int    nSuccess      = 0;
    double rankSum        = 0.0; // "SR" in the slides
    double uScore         = 0.0; // "U-score" = SR - n(n+1)/2
    int    rankOnProblem  = 0;   // 1 = best algorithm on this problem
};

// Computes the U-score / per-problem rank for every algorithm in
// `methods`, all compared on the SAME problem. Algorithms may have
// different numbers of trials (the correction factor n(n+1)/2 is applied
// per algorithm using its own trial count). Trials the paper considers
// "identical" (same success/failure status and the same hitEvals, or the
// same finalBestF) share the average of the ranks their tied group spans
// (Section 3.2, "Ties", of Price/Kumar/Suganthan 2023) rather than being
// given arbitrary sequential ranks. Returns one ProblemScore per input
// MethodTrials, in the same order they were supplied.
std::vector<ProblemScore> computeUScoreForProblem(const std::vector<MethodTrials>& methods);

// Per-algorithm result aggregated across every problem it was scored on.
struct OverallScore {
    std::string methodName;
    int    nProblems   = 0;
    double sumUScore    = 0.0; // "Sum" in the slides
    double sumRank      = 0.0; // "RS" in the slides
    int    overallRank  = 0;   // 1 = best (ascending sumRank)
};

// Aggregates a set of per-problem results (one std::vector<ProblemScore>
// per problem, as produced by computeUScoreForProblem()) into one
// OverallScore per algorithm name, and ranks algorithms by ascending
// sumRank (ties broken by descending sumUScore, then by first-seen
// order) -- the "Ranking result" seen in the slides.
std::vector<OverallScore> aggregateUScore(const std::vector<std::vector<ProblemScore>>& perProblemScores);

} // namespace uscore
} // namespace optimsolution
