#include "uscore.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace optimsolution {
namespace uscore {

static inline bool isFiniteNum(double v) {
    return std::isfinite(v);
}

TrialOutcome resolveTrial(const std::vector<EvalPoint>& trace, double target, double tol) {
    TrialOutcome out;
    if (trace.empty()) {
        out.success   = false;
        out.finalBestF = std::numeric_limits<double>::infinity();
        return out;
    }

    for (const auto& p : trace) {
        if (isFiniteNum(p.bestF) && p.bestF <= target + tol) {
            out.success  = true;
            out.hitEvals = p.evals;
            break;
        }
    }

    // finalBestF is always the last sampled point of the trace, regardless
    // of whether the run succeeded (used as the failure tie-break, and
    // harmless -- unused -- when the run succeeded).
    out.finalBestF = trace.back().bestF;
    return out;
}

void computeSharedTarget(const std::vector<double>& allFinalBestF,
                          double& targetOut, double& tolOut) {
    double best = std::numeric_limits<double>::infinity();
    for (double v : allFinalBestF) {
        if (isFiniteNum(v) && v < best) best = v;
    }
    if (!isFiniteNum(best)) best = 0.0; // degenerate fallback: no finite values at all
    targetOut = best;
    tolOut    = std::max(1e-12, 1e-9 * std::fabs(best));
}

std::vector<ProblemScore> computeUScoreForProblem(const std::vector<MethodTrials>& methods) {
    std::vector<ProblemScore> out;
    out.reserve(methods.size());
    for (const auto& m : methods) {
        ProblemScore s;
        s.methodName = m.methodName;
        s.nTrials    = (int)m.trials.size();
        out.push_back(s);
    }
    if (methods.empty()) return out;

    // Flatten every trial of every method into one pool, remembering which
    // method each trial belongs to.
    struct FlatTrial {
        int    methodIdx;
        bool   success;
        double hitEvals;
        double finalBestF;
    };
    std::vector<FlatTrial> flat;
    for (int mi = 0; mi < (int)methods.size(); ++mi) {
        for (const auto& t : methods[mi].trials) {
            flat.push_back(FlatTrial{mi, t.success, t.hitEvals, t.finalBestF});
        }
    }
    const int total = (int)flat.size();
    if (total == 0) return out;

    // Order trial indices from WORST to BEST:
    //   * a failed trial is always worse than a successful one;
    //   * among successes, a larger hitEvals (slower) is worse;
    //   * among failures, a larger finalBestF (less accurate) is worse.
    // std::stable_sort keeps input order for anything the rule below
    // treats as equal, which makes results reproducible.
    std::vector<int> order(total);
    for (int i = 0; i < total; ++i) order[i] = i;

    auto isWorse = [&](int a, int b) {
        const FlatTrial& A = flat[a];
        const FlatTrial& B = flat[b];
        if (A.success != B.success) {
            return !A.success; // A is worse iff A failed and B did not
        }
        if (A.success) {
            if (A.hitEvals != B.hitEvals) return A.hitEvals > B.hitEvals;
        } else {
            if (A.finalBestF != B.finalBestF) return A.finalBestF > B.finalBestF;
        }
        return false; // treated as equal -> keep original relative order
    };
    std::stable_sort(order.begin(), order.end(), isWorse);

    // Rank 1 = worst .. rank `total` = best, EXCEPT that trials the paper
    // considers "identical" (same success/failure status and the same
    // hitEvals, or the same finalBestF) must share the AVERAGE of the
    // ranks their tied group spans -- see Section 3.2 ("Ties") of Price,
    // Kumar & Suganthan (2023): "Resolve ties by assigning average ranks
    // to identical trials." This is the standard midrank convention used
    // by the Mann-Whitney U / Wilcoxon rank-sum test, and is required for
    // U-scores to match the paper's worked examples exactly -- e.g. the
    // m=2,n=2 "p q q p" tie example in Section 3.2, where the two tied
    // "q" trials each get rank 2.5 (not ranks 2 and 3), giving U_P = U_Q
    // = 2 as stated in the paper (see tests/test_uscore.cpp).
    auto tied = [&](int a, int b) { return !isWorse(a, b) && !isWorse(b, a); };
    std::vector<double> rankOfFlatIdx(total, 0.0);
    int i = 0;
    while (i < total) {
        int j = i;
        while (j + 1 < total && tied(order[j], order[j + 1])) ++j;
        // Positions i..j (0-indexed) occupy ranks (i+1)..(j+1); every
        // trial in this tied group gets their average.
        const double avgRank = double(i + 1 + j + 1) / 2.0;
        for (int pos = i; pos <= j; ++pos) rankOfFlatIdx[order[pos]] = avgRank;
        i = j + 1;
    }

    // Per-method rank-sum ("SR") and success count.
    int flatIdx = 0;
    for (int mi = 0; mi < (int)methods.size(); ++mi) {
        for (const auto& t : methods[mi].trials) {
            if (t.success) out[mi].nSuccess += 1;
            out[mi].rankSum += rankOfFlatIdx[flatIdx];
            ++flatIdx;
        }
    }

    // U-score = SR - n(n+1)/2, using each method's OWN trial count for n
    // (this is exactly the Mann-Whitney U of that method's trials against
    // the pooled remainder).
    for (auto& s : out) {
        const double n = double(s.nTrials);
        s.uScore = s.rankSum - n * (n + 1.0) / 2.0;
    }

    // Rank the methods on this one problem: 1 = best (highest U-score).
    std::vector<int> methodOrder(out.size());
    for (size_t i = 0; i < out.size(); ++i) methodOrder[i] = (int)i;
    std::stable_sort(methodOrder.begin(), methodOrder.end(), [&](int a, int b) {
        return out[a].uScore > out[b].uScore;
    });
    for (size_t pos = 0; pos < methodOrder.size(); ++pos) {
        out[methodOrder[pos]].rankOnProblem = (int)pos + 1;
    }

    return out;
}

std::vector<OverallScore> aggregateUScore(const std::vector<std::vector<ProblemScore>>& perProblemScores) {
    std::vector<OverallScore> acc;
    std::unordered_map<std::string, size_t> indexOf;

    for (const auto& problemRow : perProblemScores) {
        for (const auto& s : problemRow) {
            auto it = indexOf.find(s.methodName);
            size_t idx;
            if (it == indexOf.end()) {
                OverallScore o;
                o.methodName = s.methodName;
                acc.push_back(o);
                idx = acc.size() - 1;
                indexOf[s.methodName] = idx;
            } else {
                idx = it->second;
            }
            acc[idx].nProblems += 1;
            acc[idx].sumUScore += s.uScore;
            acc[idx].sumRank   += double(s.rankOnProblem);
        }
    }

    // Overall ranking: ascending sumRank (lower average/summed rank is
    // better), ties broken by descending sumUScore, then by the order the
    // algorithm was first encountered (stable_sort keeps that for any
    // remaining tie).
    std::vector<int> order(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (acc[a].sumRank != acc[b].sumRank) return acc[a].sumRank < acc[b].sumRank;
        if (acc[a].sumUScore != acc[b].sumUScore) return acc[a].sumUScore > acc[b].sumUScore;
        return false;
    });
    for (size_t pos = 0; pos < order.size(); ++pos) {
        acc[order[pos]].overallRank = (int)pos + 1;
    }

    return acc;
}

} // namespace uscore
} // namespace optimsolution
