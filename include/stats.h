#pragma once
// ---------------------------------------------------------------------
// Shared statistical tests for method/variant comparison.
//
// These implementations were originally written for and validated as part
// of Mode 6 ("--rank", see src/main.cpp): chi-square p-value via the
// regularized incomplete gamma function, the Friedman omnibus test with
// the standard tie correction, the Wilcoxon signed-rank test using the
// normal approximation with continuity correction and tie handling, the
// Holm-Bonferroni step-down adjustment, and both pandas-compatible
// ranking conventions (average / competition). Every one of them was
// checked against scipy.stats / pandas on synthetic reference data --
// chi-square vs scipy.stats.chi2.sf, Friedman vs
// scipy.stats.friedmanchisquare (with AND without ties), Wilcoxon vs
// scipy.stats.wilcoxon(mode="approx"), Holm vs a manual reference, and
// the ranking functions vs pandas .rank() -- and matched to
// floating-point precision.
//
// They were moved here verbatim (only `static` -> `inline`, plus a
// namespace) when Mode 8 ("--portfolio", src/mode_portfolio.cpp) needed
// the exact same protocol to compare switching policies against their
// baselines. Keeping ONE copy matters: two modes whose published numbers
// come from silently diverging copies of "the same" test is precisely
// the kind of bug that is invisible until someone tries to reproduce a
// table.
// ---------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace optimsolution {
namespace stats {

// ---- Regularized incomplete gamma (Numerical-Recipes-style) -> chi-square
// upper-tail p-value: p = Q(df/2, stat/2). ----
inline double rankGammln(double xx) {
    static const double cof[6] = {
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
    };
    double x = xx, y = xx;
    double tmp = x + 5.5;
    tmp -= (x + 0.5) * std::log(tmp);
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; j++) { y += 1.0; ser += cof[j] / y; }
    return -tmp + std::log(2.5066282746310005 * ser / x);
}
inline void rankGser(double& gamser, double a, double x, double& gln) {
    const int ITMAX = 200; const double EPS = 3.0e-12;
    gln = rankGammln(a);
    if (x <= 0.0) { gamser = 0.0; return; }
    double ap = a, sum = 1.0 / a, del = sum;
    for (int n = 1; n <= ITMAX; n++) {
        ap += 1.0; del *= x / ap; sum += del;
        if (std::fabs(del) < std::fabs(sum) * EPS) break;
    }
    gamser = sum * std::exp(-x + a * std::log(x) - gln);
}
inline void rankGcf(double& gammcf, double a, double x, double& gln) {
    const int ITMAX = 200; const double EPS = 3.0e-12; const double FPMIN = 1.0e-300;
    gln = rankGammln(a);
    double b = x + 1.0 - a, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
    for (int i = 1; i <= ITMAX; i++) {
        double an = -i * (i - a); b += 2.0; d = an * d + b;
        if (std::fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; double delta = d * c; h *= delta;
        if (std::fabs(delta - 1.0) < EPS) break;
    }
    gammcf = std::exp(-x + a * std::log(x) - gln) * h;
}
inline double rankGammq(double a, double x) {
    double gln;
    if (x < a + 1.0) { double gamser; rankGser(gamser, a, x, gln); return 1.0 - gamser; }
    double gammcf; rankGcf(gammcf, a, x, gln); return gammcf;
}
inline double chiSquareSf(double stat, int df) {
    if (stat <= 0.0 || df <= 0) return 1.0;
    return rankGammq((double)df / 2.0, stat / 2.0);
}
inline double normalSf2sided(double z) {
    return std::erfc(std::fabs(z) / std::sqrt(2.0));
}

// Average-rank (pandas rank(method='average')): ties share the mean of the
// positions they occupy. Used for the Friedman test's rank sums.
// NaN-safe ordering: a raw `a < b` comparator is undefined behavior for
// std::sort whenever NaN is present (NaN compares false to everything,
// including itself, breaking the strict-weak-ordering std::sort requires).
// This matters here specifically because a method can legitimately report
// a NaN/Inf mean_f (e.g. it diverged or overflowed on a given problem), and
// with the raw comparator that group's sort becomes undefined -- in
// practice, on real data, this has produced groups where NO row ends up
// with rank==1/is_best==1 at all, which then crashes the downstream
// Python meta-learner (IndexError on an empty "tied for best" list).
// Treating NaN as unconditionally "worse than any real number" (and tied
// with other NaNs) restores a valid ordering and guarantees at least one
// row per group gets the best rank -- a real winner if one exists, or all
// tied for best only in the degenerate case where every value in the
// group is NaN.
inline bool lessNanSafe(double a, double b) {
    const bool an = std::isnan(a), bn = std::isnan(b);
    if (an && bn) return false; // both NaN: equal, neither "less"
    if (an) return false;        // a is NaN, b is real: NaN never sorts before a real value
    if (bn) return true;         // a is real, b is NaN: a real value always sorts before NaN
    return a < b;
}
inline bool equalNanSafe(double a, double b) {
    const bool an = std::isnan(a), bn = std::isnan(b);
    if (an && bn) return true;
    if (an || bn) return false;
    return a == b;
}
inline std::vector<double> averageRank(const std::vector<double>& v) {
    const int n = (int)v.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return lessNanSafe(v[a], v[b]); });
    std::vector<double> rank(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && equalNanSafe(v[idx[j + 1]], v[idx[i]])) ++j;
        double avgRank = (i + j) / 2.0 + 1.0;
        for (int t = i; t <= j; ++t) rank[idx[t]] = avgRank;
        i = j + 1;
    }
    return rank;
}
// Competition rank (pandas rank(method='min')): tied values all get the
// LOWEST rank in their tie group. Used for the "is_best" / wins label.
inline std::vector<double> competitionRank(const std::vector<double>& v) {
    const int n = (int)v.size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return lessNanSafe(v[a], v[b]); });
    std::vector<double> rank(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && equalNanSafe(v[idx[j + 1]], v[idx[i]])) ++j;
        double minRank = (double)(i + 1);
        for (int t = i; t <= j; ++t) rank[idx[t]] = minRank;
        i = j + 1;
    }
    return rank;
}

struct FriedmanResult { bool valid; double chi2; int df; double p; int N; std::vector<double> rankSums; };
inline FriedmanResult friedmanTest(const std::vector<std::vector<double>>& ranks) {
    const int N = (int)ranks.size();
    const int k = N > 0 ? (int)ranks[0].size() : 0;
    if (N < 2 || k < 2) return { false, 0, 0, 1.0, N, {} };
    std::vector<double> Rj(k, 0.0);
    for (const auto& row : ranks)
        for (int j = 0; j < k; ++j) Rj[j] += row[j];
    double sumSq = 0.0;
    for (double r : Rj) sumSq += r * r;
    double chi2 = (12.0 / ((double)N * k * (k + 1))) * sumSq - 3.0 * N * (k + 1);

    // Tie correction: whenever two or more methods land on the EXACT same
    // metric value within a problem/dim block (very common in optimization
    // benchmarks -- multiple methods converging to the same machine-epsilon
    // value, e.g. both reaching exactly 0.0), the plain Friedman formula
    // above is a biased approximation. The standard correction divides by
    // C = 1 - sum(t^3 - t) / (N*k*(k^2-1)), where t ranges over the sizes
    // of each tied group within each block (ties are detected directly from
    // the already-computed average ranks: a tied group shares one rank
    // value, e.g. two methods tied for best both get rank 1.5). Verified to
    // reproduce scipy.stats.friedmanchisquare exactly on both tied and
    // untied synthetic data (see the isolated test suite).
    double tieTerm = 0.0;
    for (const auto& row : ranks) {
        std::vector<double> sortedRow = row;
        std::sort(sortedRow.begin(), sortedRow.end());
        int i = 0;
        while (i < k) {
            int j = i;
            while (j + 1 < k && sortedRow[j + 1] == sortedRow[i]) ++j;
            double t = (double)(j - i + 1);
            if (t > 1) tieTerm += (t * t * t - t);
            i = j + 1;
        }
    }
    const double C = 1.0 - tieTerm / ((double)N * k * ((double)k * k - 1));
    if (C > 0.0) chi2 /= C; // if C<=0 (pathological all-tied data), fall back to the uncorrected value

    int df = k - 1;
    double p = chiSquareSf(chi2, df);
    return { true, chi2, df, p, N, Rj };
}

struct WilcoxonResult { bool valid; double statistic; double z; double p; int n; };
inline WilcoxonResult wilcoxonSignedRank(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> d;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        if (diff != 0.0) d.push_back(diff);
    }
    const int n = (int)d.size();
    if (n < 1) return { false, 0.0, 0.0, 1.0, 0 };

    std::vector<double> absd(n);
    for (int i = 0; i < n; ++i) absd[i] = std::fabs(d[i]);
    std::vector<double> ranks = averageRank(absd);

    double Wpos = 0.0, Wneg = 0.0;
    for (int i = 0; i < n; ++i) { if (d[i] > 0) Wpos += ranks[i]; else Wneg += ranks[i]; }
    double T = std::min(Wpos, Wneg);

    std::vector<double> sortedAbs = absd;
    std::sort(sortedAbs.begin(), sortedAbs.end());
    double tieTerm = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && sortedAbs[j + 1] == sortedAbs[i]) ++j;
        double t = (double)(j - i + 1);
        if (t > 1) tieTerm += (t * t * t - t);
        i = j + 1;
    }

    double mean = n * (n + 1) / 4.0;
    double variance = n * (n + 1) * (2 * n + 1) / 24.0 - tieTerm / 48.0;
    if (variance <= 0.0) return { false, T, 0.0, 1.0, n };
    double sd = std::sqrt(variance);

    double numerator = Wpos - mean; // continuity-corrected, matches scipy default
    if (numerator > 0) numerator -= 0.5;
    else if (numerator < 0) numerator += 0.5;
    double z = numerator / sd;
    double p = normalSf2sided(z);
    return { true, T, z, p, n };
}

inline std::vector<double> holmAdjust(const std::vector<double>& pvals) {
    const int m = (int)pvals.size();
    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return pvals[a] < pvals[b]; });
    std::vector<double> adjSorted(m);
    double runningMax = 0.0;
    for (int rank = 0; rank < m; ++rank) {
        double factor = (double)(m - rank);
        double val = factor * pvals[order[rank]];
        runningMax = std::max(runningMax, val);
        adjSorted[rank] = std::min(1.0, runningMax);
    }
    std::vector<double> adj(m);
    for (int rank = 0; rank < m; ++rank) adj[order[rank]] = adjSorted[rank];
    return adj;
}

} // namespace stats
} // namespace optimsolution
