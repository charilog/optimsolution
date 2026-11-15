#pragma once
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <limits>
#include <cmath>
#include <unordered_map>
#include "problem.h"
#include "utils.h"

namespace optimsolution {

using Vec = std::vector<double>;

struct LineSearchResult {
    Vec    x_new;
    double f_new;
    Vec    g_new;
    double alpha = 0.0;   // accepted step (0 when the line search failed)
};

// FIX #4: inline so the definition lives in the header and avoids ODR violations
//         when multiple TUs include this header.
inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// FIX #1: was declared but never defined in .cpp — defined here as inline.
inline double l2norm(const Vec& a) {
    return std::sqrt(dot(a, a));
}

// BUGFIX: project_to_bounds() used to be declared here but defined nowhere
// and never called, so gd/lbfgs/bfgs line-search trial points (x + alpha*d)
// could leave the search box and the local searches could RETURN a point
// outside the bounds (which then became the method's reported best).
inline void project_to_bounds(Vec& x, const Problem* prob) {
    if (!prob) return;
    const Vec& lb = prob->lb();
    const Vec& ub = prob->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (j < lb.size() && x[j] < lb[j]) x[j] = lb[j];
        if (j < ub.size() && x[j] > ub[j]) x[j] = ub[j];
        if (!std::isfinite(x[j]) && j < lb.size() && j < ub.size()) x[j] = 0.5 * (lb[j] + ub[j]);
    }
}

// Projected backtracking (Armijo) line search.
// BUGFIXES: (1) trial points are projected onto the box and the Armijo test
// uses the ACTUAL step s = P(x + alpha*d) - x, i.e. f(x+s) <= f0 + c1*g's;
// (2) f(x) is passed in (f0_known) instead of being re-evaluated on every
// call -- the old version spent one extra objective evaluation per
// iteration; (3) an optional evaluation cap (maxEvalsCap > 0) is honoured.
inline LineSearchResult backtrackingArmijo(
    Problem*    prob,
    const Vec&  x,
    const Vec&  g,
    const Vec&  d,
    double      alpha0,
    double      c1,
    int         max_backtracks,
    double      f0_known = std::numeric_limits<double>::quiet_NaN(),
    long long   maxEvalsCap = -1)
{
    const int D   = (int)x.size();
    const double f0 = std::isfinite(f0_known) ? f0_known : prob->evaluate(x);
    double alpha    = alpha0;

    for (int bt = 0; bt < max_backtracks; ++bt) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        Vec xn(D);
        for (int j = 0; j < D; ++j) xn[j] = x[j] + alpha * d[j];
        project_to_bounds(xn, prob);
        double gs = 0.0;
        bool moved = false;
        for (int j = 0; j < D; ++j) { const double sj = xn[j] - x[j]; gs += g[j] * sj; if (sj != 0.0) moved = true; }
        if (!moved) break;                     // projection removed the whole step
        if (gs >= 0.0) { alpha *= 0.5; continue; }   // not a descent step after projection
        double fn = prob->evaluate(xn);
        if (fn <= f0 + c1 * gs) {
            Vec gn = prob->gradient(xn);
            return { std::move(xn), fn, std::move(gn), alpha };
        }
        alpha *= 0.5;
    }
    // Line search failed — return current point unchanged
    return { x, f0, g, 0.0 };
}

// Gradient Descent with Armijo backtracking. `section` selects which
// optimsolution.cfg section supplies max_iters/tol/alpha0/c1/backtracks --
// defaults to "gd" (unchanged behavior for every existing caller); pass a
// different section name to read separate settings.
// `maxItersOverride`/`maxEvalsCap` (both -1 by default = "no override, use
// the cfg section's own values / no extra cap") let a caller that already
// has its own iteration/evaluation budget -- e.g. mode 7's GUI Iterations
// field -- drive this local search's stopping criteria directly instead of
// silently falling back to the cfg section's own defaults, so gd/lbfgs/
// bfgs/nm are directly comparable to every other optimizer under the same
// nominal budget. The single-run CLI's own end-of-run local refinement
// (main.cpp) does not pass these, so its behavior is completely unchanged.
std::pair<Vec, double> localGD(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "gd",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// L-BFGS (quasi-Newton, two-loop recursion)
std::pair<Vec, double> localLBFGS(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "lbfgs",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// Full BFGS (quasi-Newton, dense inverse-Hessian)
// FIX #3: localBFGS was implemented in .cpp but missing from the header.
std::pair<Vec, double> localBFGS(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "bfgs",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// Nelder–Mead simplex (derivative-free)
std::pair<Vec, double> localNM(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "nm",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// ---------------------------------------------------------------------------
// Gradient-based local searches added after gd/lbfgs/bfgs. Same signature and
// same conventions as the four above: `section` selects the optimsolution.cfg
// section, `maxItersOverride`/`maxEvalsCap` (-1 = off) behave exactly as
// described at localGD(), every trial point is projected onto the box, and
// the returned point is always inside the box.
//
// All three work with the box-PROJECTED gradient: a coordinate that sits on a
// bound (within 1e-10 * max(1, ub-lb)) while -g pushes it out of the box is
// treated as fixed (its gradient component is zeroed for the direction and
// for the stopping test ||P(g)|| <= tol), so the methods do not stall against
// an active bound. Out-of-range settings fall back to the default with one
// warning on stderr.
// ---------------------------------------------------------------------------

// Nonlinear Conjugate Gradient (default beta rule: PR+, Polak-Ribiere with
// max(0,.) truncation; also fr / pr / hs / dy / hz). Strong-Wolfe line search
// on the feasible part of the direction (Nocedal-Wright initial step), with a
// projected-Armijo fallback; automatic restarts (every n iterations, Powell's
// orthogonality test, active-set change, projected step, non-descent) and a
// stagnation stop (ftol / stall_iters).
std::pair<Vec, double> localCG(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "cg",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// Truncated Newton / line-search Newton-CG (Nocedal & Wright Alg. 7.1).
// The Hessian is never formed: each Hessian-vector product is the forward
// difference of two gradients, H v ~ (g(x + d u) - g(x)) * |v| / d,
// u = v/|v|. Inner CG stops on the forcing term
// |r| <= min(0.5, sqrt|g|)*|g|, on negative curvature, or after max_cg steps.
// Projected-Armijo line search from the full step alpha = 1; stagnation stop
// as in cg.
std::pair<Vec, double> localNCG(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "ncg",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// Adam (Kingma & Ba 2015), optional AMSGrad (Reddi et al. 2018) and learning
// rate schedules. Not a descent method (no line search), so it evaluates f
// once per iteration and returns the BEST point it visited.
std::pair<Vec, double> localAdam(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0,
    const std::string& section = "adam",
    int            maxItersOverride = -1,
    long long      maxEvalsCap = -1);

// Path of the settings file that the local searches read their [gd]/[lbfgs]/
// [bfgs]/[nm] sections from (default: "optimsolution.cfg" in the current
// directory). main() sets it to the file it actually located.
void setLocalSearchConfigPath(const std::string& path);

// Scoped, per-thread override of one local-search section's parameters.
// While an instance is alive, every local search running on the SAME thread
// that reads `section` sees the settings-file values overlaid with `kv`.
// Used by the local methods when they run as stand-alone methods
// (src/methods/local_methods.cpp), so that parameter values supplied
// through MethodConfig -- e.g. a sensitivity sweep over [adam] lr -- really
// apply. thread_local, hence safe in the OpenMP parallel-runs modes.
class ScopedLocalSearchOverride {
public:
    ScopedLocalSearchOverride(const std::string& section,
                              const std::unordered_map<std::string, std::string>& kv);
    ~ScopedLocalSearchOverride();
    ScopedLocalSearchOverride(const ScopedLocalSearchOverride&) = delete;
    ScopedLocalSearchOverride& operator=(const ScopedLocalSearchOverride&) = delete;
private:
    const void* prev_kv_;
    std::string prev_section_;
};

// Dispatcher. `maxItersOverride`/`maxEvalsCap` are forwarded verbatim to
// whichever local*() function `name` selects -- see localGD()'s own comment
// above for what they do and why they default to "no override".
std::pair<Vec, double> runLocalSearch(
    const std::string& name,
    Problem*           prob,
    std::mt19937_64&   rng,
    const Vec&         x0,
    int                maxItersOverride = -1,
    long long          maxEvalsCap = -1);

} // namespace optimsolution
