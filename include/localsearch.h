#pragma once
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <limits>
#include <cmath>
#include "problem.h"
#include "utils.h"

namespace optimsolution {

using Vec = std::vector<double>;

struct LineSearchResult {
    Vec    x_new;
    double f_new;
    Vec    g_new;
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

// project_to_bounds: declaration only.
// Implement in localsearch.cpp when Problem exposes bound accessors.
void project_to_bounds(Vec& x, const Problem* prob);

// FIX #2: was declared but never defined in .cpp — defined here as inline.
inline LineSearchResult backtrackingArmijo(
    Problem*    prob,
    const Vec&  x,
    const Vec&  g,
    const Vec&  d,
    double      alpha0,
    double      c1,
    int         max_backtracks)
{
    const int D   = (int)x.size();
    const double dg = dot(g, d);
    const double f0 = prob->evaluate(x);
    double alpha    = alpha0;

    for (int bt = 0; bt < max_backtracks; ++bt) {
        Vec xn(D);
        for (int j = 0; j < D; ++j) xn[j] = x[j] + alpha * d[j];
        double fn = prob->evaluate(xn);
        if (fn <= f0 + c1 * alpha * dg) {
            Vec gn = prob->gradient(xn);
            return { std::move(xn), fn, std::move(gn) };
        }
        alpha *= 0.5;
    }
    // Line search failed — return current point unchanged
    return { x, f0, g };
}

// Gradient Descent with Armijo backtracking
std::pair<Vec, double> localGD(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0);

// L-BFGS (quasi-Newton, two-loop recursion)
std::pair<Vec, double> localLBFGS(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0);

// Full BFGS (quasi-Newton, dense inverse-Hessian)
// FIX #3: localBFGS was implemented in .cpp but missing from the header.
std::pair<Vec, double> localBFGS(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0);

// Nelder–Mead simplex (derivative-free)
std::pair<Vec, double> localNM(
    Problem*       prob,
    std::mt19937_64& rng,
    const Vec&     x0);

// Dispatcher
std::pair<Vec, double> runLocalSearch(
    const std::string& name,
    Problem*           prob,
    std::mt19937_64&   rng,
    const Vec&         x0);

} // namespace optimsolution
