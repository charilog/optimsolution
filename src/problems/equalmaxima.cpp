#include "equalmaxima.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Equal Maxima (classic niching / multimodal-optimization benchmark, e.g.
// CEC niching competition F1): g(x) = sin^6(5*pi*x) on [0,1], which has FIVE
// equally-good global MAXIMA at x = 0.1, 0.3, 0.5, 0.7, 0.9 (g=1 there). The
// entire point of the benchmark is testing whether a search/niching method
// can locate ALL FIVE separated peaks, not just any one of them.
//
// FIX: this is a minimization framework, but the previous version returned
// g(x) directly, UNNEGATED. Minimizing sin^6(5*pi*x) as-is targets its
// zeros instead — x = 0, 0.2, 0.4, 0.6, 0.8, 1.0 — which are SIX trivial,
// evenly-spaced points where the function is identically flat-bottomed
// (sin=0), a completely different and much easier landscape that has
// nothing to do with the "five separated equal peaks" the function's name
// and literature usage refer to. The class even declared its optimum at
// x=0 with value 0, i.e. one of those trivial zeros, not one of the true
// peaks. The function must be NEGATED — f(x) = -sin^6(5*pi*x), minimized at
// the same five x-locations where g was maximal, value -1.
//
// Also FIX: evaluate_core only ever reads x[0], but init() accepted an
// arbitrary dim and left x[1..D-1] completely unconstrained/irrelevant —
// inconsistent with the class's own "1D benchmark test function" label.
// init() now forces dimension = 1, matching how other fixed-dimension
// problems in this codebase (e.g. BifunctionalCatalyst) are handled.
// -----------------------------------------------------------------------------

EqualMaxima::EqualMaxima()
{
    setName("equalmaxima");
    setFullName("Equal Maxima function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("1D benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void EqualMaxima::init(int /*dim*/)
{
    // Forced 1D: evaluate_core only ever uses x[0], per the function's own
    // definition and documented purpose.
    Problem::init(1);

    Vec l(1, 0.0);
    Vec u(1, 1.0);
    setBounds(l, u);

    // FIX: one of the five true peaks (x=0.1, value -1 after negation),
    // not x=0 (a trivial zero of the un-negated function).
    Vec xopt(1, 0.1);
    setKnownGlobalOptimum(-1.0, xopt);
}

double EqualMaxima::evaluate_core(const Vec& x)
{
    const double t = std::sin(5.0 * PI * x[0]);
    const double t6 = t * t * t * t * t * t; // sin^6
    // FIX: negated — see class-level comment. Minimizing -sin^6 puts the
    // global optima at the five true peaks x=0.1,0.3,0.5,0.7,0.9 (value
    // -1), instead of the six trivial zeros of the un-negated function.
    return -t6;
}

void EqualMaxima::gradient_core(const Vec& x, Vec& g)
{
    g.resize(1);

    const double t  = std::sin(5.0 * PI * x[0]);
    const double dt = std::cos(5.0 * PI * x[0]) * 5.0 * PI;

    // d/dx [ -sin^6 ] = -6*sin^5*cos*d(5*pi*x)/dx
    g[0] = -6.0 * std::pow(t, 5) * dt;
}

} // namespace optimsolution
