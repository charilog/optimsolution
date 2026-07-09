#include "cosinemixture.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Cosine Mixture (Ali, Khompatraporn & Zabinsky test-function collection)
//
// Standard definition (posed as MAXIMIZATION in the original literature):
//   g(x) = sum_i [ 0.1*cos(5*pi*x_i) ] - sum_i x_i^2
// with a global MAXIMUM of 0.1*D at x = 0.
//
// FIX: this file is a minimization problem (every other Problem in this
// codebase returns "smaller is better"), but evaluate_core previously
// returned g(x) UNNEGATED. Since g(x) -> -infinity as x moves away from the
// origin (the -x_i^2 term dominates), the AS-WRITTEN function's minimum
// over the bounded domain sits at the domain's corners, not at x = 0 as
// setKnownGlobalOptimum(0.1*dim, ...) claimed — easily demonstrated:
// f(0) = 0.5 but f(1,1,1,1,1) = -5.5 for D=5, i.e. the boundary is far
// "better" than the declared optimum under the old sign convention. The
// function must be NEGATED for the declared optimum at x=0 to actually be
// a minimum: f(x) = -g(x) = sum_i [x_i^2 - 0.1*cos(5*pi*x_i)], whose minimum
// is -0.1*D at x = 0 (not +0.1*D).
// -----------------------------------------------------------------------------

CosineMixture::CosineMixture()
{
    setName("cm");
    setFullName("Cosine Mixture function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void CosineMixture::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -1.0);
    Vec u(dim,  1.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    // FIX: the minimum of the negated function is -0.1*dim, not +0.1*dim
    // (see rationale above) — at x=0 every cos term is 1, giving
    // sum(0 - 0.1*1) = -0.1*dim.
    setKnownGlobalOptimum(-0.1 * dim, xopt);
}

double CosineMixture::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    // FIX: negated relative to the previous version — see class-level
    // comment. f(x) = sum_i [ x_i^2 - 0.1*cos(5*pi*x_i) ], minimized at x=0.
    for (int i = 0; i < D; ++i)
        sum += x[i] * x[i] - 0.1 * std::cos(5.0 * PI * x[i]);

    return sum;
}

void CosineMixture::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);

    // d/dx_i [ x_i^2 - 0.1*cos(5*pi*x_i) ] = 2*x_i + 0.5*pi*sin(5*pi*x_i)
    for (int i = 0; i < D; ++i)
        g[i] = 2.0 * x[i] + 0.1 * 5.0 * PI * std::sin(5.0 * PI * x[i]);
}

} // namespace optimsolution
