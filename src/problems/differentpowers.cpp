#include "differentpowers.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Different Powers function
//
// f(x) = sum_i |x_i|^{2 + 4 * (i-1)/(D-1)}
//
// Global minimum: 0 at x = 0
// Domain: [-5, 5]^D
//
// Properties:
//   - unimodal
//   - separable: each term depends only on its own x_i (the exponent p_i
//     is a fixed, precomputed per-dimension constant — it does not depend
//     on any other coordinate), so the function decomposes into D
//     independent 1-D problems. The dimension-DEPENDENT exponent affects
//     conditioning/difficulty, not separability; the previous version's
//     "non-separable (dimension-dependent exponents)" label was a
//     contradiction — a function this codebase treats consistently
//     elsewhere (e.g. differing per-unit coefficients in the ELD/DED family
//     don't make those separable terms non-separable either).
// -----------------------------------------------------------------------------

DifferentPowers::DifferentPowers()
{
    setName("differentpowers");
    setFullName("Different Powers function");
    setModality("unimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void DifferentPowers::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);

    exp_.resize(dim);
    if (dim == 1)
        exp_[0] = 2.0;
    else
        for (int i = 0; i < dim; ++i)
            exp_[i] = 2.0 + 4.0 * (double(i) / double(dim - 1));
}

double DifferentPowers::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i)
        sum += std::pow(std::abs(x[i]), exp_[i]);

    return sum;
}

void DifferentPowers::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i)
    {
        double absxi = std::abs(x[i]);
        if (absxi < 1e-16) {
            g[i] = 0.0;
            continue;
        }

        double exponent = exp_[i];
        double coeff = exponent * std::pow(absxi, exponent - 1.0);
        g[i] = (x[i] >= 0.0 ? coeff : -coeff);
    }
}

} // namespace optimsolution
