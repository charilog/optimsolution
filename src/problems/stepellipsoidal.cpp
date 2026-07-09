#include "stepellipsoidal.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Step Ellipsoidal (BBOB variant, deterministic)
//
// Domain: [-5,5]^D
//
// Properties:
//   - unimodal
//   - non-separable
//
// Global optimum: 0 at x = 0
// -----------------------------------------------------------------------------

StepEllipsoidal::StepEllipsoidal()
{
    setName("stepellipsoidal");
    setFullName("Step-Ellipsoidal function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("BBOB synthetic benchmark");

    setKnownGlobalOptimum(0.0);
}

void StepEllipsoidal::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);

    // weights
    w_.resize(dim);
    if (dim == 1)
        w_[0] = 1.0;
    else
        for (int i = 0; i < dim; ++i)
            w_[i] = std::pow(10.0, 2.0 * double(i) / double(dim - 1));
}

double StepEllipsoidal::evaluate_core(const Vec& x)
{
    const int D = dimension();

    // Step transform
    Vec z(D);
    for (int i = 0; i < D; ++i)
        z[i] = (std::floor(0.5 + x[i])) * 1.0;

    double maxabs = 0.0;
    for (int i = 0; i < D; ++i)
        maxabs = std::max(maxabs, std::abs(z[i]));

    double sum = 0.0;
    for (int i = 0; i < D; ++i)
        sum += w_[i] * z[i] * z[i];

    return 0.1 * maxabs + sum;
}

void StepEllipsoidal::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    // derivative undefined at grid lines; use straight-through estimator
    // (treat dz_i/dx_i ~= 1) for both terms of
    //   f(x) = 0.1 * max_i |z_i| + sum_i w_i * z_i^2
    Vec z(D);
    int argmax = 0;
    double maxabs = -1.0;
    for (int i = 0; i < D; ++i) {
        z[i] = std::floor(0.5 + x[i]);
        // straight-through gradient of the sum-of-squares term
        g[i] = 2.0 * w_[i] * z[i];
        if (std::fabs(z[i]) > maxabs) {
            maxabs = std::fabs(z[i]);
            argmax = i;
        }
    }

    // straight-through gradient of the 0.1 * max_i |z_i| term: only the
    // argmax coordinate carries this contribution.
    if (D > 0 && z[argmax] != 0.0) {
        g[argmax] += 0.1 * (z[argmax] > 0.0 ? 1.0 : -1.0);
    }
}

} // namespace optimsolution
