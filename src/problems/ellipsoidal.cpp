#include "ellipsoidal.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Ellipsoidal (weighted sphere)
//
// f(x) = sum_i 10^{6 * (i-1)/(D-1)} * x_i^2
//
// Global minimum: 0 at x=0
// Domain: [-5, 5]^D
//
// Properties:
//   - unimodal
//   - non-separable
// -----------------------------------------------------------------------------

Ellipsoidal::Ellipsoidal()
{
    setName("ellipsoidal");
    setFullName("Ellipsoidal function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Ellipsoidal::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);

    // Precompute weights
    w_.resize(dim);
    if (dim == 1)
        w_[0] = 1.0;
    else
        for (int i = 0; i < dim; ++i)
            w_[i] = std::pow(10.0, 6.0 * double(i) / double(dim - 1));
}

double Ellipsoidal::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i)
        sum += w_[i] * x[i] * x[i];

    return sum;
}

void Ellipsoidal::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);

    for (int i = 0; i < D; ++i)
        g[i] = 2.0 * w_[i] * x[i];
}

} // namespace optimsolution
