#include "expotential.h"
#include <cmath>

namespace optimsolution {

Expotential::Expotential()
{
    setName("expotential");
    setFullName("Expotential function  f(x)=1-exp(-0.5||x||^2)");
    setModality("unimodal");
    setSeparability("non-separable"); // because ||x|| couples dimensions
    setCategory("continuous benchmark test function");

    // Optimum is known but dimension is unknown until init()
    // so setKnownGlobalOptimum(...) will be finalized in init().
}

void Expotential::init(int dim)
{
    if (dim < 1) dim = 1;
    Problem::init(dim);

    // Standard domain as used in original file:
    Vec lo(dim, -1.0), hi(dim, 1.0);
    setBounds(lo, hi);

    // Optimum = 0 at x = (0,...,0)
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

// f(x) = 1 - exp( -0.5 * ||x||^2 )
double Expotential::evaluate_core(const Vec& x)
{
    double r2 = 0.0;
    for (double xi : x)
        r2 += xi * xi;

    const double E = std::exp(-0.5 * r2);
    const double f = 1.0 - E;

    return std::isfinite(f) ? f : 1e12;
}

// ∇f(x) = exp(-0.5 * ||x||^2) * x
void Expotential::gradient_core(const Vec& x, Vec& g)
{
    const int D = (int)x.size();
    g.assign(D, 0.0);

    double r2 = 0.0;
    for (double xi : x)
        r2 += xi * xi;

    const double E = std::exp(-0.5 * r2);

    for (int i = 0; i < D; ++i)
        g[i] = E * x[i];
}

} // namespace optimsolution
