#include "cigar.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Cigar function
//
// f(x) = x1^2 + 1e6 * sum_{i=2..D} x_i^2
//
// Global minimum: f* = 0  at x = 0
// Domain: [-5, 5]^D
//
// Properties:
//   - unimodal
//   - non-separable (very ill-conditioned)
// -----------------------------------------------------------------------------

Cigar::Cigar()
{
    setName("cigar");
    setFullName("Cigar function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Cigar::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Cigar::evaluate_core(const Vec& x)
{
    const int D = dimension();

    double sum = x[0] * x[0]; // first dimension

    double tail = 0.0;
    for (int i = 1; i < D; ++i)
        tail += x[i] * x[i];

    return sum + 1e6 * tail;
}

void Cigar::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    g[0] = 2.0 * x[0];

    for (int i = 1; i < D; ++i)
        g[i] = 2.0 * 1e6 * x[i];
}

} // namespace optimsolution
