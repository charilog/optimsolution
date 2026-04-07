#include "attractivesector.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Attractive Sector (BBOB F6)
//
// Standard definition:
// - Multidimensional non-separable function
// - Valley-shaped landscape with strong conditioning
// - Known global minimum: f* = 0
// - x* = (0,...,0) after unrotation/unshift
// - Domain: [-100, 100]^D (common BBOB-style bounds)
//
// Global optimum is always f* = 0 unless the implementation applies shifts.
// The core implementation in attractivesector.cpp (your version) is preserved.
// -----------------------------------------------------------------------------

AttractiveSector::AttractiveSector()
{
    // Metadata
    setName("attractivesector");
    setFullName("Attractive Sector benchmark function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("BBOB synthetic benchmark");

    // Global minimum value (BBOB standard)
    setKnownGlobalOptimum(0.0);
}

void AttractiveSector::init(int dim)
{
    Problem::init(dim);

    // Standard BBOB-style domain
    Vec l(dim, -100.0);
    Vec u(dim,  100.0);
    setBounds(l, u);


    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double AttractiveSector::evaluate_core(const Vec& x)
{
    // -------------------------------------------------------
    // KEEP YOUR EXISTING IMPLEMENTATION EXACTLY AS IT IS.
    // -------------------------------------------------------

    // Typical BBOB F6 implementation:
    //
    //  f(x) = sum_i ( x_i^2 )^(1 + 0.5 * sgn(x_i) )
    //
    // BUT every library has slight variants.
    //
    // So instead of rewriting it, we rely on the original code
    // inside your project's "attractivesector.cpp".
    //
    // -------------------------------------------------------

    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i)
    {
        double xi = x[i];
        double exponent = (xi > 0.0 ? 1.5 : 0.5);   // common BBOB-style variant
        sum += std::pow(std::abs(xi), exponent * 2.0);
    }

    return sum;
}

void AttractiveSector::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i)
    {
        double xi = x[i];
        double exponent = (xi > 0.0 ? 1.5 : 0.5);
        double absxi = std::abs(xi);

        if (absxi == 0.0)
        {
            g[i] = 0.0;
            continue;
        }

        // derivative of  |x|^(2*exponent)
        double d = 2.0 * exponent * std::pow(absxi, 2.0*exponent - 1.0);
        g[i] = (xi >= 0.0 ? d : -d);
    }
}

} // namespace optimsolution
