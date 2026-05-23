#include "bucherastrigin.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Buche–Rastrigin (BBOB-style deterministic variant)
//
// Classic definition:
//
//   f(x) = 10*D + sum_i [ z_i^2 - 10*cos(2π z_i) ]
//
// where:
//   z_i = s_i * y_i
//   y_i = x_i + bias_i   (here bias = 0 for deterministic version)
//
//   s_i = 10^( 0.5 * (i/(D-1)) )    // asymmetric scaling
//
//   Additionally, coordinates with x_i > 0 are stretched:
//
//   if (x_i > 0) z_i *= 10
//
// Global optimum:
//   f* = 0  at x* = 0
//
// Domain:
//   [-5, 5]^D (standard BBOB range)
//
// Properties:
//   - multimodal
//   - non-separable
//   - asymmetric landscape
// -----------------------------------------------------------------------------

BucheRastrigin::BucheRastrigin()
{
    setName("bucherastrigin");
    setFullName("Buche-Rastrigin function (BBOB-style variant)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("BBOB synthetic benchmark");

    setKnownGlobalOptimum(0.0);   // value
}

void BucheRastrigin::init(int dim)
{
    Problem::init(dim);

    // Standard domain for BBOB F4
    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    // Global minimizer (deterministic version)
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);

    // Precompute scaling factors s_i
    scale_.resize(dim);
    if (dim > 1) {
        for (int i = 0; i < dim; ++i)
            scale_[i] = std::pow(10.0, 0.5 * (double(i) / double(dim - 1)));
    } 
    else
        scale_[0] = 1.0;
}

double BucheRastrigin::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i) {
        double z = x[i] * scale_[i];

        // asymmetric stretch (BBOB variant)
        if (x[i] > 0.0)
            z *= 10.0;

        sum += (z*z - 10.0 * std::cos(2.0 * PI * z));
    }

    return 10.0 * D + sum;
}

void BucheRastrigin::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D; ++i)
    {
        double base = x[i] * scale_[i];
        double z = base;

        bool positive = (x[i] > 0.0);
        if (positive)
            z *= 10.0;

        // dz/dx = scale * (10 if x>0 else 1)
        double dzdx = scale_[i] * (positive ? 10.0 : 1.0);

        // derivative: d/dz( z^2 - 10*cos(2πz) )
        double df_dz = 2.0 * z + 20.0 * PI * std::sin(2.0 * PI * z);

        g[i] = df_dz * dzdx;
    }
}

} // namespace optimsolution
