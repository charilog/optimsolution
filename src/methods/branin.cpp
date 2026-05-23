#include "branin.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Branin (Branin–Hoo) function (2D)
//
// f(x1,x2) = (x2 - b*x1^2 + c*x1 - r)^2 + s*(1 - t)*cos(x1) + s
//
// Standard parameters:
//   a = 1
//   b = 5.1 / (4π²)
//   c = 5 / π
//   r = 6
//   s = 10
//   t = 1 / (8π)
//
// Domain:
//   x1 ∈ [-5, 10]
//   x2 ∈ [ 0, 15 ]
//
// Global minima (3 total):
//   f* ≈ 0.397887
//
// Properties:
//   - multimodal
//   - non-separable
//   - classic 2D benchmark
// -----------------------------------------------------------------------------

Branin::Branin()
{
    setName("branin");
    setFullName("Branin (Branin-Hoo) function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("2D benchmark test function");

    // Known minimum value (same for all 3 global minima)
    setKnownGlobalOptimum(0.397887);
}

void Branin::init(int dim)
{
    // Standard: Branin is strictly 2D
    Problem::init(2);

    Vec l = { -5.0,  0.0 };
    Vec u = { 10.0, 15.0 };
    setBounds(l, u);

    // One of the three known global minimizers:
    // (−π, 12.275), (π, 2.275), (9.42478, 2.475)
    Vec xopt = { -PI, 12.275 };
    setKnownGlobalOptimum(0.397887, xopt);
}

double Branin::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    const double a = 1.0;
    const double b = 5.1 / (4.0 * PI * PI);
    const double c = 5.0 / PI;
    const double r = 6.0;
    const double s = 10.0;
    const double t = 1.0 / (8.0 * PI);

    double term1 = x2 - b * x1 * x1 + c * x1 - r;
    double term2 = s * (1.0 - t) * std::cos(x1);

    return a * term1 * term1 + term2 + s;
}

void Branin::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    const double b = 5.1 / (4.0 * PI * PI);
    const double c = 5.0 / PI;
    const double r = 6.0;
    const double s = 10.0;
    const double t = 1.0 / (8.0 * PI);

    // Common subexpression
    double term1 = x2 - b * x1 * x1 + c * x1 - r;

    // df/dx1
    double df_dx1 =
        2.0 * term1 * (-2.0 * b * x1 + c)
        - s * (1.0 - t) * std::sin(x1);

    // df/dx2
    double df_dx2 = 2.0 * term1;

    g[0] = df_dx1;
    g[1] = df_dx2;
}

} // namespace optimsolution
