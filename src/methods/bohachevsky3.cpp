#include "bohachevsky3.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Bohachevsky 3 function
//
// f(x) = x1^2 + 2*x2^2 - 0.3*cos(3πx1 + 4πx2) + 0.3
//
// Global minimum: f* = 0 at (0,0)
// Domain: typically [-50, 50]^2
//
// Properties:
// - multimodal
// - non-separable (cos(3πx1 + 4πx2))
// -----------------------------------------------------------------------------

Bohachevsky3::Bohachevsky3()
{
    setName("bohachevsky3");
    setFullName("Bohachevsky function 3");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Bohachevsky3::init(int dim)
{
    Problem::init(2);

    Vec l = {-50.0, -50.0};
    Vec u = { 50.0,  50.0};
    setBounds(l, u);

    Vec xopt = {0.0, 0.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double Bohachevsky3::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    return x1*x1
         + 2.0 * x2*x2
         - 0.3 * std::cos(3.0 * PI * x1 + 4.0 * PI * x2)
         + 0.3;
}

void Bohachevsky3::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    double inner = 3.0 * PI * x1 + 4.0 * PI * x2;
    double sin_inner = std::sin(inner);

    // df/dx1:
    // 2x1 - 0.3 * (-sin(inner) * 3π)
    double df_dx1 = 2.0 * x1
                    + 0.3 * (3.0 * PI) * sin_inner;

    // df/dx2:
    // 4x2 - 0.3 * (-sin(inner) * 4π)
    double df_dx2 = 4.0 * x2
                    + 0.3 * (4.0 * PI) * sin_inner;

    g[0] = df_dx1;
    g[1] = df_dx2;
}

} // namespace optimsolution
