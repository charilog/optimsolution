#include "bohachevsky1.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Bohachevsky 1 function
//
// f(x) = x1^2 + 2*x2^2 - 0.3*cos(3πx1) - 0.4*cos(4πx2) + 0.7
//
// Global minimum: f* = 0 at (0,0)
// Domain: typically [-50, 50]^2
//
// Properties:
// - multimodal
// - non-separable
// -----------------------------------------------------------------------------

Bohachevsky1::Bohachevsky1()
{
    setName("bohachevsky1");
    setFullName("Bohachevsky function 1");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Bohachevsky1::init(int dim)
{
    Problem::init(2);

    Vec l = {-50.0, -50.0};
    Vec u = { 50.0,  50.0};
    setBounds(l, u);

    Vec xopt = {0.0, 0.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double Bohachevsky1::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    return x1*x1
         + 2.0 * x2*x2
         - 0.3 * std::cos(3.0 * PI * x1)
         - 0.4 * std::cos(4.0 * PI * x2)
         + 0.7;
}

void Bohachevsky1::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    // d/dx1
    double df_dx1 = 2.0 * x1
                    + 0.3 * 3.0 * PI * std::sin(3.0 * PI * x1);

    // d/dx2
    double df_dx2 = 4.0 * x2
                    + 0.4 * 4.0 * PI * std::sin(4.0 * PI * x2);

    g[0] = df_dx1;
    g[1] = df_dx2;
}

} // namespace optimsolution
