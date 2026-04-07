#include "bohachevsky2.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Bohachevsky 2 function
//
// f(x) = x1^2 + x2^2 - 0.3 * cos(3 * π * x1) * cos(4 * π * x2) + 0.3
//
// Global minimum: f* = 0 at (0,0)
// Domain: typically [-50, 50]^2
//
// Properties:
// - multimodal
// - non-separable (product of cosines couples x1,x2)
// -----------------------------------------------------------------------------

Bohachevsky2::Bohachevsky2()
{
    setName("bohachevsky2");
    setFullName("Bohachevsky function 2");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Bohachevsky2::init(int dim)
{
    Problem::init(2);

    Vec l = {-50.0, -50.0};
    Vec u = { 50.0,  50.0};
    setBounds(l, u);

    Vec xopt = {0.0, 0.0};
    setKnownGlobalOptimum(0.0, xopt);
}

double Bohachevsky2::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    return x1*x1
         + x2*x2
         - 0.3 * std::cos(3.0 * PI * x1) * std::cos(4.0 * PI * x2)
         + 0.3;
}

void Bohachevsky2::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    // Product term:
    // h = cos(3πx1) * cos(4πx2)
    double c1 = std::cos(3.0 * PI * x1);
    double c2 = std::cos(4.0 * PI * x2);
    double s1 = std::sin(3.0 * PI * x1);
    double s2 = std::sin(4.0 * PI * x2);

    // f = x1^2 + x2^2 - 0.3 h + 0.3

    // df/dx1:
    // 2x1 - 0.3 * [ - (3π)*sin(3πx1) * cos(4πx2) ]
    double df_dx1 = 2.0 * x1
                    + 0.3 * (3.0 * PI) * s1 * c2;

    // df/dx2:
    // 2x2 - 0.3 * [ cos(3πx1) * - (4π)*sin(4πx2) ]
    double df_dx2 = 2.0 * x2
                    + 0.3 * (4.0 * PI) * c1 * s2;

    g[0] = df_dx1;
    g[1] = df_dx2;
}

} // namespace optimsolution
