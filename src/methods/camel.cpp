#include "camel.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Six-Hump Camel Function
//
// f(x1,x2) = (4 - 2.1 x1^2 + x1^4 / 3) * x1^2 + x1 x2 + (-4 + 4 x2^2) * x2^2
//
// Domain:
//   x1 ∈ [-3, 3]
//   x2 ∈ [-2, 2]
//
// Properties:
//   - multimodal
//   - non-separable
//
// Global minima (two):
//   f* = -1.031628453489877
//   (0.089842, -0.712656)
//   (-0.089842, 0.712656)
// -----------------------------------------------------------------------------

namespace {
    constexpr double PI = 3.14159265358979323846; // not used, but kept for uniformity
}

Camel::Camel()
{
    setName("camel");
    setFullName("Six-Hump Camel function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous 2D benchmark test function");

    setKnownGlobalOptimum(-1.031628453489877);
}

void Camel::init(int dim)
{
    Problem::init(2); // strictly 2D

    Vec l = { -3.0, -2.0 };
    Vec u = {  3.0,  2.0 };
    setBounds(l, u);

    // We store one of the two global minimizers
    Vec xopt = { 0.089842, -0.712656 };
    setKnownGlobalOptimum(-1.031628453489877, xopt);
}

double Camel::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    double term1 = (4.0 - 2.1 * x1 * x1 + (x1 * x1 * x1 * x1) / 3.0) * (x1 * x1);
    double term2 = x1 * x2;
    double term3 = (-4.0 + 4.0 * x2 * x2) * (x2 * x2);

    return term1 + term2 + term3;
}

void Camel::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    // f = A*x1^2 + x1*x2 + B*x2^2
    // A = 4 - 2.1x1^2 + x1^4/3
    // df/dx1:
    double A = 4.0 - 2.1 * x1 * x1 + (x1 * x1 * x1 * x1) / 3.0;
    double dA_dx1 = -4.2 * x1 + (4.0 / 3.0) * x1 * x1 * x1;

    double df_dx1 = 2.0 * x1 * A + x1 * x1 * dA_dx1 + x2;

    // df/dx2:
    double B = -4.0 + 4.0 * x2 * x2;
    double dB_dx2 = 8.0 * x2;

    double df_dx2 = x1 + 2.0 * x2 * B + x2 * x2 * dB_dx2;

    g[0] = df_dx1;
    g[1] = df_dx2;
}

} // namespace optimsolution
