#include "easom.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------------------------------
// Easom function (2D)
//
// f(x1, x2) = -cos(x1)*cos(x2)*exp(-(x1-π)^2 - (x2-π)^2)
//
// Domain: [-100, 100]^2
// Global minimum: f* = -1 at (π, π)
//
// Properties:
//   - unimodal (but extremely sharp)
//   - non-separable
// -----------------------------------------------------------------------------

Easom::Easom()
{
    setName("easom");
    setFullName("Easom function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous test function");

    setKnownGlobalOptimum(-1.0);
}

void Easom::init(int dim)
{
    Problem::init(2);

    Vec l = { -100.0, -100.0 };
    Vec u = {  100.0,  100.0 };
    setBounds(l, u);

    Vec xopt = { PI, PI };
    setKnownGlobalOptimum(-1.0, xopt);
}

double Easom::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];

    double a = std::cos(x1) * std::cos(x2);
    double b = std::exp(-(x1 - PI)*(x1 - PI) - (x2 - PI)*(x2 - PI));

    return -a * b;
}

void Easom::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double x1 = x[0];
    const double x2 = x[1];

    double dx1 = x1 - PI;
    double dx2 = x2 - PI;

    double c1 = std::cos(x1);
    double c2 = std::cos(x2);
    double s1 = std::sin(x1);
    double s2 = std::sin(x2);

    double expTerm = std::exp(-(dx1*dx1 + dx2*dx2));

    // df/dx1
    g[0] = -(
        (-s1) * c2 * expTerm    // derivative of cos(x1)
        + c1 * c2 * expTerm * (-2.0 * dx1)
    );

    // df/dx2
    g[1] = -(
        c1 * (-s2) * expTerm
        + c1 * c2 * expTerm * (-2.0 * dx2)
    );
}

} // namespace optimsolution
