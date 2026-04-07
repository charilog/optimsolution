#include "equalmaxima.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

EqualMaxima::EqualMaxima()
{
    setName("equalmaxima");
    setFullName("Equal Maxima function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("1D benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void EqualMaxima::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, 0.0);
    Vec u(dim, 1.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double EqualMaxima::evaluate_core(const Vec& x)
{
    double t = std::sin(5.0 * PI * x[0]);
    return t * t * t * t * t * t; // sin^6
}

void EqualMaxima::gradient_core(const Vec& x, Vec& g)
{
    g.resize(1);

    double t = std::sin(5.0 * PI * x[0]);
    double dt = std::cos(5.0 * PI * x[0]) * 5.0 * PI;

    // derivative of sin^6 = 6 sin^5 * cos * d(x)
    g[0] = 6.0 * std::pow(t, 5) * dt;
}

} // namespace optimsolution
