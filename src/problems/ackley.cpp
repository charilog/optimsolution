#include "ackley.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// ------------------------------------------------------------------
// Ackley function
// f(x) = -20 * exp( -0.2 * sqrt( (1/D)*sum(x_i^2) ) )
//        - exp( (1/D)*sum(cos(2π x_i)) )
//        + 20 + e
//
// Global minimum: f* = 0 at x* = (0,...,0)
// Type: multimodal, non-separable
// Typical domain: [-32.768, 32.768]^D
// ------------------------------------------------------------------

Ackley::Ackley()
{
    // Metadata
    setName("ackley");
    setFullName("Ackley benchmark function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // Known optimum value
    setKnownGlobalOptimum(0.0);
}

void Ackley::init(int dim)
{
    Problem::init(dim);

    // Bounds: [-32.768, 32.768]^D
    Vec l(dim, -32.768);
    Vec u(dim,  32.768);
    setBounds(l, u);

    // Global optimum location
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Ackley::evaluate_core(const Vec& x)
{
    const int D = dimension();

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < D; ++i) {
        sum1 += x[i] * x[i];
        sum2 += std::cos(2.0 * PI * x[i]);
    }

    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum1 / D));
    double term2 = -std::exp(sum2 / D);

    return term1 + term2 + 20.0 + std::exp(1.0);
}

// Gradient of Ackley
void Ackley::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < D; ++i) {
        sum1 += x[i] * x[i];
        sum2 += std::cos(2.0 * PI * x[i]);
    }

    double sqrt_sum1_D = std::sqrt(sum1 / D);
    double exp1 = std::exp(-0.2 * sqrt_sum1_D);
    double exp2 = std::exp(sum2 / D);

    double common1 = (4.0 * x[0]); // not used directly, placeholder

    for (int i = 0; i < D; ++i) {
        double d1 = 0.0;
        double d2 = 0.0;

        // derivative of first term
        if (sqrt_sum1_D > 0.0) {
            d1 = -20.0 * exp1 * (-0.2) * (1.0 / (2.0 * sqrt_sum1_D * D)) * (2.0 * x[i]);
        }

        // derivative of second term
        d2 = -exp2 * (1.0 / D) * (-2.0 * PI * std::sin(2.0 * PI * x[i]));

        g[i] = d1 + d2;
    }
}

} // namespace optimsolution
