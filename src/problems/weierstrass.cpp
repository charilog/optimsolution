#include "weierstrass.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.1415926535897932384626433832795; }

// -----------------------------------------------------------------------------
// Weierstrass function
//
// f(x) = sum_{i=1}^D [ sum_{k=0}^{kmax} a^k cos( 2π b^k (x_i + 0.5) ) ]
//        - D * sum_{k=0}^{kmax} a^k cos( 2π b^k * 0.5 )
//
// Typical: a=0.5, b=3, kmax=20
// Domain: [-0.5, 0.5]^D
// Global minimum: f(x*) = 0 at x* = (0,...,0)
// -----------------------------------------------------------------------------

Weierstrass::Weierstrass()
    : kmax_(20), a_(0.5), b_(3.0)
{
    setName("weierstrass");
    setFullName("Weierstrass function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // Will be refined in init(dim) once D is known
    setKnownGlobalOptimum(0.0);
}

void Weierstrass::init(int dim)
{
    const int D = (dim >= 1 ? dim : 10);   // default D = 10 if caller passes non-positive
    Problem::init(D);

    // Typical Weierstrass search box
    Vec lo(D, -0.5), hi(D, 0.5);
    setBounds(lo, hi);

    // Global optimum x* = (0,...,0), f* = 0
    Vec xopt(D, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Weierstrass::constant_term() const
{
    double c = 0.0;
    for (int k = 0; k <= kmax_; ++k) {
        c += std::pow(a_, k) * std::cos(2.0 * PI * std::pow(b_, k) * 0.5);
    }
    return c;
}

double Weierstrass::evaluate_core(const Vec& x)
{
    const int D = static_cast<int>(x.size());

    // Guard parameters (avoid statics; keep cheap checks here)
    if (!(a_ > 0.0 && a_ < 1.0)) return 1e12;
    if (!(b_ > 0.0))             return 1e12;
    if (kmax_ < 0)               return 1e12;

    const double C = constant_term();

    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        // Stay within the standard box to keep numerics well-behaved
        const double xi = clampd(x[i], -0.5, 0.5);
        double term = 0.0;
        for (int k = 0; k <= kmax_; ++k) {
            term += std::pow(a_, k) *
                    std::cos(2.0 * PI * std::pow(b_, k) * (xi + 0.5));
        }
        sum += term;
    }

    double f = sum - static_cast<double>(D) * C;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Weierstrass::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences (consistent with other problems)
    const int D = static_cast<int>(x.size());
    g.assign(D, 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < D; ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
