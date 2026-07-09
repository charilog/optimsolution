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

// Exact analytic gradient (previously computed via forward differences).
//
// f(x) = sum_i [ sum_k a^k cos(2*pi*b^k*(x_i+0.5)) ] - D*C, so each coordinate
// is independent and:
//   d/dx_i [ a^k cos(2*pi*b^k*(x_i+0.5)) ] = -2*pi*(a*b)^k * sin(2*pi*b^k*(x_i+0.5))
//
// At the default kmax=20, b=3, the highest-order term oscillates with angular
// frequency 2*pi*b^20 (~2.2e10 rad per unit x), so a finite-difference step of
// h~1e-6 (as used elsewhere in this codebase, and by any similarly-sized
// generic step) advances that term's phase by ~21,900 radians -- over 3000
// full cycles within a single step. A forward/central-difference "gradient"
// computed that way is therefore essentially noise, not a usable search
// direction, for any gradient-based method. The closed-form derivative below
// has no such step-size sensitivity.
void Weierstrass::gradient_core(const Vec& x, Vec& g)
{
    const int D = static_cast<int>(x.size());
    g.assign(D, 0.0);

    if (!(a_ > 0.0 && a_ < 1.0) || !(b_ > 0.0) || kmax_ < 0) {
        return; // matches evaluate_core's invalid-parameter guard (f = 1e12 there)
    }

    for (int i = 0; i < D; ++i) {
        if (x[i] < -0.5 || x[i] > 0.5) {
            g[i] = 0.0; // evaluate_core clamps here, so f is locally flat
            continue;
        }
        const double xi = x[i];
        double dterm = 0.0;
        for (int k = 0; k <= kmax_; ++k) {
            const double ak = std::pow(a_, k);
            const double bk = std::pow(b_, k);
            dterm += ak * bk * std::sin(2.0 * PI * bk * (xi + 0.5));
        }
        g[i] = -2.0 * PI * dterm;
    }
}

} // namespace optimsolution
