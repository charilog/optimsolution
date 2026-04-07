#include "katsuura.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Katsuura function
//
// f(x) = ∏_{i=1}^n ( 1 + i * Σ_{k=1}^{32} |2^k x_i - round(2^k x_i)| / 2^k )^{10 / n^{1.2}} - 1
//
// Global minimum: f* = 0 at x* = (0, ..., 0)
// Domain: [-100, 100]^n
// -----------------------------------------------------------------------------

Katsuura::Katsuura()
{
    setName("katsuura");
    setFullName("Katsuura function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // Will be refined in init(dim) with the actual dimension and x*
    setKnownGlobalOptimum(0.0);
}

void Katsuura::init(int dim)
{
    if (dim < 1)
        dim = 1;

    Problem::init(dim);

    // Standard Katsuura domain: [-100, 100]^n
    Vec lo(dim, -100.0);
    Vec hi(dim,  100.0);
    setBounds(lo, hi);

    // Global optimum at x* = (0, ..., 0), f* = 0
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Katsuura::evaluate_core(const Vec& x)
{
    const int n = static_cast<int>(x.size());
    if (n == 0)
        return 0.0;

    // exponent 10 / n^{1.2}
    const double expo = 10.0 / std::pow(static_cast<double>(n), 1.2);

    double prod = 1.0;
    for (int i = 0; i < n; ++i) {
        // Σ_{k=1}^{32} |2^k x_i - round(2^k x_i)| / 2^k
        double sumk = 0.0;
        const double xi = x[i];

        for (int k = 1; k <= 32; ++k) {
            // 2^k * x_i computed robustly via ldexp
            const double two_k_x = std::ldexp(xi, k);
            const double frac    = std::fabs(two_k_x - std::round(two_k_x));
            sumk += frac / std::ldexp(1.0, k);  // divide by 2^k
        }

        // (1 + i * sumk) with i starting at 1
        const double base = 1.0 + static_cast<double>(i + 1) * sumk;

        // raise to expo and multiply
        prod *= std::pow(base, expo);

        // guard against overflow/NaN
        if (!std::isfinite(prod))
            return 1e12;
    }

    const double f = prod - 1.0;
    if (!std::isfinite(f))
        return 1e12;

    return f;
}

void Katsuura::gradient_core(const Vec& x, Vec& g)
{
    const int n = static_cast<int>(x.size());
    g.assign(n, 0.0);

    if (n == 0)
        return;

    const double f0 = evaluate_core(x);
    Vec xt = x;

    // forward-diff step (small; Katsuura is non-differentiable in places)
    for (int i = 0; i < n; ++i) {
        double h = std::max(1e-8, std::abs(x[i]) * 1e-6);

        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;

        // restore
        xt[i] = x[i];
    }
}

} // namespace optimsolution
