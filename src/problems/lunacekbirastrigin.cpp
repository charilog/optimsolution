#include "lunacekbirastrigin.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.141592653589793238462643383279502884; }

// -----------------------------------------------------------------------------
// Lunacek bi-Rastrigin
//
// Non-rotated BBOB form.
// Domain: [-5, 5]^n
// Global minimum ≈ 0 at x* = (mu1, ..., mu1)
// -----------------------------------------------------------------------------

LunacekBiRastrigin::LunacekBiRastrigin()
{
    setName("lunacek_bi_rastrigin");
    setFullName("Lunacek bi-Rastrigin function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // will be refined in init(dim)
    setKnownGlobalOptimum(0.0);
}

void LunacekBiRastrigin::init(int dim)
{
    if (dim < 1)
        dim = 1;

    n_ = dim;
    Problem::init(n_);

    // Parameterization
    s_ = 1.0 - 1.0 / (2.0 * std::sqrt(n_ + 20.0) - 8.2);
    s_ = std::max(s_, 1e-12);

    double inside = (mu1_ * mu1_ - d_) / s_;
    inside = std::max(inside, 0.0);
    mu2_ = -std::sqrt(inside);

    // Bounds [-5, 5]^n
    Vec lo(n_, -5.0), hi(n_, 5.0);
    setBounds(lo, hi);

    // Known optimum location (approximate)
    Vec xopt(n_, mu1_);
    setKnownGlobalOptimum(0.0, xopt);
}

double LunacekBiRastrigin::evaluate_core(const Vec& x)
{
    double sum1 = 0.0, sum2 = 0.0, csum = 0.0;

    for (int i = 0; i < n_; ++i) {
        const double t1 = x[i] - mu1_;
        const double t2 = x[i] - mu2_;
        sum1 += t1 * t1;
        sum2 += t2 * t2;
        csum += std::cos(2.0 * PI * t1);
    }

    const double bowl = std::min(sum1, d_ * n_ + s_ * sum2);
    const double ras  = 10.0 * (n_ - csum);

    double f = bowl + ras;
    if (!std::isfinite(f))
        f = 1e12;

    return f;
}

void LunacekBiRastrigin::gradient_core(const Vec& x, Vec& g)
{
    g.assign(n_, 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < n_; ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * 1e-6);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
