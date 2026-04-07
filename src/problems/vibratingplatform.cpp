#include "vibratingplatform.h"

#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PI = 3.1415926535897932384626433832795; }

VibratingPlatform::VibratingPlatform()
{
    setName("vibratingplatform");
    setFullName("Vibrating platform design");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("mechanical design / benchmark");

    // Standard benchmark constants
    rho1_ = 100.0;
    rho2_ = 2770.0;
    rho3_ = 7780.0;

    E1_ = 1.6;
    E2_ = 70.0;
    E3_ = 200.0;

    c1_ = 500.0;
    c2_ = 1500.0;
    c3_ = 800.0;

    // Standard benchmark bounds
    d1_min_ = 0.05; d1_max_ = 0.50;
    d2_min_ = 0.20; d2_max_ = 0.50;
    d3_min_ = 0.20; d3_max_ = 0.60;
    b_min_  = 0.35; b_max_  = 0.50;
    L_min_  = 3.00; L_max_  = 6.00;

    // Single-objective scalarization parameters.
    // The original benchmark is bi-objective; these values only provide a stable
    // embedding into optimsolution's scalar Problem interface.
    w_f1_  = 0.50;
    w_f2_  = 0.50;
    w_con_ = 1000.0;
}

void VibratingPlatform::init(int /*dim*/) {
    // Force the standard benchmark dimension D=5.
    Problem::init(5);
    Vec lo = { d1_min_, d2_min_, d3_min_, b_min_, L_min_ };
    Vec hi = { d1_max_, d2_max_, d3_max_, b_max_, L_max_ };
    setBounds(lo, hi);
}

double VibratingPlatform::evaluate_core(const Vec& x) {
    // Clamp inside the benchmark box for numerical stability.
    const double d1 = clampd(x[0], d1_min_, d1_max_);
    const double d2 = clampd(x[1], d2_min_, d2_max_);
    const double d3 = clampd(x[2], d3_min_, d3_max_);
    const double b  = clampd(x[3], b_min_,  b_max_ );
    const double L  = clampd(x[4], L_min_,  L_max_ );

    // Auxiliary expressions from the standard formulation.
    const double mu = 2.0 * b * (rho1_ * d1 + rho2_ * (d2 - d1) + rho3_ * (d3 - d2));
    const double EI = (2.0 * b / 3.0) *
                      (E1_ * d1 * d1 * d1 +
                       E2_ * (d2 * d2 * d2 - d1 * d1 * d1) +
                       E3_ * (d3 * d3 * d3 - d2 * d2 * d2));

    // Guard against invalid intermediate values.
    if (!(mu > 0.0) || !(EI > 0.0) || !std::isfinite(mu) || !std::isfinite(EI)) {
        return 1e12;
    }

    // Original bi-objective benchmark.
    const double f1 = -(PI / (2.0 * L * L)) * std::sqrt(EI / mu);
    const double f2 =  2.0 * b * L * (c1_ * d1 + c2_ * (d2 - d1) + c3_ * (d3 - d2));

    // Original inequality constraints g_i(x) <= 0.
    const double g1 = mu * L - 2800.0;
    const double g2 = d1 - d2;
    const double g3 = d2 - d1 - 0.15;
    const double g4 = d2 - d3;
    const double g5 = d3 - d2 - 0.01;

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };

    // Relative smooth penalty scaling to keep terms numerically balanced.
    double penalty = 0.0;
    penalty += std::pow(pos(g1) / 2800.0, 2.0);
    penalty += std::pow(pos(g2) / 0.50,   2.0);
    penalty += std::pow(pos(g3) / 0.15,   2.0);
    penalty += std::pow(pos(g4) / 0.60,   2.0);
    penalty += std::pow(pos(g5) / 0.01,   2.0);

    // Normalized scalarization for the single-objective framework.
    // We minimize both terms; since f1 is negative in the benchmark, we minimize -f1.
    const double nf1 = (-f1) / 0.02;     // order-of-magnitude normalization
    const double nf2 =   f2  / 1000.0;   // order-of-magnitude normalization

    double cost = w_f1_ * nf1 + w_f2_ * nf2 + w_con_ * penalty;
    if (!std::isfinite(cost)) cost = 1e12;
    return cost;
}

void VibratingPlatform::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    const double abs = 1e-8;
    const int n = std::min<int>(5, static_cast<int>(x.size()));

    for (int i = 0; i < n; ++i) {
        const double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
