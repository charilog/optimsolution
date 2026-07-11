#include "threebartruss.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

ThreeBarTruss::ThreeBarTruss()
    : l_(100.0), P_(2.0), sigma_(2.0), w_pen_(1.0e6)
{
    setName("threebartruss");
    setFullName("Three-Bar Truss Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(263.8958);
}

void ThreeBarTruss::init(int /*dim*/) {
    Problem::init(2);

    Vec lo = {1e-4, 1e-4};
    Vec hi = {1.0,  1.0};
    setBounds(lo, hi);

    Vec xopt = {0.78867, 0.40902};
    setKnownGlobalOptimum(263.8958, xopt);
}

double ThreeBarTruss::evaluate_core(const Vec& x) {
    const double x0 = clampd(x[0], 1e-4, 1.0);
    const double x1 = clampd(x[1], 1e-4, 1.0);
    const double sq2 = std::sqrt(2.0);

    const double cost = l_ * (2.0 * sq2 * x0 + x1);

    const double denom = std::max(1e-9, sq2 * x0 * x0 + 2.0 * x0 * x1);

    const double g1 = ((sq2 * x0 + x1) / denom) * P_ - sigma_;
    const double g2 = (x1 / denom) * P_ - sigma_;
    const double g3 = (1.0 / std::max(1e-9, x0 + sq2 * x1)) * P_ - sigma_;

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };
    double penalty = 0.0;
    penalty += std::pow(pos(g1), 2.0);
    penalty += std::pow(pos(g2), 2.0);
    penalty += std::pow(pos(g3), 2.0);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void ThreeBarTruss::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
