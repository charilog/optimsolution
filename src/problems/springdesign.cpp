#include "springdesign.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

SpringDesign::SpringDesign()
    : w_pen_(1.0e6)
{
    setName("springdesign");
    setFullName("Tension/Compression Spring Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(0.012665);
}

void SpringDesign::init(int /*dim*/) {
    Problem::init(3);

    Vec lo = {0.05, 0.25, 2.0};
    Vec hi = {2.0,  1.3,  15.0};
    setBounds(lo, hi);

    Vec xopt = {0.05169, 0.35673, 11.28896};
    setKnownGlobalOptimum(0.012665, xopt);
}

double SpringDesign::evaluate_core(const Vec& x) {
    const double d = clampd(x[0], 0.05, 2.0);
    const double D = clampd(x[1], 0.25, 1.3);
    const double N = clampd(x[2], 2.0,  15.0);

    const double cost = (N + 2.0) * D * d * d;

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };

    const double g1 = 1.0 - (D * D * D * N) / (71785.0 * std::pow(d, 4.0));
    const double denom2 = 12566.0 * (D * d * d * d - std::pow(d, 4.0));
    const double g2 = (4.0 * D * D - d * D) / std::max(1e-9, denom2)
                     + 1.0 / (5108.0 * d * d) - 1.0;
    const double g3 = 1.0 - (140.45 * d) / (D * D * N);
    const double g4 = (d + D) / 1.5 - 1.0;

    double penalty = 0.0;
    penalty += std::pow(pos(g1), 2.0);
    penalty += std::pow(pos(g2), 2.0);
    penalty += std::pow(pos(g3), 2.0);
    penalty += std::pow(pos(g4), 2.0);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void SpringDesign::gradient_core(const Vec& x, Vec& g) {
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
