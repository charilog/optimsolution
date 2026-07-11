#include "pressurevessel.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

namespace { constexpr double PV_PI = 3.141592653589793238462643383279502884; }

PressureVessel::PressureVessel()
    : w_pen_(1.0e6)
{
    setName("pressurevessel");
    setFullName("Pressure Vessel Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(6059.71);
}

void PressureVessel::init(int /*dim*/) {
    Problem::init(4);

    Vec lo = {0.0, 0.0, 10.0, 10.0};
    Vec hi = {99.0, 99.0, 200.0, 200.0};
    setBounds(lo, hi);

    Vec xopt = {0.8125, 0.4375, 42.0984, 176.6366};
    setKnownGlobalOptimum(6059.71, xopt);
}

double PressureVessel::evaluate_core(const Vec& x) {
    const double Ts = clampd(x[0], 0.0, 99.0);
    const double Th = clampd(x[1], 0.0, 99.0);
    const double R  = clampd(x[2], 10.0, 200.0);
    const double L  = clampd(x[3], 10.0, 200.0);

    const double cost = 0.6224 * Ts * R * L
                       + 1.7781 * Th * R * R
                       + 3.1661 * Ts * Ts * L
                       + 19.84  * Ts * Ts * R;

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };

    const double g1 = -Ts + 0.0193 * R;
    const double g2 = -Th + 0.00954 * R;
    const double g3 = -PV_PI * R * R * L - (4.0 / 3.0) * PV_PI * R * R * R + 1296000.0;
    const double g4 = L - 240.0;

    double penalty = 0.0;
    penalty += std::pow(pos(g1), 2.0);
    penalty += std::pow(pos(g2), 2.0);
    penalty += std::pow(pos(g3) / 1.0e6, 2.0); // g3 has huge magnitude; normalize
    penalty += std::pow(pos(g4), 2.0);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void PressureVessel::gradient_core(const Vec& x, Vec& g) {
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
