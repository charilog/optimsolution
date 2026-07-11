#include "speedreducer.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

SpeedReducer::SpeedReducer()
    : w_pen_(1.0e6)
{
    setName("speedreducer");
    setFullName("Speed Reducer (Gearbox) Design");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(2994.47);
}

void SpeedReducer::init(int /*dim*/) {
    Problem::init(7);

    Vec lo = {2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0};
    Vec hi = {3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5};
    setBounds(lo, hi);

    Vec xopt = {3.5, 0.7, 17.0, 7.3, 7.71532, 3.35054, 5.28665};
    setKnownGlobalOptimum(2994.47, xopt);
}

double SpeedReducer::evaluate_core(const Vec& x) {
    const double x0 = clampd(x[0], 2.6, 3.6);
    const double x1 = clampd(x[1], 0.7, 0.8);
    const double x2 = clampd(x[2], 17.0, 28.0);
    const double x3 = clampd(x[3], 7.3, 8.3);
    const double x4 = clampd(x[4], 7.3, 8.3);
    const double x5 = clampd(x[5], 2.9, 3.9);
    const double x6 = clampd(x[6], 5.0, 5.5);

    const double cost =
          0.7854 * x0 * x1 * x1 * (3.3333 * x2 * x2 + 14.9334 * x2 - 43.0934)
        - 1.508 * x0 * (x5 * x5 + x6 * x6)
        + 7.4777 * (x5 * x5 * x5 + x6 * x6 * x6)
        + 0.7854 * (x3 * x5 * x5 + x4 * x6 * x6);

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };

    const double g1 = 27.0 / (x0 * x1 * x1 * x2) - 1.0;
    const double g2 = 397.5 / (x0 * x1 * x1 * x2 * x2) - 1.0;
    const double g3 = (1.93 * x3 * x3 * x3) / (x1 * x2 * std::pow(x5, 4.0)) - 1.0;
    const double g4 = (1.93 * x4 * x4 * x4) / (x1 * x2 * std::pow(x6, 4.0)) - 1.0;
    const double t5 = std::sqrt(std::pow(745.0 * x3 / (x1 * x2), 2.0) + 16.9e6);
    const double g5 = t5 / (110.0 * x5 * x5 * x5) - 1.0;
    const double t6 = std::sqrt(std::pow(745.0 * x4 / (x1 * x2), 2.0) + 157.5e6);
    const double g6 = t6 / (85.0 * x6 * x6 * x6) - 1.0;
    const double g7 = (x1 * x2) / 40.0 - 1.0;
    const double g8 = (5.0 * x1) / x0 - 1.0;
    const double g9 = x0 / (12.0 * x1) - 1.0;
    const double g10 = (1.5 * x5 + 1.9) / x3 - 1.0;
    // NOTE: the source paper (Peng et al., Eq. 16) prints this constraint's
    // coefficient as 1.5 (matching g10's coefficient exactly), but that
    // makes the constraint unsatisfiable near the known optimum (and near
    // the box's upper bound on x4/x6, no feasible point exists at all).
    // The standard Golinski speed-reducer formulation (e.g. arXiv:2202.06017)
    // uses coefficient 1.1 here; with 1.1 this constraint becomes active
    // (approximately 0) at the literature optimum, confirming it is correct.
    const double g11 = (1.1 * x6 + 1.9) / x4 - 1.0;

    double penalty = 0.0;
    penalty += pos(g1) * pos(g1);
    penalty += pos(g2) * pos(g2);
    penalty += pos(g3) * pos(g3);
    penalty += pos(g4) * pos(g4);
    penalty += pos(g5) * pos(g5);
    penalty += pos(g6) * pos(g6);
    penalty += pos(g7) * pos(g7);
    penalty += pos(g8) * pos(g8);
    penalty += pos(g9) * pos(g9);
    penalty += pos(g10) * pos(g10);
    penalty += pos(g11) * pos(g11);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void SpeedReducer::gradient_core(const Vec& x, Vec& g) {
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
