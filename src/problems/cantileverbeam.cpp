#include "cantileverbeam.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

CantileverBeam::CantileverBeam()
    : w_pen_(1.0e6)
{
    setName("cantileverbeam");
    setFullName("Cantilever Beam Design");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("mechanical engineering design benchmark");

    setKnownGlobalOptimum(1.33996);
}

void CantileverBeam::init(int /*dim*/) {
    Problem::init(5);

    Vec lo(5, 0.01), hi(5, 100.0);
    setBounds(lo, hi);

    Vec xopt = {6.0160, 5.3092, 4.4950, 3.4967, 2.1526};
    setKnownGlobalOptimum(1.33996, xopt);
}

double CantileverBeam::evaluate_core(const Vec& x) {
    double xs[5];
    for (int i = 0; i < 5; ++i) xs[i] = clampd(x[i], 0.01, 100.0);

    double cost = 0.0;
    for (int i = 0; i < 5; ++i) cost += xs[i];
    cost *= 0.0624;

    const double coeff[5] = {61.0, 37.0, 19.0, 7.0, 1.0};
    double g1 = -1.0;
    for (int i = 0; i < 5; ++i) g1 += coeff[i] / (xs[i] * xs[i] * xs[i]);

    auto pos = [](double v) { return v > 0.0 ? v : 0.0; };
    double penalty = std::pow(pos(g1), 2.0);

    double f = cost + w_pen_ * penalty;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void CantileverBeam::gradient_core(const Vec& x, Vec& g) {
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
