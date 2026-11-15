#include "schaffer.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Schaffer::Schaffer()
{
    setName("schaffer");
    setFullName("Schaffer N.2 (F6) function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

}

void Schaffer::init(int /*dim*/) {

    Problem::init(2);

    Vec lo(2, -100.0), hi(2, 100.0);
    setBounds(lo, hi);

    // Global optimum: (0,0) με f = 0
    Vec xopt(2, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Schaffer::evaluate_core(const Vec& x) {
    const double x1 = x[0];
    const double x2 = x[1];

    const double t   = x1 * x1 - x2 * x2;
    const double num = std::sin(t) * std::sin(t) - 0.5;
    const double den = std::pow(1.0 + 0.001 * (x1 * x1 + x2 * x2), 2.0);

    double f = 0.5 + num / den;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Schaffer::gradient_core(const Vec& x, Vec& g) {
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
