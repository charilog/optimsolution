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
    const int D = static_cast<int>(x.size());
    g.assign(D, 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < D; ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
